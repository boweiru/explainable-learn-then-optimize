import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import sobol_seq
from scipy.optimize import minimize
from scipy.stats import norm
import Running
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import lhsmdu
import time
import random
from matplotlib import rcParams
from pymoo.problems.many import WFG1
from scipy.stats.qmc import Sobol
from scipy.optimize import differential_evolution
import pickle
from sklearn.preprocessing import StandardScaler
import os
from sko.GA import GA


def EIM(x, GP, Pareto_front, x_scaler, y_scalers):
    x_scaled = x_scaler.transform(x.reshape(1, -1))
    y_mean = np.zeros(Pareto_front.shape[1])
    y_std = np.zeros(Pareto_front.shape[1])
    for i in range(Pareto_front.shape[1]):
        mean, var = GP[i].predict(x_scaled, return_std=True)
        y_mean[i] = y_scalers[i].inverse_transform(mean.reshape(-1, 1)).ravel()[0]
        y_std[i] = var[0] * y_scalers[i].scale_

    eim = np.zeros(Pareto_front.shape)
    if np.sum(y_std) == 0:
        return eim
    else:
        for i in range(Pareto_front.shape[0]):
            for j in range(Pareto_front.shape[1]):
                eim[i, j] = (Pareto_front[i, j] - y_mean[j]) * norm.cdf((Pareto_front[i, j] - y_mean[j]) / y_std[j]) + \
                            y_std[j] * norm.pdf((Pareto_front[i, j] - y_mean[j]) / y_std[j])
    return eim


def EIM_h(x, GP, r, Pareto_front, x_scaler, y_scalers):
    eim = EIM(x, GP, Pareto_front, x_scaler, y_scalers)
    eim_h = []
    for i in range(eim.shape[0]):
        v1 = 1
        v2 = 1
        for j in range(eim.shape[1]):
            v1 *= r[j] + eim[i, j] - Pareto_front[i][j]
            v2 *= r[j] - Pareto_front[i][j]
        eim_h.append(v1 - v2)
    return -min(eim_h)


def Acquisition_opt(num_vars, bounds, GP, r, Pareto_front, x_scaler, y_scalers, ga_seed,
                    ga_pop=440,
                    ga_iter=150,
                    ga_mutation_prob=0.08,
                    ga_precision=1e-5,
                    ):
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    def objective(x):
        return EIM_h(
            np.array(x, dtype=float),
            GP, r, Pareto_front, x_scaler, y_scalers
        )

    np.random.seed(ga_seed)

    ga = GA(
        func=objective,
        n_dim=num_vars,
        size_pop=ga_pop,
        max_iter=ga_iter,
        prob_mut=ga_mutation_prob,
        lb=lb,
        ub=ub,
        precision=ga_precision
    )

    best_x, best_y = ga.run()

    return np.array(best_x, dtype=float)


def fast_non_dominated_sort(y):
    y_0 = y[0]  # 目标函数1
    Dominant_solution = [[] for i in range(len(y_0))]
    Pareto_front_index = []
    num_of_Dominated_solutions = np.zeros(len(y_0))

    for i in range(len(y_0)):
        for j in range(len(y_0)):  # 遍历每一个个体
            if j == i:
                continue
            less = 0
            equal = 0
            greater = 0
            for k in range(len(y)):
                if y[k][i] > y[k][j]:
                    less = less + 1
                if y[k][i] == y[k][j]:
                    equal = equal + 1
                if y[k][i] < y[k][j]:
                    greater = greater + 1

            if (less + equal == len(y)) and (equal != len(y)):
                num_of_Dominated_solutions[i] = num_of_Dominated_solutions[i] + 1
            elif (greater + equal == len(y)) and (equal != len(y)):
                Dominant_solution[i].append(j)

        if num_of_Dominated_solutions[i] == 0:
            if i not in Pareto_front_index:
                Pareto_front_index.append(i)

    return Pareto_front_index


def Relative_Hypervolume(Pareto_front, r, r_good):
    N_sim = 10000
    n = 0
    rdm = RandomState(42)
    for i in range(N_sim):
        x_sim = [rdm.uniform(r_good[i], r[i]) for i in range(len(r))]
        for j in range(Pareto_front.shape[0]):
            flag = True
            for k in range(len(x_sim)):
                if x_sim[k] < Pareto_front[j, k]:
                    flag = False
                    break
            if flag:
                n += 1
                break
    return 1 - (n / N_sim)


def Bayesian_optimization_Simulation_scenario(num_vars, names, bounds, initial_points,
                                              y_initial_points, simulation_path, obs_path,
                                               start_time, end_time,
                                               signal, obs_delta_time, true_flow,
                                               true_time, true_speed, true_queue, true_timeLoss, r, r_good, max_iteration, time_start):
    time_Bayesian_optimization = []
    x = initial_points
    y = y_initial_points
    obj = y.shape[1]
    All_PF = []
    Pareto_front_index = fast_non_dominated_sort(y.T)
    Pareto_front = np.zeros((len(Pareto_front_index), y.shape[1]))
    Pareto_solution = np.zeros((len(Pareto_front_index), x.shape[1]))
    for i in range(len(Pareto_front_index)):
        Pareto_solution[i, :] = x[Pareto_front_index[i], :]
        Pareto_front[i, :] = y[Pareto_front_index[i], :]
    All_PF.append(Pareto_front)

    for i in range(max_iteration):
        print('iteration ' + str(i))
        x_scaler = StandardScaler()
        y_scalers = [StandardScaler() for _ in range(obj)]
        x_scaled = x_scaler.fit_transform(x)
        GP = []
        for j in range(obj):
            y_j_scaled = y_scalers[j].fit_transform(y[:, j].reshape(-1, 1))
            gpr = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), optimizer="fmin_l_bfgs_b",
                                           n_restarts_optimizer=10)
            gpr.fit(x_scaled, y_j_scaled)
            GP.append(gpr)
        x_new = Acquisition_opt(num_vars, bounds, GP, r, Pareto_front, x_scaler, y_scalers, 42 + repetition)
        print('The new point is ', x_new)
        x_new = x_new.reshape((1, len(x_new)))
        for j in range(obj):
            print(y_scalers[j].inverse_transform(GP[j].predict(x_scaler.transform(x_new.reshape(1, -1))).reshape(-1, 1)))

        x = np.concatenate((x, x_new), axis=0)

        y_new = Running.objective_function(num_vars, names, x_new, simulation_path, obs_path,
                                               start_time, end_time,
                                               signal, obs_delta_time, true_flow,
                                               true_time, true_speed, true_queue, true_timeLoss, obj)
        print('The objective function is', y_new)
        y_new = y_new.reshape((1, obj))
        y = np.concatenate((y, y_new), axis=0)

        Pareto_front = np.concatenate((Pareto_front, y_new), axis=0)
        Pareto_solution = np.concatenate((Pareto_solution, x_new), axis=0)

        Pareto_front_index = fast_non_dominated_sort(Pareto_front.T)
        PF = np.zeros((len(Pareto_front_index), y.shape[1]))
        PS = np.zeros((len(Pareto_front_index), num_vars))
        for j in range(len(Pareto_front_index)):
            PS[j, :] = Pareto_solution[Pareto_front_index[j], :]
            PF[j, :] = Pareto_front[Pareto_front_index[j], :]

        Pareto_front = PF
        Pareto_solution = PS
        All_PF.append(Pareto_front)

        time_Bayesian_optimization.append(time.time() - time_start)
    return Pareto_front, Pareto_solution, All_PF, time_Bayesian_optimization


if __name__ == '__main__':
    num_vars_all = 29

    names_all = []
    names_all.append('speedFactor')
    names_all.append('speedDev')
    names_all.append('minGap')
    names_all.append('accel')
    names_all.append('decel')
    names_all.append('tau')
    names_all.append('delta')
    names_all.append('emergencyDecel')
    names_all.append('lcStrategic')
    names_all.append('lcCooperative')
    names_all.append('lcSpeedGain')
    names_all.append('lcKeepRight')
    names_all.append('lcOvertakeRight')
    names_all.append('lcOpposite')
    names_all.append('lcLookaheadLeft')
    names_all.append('lcSpeedGainRight')
    names_all.append('lcSpeedGainLookahead')
    names_all.append('lcOvertakeDeltaSpeedFactor')
    names_all.append('lcKeepRightAcceptanceTime')
    names_all.append('lcCooperativeSpeed')
    names_all.append('lcAssertive')
    names_all.append('jmCrossingGap')
    names_all.append('jmIgnoreFoeProb')
    names_all.append('jmIgnoreFoeSpeed')
    names_all.append('jmIgnoreJunctionFoeProb')
    names_all.append('jmSigmaMinor')
    names_all.append('jmStoplineGap')
    names_all.append('jmTimegapMinor')
    names_all.append('impatience')

    bounds_all = [[0.2, 2],
                  [0, 1],
                  [1, 10],
                  [1, 3],
                  [1, 6],
                  [1e-09, 3],
                  [0.1, 10],
                  [6, 10],
                  [0, 10],
                  [0, 1],
                  [0, 10],
                  [0, 10],
                  [0, 1],
                  [0, 10],
                  [0, 10],
                  [0, 10],
                  [0, 10],
                  [-1, 1],
                  [-1, 10],
                  [0, 1],
                  [1e-09, 10],
                  [0, 30],
                  [0, 1],
                  [0, 20 / 3.6],
                  [0, 1],
                  [0, 1],
                  [0, 6],
                  [0, 3],
                  [0, 10]]
    bounds_all = np.array(bounds_all)

    default_values = [1.0, 0.1, 2.5, 2.6, 4.5, 1.0, 4, 9, 1.0, 1.0, 1.0, 1.0, 0, 1.0, 2.0, 0.1, 0, 0, -1, 1.0, 1.0, 10,
                      0, 0, 0, 0.5, 1, 1, 0.0]
    default_values = np.array(default_values)

    # 读取 Excel 文件
    Training_set = pd.read_excel('Training_set.xlsx', header=None)
    y_Training_set = pd.read_excel('y_Training_set.xlsx', header=None)

    # 将DataFrame转换为数组
    Training_set = Training_set.values
    y_Training_set = y_Training_set.values

    whether_to_optimize = np.ones(num_vars_all)
    num_vars = np.sum(whether_to_optimize == 1)
    names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
    bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

    obj = 4

    simulation_path = os.path.dirname(os.path.abspath(__file__))
    obs_path = os.path.dirname(os.path.abspath(__file__))
    start_time = 0
    end_time = 900
    duration = end_time - start_time
    obs_delta_time = 185
    signal_df = pd.read_csv('IntersectionA-Signal.csv')
    signal = {'startTime': list(signal_df['startTime']), 'eventDuration': list(signal_df['eventDuration']),
              'index': list(signal_df['index'])}

    true_flow, true_time, true_speed, true_queue, true_timeLoss = Running.get_True_data(obs_path)

    r = np.array([10 for _ in range(obj)])
    r_good = np.zeros(obj)
    initial_size = 70
    max_iteration = 130

    repetition = 10

    for j in range(repetition):
        rhd = []
        time_start = time.time()
        initial_points = Training_set[:initial_size, :]
        y_initial_points = y_Training_set[:initial_size, :]

        Pareto_front, Pareto_solution, All_PF, t = Bayesian_optimization_Simulation_scenario(num_vars, names, bounds, initial_points,
                                                  y_initial_points, simulation_path, obs_path,
                                                   start_time, end_time,
                                                   signal, obs_delta_time, true_flow,
                                                   true_time, true_speed, true_queue, true_timeLoss, r, r_good, max_iteration, time_start)
        for i in range(max_iteration + 1):
            rhd.append(Relative_Hypervolume(All_PF[i], r, r_good))
            print(i, rhd[-1])

        np.save('.\\Results\\PF-MOBO29-Junction-repetition-' + str(j) + '.npy', Pareto_front)
        np.save('.\\Results\\PS-MOBO29-Junction-repetition-' + str(j) + '.npy', Pareto_solution)
        with open('.\\Results\\All_PF-MOBO29-Junction-repetition-' + str(j) + '.pkl', 'wb') as f:
            pickle.dump(All_PF, f)

        np.save('.\\Results\\RHD-MOBO29-Junction-repetition-' + str(j) + '.npy', rhd)
        np.save('.\\Results\\Time-MOBO29-Junction-repetition-' + str(j) + '.npy', np.array(t))


