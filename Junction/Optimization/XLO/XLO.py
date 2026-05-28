import numpy as np
import pandas as pd
from MSG import BaggingMeta
from numpy.random import RandomState
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import sobol_seq
from scipy.optimize import minimize
from scipy.stats import norm
import Running
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import lhsmdu
import Running
import time
import random
from matplotlib import rcParams
import joblib
import pickle
import os
from sko.GA import GA


def EIM(x_new, whether_to_optimize, default_values, Model, Pareto_front):
    j = 0
    for i in range(len(whether_to_optimize)):
        if whether_to_optimize[i] > 0:
            default_values[i] = x_new[j]
            j += 1
    x_new = default_values.reshape((1, len(default_values)))
    y_mean, y_std = Model.predict(x_new)
    y_mean = y_mean.reshape(Pareto_front.shape[1])
    y_std = y_std.reshape(Pareto_front.shape[1])

    eim = np.zeros(Pareto_front.shape)
    if np.sum(y_std) == 0:
        return eim
    else:
        for i in range(Pareto_front.shape[0]):
            for j in range(Pareto_front.shape[1]):
                eim[i, j] = (Pareto_front[i, j] - y_mean[j]) * norm.cdf((Pareto_front[i, j] - y_mean[j]) / y_std[j]) + \
                            y_std[j] * norm.pdf((Pareto_front[i, j] - y_mean[j]) / y_std[j])
    return eim


def EIM_h(x_new, whether_to_optimize, default_values, Model, r, Pareto_front):
    eim = EIM(x_new, whether_to_optimize, default_values, Model, Pareto_front)
    eim_h = []
    for i in range(eim.shape[0]):
        v1 = 1
        v2 = 1
        for j in range(eim.shape[1]):
            v1 *= r[j] + eim[i, j] - Pareto_front[i][j]
            v2 *= r[j] - Pareto_front[i][j]
        eim_h.append(v1 - v2)
    return -min(eim_h)


def Acquisition_opt(num_vars, whether_to_optimize, default_values, bounds, Model, r, Pareto_front, ga_seed,
                    ga_pop=60,
                    ga_iter=80,
                    ga_mutation_prob=0.12,
                    ga_precision=1e-6,
                    ):
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    def objective(x):
        return EIM_h(
            np.array(x, dtype=float),
            whether_to_optimize,
            default_values.copy(),
            Model,
            r,
            Pareto_front
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
    y_0 = y[0]
    Pareto_front_index = set()
    num_of_Dominated_solutions = np.zeros(len(y_0))

    for i in range(len(y_0)):
        for j in range(len(y_0)):
            if j == i:
                continue
            less = 0
            equal = 0
            greater = 0
            for k in range(len(y)):
                if y[k][i] > y[k][j]:
                    less = less + 1
                elif y[k][i] == y[k][j]:
                    equal = equal + 1
                elif y[k][i] < y[k][j]:
                    greater = greater + 1

            if (less + equal == len(y)) and (equal != len(y)):
                num_of_Dominated_solutions[i] = num_of_Dominated_solutions[i] + 1

        if num_of_Dominated_solutions[i] == 0:
            Pareto_front_index.add(i)

    return list(Pareto_front_index)


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


def Bayesian_optimization_Simulation_scenario(num_vars, names, bounds, num_vars_all, Training_set,
                                              y_Training_set, default_values, Model, r,
                                              start_time,
                                              end_time, signal,
                                              obs_delta_time, true_flow, true_time, true_speed, true_queue,
                                              true_timeLoss, time_start, seed):
    time_Bayesian_optimization = []
    x = Training_set
    y = y_Training_set
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
        if i > 0:
            Model = BaggingMeta(obj)
            Model.fit(x, y)
        part_x_new = Acquisition_opt(num_vars, whether_to_optimize, default_values, bounds, Model, r, Pareto_front,
                                     42 + seed)
        print('The part new point is ', part_x_new)
        part_x_new = part_x_new.reshape((1, len(part_x_new)))

        x_new = default_values
        j = 0
        for k in range(len(whether_to_optimize)):
            if whether_to_optimize[k] > 0:
                x_new[k] = part_x_new[0, j]
                j += 1
        x_new = x_new.reshape((1, len(default_values)))
        print('The new point is ', x_new)

        x = np.concatenate((x, x_new), axis=0)

        y_new = Running.objective_function(num_vars, names, part_x_new, simulation_path, obs_path,
                                               start_time, end_time,
                                               signal, obs_delta_time, true_flow,
                                               true_time, true_speed, true_queue, true_timeLoss, obj)
        print('The objective function is', y_new)
        y_new = np.array(y_new)
        y_new = y_new.reshape((1, len(y_new)))
        y = np.concatenate((y, y_new), axis=0)

        Pareto_front = np.concatenate((Pareto_front, y_new), axis=0)
        Pareto_solution = np.concatenate((Pareto_solution, x_new), axis=0)

        Pareto_front_index = fast_non_dominated_sort(Pareto_front.T)
        PF = np.zeros((len(Pareto_front_index), y.shape[1]))
        PS = np.zeros((len(Pareto_front_index), num_vars_all))
        for j in range(len(Pareto_front_index)):
            PS[j, :] = Pareto_solution[Pareto_front_index[j], :]
            PF[j, :] = Pareto_front[Pareto_front_index[j], :]

        Pareto_front = PF
        Pareto_solution = PS
        All_PF.append(Pareto_front)

        time_Bayesian_optimization.append(time.time() - time_start)

    return Pareto_front, Pareto_solution, All_PF, time_Bayesian_optimization


if __name__ == "__main__":
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

    Training_set = Training_set[:60, :]
    y_Training_set = y_Training_set[:60, :]

    obj = y_Training_set.shape[1]

    Model = joblib.load('MSG.pkl')

    whether_to_optimize = np.load('whether_to_optimize-Natural-Breaks.npy')

    num_vars = np.sum(whether_to_optimize == 1)
    names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
    bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

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
    max_iteration = 140

    repetition = 10

    for j in range(repetition):
        rhd = []
        time_start = time.time()
        Pareto_front, Pareto_solution, All_PF, t = Bayesian_optimization_Simulation_scenario(num_vars, names, bounds,
                                                                                             num_vars_all, Training_set,
                                                                                             y_Training_set, default_values,
                                                                                             Model, r,
                                                                                             start_time,
                                                                                             end_time, signal,
                                                                                             obs_delta_time, true_flow,
                                                                                             true_time, true_speed,
                                                                                             true_queue,
                                                                                             true_timeLoss, time_start, repetition)
        for i in range(max_iteration + 1):
            rhd.append(Relative_Hypervolume(All_PF[i], r, r_good))
            print(i, rhd[-1])

        np.save('.\\Results\\PF-XLO-Junction-repetition-' + str(j) + '.npy', Pareto_front)
        np.save('.\\Results\\PS-XLO-Junction-repetition-' + str(j) + '.npy', Pareto_solution)
        with open('.\\Results\\All_PF-XLO-Junction-repetition-' + str(j) + '.pkl', 'wb') as f:
            pickle.dump(All_PF, f)

        np.save('.\\Results\\RHD-XLO-Junction-repetition-' + str(j) + '.npy', rhd)
        np.save('.\\Results\\Time-XLO-Junction-repetition-' + str(j) + '.npy', np.array(t))

