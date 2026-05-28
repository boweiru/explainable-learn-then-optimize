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
                    ga_pop=400,
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

    print("Best GA objective (EIM_h):", best_y)

    return np.array(best_x, dtype=float)


def fast_non_dominated_sort(y):
    y_0 = y[0]
    Dominant_solution = [[] for i in range(len(y_0))]
    Pareto_front_index = []
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
                                              y_initial_points, cygwin_path, field_data, r, max_iteration, time_start):
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
    print(Relative_Hypervolume(All_PF[-1], r, r_good))

    for i in range(max_iteration):
        print('iteration ' + str(i))
        x_scaler = StandardScaler()
        y_scalers = [StandardScaler() for _ in range(obj)]
        x_scaled = x_scaler.fit_transform(x)
        GP = []
        for j in range(obj):
            y_j_scaled = y_scalers[j].fit_transform(y[:, j].reshape(-1, 1))
            gpr = GaussianProcessRegressor(kernel=Matern(nu=1.5), optimizer="fmin_l_bfgs_b",
                                           n_restarts_optimizer=10)
            gpr.fit(x_scaled, y_j_scaled)
            GP.append(gpr)
        x_new = Acquisition_opt(num_vars, bounds, GP, r, Pareto_front, x_scaler, y_scalers, 42 + repetition)
        print('The new point is ', x_new)
        x_new = x_new.reshape((1, len(x_new)))
        for j in range(obj):
            print(y_scalers[j].inverse_transform(GP[j].predict(x_scaler.transform(x_new.reshape(1, -1))).reshape(-1, 1)))

        x = np.concatenate((x, x_new), axis=0)

        y_new = Running.objective_function(num_vars, names, x_new, field_data, cygwin_path)
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
        print(Relative_Hypervolume(All_PF[-1], r, r_good))

        time_Bayesian_optimization.append(time.time() - time_start)

    return Pareto_front, Pareto_solution, All_PF, time_Bayesian_optimization


if __name__ == '__main__':
    num_vars_all = 23
    names_all = [
        "Simm", "Xp", "Yp", "sigma2_i", "Xy_star", "Xh_star",
        "ln_1_minus_alpha_m", "alpha_m_decay", "sigma2_0", "Xv_star",
        "Y2_star", "alpha", "v1", "log_phi1", "Q_D", "Q_n",
        "v0", "YB1_star", "F0", "log2_over_omega",
        "Y1_star", "Y0_star", "alpha_F_star",
    ]
    bounds_all = [
        [0, 6],  # Simm
        [0, 10000],  # Xp
        [0, 14],  # Yp
        [1, 75],  # sigma2_i
        [0, 1000],  # Xy_star
        [0, 200],  # Xh_star
        [0, 5],  # ln_1_minus_alpha_m
        [0, 11],  # alpha_m_decay
        [0, 26],  # sigma2_0
        [0, 66],  # Xv_star
        [0, 20000],  # Y2_star
        [10000, 200000],  # alpha
        [0, 2],  # v1
        [0, 9],  # log_phi1
        [0, 1],  # Q_D
        [3, 651],  # Q_n
        [1, 16],  # v0
        [20000, 1600000],  # YB1_star
        [0, 1],  # F0
        [0, 4],  # log2_over_omega
        [0, 10],  # Y1_star
        [0, 600],  # Y0_star
        [0, 3],  # alpha_F_star
    ]
    bounds_all = np.array(bounds_all)

    default_values = [
        0.196,  # Simm
        1954.8,  # Xp
        1.291,  # Yp
        11.729,  # sigma2_i
        593.661,  # Xy_star
        54.082,  # Xh_star
        1.770,  # ln_1_minus_alpha_m
        1.279,  # alpha_m_decay
        5.838,  # sigma2_0
        3.959,  # Xv_star
        6560.08,  # Y2_star
        63220.5,  # alpha
        0.123,  # v1
        0.340,  # log_phi1
        0.019,  # Q_D
        46.5095,  # Q_n
        3.739,  # v0
        849046,  # YB1_star
        0.078,  # F0
        0.468,  # log2_over_omega
        1.665,  # Y1_star
        90.938,  # Y0_star
        0.138,  # alpha_F_star
    ]
    default_values = np.array(default_values).astype(float)

    col_field = ['Scen', 'Survey', 'Age', 'Type', 'Val']
    field_data = pd.read_csv('fieldData.txt', sep=r"\s+", header=None, names=col_field)
    field_data = field_data[field_data['Scen'] == 28]
    field_data = field_data.drop(columns='Scen')
    field_data = field_data.reset_index(drop=True)

    Training_set = pd.read_excel('Training_set.xlsx', header=None)
    y_Training_set = pd.read_excel('y_Training_set.xlsx', header=None)

    Training_set = Training_set.values
    y_Training_set = y_Training_set.values

    whether_to_optimize = np.ones(num_vars_all)
    num_vars = np.sum(whether_to_optimize == 1)
    names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
    bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 替换为 Cygwin 路径格式
    drive_letter = script_dir[0].lower()
    path_rest = script_dir[2:].replace("\\", "/")

    cygwin_path = f'/cygdrive/{drive_letter}{path_rest}'

    obj = 2

    repetition = 10

    r = np.array([3000, 300])
    r_good = np.zeros(obj)
    initial_size = 50
    max_iteration = 150

    for j in range(repetition):
        rhd = []
        time_start = time.time()
        initial_points = Training_set[:initial_size, :]
        y_initial_points = y_Training_set[:initial_size, :]

        Pareto_front, Pareto_solution, All_PF, t = Bayesian_optimization_Simulation_scenario(num_vars, names, bounds,
                                                                                             initial_points,
                                                                                             y_initial_points,
                                                                                             cygwin_path, field_data, r,
                                                                                             max_iteration, time_start)
        for i in range(max_iteration + 1):
            rhd.append(Relative_Hypervolume(All_PF[i], r, r_good))
            print(i, rhd[-1])

        np.save('.\\Results\\PF-MOBO23-Malaria-repetition-' + str(j) + '.npy', Pareto_front)
        np.save('.\\Results\\PS-MOBO23-Malaria-repetition-' + str(j) + '.npy', Pareto_solution)
        with open('.\\Results\\All_PF-MOBO23-Malaria-repetition-' + str(j) + '.pkl', 'wb') as f:
            pickle.dump(All_PF, f)

        np.save('.\\Results\\RHD-MOBO23-Malaria-repetition-' + str(j) + '.npy', rhd)
        np.save('.\\Results\\Time-MOBO23-Malaria-repetition-' + str(j) + '.npy', np.array(t))
