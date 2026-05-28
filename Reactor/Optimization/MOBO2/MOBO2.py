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
        std_pred = var[0].item()
        scale_val = y_scalers[i].scale_[0]
        y_std[i] = std_pred * scale_val

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
    y_0 = y[0]
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


def Bayesian_optimization_Simulation_scenario(num_vars, bounds, initial_points,
                                              y_initial_points, defaults, whether_to_optimize, Ports, container_name, r,
                                              max_iteration, time_start):
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

        x = np.concatenate((x, x_new), axis=0)

        y_new = Running.objective_function(x_new, defaults, whether_to_optimize, Ports, container_name)

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

        with open('time-' + str(repetition) + '.pkl', 'wb') as f:
            pickle.dump(time_Bayesian_optimization, f)

    return Pareto_front, Pareto_solution, All_PF, time_Bayesian_optimization


if __name__ == '__main__':
    num_vars_all = 50

    names_all = ['x' + str(i + 1) for i in range(num_vars_all)]

    bounds_all = np.zeros((num_vars_all, 2))
    for i in range(num_vars_all):
        bounds_all[i, 1] = 1

    defaults = np.array([0.00537113, 0.01956192, 0.09735945, 0.88776312, 0.88714514, 0.57085091,
                         0.42610494, 0.02036301, 0.26047715, 0.75830657, 0.70496143, 0.83048882,
                         0.89518658, 0.11090935, 0.04907264, 0.79150232, 0.68261477, 0.96839118,
                         0.02058426, 0.96722387, 0.88073639, 0.00337222, 0.3072991, 0.41926895,
                         0.59300684, 0.15250513, 0.86574452, 0.0456165, 0.43630551, 0.6753973,
                         0.87134454, 0.00786597, 0.1656049, 0.71920562, 0.97830184, 0.87216089,
                         0.92657415, 0.4473377, 0.69654615, 0.74506947, 0.79340205, 0.75310328,
                         0.1619275, 0.87130639, 0.1532223, 0.95022545, 0.21889663, 0.02237718,
                         0.06472828, 0.41742262])

    whether_to_optimize = np.load('whether_to_optimize-Natural-Breaks.npy')
    num_vars = np.sum(whether_to_optimize == 1)
    names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
    bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

    obj = 2
    repetition = 5


    r = np.array([0, 1000])
    r_good = np.array([-200, 0])
    initial_size = 10
    max_iteration = 340

    for j in range(repetition):
        rhd = []
        time_start = time.time()
        Ports = 8001 + repetition
        container_name = 'MOBO2' + str(Ports)
        initial_points = sobol_seq.i4_sobol_generate(num_vars, initial_size)
        initial_points = bounds[:, 0] + initial_points * (bounds[:, 1] - bounds[:, 0])
        y_initial_points = np.zeros((initial_size, obj))

        for i in range(initial_size):
            y_initial_points[i, :] = Running.objective_function(initial_points[i, :], defaults, whether_to_optimize,
                                                                    Ports, container_name)

        Pareto_front, Pareto_solution, All_PF, t = Bayesian_optimization_Simulation_scenario(num_vars, bounds,
                                                                                             initial_points,
                                                                                             y_initial_points, defaults,
                                                                                             whether_to_optimize, Ports,
                                                                                             container_name, r,
                                                                                             max_iteration, time_start)
        for i in range(max_iteration + 1):
            rhd.append(Relative_Hypervolume(All_PF[i], r, r_good))
            print(i, rhd[-1])

        np.save('.\\Results\\PF-MOBO2-Reactor-repetition-' + str(j) + '.npy', Pareto_front)
        np.save('.\\Results\\PS-MOBO2-Reactor-repetition-' + str(j) + '.npy', Pareto_solution)
        with open('.\\Results\\All_PF-MOBO2-Reactor-repetition-' + str(j) + '.pkl', 'wb') as f:
            pickle.dump(All_PF, f)

        np.save('.\\Results\\RHD-MOBO2-Reactor-repetition-' + str(j) + '.npy', rhd)
        np.save('.\\Results\\Time-MOBO2-Reactor-repetition-' + str(j) + '.npy', np.array(t))
