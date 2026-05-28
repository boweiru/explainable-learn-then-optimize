import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from numpy.random import RandomState
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from pymoo.problems.many import WFG3
from scipy.stats.qmc import Sobol
import pickle
import pandas as pd
import time


def EIM(x, GP, Pareto_front):
    y_mean = np.zeros(Pareto_front.shape[1])
    y_std = np.zeros(Pareto_front.shape[1])
    for i in range(Pareto_front.shape[1]):
        mean, var = GP[i].predict(x.reshape(1, len(x)), return_std=True)
        y_mean[i] = mean[0]
        y_std[i] = var[0]

    eim = np.zeros(Pareto_front.shape)
    if np.sum(y_std) == 0:
        return eim
    else:
        for i in range(Pareto_front.shape[0]):
            for j in range(Pareto_front.shape[1]):
                eim[i, j] = (Pareto_front[i, j] - y_mean[j]) * norm.cdf((Pareto_front[i, j] - y_mean[j]) / y_std[j]) + \
                            y_std[j] * norm.pdf((Pareto_front[i, j] - y_mean[j]) / y_std[j])
    return eim


def EIM_h(x, GP, r, Pareto_front):
    eim = EIM(x, GP, Pareto_front)
    eim_h = []
    for i in range(eim.shape[0]):
        v1 = 1
        v2 = 1
        for j in range(eim.shape[1]):
            v1 *= r[j] + eim[i, j] - Pareto_front[i][j]
            v2 *= r[j] - Pareto_front[i][j]
        eim_h.append(v1 - v2)
    return -min(eim_h)


def multi_start_opt(num_vars, bounds, GP, r, Pareto_front, multi_start=10):
    x_start = np.random.rand(multi_start, num_vars)
    for i in range(num_vars):
        x_start[:, i] = bounds[i][0] + (bounds[i][1] - bounds[i][0]) * x_start[:, i]

    min_negative_EIM = 1e09
    for i in range(multi_start):
        res = minimize(EIM_h, x_start[i, :], (GP, r, Pareto_front), bounds=bounds, method='L-BFGS-B')

        x_Local = res.x
        y_Local = res.fun

        if y_Local < min_negative_EIM:
            min_negative_EIM = y_Local
            x_new = x_Local

    return x_new

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


def Bayesian_optimization_Numerical_experiments(num_vars, bounds, initial_points, y_initial_points, problem,
                                                default_values, r, max_iteration, time_start):
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
        GP = []
        for j in range(obj):
            gpr = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), optimizer="fmin_l_bfgs_b",
                                           n_restarts_optimizer=10)
            gpr.fit(x, y[:, j])
            GP.append(gpr)
        x_new = multi_start_opt(num_vars, bounds, GP, r, Pareto_front)
        print('The new point is ', x_new)
        x_new = x_new.reshape((1, len(x_new)))
        x = np.concatenate((x, x_new), axis=0)

        all_x_new = default_values
        j = 0
        for k in range(len(whether_to_optimize)):
            if whether_to_optimize[k] > 0:
                all_x_new[k] = x_new[0, j]
                j += 1

        print('The all new point is ', all_x_new)

        y_new = problem.evaluate(all_x_new)
        print('The objective function is', y_new)
        y_new = y_new.reshape((1, len(y_new)))
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
    obj = 5
    num_vars_all = 100
    k = 4

    problem = WFG3(n_var=num_vars_all, n_obj=obj, k=k)

    names_all = ['x' + str(i + 1) for i in range(num_vars_all)]
    names_all = np.array(names_all)

    bounds_all = np.zeros((num_vars_all, 2))
    for i in range(num_vars_all):
        bounds_all[i, 1] = 2 * (i + 1)

    default_values = np.array([float(2 * (i + 1) * 0.35) for i in range(num_vars_all)])

    Training_set = pd.read_excel('Training_set.xlsx', header=None)
    y_Training_set = pd.read_excel('y_Training_set.xlsx', header=None)

    Training_set = Training_set.values
    y_Training_set = y_Training_set.values

    whether_to_optimize = np.ones(num_vars_all)
    num_vars = np.sum(whether_to_optimize == 1)
    names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
    bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

    r = np.array([20 for _ in range(obj)])
    r_good = np.zeros(obj)
    initial_size = 410
    max_iteration = 590

    repetition = 10
    for j in range(repetition):
        rhd = []
        time_start = time.time()
        initial_points = Training_set[:initial_size, :]
        y_initial_points = y_Training_set[:initial_size, :]

        Pareto_front, Pareto_solution, All_PF, t = Bayesian_optimization_Numerical_experiments(num_vars, bounds,
                                                                                               initial_points,
                                                                                               y_initial_points,
                                                                                               problem,
                                                                                               default_values, r,
                                                                                               max_iteration,
                                                                                               time_start)

        for i in range(max_iteration + 1):
            rhd.append(Relative_Hypervolume(All_PF[i], r, r_good))
            print(i, rhd[-1])

        np.save('.\\Results\\PF-MOBO100-WFG3-repetition-' + str(j) + '.npy', Pareto_front)
        np.save('.\\Results\\PS-MOBO100-WFG3-repetition-' + str(j) + '.npy', Pareto_solution)

        np.save('.\\Results\\RHD-MOBO100-WFG3-repetition-' + str(j) + '.npy', rhd)
        np.save('.\\Results\\Time-MOBO100-WFG3-repetition-' + str(j) + '.npy', np.array(t))