import time
from numpy.random import RandomState
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
import os, sys
import numpy as np
import Running
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pymoo.core.sampling import Sampling
import sobol_seq
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import random
import lhsmdu
from scipy.stats.qmc import Sobol
import pickle


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

class MyProblem(ElementwiseProblem):
    def __init__(self, defaults, whether_to_optimize, Ports, container_name):
        super().__init__(n_var=50,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([
                             0 for i in range(50)
                         ]),
                         xu=np.array([
                             1 for i in range(50)
                         ]))
        self.num_vars_all = 50
        self.names_all = ['x' + str(i + 1) for i in range(self.num_vars_all)]
        self.obj = 2
        self.defaults = defaults
        self.whether_to_optimize = whether_to_optimize
        self.Ports = Ports
        self.container_name = container_name

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = Running.objective_function(x, self.defaults, self.whether_to_optimize, self.Ports, self.container_name)

class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        sobol_seq = Sobol(d=problem.n_var, seed=42)
        X = sobol_seq.random(n_samples)
        X = xl + X * (xu - xl)
        return X

repetition = 5

num_vars_all = 50
obj = 2

Ports = 5001 + repetition
container_name = 'NSGA50' + str(Ports)
bounds_all = np.zeros((50, 2))
for i in range(50):
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
whether_to_optimize = np.ones(num_vars_all)
num_vars = np.sum(whether_to_optimize == 1)
bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]
problem = MyProblem(defaults, whether_to_optimize, Ports, container_name)

r = np.array([0, 1000])
r_good = np.array([-200, 0])

for i in range(repetition):
    start_time = time.time()
    np.random.seed(20 + i)
    random.seed(30 + i)
    algorithm = NSGA2(pop_size=160, n_offsprings=10, sampling=MySampling(), eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 35),
                   save_history=True,
                   verbose=True)

    np.save('.\\Results\\Time-NSGA50-Reactor-repetition-' + str(i) + '.npy', time.time() - start_time)

    hist = res.history
    hist_PF = []
    for algo in hist:
        opt = algo.opt
        hist_PF.append(opt.get("F"))

    rhd = []
    for j in range(len(hist_PF)):
        rhd.append(Relative_Hypervolume(hist_PF[j], r, r_good))
        print(j, rhd[-1])

    Pareto_front = np.array(res.F)
    Pareto_solution = np.array(res.X)

    np.save('.\\Results\\PF-NSGA50-Reactor-repetition-' + str(i) + '.npy', Pareto_front)
    np.save('.\\Results\\PS-NSGA50-Reactor-repetition-' + str(i) + '.npy', Pareto_solution)

    np.save('.\\Results\\RHD-NSGA50-Reactor-repetition-' + str(i) + '.npy', np.array(rhd))

