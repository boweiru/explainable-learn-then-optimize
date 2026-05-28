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
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import random
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
    def __init__(self):
        super().__init__(n_var=5,
                         n_obj=3,
                         n_constr=0,
                         xl=np.array(
                             [0.0 for _ in range(5)]),
                         xu=np.array([1.0 for _ in range(5)]))
        self.num_vars = 5
        self.names = ['x' + str(i + 1) for i in range(55)]
        self.obj = 3
        self.default_values = np.array(
            [0.73921768, 0.10898673, 0.16005066, 0.14704244, 0.5119668, 0.95435298, 0.80380099, 0.96993233, 0.8800116,
             0.83822508, 0.61731428, 0.9553677, 0.38357837, 0.39428249, 0.21374675, 0.17324961, 0.69438701, 0.60017853,
             0.03212, 0.23807707, 0.84235262, 0.77054421, 0.02141587, 0.84265017, 0.07335719, 0.38209062, 0.69638593,
             0.40550541, 0.06966453, 0.31102227, 0.30842063, 0.18020767, 0.47142388, 0.85900008, 0.48740759, 0.65456127,
             0.82388171, 0.65475963, 0.23720732, 0.74740408, 0.37484264, 0.84814337, 0.9433742, 0.22306231, 0.9753645,
             0.64568059, 0.01324473, 0.07003075, 0.4040253, 0.68442295, 0.44221834, 0.79183038, 0.53905135, 0.46450397,
             0.94827994])

        self.whether_to_optimize = np.load('whether_to_optimize-Natural-Breaks.npy')

    def _evaluate(self, x, out, *args, **kwargs):
        Result = Running.objective_function(x, self.default_values, self.whether_to_optimize)

        out["F"] = [Result[0], Result[1], Result[2]]

problem = MyProblem()


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        sobol_seq = Sobol(d=problem.n_var, seed=42)
        X = sobol_seq.random(n_samples)
        X = xl + X * (xu - xl)
        return X


repetition = 5
obj = 3

r = np.array([250, 200, 2000])
r_good = np.array([0, 0, -100])

for i in range(repetition):
    start_time = time.time()
    np.random.seed(20 + repetition)
    random.seed(30 + repetition)
    algorithm = NSGA2(pop_size=10, n_offsprings=10, sampling=MySampling(), eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 32),
                   save_history=True,
                   verbose=True)


    np.save('.\\Results\\Time-NSGA5-Aircraft-repetition-' + str(i) + '.npy', time.time() - start_time)

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

    np.save('.\\Results\\PF-NSGA5-Aircraft-repetition-' + str(i) + '.npy', Pareto_front)
    np.save('.\\Results\\PS-NSGA5-Aircraft-repetition-' + str(i) + '.npy', Pareto_solution)

    with open('All_PF-NSGA5-Aircraft-repetition-' + str(repetition) + '.pkl', 'wb') as f:
        pickle.dump(hist_PF, f)

    np.save('.\\Results\\RHD-NSGA5-Aircraft-repetition-' + str(i) + '.npy', np.array(rhd))



