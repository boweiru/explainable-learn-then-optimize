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
    def __init__(self):
        super().__init__(n_var=5,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([
                             1.0,  # sigma2_i
                             0.0,  # sigma2_0
                             10000.0,  # alpha
                             0.0,  # Q_D
                             0.0,  # log2_over_omega
                         ]),
                         xu=np.array([
                             75.0,  # sigma2_i
                             26.0,  # sigma2_0
                             200000.0,  # alpha
                             1.0,  # Q_D
                             4.0,  # log2_over_omega
                         ]))
        self.num_vars_all = 5
        self.names_all = [
            "sigma2_i", "sigma2_0", 'alpha', 'Q_D', "log2_over_omega"
        ]

        self.obj = 2
        col_field = ['Scen', 'Survey', 'Age', 'Type', 'Val']
        self.field_data = pd.read_csv('fieldData.txt', sep=r"\s+", header=None, names=col_field)
        self.field_data = self.field_data[self.field_data['Scen'] == 28]
        self.field_data = self.field_data.drop(columns='Scen')
        self.field_data = self.field_data.reset_index(drop=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        drive_letter = script_dir[0].lower()
        path_rest = script_dir[2:].replace("\\", "/")

        self.cygwin_path = f'/cygdrive/{drive_letter}{path_rest}'

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = Running.objective_function(self.num_vars_all, self.names_all, x, self.field_data, self.cygwin_path)

problem = MyProblem()


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        sobol_seq = Sobol(d=problem.n_var, seed=42)
        X = sobol_seq.random(n_samples)
        X = xl + X * (xu - xl)
        return X


repetition = 0
obj = 2

r = np.array([3000, 300])
r_good = np.zeros(obj)

RHD = []
Time = []

for j in range(10):
    start_time = time.time()
    np.random.seed(20 + repetition)
    random.seed(30 + repetition)
    algorithm = NSGA2(pop_size=10, n_offsprings=10, sampling=MySampling(), eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 16),
                   save_history=True,
                   verbose=True)

    np.save('.\\Results\\Time-NSGA5-malaria-repetition-' + str(j) + '.npy', time.time() - start_time)

    hist = res.history
    hist_PF = []
    for algo in hist:
        opt = algo.opt
        hist_PF.append(opt.get("F"))

    rhd = []
    for j in range(len(hist_PF)):
        rhd.append(Relative_Hypervolume(hist_PF[j], r, r_good))
        print(j, rhd[-1])
    RHD.append(rhd)

    Pareto_front = np.array(res.F)
    Pareto_solution = np.array(res.X)

    np.save('.\\Results\\PF-NSGA5-Malaria-repetition-' + str(j) + '.npy', Pareto_front)
    np.save('.\\Results\\PS-NSGA5-Malaria-repetition-' + str(j) + '.npy', Pareto_solution)

    np.save('.\\Results\\RHD-NSGA5-Malaria-repetition-' + str(j) + '.npy', np.array(rhd))

    with open('.\\Results\\All_PF-NSGA5-Malaria-repetition-' + str(j) + '.pkl', 'wb') as f:
        pickle.dump(hist_PF, f)







