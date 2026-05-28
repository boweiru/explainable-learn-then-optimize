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
        super().__init__(n_var=23,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([
                             0.0,  # Simm
                             0.0,  # Xp
                             0.0,  # Yp
                             1.0,  # sigma2_i
                             0.0,  # Xy_star
                             0.0,  # Xh_star
                             0.0,  # ln_1_minus_alpha_m
                             0.0,  # alpha_m_decay
                             0.0,  # sigma2_0
                             0.0,  # Xv_star
                             0.0,  # Y2_star
                             10000.0,  # alpha
                             0.0,  # v1
                             0.0,  # log_phi1
                             0.0,  # Q_D
                             3.0,  # Q_n
                             1.0,  # v0
                             20000.0,  # YB1_star
                             0.0,  # F0
                             0.0,  # log2_over_omega
                             0.0,  # Y1_star
                             0.0,  # Y0_star
                             0.0  # alpha_F_star
                         ]),
                         xu=np.array([
                             6.0,  # Simm
                             10000.0,  # Xp
                             14.0,  # Yp
                             75.0,  # sigma2_i
                             1000.0,  # Xy_star
                             200.0,  # Xh_star
                             5.0,  # ln_1_minus_alpha_m
                             11.0,  # alpha_m_decay
                             26.0,  # sigma2_0
                             66.0,  # Xv_star
                             20000.0,  # Y2_star
                             200000.0,  # alpha
                             2.0,  # v1
                             9.0,  # log_phi1
                             1.0,  # Q_D
                             651.0,  # Q_n
                             16.0,  # v0
                             1600000.0,  # YB1_star
                             1.0,  # F0
                             4.0,  # log2_over_omega
                             10.0,  # Y1_star
                             600.0,  # Y0_star
                             3.0  # alpha_F_star
                         ]))
        self.num_vars_all = 23
        self.names_all = [
            "Simm", "Xp", "Yp", "sigma2_i", "Xy_star", "Xh_star",
            "ln_1_minus_alpha_m", "alpha_m_decay", "sigma2_0", "Xv_star",
            "Y2_star", "alpha", "v1", "log_phi1", "Q_D", "Q_n",
            "v0", "YB1_star", "F0", "log2_over_omega",
            "Y1_star", "Y0_star", "alpha_F_star",
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

        # 拼接成 Cygwin 路径
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


obj = 2

r = np.array([3000, 300])
r_good = np.zeros(obj)

for j in range(10):
    start_time = time.time()
    np.random.seed(20 + j)
    random.seed(30 + j)
    algorithm = NSGA2(pop_size=50, n_offsprings=10, sampling=MySampling(), eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 16),
                   save_history=True,
                   verbose=True)

    np.save('.\\Results\\Time-NSGA23-malaria-repetition-' + str(j) + '.npy', time.time() - start_time)

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

    np.save('.\\Results\\PF-NSGA23-Malaria-repetition-' + str(j) + '.npy', Pareto_front)
    np.save('.\\Results\\PS-NSGA23-Malaria-repetition-' + str(j) + '.npy', Pareto_solution)

    np.save('.\\Results\\RHD-NSGA23-Malaria-repetition-' + str(j) + '.npy', np.array(rhd))

    with open('.\\Results\\All_PF-NSGA23-Malaria-repetition-' + str(j) + '.pkl', 'wb') as f:
        pickle.dump(hist_PF, f)






