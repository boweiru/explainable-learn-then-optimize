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
        super().__init__(n_var=60,
                         n_obj=4,
                         n_constr=0,
                         xl=np.array(
                             [3, 0.2, 0, 1, 1, 1, 1e-09, 0.1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1e-09, 0, 0, 0,
                              0, 0, 0, 0, 0, 7, 0.2, 0, 1, 1, 1, 1e-09, 0.1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0,
                              1e-09, 0, 0, 0, 0, 0, 0, 0, 0]),
                         xu=np.array(
                             [7, 2, 1, 10, 3, 6, 3, 10, 10, 10, 1, 10, 10, 1, 10, 10, 10, 10, 1, 10, 1, 10, 30, 1,
                              20 / 3.6, 1, 1, 6, 3, 10, 12, 2, 1, 10, 3, 6, 3, 10, 10, 10, 1, 10, 10, 1, 10, 10, 10, 10,
                              1, 10, 1, 10, 30, 1, 20 / 3.6, 1, 1, 6, 3, 10]))
        self.num_vars_all = 60
        self.names_all = []
        self.names_all.append('psg_length')
        self.names_all.append('psg_speedFactor')
        self.names_all.append('psg_speedDev')
        self.names_all.append('psg_minGap')
        self.names_all.append('psg_accel')
        self.names_all.append('psg_decel')
        self.names_all.append('psg_tau')
        self.names_all.append('psg_delta')
        self.names_all.append('psg_emergencyDecel')
        self.names_all.append('psg_lcStrategic')
        self.names_all.append('psg_lcCooperative')
        self.names_all.append('psg_lcSpeedGain')
        self.names_all.append('psg_lcKeepRight')
        self.names_all.append('psg_lcOvertakeRight')
        self.names_all.append('psg_lcOpposite')
        self.names_all.append('psg_lcLookaheadLeft')
        self.names_all.append('psg_lcSpeedGainRight')
        self.names_all.append('psg_lcSpeedGainLookahead')
        self.names_all.append('psg_lcOvertakeDeltaSpeedFactor')
        self.names_all.append('psg_lcKeepRightAcceptanceTime')
        self.names_all.append('psg_lcCooperativeSpeed')
        self.names_all.append('psg_lcAssertive')
        self.names_all.append('psg_jmCrossingGap')
        self.names_all.append('psg_jmIgnoreFoeProb')
        self.names_all.append('psg_jmIgnoreFoeSpeed')
        self.names_all.append('psg_jmIgnoreJunctionFoeProb')
        self.names_all.append('psg_jmSigmaMinor')
        self.names_all.append('psg_jmStoplineGap')
        self.names_all.append('psg_jmTimegapMinor')
        self.names_all.append('psg_impatience')
        self.names_all.append('trk_length')
        self.names_all.append('trk_speedFactor')
        self.names_all.append('trk_speedDev')
        self.names_all.append('trk_minGap')
        self.names_all.append('trk_accel')
        self.names_all.append('trk_decel')
        self.names_all.append('trk_tau')
        self.names_all.append('trk_delta')
        self.names_all.append('trk_emergencyDecel')
        self.names_all.append('trk_lcStrategic')
        self.names_all.append('trk_lcCooperative')
        self.names_all.append('trk_lcSpeedGain')
        self.names_all.append('trk_lcKeepRight')
        self.names_all.append('trk_lcOvertakeRight')
        self.names_all.append('trk_lcOpposite')
        self.names_all.append('trk_lcLookaheadLeft')
        self.names_all.append('trk_lcSpeedGainRight')
        self.names_all.append('trk_lcSpeedGainLookahead')
        self.names_all.append('trk_lcOvertakeDeltaSpeedFactor')
        self.names_all.append('trk_lcKeepRightAcceptanceTime')
        self.names_all.append('trk_lcCooperativeSpeed')
        self.names_all.append('trk_lcAssertive')
        self.names_all.append('trk_jmCrossingGap')
        self.names_all.append('trk_jmIgnoreFoeProb')
        self.names_all.append('trk_jmIgnoreFoeSpeed')
        self.names_all.append('trk_jmIgnoreJunctionFoeProb')
        self.names_all.append('trk_jmSigmaMinor')
        self.names_all.append('trk_jmStoplineGap')
        self.names_all.append('trk_jmTimegapMinor')
        self.names_all.append('trk_impatience')

        self.obj = 4

        self.simulation_path = os.getcwd()
        self.obs_path = os.getcwd()
        self.start_time = 0
        self.end_time = 900
        self.obs_detal_time = 160

        self.true_flow, self.true_time, self.true_speed, self.true_queue = Running.get_True_data(self.obs_path)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = Running.objective_function(self.num_vars_all,
                                                  self.names_all, x,
                                                  self.simulation_path,
                                                  self.obs_path,
                                                  self.start_time,
                                                  self.end_time,
                                                  self.obs_detal_time,
                                                  self.true_flow,
                                                  self.true_time,
                                                  self.true_speed,
                                                  self.true_queue)


problem = MyProblem()


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        sobol_seq = Sobol(d=problem.n_var, seed=42)
        X = sobol_seq.random(n_samples)
        X = xl + X * (xu - xl)
        return X


repetitions = 10
obj = 4

r = np.array([10 for _ in range(obj)])
r_good = np.zeros(obj)

RHD = []
Time = []

for i in range(repetitions):
    start_time = time.time()
    np.random.seed(20 + i)
    random.seed(30 + i)
    algorithm = NSGA2(pop_size=200, n_offsprings=10, sampling=MySampling(), eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 31),
                   save_history=True,
                   verbose=True)

    np.save('.\\Results\\Time-NSGA60-Corridor-repetition-' + str(i) + '.npy', time.time() - start_time)

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

    np.save('.\\Results\\PF-NSGA60-Corridor-repetition-' + str(i) + '.npy', Pareto_front)
    np.save('.\\Results\\PS-NSGA60-Corridor-repetition-' + str(i) + '.npy', Pareto_solution)

    np.save('.\\Results\\RHD-NSGA60-Corridor-repetition-' + str(i) + '.npy', np.array(rhd))



