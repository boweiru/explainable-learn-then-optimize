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
        super().__init__(n_var=29,
                         n_obj=4,
                         n_constr=0,
                         xl=np.array(
                             [0.2, 0, 1, 1, 1, 1e-09, 0.1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1e-09, 0, 0, 0, 0,
                              0, 0, 0, 0]),
                         xu=np.array([2, 1, 10, 3, 6, 3, 10, 10, 10, 1, 10, 10, 1, 10, 10, 10, 10, 1, 10, 1, 10, 30, 1,
                                      20 / 3.6, 1, 1, 6, 3, 10]))
        self.num_vars = 29
        self.names = []
        self.names.append('speedFactor')
        self.names.append('speedDev')
        self.names.append('minGap')
        self.names.append('accel')
        self.names.append('decel')
        self.names.append('tau')
        self.names.append('delta')
        self.names.append('emergencyDecel')
        self.names.append('lcStrategic')
        self.names.append('lcCooperative')
        self.names.append('lcSpeedGain')
        self.names.append('lcKeepRight')
        self.names.append('lcOvertakeRight')
        self.names.append('lcOpposite')
        self.names.append('lcLookaheadLeft')
        self.names.append('lcSpeedGainRight')
        self.names.append('lcSpeedGainLookahead')
        self.names.append('lcOvertakeDeltaSpeedFactor')
        self.names.append('lcKeepRightAcceptanceTime')
        self.names.append('lcCooperativeSpeed')
        self.names.append('lcAssertive')
        self.names.append('jmCrossingGap')
        self.names.append('jmIgnoreFoeProb')
        self.names.append('jmIgnoreFoeSpeed')
        self.names.append('jmIgnoreJunctionFoeProb')
        self.names.append('jmSigmaMinor')
        self.names.append('jmStoplineGap')
        self.names.append('jmTimegapMinor')
        self.names.append('impatience')
        self.obj = 4
        self.simulation_path = os.path.dirname(os.path.abspath(__file__))
        self.obs_path = os.path.dirname(os.path.abspath(__file__))
        self.start_time = 0
        self.end_time = 900
        self.obs_detal_time = 185
        signal_df = pd.read_csv('IntersectionA-Signal.csv')
        self.signal_dict = {'startTime': list(signal_df['startTime']),
                            'eventDuration': list(signal_df['eventDuration']), 'index': list(signal_df['index'])}
        self.true_flow, self.true_time, self.true_speed, self.true_queue, self.true_timeLoss = Running.get_True_data(
            self.obs_path)

    def _evaluate(self, x, out, *args, **kwargs):
        WMAPE_flow, WMAPE_speed, WMAPE_time, WMAPE_queue = Running.objective_function(self.num_vars, self.names, x,
                                                                                          self.simulation_path,
                                                                                          self.obs_path,
                                                                                          self.start_time,
                                                                                          self.end_time,
                                                                                          self.signal_dict,
                                                                                          self.obs_detal_time,
                                                                                          self.true_flow,
                                                                                          self.true_time,
                                                                                          self.true_speed,
                                                                                          self.true_queue,
                                                                                          self.true_timeLoss, self.obj)

        out["F"] = [WMAPE_flow, WMAPE_speed, WMAPE_time, WMAPE_queue]

problem = MyProblem()


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        X = sobol_seq.i4_sobol_generate(problem.n_var, n_samples)
        X = xl + X * (xu - xl)
        return X


obj = 4

r = np.array([10 for _ in range(obj)])
r_good = np.zeros(obj)

for j in range(10):
    start_time = time.time()
    np.random.seed(20 + j)
    random.seed(30 + j)
    algorithm = NSGA2(pop_size=70, n_offsprings=10, sampling=MySampling(), eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 14),
                   save_history=True,
                   verbose=True)

    np.save('.\\Results\\Time-NSGA29-Junction-repetition-' + str(j) + '.npy', time.time() - start_time)

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

    np.save('.\\Results\\PF-NSGA29-Junction-repetition-' + str(j) + '.npy', Pareto_front)
    np.save('.\\Results\\PS-NSGA29-Junction-repetition-' + str(j) + '.npy', Pareto_solution)

    np.save('.\\Results\\RHD-NSGA29-Junction-repetition-' + str(j) + '.npy', np.array(rhd))

    with open('.\\Results\\All_PF-NSGA29-Junction-repetition-' + str(j) + '.pkl', 'wb') as f:
        pickle.dump(hist_PF, f)






