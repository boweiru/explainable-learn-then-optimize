from numpy.random import RandomState
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import random
from scipy.stats.qmc import Sobol
from pymoo.problems.many import WFG3
import time


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
        super().__init__(n_var=4,
                         n_obj=5,
                         n_constr=0,
                         xl=np.zeros(4),
                         xu=np.array(
                             [2 * (i + 1) for i in range(4)]))
        self.num_vars_all = 100
        self.names_all = np.array(['x' + str(i + 1) for i in range(self.num_vars_all)])
        self.obj = 5
        self.k = 4
        self.default_values = np.array([float(2 * 0.35 * (i + 1)) for i in range(self.num_vars_all)])
        self.whether_to_optimize = np.zeros(self.num_vars_all)
        for i in range(self.k):
            self.whether_to_optimize[i] = 1
        self.problem = WFG3(n_var=self.num_vars_all, n_obj=self.obj, k=self.k)

    def _evaluate(self, x, out, *args, **kwargs):
        all_x = self.default_values
        j = 0
        for i in range(len(self.whether_to_optimize)):
            if self.whether_to_optimize[i] > 0:
                all_x[i] = x[j]
                j += 1
        out["F"] = self.problem.evaluate(all_x)


problem = MyProblem()


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        sobol_seq = Sobol(d=problem.n_var, seed=42)
        X = sobol_seq.random(n_samples)
        X = xl + X * (xu - xl)
        return X


repetitions = 10
obj = 5

r = np.array([20 for _ in range(obj)])
r_good = np.zeros(obj)

for i in range(repetitions):
    start_time = time.time()
    np.random.seed(20 + i)
    random.seed(30 + i)
    algorithm = NSGA2(pop_size=10, n_offsprings=10, sampling=MySampling(), eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 60),
                   save_history=True,
                   verbose=True)

    np.save('.\\Results\\Time-NSGA4-WFG3-repetition-' + str(i) + '.npy', time.time() - start_time)

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

    np.save('.\\Results\\PF-NSGA4-WFG3-repetition-' + str(i) + '.npy', Pareto_front)
    np.save('.\\Results\\PS-NSGA4-WFG3-repetition-' + str(i) + '.npy', Pareto_solution)

    np.save('.\\Results\\RHD-NSGA4-WFG3-repetition-' + str(i) + '.npy', np.array(rhd))

