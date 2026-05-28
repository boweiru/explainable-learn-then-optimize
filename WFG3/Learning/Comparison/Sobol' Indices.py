import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from pymoo.problems.many import WFG3
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score
import time

def top_k_accuracy(true_rank, predicted_rank, k):
    true_top_k = set(true_rank[:k])
    predicted_top_k = set(predicted_rank[:k])

    overlap_count = len(true_top_k & predicted_top_k)

    accuracy = overlap_count / k
    return accuracy


obj = 5
num_vars_all = 100
k = 4
bounds = [[0, 2 * (i + 1)] for i in range(num_vars_all)]

model = WFG3(n_var=num_vars_all, n_obj=obj, k=k)

start_time = time.time()

problem = {
    'num_vars': num_vars_all,
    'names': ['x' + str(i + 1) for i in range(100)],
    'bounds': bounds
}

param_values = saltelli.sample(problem, 2)  # N*(2D+2)

Y_all = model.evaluate(param_values)

var_Y = np.var(Y_all, axis=0)

num_inputs = problem['num_vars']
sobol_results_list = []

for j in range(obj):
    Y = Y_all[:, j]
    Si = sobol.analyze(problem, Y, print_to_console=False)
    sobol_results_list.append(pd.DataFrame({
        'S1': Si['S1'],
        'ST': Si['ST'],
    }, index=problem['names']))

sobol_results_df = pd.concat(sobol_results_list, keys=[f'Y{j + 1}' for j in range(obj)], axis=1)

S1_matrix = np.array([result['S1'].values for result in sobol_results_list]).T
ST_matrix = np.array([result['ST'].values for result in sobol_results_list]).T

S1_weighted = np.dot(S1_matrix, var_Y) / np.sum(var_Y)
ST_weighted = np.dot(ST_matrix, var_Y) / np.sum(var_Y)

aggregated_results = pd.DataFrame({
    'Aggregated_S1': S1_weighted,
    'Aggregated_ST': ST_weighted
}, index=problem['names'])

print(time.time() - start_time)

predicted_rank = [sorted(ST_weighted, reverse=True).index(x) + 1 for x in ST_weighted]

true_PT = np.load(
    'Real_PT-WFG3.npy')
true_rank = [sorted(true_PT, reverse=True).index(x) + 1 for x in true_PT]

top_k_acc = top_k_accuracy(true_rank, predicted_rank, 4)
ndcg = ndcg_score(np.array([true_PT]), np.array([ST_weighted]))

print('Top K Accuracy: ', top_k_acc)
print('NDCG: ', ndcg)
