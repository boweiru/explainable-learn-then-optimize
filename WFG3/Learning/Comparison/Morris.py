from SALib.sample.morris import sample
from SALib.analyze.morris import analyze
import numpy as np
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


problem = {
    'num_vars': 100,
    'names': ['x' + str(i + 1) for i in range(100)],
    'bounds': [[0, 2 * (i + 1)] for i in range(100)]
}

param_values = sample(problem, N=4, num_levels=4, optimal_trajectories=None)  # N*(D+1)

obj = 5
num_vars_all = 100
k = 4

model = WFG3(n_var=num_vars_all, n_obj=obj, k=k)

start_time = time.time()

Y = model.evaluate(param_values)

var_y = np.var(Y, axis=0)
weights = var_y / np.sum(var_y)

mu_star_matrix = np.zeros((problem['num_vars'], obj))

for i in range(obj):
    Si = analyze(problem, param_values, Y[:, i], print_to_console=False)
    mu_star_matrix[:, i] = Si['mu_star']

weighted_mu_star = np.dot(mu_star_matrix, weights)
print(time.time() - start_time)

predicted_rank = [sorted(weighted_mu_star, reverse=True).index(x) + 1 for x in weighted_mu_star]

true_PT = np.load(
    'Real_PT-WFG3.npy')
true_rank = [sorted(true_PT, reverse=True).index(x) + 1 for x in true_PT]


top_k_acc = top_k_accuracy(true_rank, predicted_rank, 4)
ndcg = ndcg_score(np.array([true_PT]), np.array([weighted_mu_star]))

print('Top K Accuracy: ', top_k_acc)
print('NDCG: ', ndcg)
