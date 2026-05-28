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


obj = 5
num_vars_all = 100
k = 4
bounds = np.array([[0, 2 * (i + 1)] for i in range(num_vars_all)])

model = WFG3(n_var=num_vars_all, n_obj=obj, k=k)
start_time = time.time()

x0 = np.array([i + 1 for i in range(num_vars_all)])

epsilon = np.array([0.2 * (i + 1) for i in range(num_vars_all)])

n_samples = 200
X_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_samples, num_vars_all))
Y_samples = model.evaluate(X_samples)
sigma_squared = np.var(Y_samples, axis=0)

Di_matrix = np.zeros((num_vars_all, obj))

Di_num = np.zeros((num_vars_all, obj))
for i in range(num_vars_all):
    x_plus = np.copy(x0)
    x_minus = np.copy(x0)

    x_plus[i] += epsilon[i]
    x_minus[i] -= epsilon[i]

    y_plus = model.evaluate(x_plus)
    y_minus = model.evaluate(x_minus)

    gradient_i = (y_plus - y_minus) / (2 * epsilon[i])

    Di_num[i, :] = gradient_i * epsilon[i]

Di_denom = np.sum(Di_num, axis=1)
for i in range(obj):
    Di_matrix[:, i] = Di_num[:, i] / Di_denom[i]

Di_agg = np.dot(Di_matrix, sigma_squared) / np.sum(sigma_squared)
print(time.time() - start_time)

predicted_rank = [sorted(Di_agg, reverse=True).index(x) + 1 for x in Di_agg]

true_PT = np.load(
    'Real_PT-WFG3.npy')
true_rank = [sorted(true_PT, reverse=True).index(x) + 1 for x in true_PT]

top_k_acc = top_k_accuracy(true_rank, predicted_rank, 4)
ndcg = ndcg_score(np.array([true_PT]), np.array([Di_agg]))

print('Top K Accuracy: ', top_k_acc)
print('NDCG: ', ndcg)
