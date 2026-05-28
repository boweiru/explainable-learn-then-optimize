import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import simps
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

num_samples = 2
Copy = 2

X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_samples, num_vars_all))
Y_all = model.evaluate(X)

y_vals_list = []
pdf_Y_list = []
var_Y = np.std(Y_all, axis=0)

for j in range(obj):
    kde_Y = gaussian_kde(
        Y_all[:, j])
    y_vals = np.linspace(min(Y_all[:, j]), max(Y_all[:, j]), 1000)
    pdf_Y = kde_Y(y_vals)

    y_vals_list.append(y_vals)
    pdf_Y_list.append(pdf_Y)

aggregated_delta = np.zeros(num_vars_all)
delta = np.zeros((num_vars_all, obj))

for i in range(num_vars_all):
    delta_i = np.zeros((Copy, obj))
    for j in range(Copy):
        X_fixed = X.copy()
        X_fixed[:, i] = np.random.choice(X[:, i])
        Y_i = model.evaluate(X_fixed)

        for k in range(obj):
            kde_Y_i_k = gaussian_kde(Y_i[:, k])
            pdf_Y_i_k = kde_Y_i_k(y_vals_list[k])

            delta_i_k = simps(np.abs(pdf_Y_list[k] - pdf_Y_i_k), y_vals_list[k])
            delta_i[j, k] = delta_i_k

    delta[i, :] = np.mean(delta_i) / 2

aggregated_delta = np.dot(delta, var_Y) / np.sum(var_Y)
print(time.time() - start_time)

predicted_rank = [sorted(aggregated_delta, reverse=True).index(x) + 1 for x in aggregated_delta]

true_PT = np.load(
    'Real_PT-WFG3.npy')
true_rank = [sorted(true_PT, reverse=True).index(x) + 1 for x in true_PT]

top_k_acc = top_k_accuracy(true_rank, predicted_rank, 4)
ndcg = ndcg_score(np.array([true_PT]), np.array([aggregated_delta]))

print('Top K Accuracy: ', top_k_acc)
print('NDCG: ', ndcg)
