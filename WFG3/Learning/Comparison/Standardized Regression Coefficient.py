import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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

np.random.seed(42)
X = np.random.rand(400, num_vars_all)
for i in range(num_vars_all):
    a, b = bounds[i]
    X[:, i] = a + (b - a) * X[:, i]

Y = model.evaluate(X)

scaler_X = StandardScaler()
X_std = scaler_X.fit_transform(X)
scaler_Y = StandardScaler()
Y_std = scaler_Y.fit_transform(Y)

SRC_matrix = np.zeros((num_vars_all, obj))
R2_scores = np.zeros(obj)
var_Y = np.zeros(obj)

for j in range(obj):
    model = LinearRegression()
    model.fit(X_std, Y_std[:, j])
    SRC_matrix[:, j] = model.coef_
    R2_scores[j] = model.score(X_std, Y_std[:, j])
    var_Y[j] = np.var(Y[:, j])

weighted_SRC = np.dot(SRC_matrix, var_Y) / np.sum(var_Y)
print(time.time() - start_time)
predicted_rank = [sorted(weighted_SRC, reverse=True).index(x) + 1 for x in weighted_SRC]

true_PT = np.load(
    'Real_PT-WFG3.npy')
true_rank = [sorted(true_PT, reverse=True).index(x) + 1 for x in true_PT]

top_k_acc = top_k_accuracy(true_rank, predicted_rank, 4)
ndcg = ndcg_score(np.array([true_PT]), np.array([weighted_SRC]))

print('Top K Accuracy: ', top_k_acc)
print('NDCG: ', ndcg)
