from pymoo.problems.many import WFG3
import numpy as np
from scipy.stats.qmc import Sobol
import xlsxwriter as xw
from numpy.random import RandomState
import time
from pymoo.util.ref_dirs import get_reference_directions


def objective_function(x, defaults, whether_to_optimize):
    x = np.array(x).flatten()
    x_new = defaults.copy()

    j = 0
    for i in range(len(whether_to_optimize)):
        if whether_to_optimize[i] == 1:
            x_new[i] = x[j]
            j += 1

    problem = WFG3(n_var=num_vars_all, n_obj=obj, k=k)

    return problem.evaluate(x_new)


if __name__ == "__main__":
    def xw_toExcel(X, fileName):
        sample_size = X.shape[0]
        workbook = xw.Workbook(fileName)
        worksheet1 = workbook.add_worksheet("sheet1")
        worksheet1.activate()
        i = 1
        for j in range(sample_size):
            row = 'A' + str(i)
            worksheet1.write_row(row, X[j, :])
            i += 1
        workbook.close()

    obj = 5
    num_vars_all = 100
    k = 4

    bounds_all = np.zeros((num_vars_all, 2))
    for i in range(num_vars_all):
        bounds_all[i, 1] = 2 * (i + 1)

    default_values = np.array([float(2 * (i + 1) * 0.35) for i in range(num_vars_all)])

    whether_to_optimize = np.ones(num_vars_all)

    sample_size = 1000
    sobol_seq = Sobol(d=num_vars_all, seed=42)
    Training_set = sobol_seq.random(sample_size)
    for i in range(num_vars_all):
        Training_set[:, i] = bounds_all[i][0] + (
                bounds_all[i][1] - bounds_all[i][0]) * Training_set[:, i]

    Training_set = np.array(Training_set)
    np.save('Training_set.npy', Training_set)
    xw_toExcel(Training_set, 'Training_set.xlsx')

    test_size = 10000
    Testing_set = np.random.uniform(low=bounds_all[:, 0], high=bounds_all[:, 1],
                                    size=(test_size, num_vars_all))

    Testing_set = np.array(Testing_set)
    np.save('Testing_set.npy', Testing_set)
    xw_toExcel(Testing_set, 'Testing_set.xlsx')

    Time = np.zeros(sample_size)
    y_Training_set = np.zeros((sample_size, obj))
    for i in range(sample_size):
        print('Training ' + str(i))
        time_to_start = time.time()
        y_Training_set[i, :] = objective_function(Training_set[i, :], default_values, whether_to_optimize)
        Time[i] = time.time() - time_to_start
        print(y_Training_set[i, :])

    np.save('Time-Training-set.npy', Time)
    xw_toExcel(y_Training_set, 'y_Training_set.xlsx')
    np.save('y_Training_set.npy', y_Training_set)

    y_Testing_set = np.zeros((test_size, obj))
    for i in range(test_size):
        print('Testing ' + str(i))
        y_Testing_set[i, :] = objective_function(Testing_set[i, :], default_values, whether_to_optimize)
        print(y_Testing_set[i, :])

    xw_toExcel(y_Testing_set, 'y_Testing_set.xlsx')
    np.save('y_Testing_set.npy', y_Testing_set)
