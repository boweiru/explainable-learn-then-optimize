import numpy as np
import time
import xlsxwriter as xw
from scipy.stats.qmc import Sobol


def objective_function(x):
    x = np.array(x).flatten()
    func =

    return func(x)


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

    num_vars_all =

    names_all = ['x' + str(i + 1) for i in range(num_vars_all)]
    names_all = np.array(names_all)

    bounds_all = np.zeros((num_vars_all, 2))
    bounds_all[:, 0] = 0
    bounds_all[:, 1] = 1

    sample_size =
    sobol_seq = Sobol(d=num_vars_all, seed=42)
    Training_set = sobol_seq.random(sample_size)
    for i in range(num_vars_all):
        Training_set[:, i] = bounds_all[i][0] + (
                bounds_all[i][1] - bounds_all[i][0]) * Training_set[:, i]

    Training_set = np.array(Training_set)
    np.save('Training_set.npy', Training_set)
    xw_toExcel(Training_set, 'Training_set.xlsx')

    test_size =
    Testing_set = np.random.uniform(low=bounds_all[:, 0], high=bounds_all[:, 1],
                                    size=(test_size, num_vars_all))

    Testing_set = np.array(Testing_set)
    np.save('Testing_set.npy', Testing_set)
    xw_toExcel(Testing_set, 'Testing_set.xlsx')

    Time = np.zeros(sample_size)
    obj =
    y_Training_set = np.zeros((sample_size, obj))
    for i in range(sample_size):
        print('Training ' + str(i))
        time_to_start = time.time()
        y_Training_set[i, :] = objective_function(Training_set[i, :])
        Time[i] = time.time() - time_to_start
        print(y_Training_set[i, :])

    np.save('Time-Training-set.npy', Time)
    xw_toExcel(y_Training_set, 'y_Training_set.xlsx')
    np.save('y_Training_set.npy', y_Training_set)

    y_Testing_set = np.zeros((test_size, obj))
    for i in range(test_size):
        print('Testing ' + str(i))
        y_Testing_set[i, :] = objective_function(Testing_set[i, :])
        print(y_Testing_set[i, :])

    xw_toExcel(y_Testing_set, 'y_Testing_set.xlsx')
    np.save('y_Testing_set.npy', y_Testing_set)

