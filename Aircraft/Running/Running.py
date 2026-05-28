import numpy as np
from designer import HPADesigner
import time
import xlsxwriter as xw
from scipy.stats.qmc import Sobol


class HPA303(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level==0 else True
        super().__init__(n_div=n_div, max_plys=20, level=level, AIRFOIL=AIRFOIL, WIRE=True, DIHEDRAL=True, PAYLOAD=True)
        self.nf = 3
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED
    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        g = 10*penalty1 + 2*penalty2 + 2*penalty3
        f = np.array([self.drag + g, self.y_aoa[0] + g, -self.payload + 10*g])
        return f


def objective_function(x):
    x = np.array(x).flatten()
    func = HPA303(n_div=7, level=1, NORMALIZED=True)

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

    num_vars_all = 55

    names_all = ['x' + str(i + 1) for i in range(num_vars_all)]
    names_all = np.array(names_all)

    bounds_all = np.zeros((num_vars_all, 2))
    bounds_all[:, 0] = 0
    bounds_all[:, 1] = 1

    sample_size = 600
    sobol_seq = Sobol(d=num_vars_all, seed=42)
    Training_set = sobol_seq.random(sample_size)
    for i in range(num_vars_all):
        Training_set[:, i] = bounds_all[i][0] + (
                bounds_all[i][1] - bounds_all[i][0]) * Training_set[:, i]

    Training_set = np.array(Training_set)
    np.save('Training_set.npy', Training_set)
    xw_toExcel(Training_set, 'Training_set.xlsx')

    test_size = 1000
    Testing_set = np.random.uniform(low=bounds_all[:, 0], high=bounds_all[:, 1],
                                    size=(test_size, num_vars_all))

    Testing_set = np.array(Testing_set)
    np.save('Testing_set.npy', Testing_set)
    xw_toExcel(Testing_set, 'Testing_set.xlsx')

    Time = np.zeros(sample_size)
    obj = 3
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