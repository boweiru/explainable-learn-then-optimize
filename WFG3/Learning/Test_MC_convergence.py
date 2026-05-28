import numpy as np
from scipy.stats.qmc import Sobol
from sklearn.utils import resample
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import time
from MSG import BaggingMeta
import jenkspy
from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib import rcParams
import joblib


class Multioutputs_Sensitivity_Analysis:
    def __init__(self, num_vars, names, bounds, Model):
        self.problem = {'num_vars': num_vars, 'names': names, 'bounds': bounds}
        self.Model = Model
        self.SI = {}

    def sample(self, dimension, sample_size):
        sobol_seq = Sobol(d=dimension, seed=123)
        Sobol_sequence = sobol_seq.random(sample_size)
        return Sobol_sequence

    def Sensitivity_indices(self, resample_size, shift_rows=4):
        Sobol_sequence = self.sample(2 * self.problem['num_vars'], resample_size)
        for i in range(self.problem['num_vars']):
            Sobol_sequence[:, i] = self.problem['bounds'][i][0] + (
                    self.problem['bounds'][i][1] - self.problem['bounds'][i][0]) * Sobol_sequence[:, i]
            Sobol_sequence[:, i + self.problem['num_vars']] = self.problem['bounds'][i][0] + (
                    self.problem['bounds'][i][1] - self.problem['bounds'][i][0]) * Sobol_sequence[:,
                                                                                   i + self.problem['num_vars']]

        start_col = self.problem['num_vars']
        shift_rows = shift_rows
        sub_arr = Sobol_sequence[:, start_col:]
        shifted_sub_arr = np.roll(sub_arr, shift_rows, axis=0)
        Sobol_sequence[:, start_col:] = shifted_sub_arr
        M1 = Sobol_sequence[:, :start_col]
        M2 = Sobol_sequence[:, start_col:]
        M1_predict, _ = self.Model.predict(M1)
        self.obj = M1_predict.shape[1]
        M1_predict = M1_predict.reshape((resample_size, self.obj))
        std_deviation_M1_predict = np.std(M1_predict, axis=0)
        M1_predict = M1_predict / std_deviation_M1_predict
        self.correlation_matrix = np.corrcoef(M1_predict, rowvar=False)
        self.correlation_matrix = self.correlation_matrix.reshape((self.obj, self.obj))

        E_square = (np.mean(M1_predict, axis=0)) ** 2
        V = np.mean(np.square(M1_predict), axis=0) - E_square
        V = V.reshape((1, self.obj))
        V_length_square = V @ self.correlation_matrix @ V.T

        PT = np.zeros(self.problem['num_vars'])

        for i in range(self.problem['num_vars']):
            Ni = M1.copy()
            Ni[:, i] = M2[:, i]
            Ni_predict, _ = self.Model.predict(Ni)
            std_deviation_Ni_predict = np.std(Ni_predict, axis=0)
            Ni_predict = Ni_predict / std_deviation_Ni_predict
            Ni_predict = Ni_predict.reshape((resample_size, self.obj))
            VTi = np.mean(np.square(M1_predict - Ni_predict), axis=0) / 2
            PT[i] = (VTi @ self.correlation_matrix @ V.T) / V_length_square

        self.SI['PT'] = PT


if __name__ == '__main__':
    def Natural_Breaks(names, X, min_gvf=0.7):
        data = np.array(X)
        for nb_class in range(1, len(data) + 1):
            jnb = jenkspy.JenksNaturalBreaks(nb_class)
            jnb.fit(data)
            if jnb.goodness_of_variance_fit(data) > min_gvf:
                break

        group_dict = {}
        for idx, label in enumerate(jnb.labels_):
            if label not in group_dict:
                group_dict[label] = []
            group_dict[label].append(idx)

        sorted_group_labels = np.zeros_like(jnb.labels_)
        for label, indices in group_dict.items():
            group_values = X[indices]
            sorted_indices = np.argsort(group_values)[::-1]

            for new_order, idx in enumerate(sorted_indices):
                sorted_group_labels[indices[idx]] = label

        jnb.labels_ = sorted_group_labels
        print(jnb.labels_)
        i = 0
        for group in jnb.groups_:
            print('cluster ' + str(i) + ': ')
            group = sorted(group, reverse=True)
            for point in group:
                print(names[np.where(X == point)[0].item()])
            i += 1
        return jnb


    obj = 5
    num_vars_all = 100
    k = 4

    names_all = ['x' + str(i + 1) for i in range(num_vars_all)]
    names_all = np.array(names_all)

    bounds_all = np.zeros((num_vars_all, 2))
    for i in range(num_vars_all):
        bounds_all[i, 1] = 2 * (i + 1)

    Model = joblib.load('MSG.pkl')
    k = 20
    PT = np.zeros((k, num_vars_all))
    for i in range(1, k + 1):
        print(i)
        MSA = Multioutputs_Sensitivity_Analysis(num_vars_all, names_all, bounds_all, Model)
        MSA.Sensitivity_indices(2 ** i, 0)

        for j in range(num_vars_all):
            PT[i - 1, j] = MSA.SI['PT'][j]

        print(MSA.SI['PT'])

    np.save('PT-convergence.npy', PT)




