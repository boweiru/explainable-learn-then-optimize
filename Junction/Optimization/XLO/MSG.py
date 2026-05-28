import time
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import joblib


class BaggingMeta(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, obj, n_estimators=10):
        self.obj = obj
        self.n_estimators = n_estimators
        self.base_models = []
        for i in range(self.n_estimators):
            Xgb = xgb.XGBRegressor(verbosity=0, n_jobs=1)
            self.base_models.append(Xgb)
        self.meta_models = []
        for i in range(self.obj):
            GP = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), optimizer="fmin_l_bfgs_b",
                                          n_restarts_optimizer=10)
            self.meta_models.append(GP)

        self.Meta_scaler = [StandardScaler() for _ in range(self.obj)]
        self.y_scalers = [StandardScaler() for _ in range(self.obj)]

    def fit(self, X, y):
        n = X.shape[0]
        for i in range(self.n_estimators):
            indices = np.random.choice(n, size=int(n * 0.75), replace=True)
            X_resampled, y_resampled = X[indices], y[indices]
            self.base_models[i].fit(X_resampled, y_resampled)

        meta_features = np.zeros((n, self.obj * self.n_estimators))
        for j in range(self.n_estimators):
            meta_features[:, j * self.obj:(j + 1) * self.obj] = self.base_models[j].predict(X)

        y_scaled = np.zeros_like(y)
        for j in range(self.obj):
            y_scaled[:, j] = self.y_scalers[j].fit_transform(y[:, j].reshape(-1, 1)).flatten()

        for j in range(self.obj):
            meta_features[:, j::self.obj] = self.Meta_scaler[j].fit_transform(meta_features[:, j::self.obj])
            self.meta_models[j].fit(meta_features[:, j::self.obj], y_scaled[:, j])

    def predict(self, X):
        n = X.shape[0]
        meta_features = np.zeros((n, self.obj * self.n_estimators))
        for j in range(self.n_estimators):
            meta_features[:, j * self.obj:(j + 1) * self.obj] = self.base_models[j].predict(X)

        second_predictions_mean = np.zeros((n, self.obj))
        second_predictions_std = np.zeros((n, self.obj))

        for j in range(self.obj):
            meta_features[:, j::self.obj] = self.Meta_scaler[j].transform(meta_features[:, j::self.obj])
            pred_mean_scaled, pred_std_scaled = self.meta_models[j].predict(
                meta_features[:, j::self.obj], return_std=True)

            second_predictions_mean[:, j] = self.y_scalers[j].inverse_transform(
                pred_mean_scaled.reshape(-1, 1)).flatten()
            second_predictions_std[:, j] = pred_std_scaled * self.y_scalers[j].scale_

        return second_predictions_mean, second_predictions_std


if __name__ == '__main__':
    Training_set = pd.read_excel('Training_set.xlsx', header=None)
    y_Training_set = pd.read_excel('y_Training_set.xlsx', header=None)
    Testing_set = pd.read_excel('Testing_set.xlsx', header=None)
    y_Testing_set = pd.read_excel('y_Testing_set.xlsx', header=None)

    Training_set = Training_set.values
    y_Training_set = y_Training_set.values
    Testing_set = Testing_set.values
    y_Testing_set = y_Testing_set.values

    Training_set = Training_set[:60, :]
    y_Training_set = y_Training_set[:60, :]

    obj = y_Training_set.shape[1]

    start_time = time.time()
    Model = BaggingMeta(obj)
    Model.fit(Training_set, y_Training_set)
    y_Testing_set_predict, _ = Model.predict(Testing_set)
    y_Training_set_predict, _ = Model.predict(Training_set)

    for i in range(obj):
        print('MSG Testing Objective ' + str(i) + ':',
              np.sum(np.abs(y_Testing_set[:, i] - y_Testing_set_predict[:, i])) / np.sum(np.abs(y_Testing_set[:, i])))
    print(time.time() - start_time)

    joblib.dump(Model, 'MSG.pkl')

