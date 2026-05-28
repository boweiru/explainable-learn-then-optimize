import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import MSG_CatBoost
import MSG_Model
import MSG_Model_Data
import MSG
from sklearn.preprocessing import StandardScaler

repetitions = 10
obj = 2

for dimension in range(20, 220, 20):
    print(dimension)

    Training_set = pd.read_excel('Training_set.xlsx', header=None).values[:dimension, :]
    y_Training_set = pd.read_excel('y_Training_set.xlsx', header=None).values[:dimension, :]
    Testing_set = pd.read_excel('Testing_set.xlsx', header=None).values

    n_test = Testing_set.shape[0]

    PRED_GP = np.zeros((repetitions, n_test, obj))
    PRED_SVR = np.zeros((repetitions, n_test, obj))
    PRED_RF = np.zeros((repetitions, n_test, obj))
    PRED_LGB = np.zeros((repetitions, n_test, obj))
    PRED_CatB = np.zeros((repetitions, n_test, obj))
    PRED_XGB = np.zeros((repetitions, n_test, obj))
    PRED_MSG_CatBoost = np.zeros((repetitions, n_test, obj))
    PRED_MSG_Model = np.zeros((repetitions, n_test, obj))
    PRED_MSG_Model_Data = np.zeros((repetitions, n_test, obj))
    PRED_MSG = np.zeros((repetitions, n_test, obj))

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(Training_set)
    X_test_scaled = X_scaler.transform(Testing_set)

    y_scalers = [StandardScaler() for _ in range(obj)]

    for i in range(repetitions):
        print(f'Repetition {i+1}/{repetitions}')

        # ===== GP =====
        print('GP')
        GP_models = []
        for j in range(obj):
            Y_j_scaled = y_scalers[j].fit_transform(y_Training_set[:, j].reshape(-1, 1)).ravel()
            gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(),
                                          optimizer="fmin_l_bfgs_b",
                                          n_restarts_optimizer=10)
            gp.fit(X_scaled, Y_j_scaled)
            y_pred_scaled = gp.predict(X_test_scaled)
            y_pred = y_scalers[j].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            PRED_GP[i, :, j] = y_pred
            GP_models.append(gp)

        # ===== SVR =====
        print('SVR')
        SVR_models = []
        for j in range(obj):
            Y_j_scaled = y_scalers[j].fit_transform(y_Training_set[:, j].reshape(-1, 1)).ravel()
            svr = SVR(kernel='rbf')
            svr.fit(X_scaled, Y_j_scaled)
            y_pred_scaled = svr.predict(X_test_scaled)
            y_pred = y_scalers[j].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            PRED_SVR[i, :, j] = y_pred
            SVR_models.append(svr)

        # ===== RandomForest =====
        print('RF')
        RF_models = []
        for j in range(obj):
            rf = RandomForestRegressor()
            rf.fit(Training_set, y_Training_set[:, j])
            y_pred = rf.predict(Testing_set)
            PRED_RF[i, :, j] = y_pred
            RF_models.append(rf)

        # ===== LightGBM =====
        print('LightGBM')
        LGB_models = []
        for j in range(obj):
            lgbm = lgb.LGBMRegressor(boosting='dart', data_sample_strategy='goss', verbosity=-1)
            lgbm.fit(Training_set, y_Training_set[:, j])
            y_pred = lgbm.predict(Testing_set)
            PRED_LGB[i, :, j] = y_pred
            LGB_models.append(lgbm)

        # ===== CatBoost =====
        print('CatBoost')
        CatB = CatBoostRegressor(loss_function='MultiRMSE', verbose=0)
        CatB.fit(Training_set, y_Training_set)
        y_pred_CatB = CatB.predict(Testing_set)  # (n_test, obj)
        PRED_CatB[i, :, :] = y_pred_CatB

        # ===== XGBoost =====
        print('XGBoost')
        Xgb = xgb.XGBRegressor(verbosity=0, n_jobs=1)
        Xgb.fit(Training_set, y_Training_set)
        y_pred_XGB = Xgb.predict(Testing_set)  # 期望 (n_test, obj)
        PRED_XGB[i, :, :] = y_pred_XGB

        # ===== MSG_CatBoost =====
        print('MSG_CatBoost')
        MSG_CatB = MSG_CatBoost.BaggingMeta(obj=obj)
        MSG_CatB.fit(Training_set, y_Training_set)
        y_pred_MSG_CatB, _ = MSG_CatB.predict(Testing_set)  # (n_test, obj)
        PRED_MSG_CatBoost[i, :, :] = y_pred_MSG_CatB

        # ===== MSG_Model =====
        print('MSG_Model')
        MSG_Model = MSG_Model.BaggingMeta(obj=obj)
        MSG_Model.fit(Training_set, y_Training_set)
        y_pred_MSG_Model, _ = MSG_Model.predict(Testing_set)
        PRED_MSG_Model[i, :, :] = y_pred_MSG_Model

        # ===== MSG_Model_Data =====
        print('MSG_Model_Data')
        MSG_Model_Data = MSG_Model_Data.BaggingMeta(obj=obj)
        MSG_Model_Data.fit(Training_set, y_Training_set)
        y_pred_MSG_Model_Data, _ = MSG_Model_Data.predict(Testing_set)
        PRED_MSG_Model_Data[i, :, :] = y_pred_MSG_Model_Data

        # ===== MSG =====
        print('MSG_XGBoost')
        MSG_XGB = MSG.BaggingMeta(obj=obj)
        MSG_XGB.fit(Training_set, y_Training_set)
        y_pred_MSG_XGB, _ = MSG_XGB.predict(Testing_set)
        PRED_MSG[i, :, :] = y_pred_MSG_XGB

    # —— 保存预测结果 —— #
    np.save(f'PRED-GP-{dimension}.npy', PRED_GP)
    np.save(f'PRED-SVR-{dimension}.npy', PRED_SVR)
    np.save(f'PRED-RF-{dimension}.npy', PRED_RF)
    np.save(f'PRED-LGB-{dimension}.npy', PRED_LGB)
    np.save(f'PRED-CatBoost-{dimension}.npy', PRED_CatB)
    np.save(f'PRED-XGB-{dimension}.npy', PRED_XGB)
    np.save(f'PRED-MSG_CatBoost-{dimension}.npy', PRED_MSG_CatBoost)
    np.save(f'PRED-MSG_Model-{dimension}.npy', PRED_MSG_Model)
    np.save(f'PRED-MSG_Model_Data-{dimension}.npy', PRED_MSG_Model_Data)
    np.save(f'PRED-MSG-{dimension}.npy', PRED_MSG)