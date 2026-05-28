import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import time
import xlsxwriter as xw
from scipy.stats.qmc import Sobol
import os


def update_xml_parameters(parameter):
    tree = ET.parse('wu_5000_28.xml')
    root = tree.getroot()

    model_elem = root.find('.//model')
    parameters_elem = model_elem.find('parameters')

    default_values = {
        "Simm": 0.196,
        "Xp": 1954.8,
        "Yp": 1.291,
        "sigma2_i": 11.729,
        "Xy_star": 593.661,
        "Xh_star": 54.082,
        "ln_1_minus_alpha_m": 1.770,
        "alpha_m_decay": 1.279,
        "sigma2_0": 5.838,
        "Xv_star": 3.959,
        "Y2_star": 6560.08,
        "alpha": 63220.5,
        "v1": 0.123,
        "log_phi1": 0.340,
        "Q_D": 0.019,
        "Q_n": 46.5095,
        "v0": 3.739,
        "YB1_star": 849046,
        "F0": 0.078,
        "log2_over_omega": 0.468,
        "Y1_star": 1.665,
        "Y0_star": 90.938,
        "alpha_F_star": 0.138,
    }

    for param in parameters_elem.findall('parameter'):
        name = param.get('name')
        if name in default_values:
            param.set("value", f"{default_values[name]:.6f}")

    for param in parameters_elem.findall('parameter'):
        name = param.get('name')
        if name in parameter:
            param.set("value", f"{parameter[name]:.6f}")

    tree.write('wu_5000_28.xml', encoding='UTF-8', xml_declaration=True)


def update_xml_seed(seed):
    tree = ET.parse('wu_5000_28.xml')
    root = tree.getroot()

    param_block = root.find('.//parameters')
    param_block.set('iseed', str(seed))
    tree.write('wu_5000_28.xml', encoding='utf-8', xml_declaration=True)


def compute_obj2(sim_output_avg, field_data):
    loss = 0.0
    age_groups = sorted(field_data['Age'].unique())
    for age in age_groups:
        k_obs = field_data.loc[(field_data['Age'] == age) & (field_data['Type'] == 3)].sort_values('Survey')[
            'Val'].values
        n_obs = field_data.loc[(field_data['Age'] == age) & (field_data['Type'] == 0)].sort_values('Survey')[
            'Val'].values
        if len(k_obs) == 0 or len(n_obs) == 0:
            continue

        k_mod = \
            sim_output_avg.loc[(sim_output_avg['Age'] == age) & (sim_output_avg['Type'] == 3)].sort_values('Survey')[
                'Val'].values
        n_mod = \
            sim_output_avg.loc[(sim_output_avg['Age'] == age) & (sim_output_avg['Type'] == 0)].sort_values('Survey')[
                'Val'].values
        if len(k_mod) == 0 or len(n_mod) == 0 or np.any(n_mod == 0):
            continue

        p = k_mod / n_mod
        p = np.clip(p, 0.001, 0.999)
        loss += -np.sum(k_obs * np.log(p) + (n_obs - k_obs) * np.log(1 - p))

    return loss


def compute_obj3(sim_output_avg, field_data, dens_bias=4.8):
    RSS = 0.0
    n = 0

    age_groups = sorted(field_data['Age'].unique())
    for age in age_groups:
        sim_sumlogdens = \
            sim_output_avg.loc[(sim_output_avg['Age'] == age) & (sim_output_avg['Type'] == 5)].sort_values('Survey')[
                'Val'].values
        sim_nPatent = \
            sim_output_avg.loc[(sim_output_avg['Age'] == age) & (sim_output_avg['Type'] == 3)].sort_values('Survey')[
                'Val'].values
        if len(sim_sumlogdens) == 0 or len(sim_nPatent) == 0:
            continue
        predV = sim_sumlogdens / sim_nPatent

        obs_sumlogdens = field_data.loc[(field_data['Age'] == age) & (field_data['Type'] == 5)].sort_values('Survey')[
            'Val'].values
        obs_nPatent = field_data.loc[(field_data['Age'] == age) & (field_data['Type'] == 3)].sort_values('Survey')[
            'Val'].values
        if len(obs_sumlogdens) == 0 or len(obs_nPatent) == 0:
            continue

        obsV = obs_sumlogdens / obs_nPatent

        valid = ~np.isnan(predV) & ~np.isnan(obsV)
        residuals = (predV[valid] - np.log(dens_bias) - obsV[valid]) ** 2
        RSS += np.sum(residuals)
        n += len(residuals)

    if n <= 1:
        return 1e6

    sigma = np.sqrt(RSS / (n - 1))
    loss = -(n * ((-0.5 * np.log(2 * np.pi)) - np.log(sigma)) - 0.5 * (n - 1))
    return loss


def objective_function(num_vars, names, x, field_data, cygwin_path, repetitions=1):
    x = np.array(x)
    x = x.reshape(-1)
    parameter = {}
    for i in range(num_vars):
        parameter[names[i]] = x[i]

    output_files = []
    update_xml_parameters(parameter)

    for i in range(repetitions):
        update_xml_seed(42 + i)

        bash_command = f'cd {cygwin_path} && ./OpenMalaria.exe --scenario wu_5000_28.xml --output output.txt'

        subprocess.run(
            [r"D:\cygwin64\bin\bash.exe", "-l", "-c", bash_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        col_field = ['Survey', 'Age', 'Type', 'Val']
        out_file = pd.read_csv('output.txt', sep=r"\s+", header=None, names=col_field)

        output_files.append(out_file)
    sim_output_avg = sum(output_files) / len(output_files)
    sim_output_avg[["Survey", "Age", "Type"]] = sim_output_avg[["Survey", "Age", "Type"]].astype(int)

    obj2 = compute_obj2(sim_output_avg, field_data)
    obj3 = compute_obj3(sim_output_avg, field_data)
    if np.any(np.isnan([obj2, obj3])) or np.any(np.isinf([obj2, obj3])):
        obj2 = obj2 if not (np.isnan(obj2) or np.isinf(obj2)) else 1e6
        obj3 = obj3 if not (np.isnan(obj3) or np.isinf(obj3)) else 1e6
    return np.array([obj2, obj3])


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


    num_vars_all = 23
    names_all = [
        "Simm", "Xp", "Yp", "sigma2_i", "Xy_star", "Xh_star",
        "ln_1_minus_alpha_m", "alpha_m_decay", "sigma2_0", "Xv_star",
        "Y2_star", "alpha", "v1", "log_phi1", "Q_D", "Q_n",
        "v0", "YB1_star", "F0", "log2_over_omega",
        "Y1_star", "Y0_star", "alpha_F_star",
    ]
    bounds_all = [
        [0, 6],  # Simm
        [0, 10000],  # Xp
        [0, 14],  # Yp
        [1, 75],  # sigma2_i
        [0, 1000],  # Xy_star
        [0, 200],  # Xh_star
        [0, 5],  # ln_1_minus_alpha_m
        [0, 11],  # alpha_m_decay
        [0, 26],  # sigma2_0
        [0, 66],  # Xv_star
        [0, 20000],  # Y2_star
        [10000, 200000],  # alpha
        [0, 2],  # v1
        [0, 9],  # log_phi1
        [0, 1],  # Q_D
        [3, 651],  # Q_n
        [1, 16],  # v0
        [20000, 1600000],  # YB1_star
        [0, 1],  # F0
        [0, 4],  # log2_over_omega
        [0, 10],  # Y1_star
        [0, 600],  # Y0_star
        [0, 3],  # alpha_F_star
    ]
    bounds_all = np.array(bounds_all)

    default_values = [
        0.196,  # Simm
        1954.8,  # Xp
        1.291,  # Yp
        11.729,  # sigma2_i
        593.661,  # Xy_star
        54.082,  # Xh_star
        1.770,  # ln_1_minus_alpha_m
        1.279,  # alpha_m_decay
        5.838,  # sigma2_0
        3.959,  # Xv_star
        6560.08,  # Y2_star
        63220.5,  # alpha
        0.123,  # v1
        0.340,  # log_phi1
        0.019,  # Q_D
        46.5095,  # Q_n
        3.739,  # v0
        849046,  # YB1_star
        0.078,  # F0
        0.468,  # log2_over_omega
        1.665,  # Y1_star
        90.938,  # Y0_star
        0.138,  # alpha_F_star
    ]
    default_values = np.array(default_values)

    col_field = ['Scen', 'Survey', 'Age', 'Type', 'Val']
    field_data = pd.read_csv('fieldData.txt', sep=r"\s+", header=None, names=col_field)
    field_data = field_data[field_data['Scen'] == 28]
    field_data = field_data.drop(columns='Scen')
    field_data = field_data.reset_index(drop=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    drive_letter = script_dir[0].lower()
    path_rest = script_dir[2:].replace("\\", "/")

    cygwin_path = f'/cygdrive/{drive_letter}{path_rest}'

    sample_size = 200
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

    Time = []
    obj = 2
    y_Training_set = np.zeros((sample_size, obj))
    for i in range(sample_size):
        print('Training ' + str(i))
        time_to_start = time.time()
        y_Training_set[i, :] = objective_function(num_vars_all, names_all, Training_set[i, :], field_data,
                                                      cygwin_path)
        Time.append(time.time() - time_to_start)
        print(y_Training_set[i, :])

    xw_toExcel(y_Training_set, 'y_Training_set.xlsx')
    np.save('y_Training_set.npy', y_Training_set)
    Time = np.array(Time)
    np.save('Time-Training-set.npy', Time)

    y_Testing_set = np.zeros((test_size, obj))
    for i in range(test_size):
        print('Testing ' + str(i))
        y_Testing_set[i, :] = objective_function(num_vars_all, names_all, Testing_set[i, :], field_data,
                                                     cygwin_path)
        print(y_Testing_set[i, :])

    xw_toExcel(y_Testing_set, 'y_Testing_set.xlsx')
    np.save('y_Testing_set.npy', y_Testing_set)









