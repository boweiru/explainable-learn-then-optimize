import scipy.io
import pandas as pd
import numpy as np
import os

def calc_yield(point1, point2, df: pd.DataFrame, offset: float = 0.002) -> float:
    x_line = df['strain']
    y_line = df['stress']
    df_line = pd.DataFrame()
    df_line['strain'] = x_line
    df_line['stress'] = y_line
    E, intercept = calc_E(df_line,point1,point2)
    offset_df = df_line.copy()
    offset_df["offset_stress"] = E * (df_line['strain'] - offset) + intercept
    offset_df["offset_delta"] = (offset_df["offset_stress"] - offset_df['stress'])
    offset_df = offset_df.reset_index(drop=True)
    yield_index = offset_df["offset_delta"].abs().idxmin()
    yield_strength = offset_df.iloc[yield_index]['stress']
    return yield_strength, E, intercept, x_line, y_line


def calc_E(cr,point1,point2):
    temp = []
    for i in range(point1):
        X = cr['strain'].values[i:i + point2]
        Y = cr['stress'].values[i:i + point2]
        E, intercept = np.polyfit(X, Y, 1)
        temp.append([E, intercept])
    temp = np.array(temp)
    temp = pd.DataFrame(temp)
    temp.columns = ['E', 'yield']
    x = temp.iloc[temp['E'].argmax()]
    E = x[0]
    intercept = x[1]
    return E, intercept


def double_points(xxx):
    xxx2 = []
    for i in range(len(xxx) - 1):
        xxx2.append(xxx[i])
        xxx2.append((xxx[i] + xxx[i + 1]) / 2)
    xxx2.append(xxx[-1])
    return np.array(xxx2).reshape(-1, 2)


def read_rpt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines_vaild = lines[3:]
    data = []
    for line in lines_vaild:
        if line.strip():
            columns = line.split()
            if len(columns) == 2:
                data.append([float(columns[0]), float(columns[1])])
    return np.array(data)


def save_to_mat(data, output_path):
    data_dict = {'data': data}
    scipy.io.savemat(output_path, data_dict)


def rpt2mat(rpt_path, mat_path):
    data = read_rpt(rpt_path)
    save_to_mat(data, mat_path)
    if os.path.exists(rpt_path):
        os.remove(rpt_path)
    else:
        print(".rpt not found")
    print(f"Data has been saved as .mat, path:{mat_path}")


def count_lines_in_file(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            return len(lines)
    except FileNotFoundError:
        print(f"Error:File '{log_file_path}' not found.")
        return 0
    except Exception as e:
        print(f"Error:{e}")
        return 0


def calculate_EY(length=6, point1=350, point2=50, target_E=5000):
    rpt_path = 'adaqusModel1.rpt'
    mat_path = 'curve.mat'
    rpt2mat(rpt_path, mat_path)
    log_file = "E&Y.log"
    input_rpt = scipy.io.loadmat('curve.mat')['data']
    input_rpt[:, 0] /= length
    input_rpt[:, 1] /= length ** 2
    input_rpt = double_points(input_rpt)
    input_rpt = double_points(input_rpt)
    input_rpt = double_points(input_rpt)
    input_df = pd.DataFrame(input_rpt).dropna(axis=0, how='any')
    input_df.columns = ['strain', 'stress']
    Y, E, intercept, x_line, y_line = calc_yield(point1,point2,input_df)

    num_lines = count_lines_in_file(log_file)
    with open(log_file, "a") as log:
        log.write(f"Run {num_lines + 1}: E = {E:.4f}, Y = {Y:.4f}\n")
    E_res = abs(E - target_E)

    return np.array([-Y, E_res])