import subprocess
import os
import pandas as pd
import numpy as np
import CurveProcess
import time
import xlsxwriter as xw
from scipy.stats.qmc import Sobol


def show_section(title):
    print("\n" + "="*50)
    print(f"==  {title}  ==")
    print("="*50 + "\n")


def restart_sim(x):
    current_dir = os.path.dirname(__file__)
    all_files = os.listdir(current_dir)
    files_to_delete = [file for file in all_files if 'aqus' in file]
    files_to_delete.extend(['1.stl', '1.txt', 'stl.msg'])
    for filename in files_to_delete:
        file_path = os.path.join(current_dir, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f'{filename} has been deleted')
            else:
                print(f'{filename} not exit')
        except:
            pass
    output = objective_function_sim(x)
    return output


def objective_function_sim(x, target_E=5000):
    matlabExe = r"D:\Matlab2024b\bin\matlab.exe"
    matScriptFull = "STL_Modeling.m"
    stlOut = "1.stl"
    hypermeshExe = r"D:\Hyperworks2019\hm\bin\win64\hmopengl.exe"
    inpScript = "3Dinp.py"
    hypermeshScript = "command_1.tcl"
    abaqusExe = r'C:\SIMULIA\Commands\abaqus.bat'
    abaqusScript = "BatchProcess_2.py"
    log_file = 'E&Y.log'
    num_lines = CurveProcess.count_lines_in_file(log_file)

    # x_list = x.flatten().tolist()
    x_list = [0.8 for i in range(27)]
    df = pd.DataFrame(x_list)
    file_path = 'Porosity.xlsx'
    df.to_excel(file_path, index=False, header=False)

    sw_total_start = time.time()
    show_section("MATLAB MODELING START")
    sw_mat_start = time.time()
    try:
        subprocess.run([matlabExe, "-batch", f"run('{matScriptFull}')"])
    except:
        with open(log_file, "a") as log:
            log.write(f"Run {num_lines + 1}: Matlab Failed!\n")
        output = restart_sim(x)
        return output
    sw_mat_end = time.time()
    print(f"MATLAB_time_spent: {sw_mat_end - sw_mat_start:.2f}s")
    show_section("MATLAB MODELING DONE")

    show_section("HYPERMESH PREPROCESSING START")
    if not os.path.exists(stlOut):
        raise FileNotFoundError(f"No STL generated: {stlOut}")
    print('HyperMesh Processing...')
    sw_hm_start = time.time()
    try:
        os.environ['ALTAIR_LICENSE_PATH'] = os.getenv('LICENSE_FILE', '')
        os.environ['HM_DISABLE_GUI_WARNINGS'] = "1"
        os.environ['HM_PREPROCESSOR_MODE'] = "batch"
        subprocess.run([hypermeshExe, "-batch", "-noconsole", "-tcl", hypermeshScript])
        subprocess.run([r"python", os.path.join(os.path.dirname(__file__), inpScript)])
    except:
        with open(log_file, "a") as log:
            log.write(f"Run {num_lines + 1}: HyperMesh Failed!\n")
        output = restart_sim(x)
        return output
    sw_hm_end = time.time()
    print(f"HyperMesh_time_spent: {sw_hm_end - sw_hm_start:.2f}s")
    show_section("HYPERMESH PREPROCESSING DONE")

    print('Abaqus Processing...')
    sw_aq_start = time.time()
    try:
        subprocess.run([abaqusExe, 'cae', 'noGUI=' + abaqusScript])
    except:
        with open(log_file, "a") as log:
            log.write(f"Run {num_lines + 1}: Abaqus Failed!\n")
        output = restart_sim(x)
        return output
    try:
        output = CurveProcess.calculate_EY(length=6, point1=350, point2=50, target_E=target_E)
    except:
        with open(log_file, "a") as log:
            log.write(f"Run {num_lines + 1}: Post-Processing Failed!\n")
        output = restart_sim(x)
        return output
    sw_total_end = time.time()
    print(f"Abaqus_time_spent: {sw_total_end - sw_aq_start:.2f}s")
    show_section("ALL TASKS COMPLETED")
    print(f"== Total_time_spent: {sw_total_end - sw_total_start:.2f}s ==")
    print('Full Process Completed Successfully!')
    return output


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


    num_vars_all = 27

    names_all = ['x' + str(i + 1) for i in range(num_vars_all)]
    names_all = np.array(names_all)

    bounds_all = np.zeros((num_vars_all, 2))
    bounds_all[:, 0] = 0.1
    bounds_all[:, 1] = 0.8

    sample_size = 300
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
    obj = 2
    y_Training_set = np.zeros((sample_size, obj))
    for i in range(sample_size):
        print('Training ' + str(i))
        time_to_start = time.time()
        y_Training_set[i, :] = objective_function_sim(Training_set[i, :])
        Time[i] = time.time() - time_to_start
        print(y_Training_set[i, :])

    xw_toExcel(y_Training_set, 'y_Training_set.xlsx')
    np.save('y_Training_set.npy', y_Training_set)
    np.save('Time-Training-set.npy', Time)


    y_Testing_set = np.zeros((test_size, obj))
    for i in range(test_size):
        print('Testing ' + str(i))
        y_Testing_set[i, :] = objective_function_sim(Testing_set[i, :])

    xw_toExcel(y_Testing_set, 'y_Testing_set.xlsx')
    np.save('y_Testing_set.npy', y_Testing_set)