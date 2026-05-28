import numpy as np
import time
import xlsxwriter as xw
from scipy.stats.qmc import Sobol
import requests
import json
import subprocess


def objective_function(x, Ports, container_name):
    docker_command = [
        "docker", "run", "-d",
        "-p", str(Ports)+":5001",
        "--name", container_name,
        "benchmark:complete",
        "bash", "-c",
        "source ~/.bashrc && micromamba activate benchmark_env && cd /root && export FLASK_APP=reactor_design_problem/functions.py && flask run --host=0.0.0.0 --port=5001"
    ]
    subprocess.run(docker_command, check=True, capture_output=True, text=True)
    max_retries = 10000
    for i in range(max_retries):
        result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
            text=True
        )
        if "Serving Flask app" in result.stdout or "Running on" in result.stderr:
            print("Flask 服务已启动")
            break
    x = np.array(x).flatten()
    url = "http://localhost:" + str(Ports) + "/full_pulsed_flow"
    d = {"x": list(x), "z": [0.1, 0.1], "keep_files": False, 'cpus': 10}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(d))
    result = json.loads(response.text)
    y = np.array([result['obj:-N'], result['obj:MSE']])
    subprocess.run(["docker", "stop", container_name], check=True, capture_output=True)
    subprocess.run(["docker", "rm", container_name], check=True, capture_output=True)
    return y


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


    num_vars_all = 50

    names_all = ['x' + str(i + 1) for i in range(num_vars_all)]

    bounds_all = np.zeros((num_vars_all, 2))
    for i in range(num_vars_all):
        bounds_all[i, 1] = 1

    sample_size = 500
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
        y_Training_set[i, :] = objective_function(Training_set[i, :], 9001, 'Running')
        print(y_Training_set[i, :])
        Time.append(time.time() - time_to_start)
    Time = np.array(Time)
    np.save('Time-Training-set.npy', Time)

    xw_toExcel(y_Training_set, 'y_Training_set.xlsx')
    np.save('y_Training_set.npy', y_Training_set)

    y_Testing_set = np.zeros((test_size, obj))
    for i in range(test_size):
        print('Testing ' + str(i))
        y_Testing_set[i, :] = objective_function(Testing_set[i, :], 9001, 'Running')
    xw_toExcel(y_Testing_set, 'y_Testing_set.xlsx')

