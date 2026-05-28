import numpy as np
import requests
import json
import subprocess


def objective_function(x, defaults, whether_to_optimize, Ports, container_name):
    x = np.array(x).flatten()
    x_new = defaults.copy()
    j = 0
    for i in range(len(whether_to_optimize)):
        if whether_to_optimize[i] == 1:
            x_new[i] = x[j]
            j += 1
    docker_command = [
        "docker", "run", "-d",
        "-p", str(Ports) + ":5001",
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
    url = "http://localhost:" + str(Ports) + "/full_pulsed_flow"
    d = {"x": list(x_new), "z": [0.1, 0.1], "keep_files": False, 'cpus': 10}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(d))
    result = json.loads(response.text)
    y = np.array([result['obj:-N'], result['obj:MSE']])
    subprocess.run(["docker", "stop", container_name], check=True, capture_output=True)
    subprocess.run(["docker", "rm", container_name], check=True, capture_output=True)
    return y

