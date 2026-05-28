# Coiled-Tube Reactor

This directory contains the CFD-based coiled-tube reactor benchmark used in the XLO experiments. The scenario is adapted from the flow reactor design benchmark introduced in *Machine learning-assisted discovery of flow reactor designs* and is used here as a 50-dimensional expensive black-box optimization problem.

## Scenario description

To enhance the plug flow performance of a coiled-tube reactor at a low Reynolds number (Re ≤ 50), we conducted a systematic optimization of its geometry and pulsed flow operational parameters. This complex problem encompasses 50 decision variables, involving cross-sectional radius, axial and radial deviations, as well as pulse amplitude, frequency and Reynolds number. System performance was rigorously evaluated through computational fluid dynamics simulations using OpenFOAM, with each simulation duration set to 30 seconds. A pulsed tracer was injected at the reactor inlet, and the time-resolved concentration profile at the outlet was recorded to construct the residence time distribution (RTD). We fitted the RTD using the tanks-in-series model and calculated the dimensionless parameter N* to characterize the plug flow performance. Simultaneously, the mean squared error between the RTD and an ideal single-peak symmetric distribution was designated as the second objective.

In this implementation, the input vector `x` is a 50-dimensional normalized design vector. The benchmark returns two objectives:

- `obj:-N`: the negative tanks-in-series metric, used so that the original maximization of plug-flow performance is converted into minimization.
- `obj:MSE`: the mean squared error between the simulated RTD and the ideal single-peak symmetric RTD.

The 50 decision variables include pulsed-flow operating parameters, axial deviations, radial deviations, and cross-sectional radii of the reactor geometry.

## Requirements

The CFD benchmark is executed through Docker. The Docker image is `benchmark_complete.tar`.  

Before running this script, make sure that:

1. Docker is installed and running.
2. The benchmark image has been built locally and tagged as:

```bash
benchmark:complete
```

You can check this with:

```bash
docker images | grep benchmark
```

The Docker image should contain the reactor benchmark environment, including OpenFOAM, the benchmark API, and the Python environment used by the original reactor benchmark code.

## Running the CFD benchmark

The script calls the CFD benchmark by creating a Docker container for each function evaluation. Internally, each call executes the following logic:

1. Start a Docker container from the local image `benchmark:complete`.
2. Map the selected host port to the Flask service inside the container.
3. Activate the benchmark environment inside the container.
4. Start the Flask API from `reactor_design_problem/functions.py`.
5. Send the 50-dimensional design vector to the `/full_pulsed_flow` endpoint.
6. Receive the CFD-based objective values.
7. Stop and remove the container.

The API request is sent to:

```text
http://localhost:9001/full_pulsed_flow
```

with the following JSON structure:

```json
{
  "x": [50-dimensional normalized design vector],
  "z": [0.1, 0.1],
  "keep_files": false,
  "cpus": 10
}
```

The script uses the Docker container name `Running` and host port `9001` by default. Make sure that no existing container with the same name is active before running the script.

If a previous run was interrupted, remove the remaining container manually:

```bash
docker stop Running
docker rm Running
```

## Run

Execute:

```bash
python Running.py
```

The default script generates:

- 500 Sobol training samples;
- 1,000 uniformly sampled testing points;
- CFD objective values for all generated samples.

Because each sample requires a CFD benchmark evaluation through Docker, the full run can be computationally expensive. 

## Outputs

After execution, the script writes the following files:

```text
Training_set.npy
Training_set.xlsx
Testing_set.npy
Testing_set.xlsx
y_Training_set.npy
y_Training_set.xlsx
y_Testing_set.xlsx
Time-Training-set.npy
```

`Training_set.npy` and `Training_set.xlsx` contain the 50-dimensional Sobol training samples.

`Testing_set.npy` and `Testing_set.xlsx` contain the 50-dimensional random testing samples.

`y_Training_set.npy` and `y_Training_set.xlsx` contain the two CFD objective values for each training sample:

```text
[obj:-N, obj:MSE]
```

`y_Testing_set.xlsx` contains the two CFD objective values for each testing sample.

`Time-Training-set.npy` records the wall-clock time required for each training-sample evaluation.

## Notes

The Docker image `benchmark:complete` is assumed to have been constructed from the original reactor benchmark environment. If your local image has a different name, modify the following line in `Running.py`:

```python
"benchmark:complete"
```

If multiple CFD evaluations are run in parallel, assign different container names and host ports to avoid Docker name and port conflicts.

## References

1. Savage, T., Basha, N., McDonough, J., Krassowski, J., Matar, O. & del Rio Chanona, E. A. *Machine learning-assisted discovery of flow reactor designs*. *Nature Chemical Engineering*, 2024. https://doi.org/10.1038/s44286-024-00099-1

2. Reactor benchmark repository: https://github.com/trsav/reactor_benchmark

3. Pulsed reactor optimisation repository: https://github.com/OptiMaL-PSE-Lab/pulsed-reactor-optimisation
