# WFG 3 Mathematical Function

This directory contains the WFG 3 benchmark scenario used in the XLO experiments. The case is configured as a five-objective, 100-variable synthetic multi-objective optimization problem with a known low effective dimensional structure. It is used to verify whether XLO can recover the ground-truth representative variables without using analytical prior knowledge.

## Scenario description

We selected the five-objective, 100-variable variant of the WFG 3 test problem with position parameters. The first four decision variables are known to be influential, giving WFG 3 an inherently low effective dimensionality. Redundant variables $x_i$ ($i = 5,6,\ldots,100$) were fixed at their Pareto-optimal default, $2(i + 1) \times 0.35$.

In this implementation, the benchmark is instantiated with:

```python
obj = 5
num_vars_all = 100
k = 4
```

where `obj` is the number of objectives, `num_vars_all` is the number of decision variables, and `k` is the number of position-related variables.

## Why WFG 3 has low effective dimensionality

WFG 3 is a standard multi-objective benchmark with an analytically known variable structure. In the WFG family, decision variables are separated into position-related variables and distance-related variables. In this case, `k = 4`, meaning that only the first four variables determine the intrinsic position on the Pareto front.

The remaining variables, `x5` to `x100`, are distance-related or degenerated dimensions. Under the Pareto-optimal default setting, these variables do not introduce independent trade-off directions for the five objectives. Therefore, the 100-dimensional search space contains only a 4-dimensional effective structure:

```text
Representative variables: x1, x2, x3, x4
Redundant variables:      x5, x6, ..., x100
```

This makes WFG 3 a controlled benchmark for evaluating factor-screening accuracy. Unlike the real-world black-box systems in the paper, the representative variables of WFG 3 are known in advance. XLO is therefore expected to identify the first four variables as representative and classify the remaining 96 variables as redundant.

## Requirements

Install the required Python packages:

The script uses:

- `pymoo` for the WFG 3 function evaluation;
- `scipy.stats.qmc.Sobol` for Sobol quasi-random sampling;
- `xlsxwriter` for exporting sample and objective matrices to Excel files;
- `numpy` for array operations.

## Decision-variable bounds and defaults

The 100 decision variables follow the standard WFG variable bounds. In the code, the upper bound of the `i`-th zero-based coordinate is:

```python
bounds_all[i, 1] = 2 * (i + 1)
```

Thus, the one-based variable `x_i` has the domain:

```text
0 <= x_i <= 2i,   i = 1, 2, ..., 100
```

The default value used for fixing redundant variables is:

```python
default_values = np.array([float(2 * (i + 1) * 0.35) for i in range(num_vars_all)])
```

That is, the default value of `x_i` is `0.35 × 2i` under one-based indexing.

## Run

Execute:

```bash
python Running.py
```

By default, the script generates:

- 1,000 Sobol training samples in the 100-dimensional WFG 3 domain;
- 1,000 uniformly sampled testing points;
- five objective values for each sample point.

The sample size can be adjusted in `Running.py`:

```python
sample_size = 1000
test_size = 10000
```

## Outputs

The script writes the following files:

```text
Training_set.npy
Training_set.xlsx
Testing_set.npy
Testing_set.xlsx
y_Training_set.npy
y_Training_set.xlsx
y_Testing_set.npy
y_Testing_set.xlsx
Time-Training-set.npy
```

`Training_set.npy` and `Training_set.xlsx` contain the 100-dimensional Sobol training samples.

`Testing_set.npy` and `Testing_set.xlsx` contain the 100-dimensional random testing samples.

`y_Training_set.npy` and `y_Training_set.xlsx` contain the five WFG 3 objective values for the training samples.

`y_Testing_set.npy` and `y_Testing_set.xlsx` contain the five WFG 3 objective values for the testing samples.

`Time-Training-set.npy` records the wall-clock time used for each training-sample evaluation.
