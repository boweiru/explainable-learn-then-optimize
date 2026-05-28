# Human-Powered Aircraft

This directory contains the human-powered aircraft (HPA) benchmark used in the XLO experiments. The scenario is adapted from the HPA303 benchmark in *Single and Multi-Objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design*. In this study, we use `n_div = 7` and `level = 1`, which gives a 55-dimensional aircraft wing design problem.

## Scenario description

Human-powered aircraft rely exclusively on the pilot’s physical strength for propulsion, thus imposing extremely stringent demands on aerodynamic efficiency and structural lightweighting. Based on wind tunnel experimental data for the DAE airfoil series, which includes 756 drag coefficient measurement points, we comprehensively modeled the coupled aerodynamic and structural behavior of the wing using analytical engineering methodologies. We utilized Prandtl’s lifting-line theory to calculate the overall lift and drag, and to determine the cruise speed. Furthermore, beam-column theory was applied to rigorously assess the structural safety of the main spar under both flight and parked conditions. This scenario segments the wing into seven distinct sections, enabling the joint optimization of 55 parameters encompassing wing geometry, sectional aerodynamic attitude, main spar structure, and global parameters. The objective is to simultaneously minimize the aircraft’s drag, angle of attack, and structural weight, with structural safety constraints seamlessly integrated into the objective function as penalty terms.

In this implementation, the input vector `x` is a 55-dimensional normalized design vector in `[0, 1]^55`. Each vector is evaluated by the HPA303 aircraft model with:

```python
n_div = 7
level = 1
NORMALIZED = True
```

The benchmark returns three objective values:

- `D`: aerodynamic drag;
- `alpha`: root angle of attack;
- `W`: structural weight.

All objectives are minimized. Structural safety constraints are incorporated into the returned objective values through the penalty formulation implemented in the original HPA benchmark.

## Files

```text
.
├── Running.py
├── problem.py
├── designer.py
├── airfoil_info/
│   ├── DAE11.xlsx
│   ├── DAE21.xlsx
│   ├── DAE31.xlsx
│   └── DAE41.xlsx
├── Training_set.npy
├── Training_set.xlsx
├── Testing_set.npy
├── Testing_set.xlsx
├── y_Training_set.npy
├── y_Training_set.xlsx
├── y_Testing_set.npy
├── y_Testing_set.xlsx
├── Time-Training-set.npy
└── README.md
```

`Running.py` generates sample points, evaluates the HPA303 benchmark, and saves the corresponding objective values.

`problem.py` defines the HPA benchmark problem classes and exposes the HPA303 evaluation interface used by this scenario.

`designer.py` contains the aircraft design evaluator. It decodes the design vector into wing geometry, aerodynamic attitude, main-spar structural parameters, and global aircraft parameters, then computes the aerodynamic and structural responses.

`airfoil_info/` stores the wind-tunnel data for the DAE airfoil series. The four Excel files contain the aerodynamic lookup tables used during section-wise lift and drag calculations:

- `DAE11.xlsx`
- `DAE21.xlsx`
- `DAE31.xlsx`
- `DAE41.xlsx`

The `.npy` and `.xlsx` files store the generated input samples and evaluated objective values.

## Requirements

The HPA benchmark is an analytical engineering benchmark.

Install the required Python packages:

```bash
pip install numpy scipy pandas openpyxl matplotlib pymoo xlsxwriter
```

The original HPA package can also be installed directly from GitHub:

```bash
pip install git+https://github.com/Nobuo-Namura/hpa
```

For this repository, however, the required benchmark source files are already included locally as `problem.py`, `designer.py`, and `airfoil_info/`. Keep these files in the same directory structure when running `Running.py`.

## System evaluation

Each aircraft design is evaluated through the following procedure:

1. `Running.py` provides a normalized 55-dimensional input vector `x`.
2. The HPA303 benchmark decodes `x` into physical aircraft design variables according to `n_div = 7` and `level = 1`.
3. The wing is divided into seven sections, covering wing geometry, sectional aerodynamic attitude, main-spar structural design, and global aircraft parameters.
4. The DAE airfoil lookup tables in `airfoil_info/` are used to retrieve section-wise aerodynamic coefficients.
5. Prandtl’s lifting-line theory is used to compute the overall lift, drag, and cruise condition.
6. Beam-column theory is used to assess the main-spar structural safety under flight and parked conditions.
7. Constraint violations are converted into penalty terms and integrated into the objective values.
8. The final three-objective response is saved as `[D, alpha, W]`.

This evaluation is deterministic and computationally lightweight, but it retains the coupled aerodynamic-structural characteristics of aircraft design.

## Run

Execute:

```bash
python Running.py
```

The default workflow evaluates the generated training and testing samples and writes the corresponding objective values to both NumPy and Excel files.

## Outputs

After execution, the script writes the following files:

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

`Training_set.npy` and `Training_set.xlsx` contain the 55-dimensional training samples.

`Testing_set.npy` and `Testing_set.xlsx` contain the 55-dimensional testing samples.

`y_Training_set.npy` and `y_Training_set.xlsx` contain the three objective values for each training sample:

```text
[D, alpha, W]
```

`y_Testing_set.npy` and `y_Testing_set.xlsx` contain the three objective values for each testing sample.

`Time-Training-set.npy` records the wall-clock time required for each training-sample evaluation.

## Notes

The benchmark uses normalized design variables. Therefore, all sampled values should remain within `[0, 1]`. If `NORMALIZED=False` is used, the input vector must be provided in the original physical variable domains defined by the HPA benchmark.

The `airfoil_info/` directory is required at runtime. If the evaluator cannot locate the DAE Excel files, check that the folder remains in the expected relative path.

## References

1. Namura, N. *Single and Multi-Objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design*. In Proceedings of the 13th International Conference on Evolutionary Multi-Criterion Optimization, 195–210. Springer, 2025.

2. Original HPA benchmark repository: https://github.com/Nobuo-Namura/hpa.
