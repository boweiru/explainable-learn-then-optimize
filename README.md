# Navigating High-Dimensional Complex Systems with Black-Box Nature via Explainable Learn-then-Optimize

This repository hosts the full implementation and experimental scripts that accompany the manuscript
"Navigating High-Dimensional Complex Systems with Black-Box Nature via Explainable Learn-then-Optimize".
It provides the universal, domain-agnostic **explainable learn-then-optimize (XLO)** framework together
with all multidisciplinary experimental scenarios used to demonstrate it. XLO simultaneously
(i) profiles the hidden input-output statistical dependencies in high-dimensional black-box systems
through a sample-efficient AI surrogate, and (ii) exploits the discovered low-dimensional effective
space to enable interpretable, accelerated system navigation.

---

![Fig. 1](F:\JinJixia\Code\Screening-Optimization\GitHub\Fig1.svg)

The framework comprises four logically coupled stages, summarized below (corresponding to panels a-d of
Fig. 1 in the main text).

| Panel | Stage | Idea                                                                                                                                                                                                                                                                                                                                                                                     |
| :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **a** | High-dimensional complex system | High-dimensional input factors drive a black-box complex system to generate multi-output responses. The exploration is constrained by two confounding computational bottlenecks: black-box opacity and the curse of dimensionality.                                                                                                                                                      |
| **b** | Sample-efficient metamodel | An ensemble AI is employed as a surrogate model to replace the original system. Its sample complexity grows **sublinearly**, rather than exponentially, with dimensionality, overcoming the computational intractability of high-dimensional spaces.                                                                                                                                     |
| **c** | Asymmetry importance learning | By quantifying the variance contribution of each factor, a ubiquitous asymmetry in factor importance is revealed: **fewer than 20% of the input factors dominate over 90% of the system's output variance**. Representative factors are automatically identified and weakly contributing factors are stripped away, uncovering the low-dimensional effective space within the black box. |
| **d** | Tractable black-box optimization | The extracted low-dimensional effective space is utilized for system navigation, rendering the optimization process (which otherwise struggles to converge in high-dimensional spaces) highly efficient and tractable. Runtime is reduced by from **29.5% to 98.6%**, accelerating the path toward interpretable scientific discovery.                                                   |

---

## Repository layout

```text
GitHub/
├── XLO/                       # The reusable XLO framework (template scripts)
├── WFG3/                      # 100-D synthetic benchmark
├── Malaria/                   # 23-D rural malaria transmission (OpenMalaria)
├── Junction/                  # 29-D isolated urban junction (SUMO)
├── Reactor/                   # 50-D coiled-tube reactor (OpenFOAM via Docker)
├── Aircraft/                  # 55-D human-powered aircraft (HPA303)
├── Corridor/                  # 60-D urban arterial corridor (SUMO)
├── Material/                  # 27-D architected material (USED-immune case, Sup. Note 6)
└── Cross_domain_insights/     # Cross-domain cNDI analysis (Fig. 6)
```

Each complex-system folder mirrors the four logical stages of XLO (`Running/`, `MSG/`, `Learning/`,
`Optimization/`); `Material/` retains only `Running/` and `Learning/` because it is used to discuss the
boundary case in which the latent effective dimensionality does *not* exist (see Supplementary Note 6).

---

## 1. `XLO/` — the reusable framework

`XLO/` contains the four template scripts that constitute the framework itself. The scripts in every
complex-system folder are scenario-specific instantiations of these templates (input dimensionality,
parameter bounds, default values, the system's `objective_function`, and hyperparameters such as
reference points are filled in for each scenario).

| Script | Functionality                                                                                                                                                                                                                                                                                                                                                                                | Key inputs | Key outputs | Hand-off to next stage |
| :--- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :--- | :--- | :--- |
| `Running.py` | Initial sampling and system evaluation. Generates Sobol quasi-random training samples and a uniformly random testing set, then calls the scenario's black-box `objective_function`.                                                                                                                                                                                                          | `num_vars_all`, `bounds_all`, `sample_size`, `test_size`, `obj` | `Training_set.npy/.xlsx`, `Testing_set.npy/.xlsx`, `y_Training_set.npy/.xlsx`, `y_Testing_set.npy/.xlsx`, `Time-Training-set.npy` | Provides the labelled samples consumed by `MSG.py`. |
| `MSG.py` | Defines the **Multi-output Stacked Generalization** metamodel (class `BaggingMeta`). <br/>Level 1: ten XGBoost base learners trained on 75% bootstrap resamples; <br/>Level 2: per-objective Gaussian Process meta-learners producing predictive mean and standard deviation.                                                                                                                | `Training_set.xlsx`, `y_Training_set.xlsx`, `Testing_set.xlsx`, `y_Testing_set.xlsx`, `obj`, `n_estimators` | `MSG.pkl` (serialized fitted metamodel), per-objective WMAPE printout | Provides the trained surrogate used by `Learning.py` and `Optimization.py`. |
| `Learning.py` | Asymmetry-importance learning. Uses Monte-Carlo Sobol resampling on the fitted MSG to compute the multi-output **total-effect indices** $P_i^{\text{total}}$, then applies the **Jenks Natural Breaks** clustering (`min_gvf` threshold) to label each factor as representative or redundant.                                                                                                | `MSG.pkl`, `num_vars_all`, `bounds_all`, `resample_size`, `min_gvf` | `PT.npy`, `whether_to_optimize-Natural-Breaks.npy`, `time-Sensitivity-Analysis.npy`, `time-Natural-Breaks.npy` | The label vector `whether_to_optimize.npy` (1 = representative, 0 = redundant) defines the low-dimensional effective space consumed by `Optimization.py`. |
| `Optimization.py` | Tractable multi-objective Bayesian black-box optimization in the **hidden** effective space. Implements the modified-hypervolume Expected Improvement Matrix (EIM) acquisition function, maximizes it through a genetic algorithm (`sko.GA`), evaluates the new candidate by calling the original `Running.objective_function`, and updates the Pareto front via fast non-dominated sorting. | `MSG.pkl`, `whether_to_optimize.npy`, `Training_set.npy`, `y_Training_set.npy`, reference points `r`, `r_good`, `defaults`, `max_iteration`, GA hyperparameters | `PF-repetition-{j}.npy`, `PS-repetition-{j}.npy`, `All_PF-repetition-{j}.pkl`, `RHD-repetition-{j}.npy`, `Time-repetition-{j}.npy` | Final Pareto fronts/sets and relative hypervolume ratio (RHR) trajectories for downstream analysis. |

The chained data dependency is therefore

```text
Running.py  ──►  Training/Testing samples  ──►  MSG.py  ──►  MSG.pkl
                                                                │
                                                                ▼
                                       Learning.py  ──►  whether_to_optimize.npy + PT.npy
                                                                │
                                                                ▼
                                       Optimization.py  ──►  PF, PS, RHR per repetition
```

`Extended Data Fig1.svg` in `XLO/` is the algorithmic schematic of this pipeline.

---

## 2. Complex systems and experiments

XLO is comprehensively evaluated on **seven complex systems** spanning an analytical synthetic
function, agent-based spatiotemporal simulations, and physics-driven engineering designs. Numbers in
parentheses below indicate the dimensionality of each system submodule.

![Fig. 2](F:\JinJixia\Code\Screening-Optimization\GitHub\Fig2.svg)

### Common subfolder layout of each complex system

Because the four XLO stages are identical in structure across systems, each complex-system folder is
organized along the same template (illustrated for `Aircraft/`):

```text
<System>/
├── Running/         # System-specific evaluator + initial sampling
├── MSG/             # Metamodel training and benchmarking
├── Learning/        # Asymmetry-importance learning
└── Optimization/    # Black-box optimization in full vs. effective space
```

The `Running/` subfolder of every system contains a dedicated **README.md** with detailed
documentation of that scenario's evaluation environment, software requirements, command-line
invocations, file dependencies, and objective conventions. Refer to those README files for full
operational details.

Below is a unified description of what each subfolder *does* and *which figure or table it serves*.
This pattern applies to all six full systems (`WFG3`, `Malaria`, `Junction`, `Reactor`, `Aircraft`,
`Corridor`); `Material` retains only `Running/` and `Learning/` because it is the USED-immune case
discussed in Supplementary Note 6.

#### 2.1 `Running/` — system evaluator and initial sampling

This subfolder wraps the **original domain-specific simulator/evaluator** as a unified
`objective_function(x)` callable, generates the initial Sobol training set and the random testing set,
and stores the corresponding multi-objective responses together with per-sample wall-clock times.
External engines (OpenMalaria, SUMO, OpenFOAM-in-Docker, Abaqus, HPA analytical solver, pymoo's WFG 3)
are dispatched from `Running.py`.

| Code | Functionality | Inputs | Outputs | Serves |
| :--- | :--- | :--- | :--- | :--- |
| `Running.py` | Generates Sobol training samples, uniformly random testing samples, and invokes the system-specific `objective_function`. | Scenario configuration (dimensionality, bounds, sample sizes); external simulator binaries / engines listed in each scenario's local README | `Training_set.npy/.xlsx`, `Testing_set.npy/.xlsx`, `y_Training_set.npy/.xlsx`, `y_Testing_set.npy/.xlsx`, `Time-Training-set.npy` | Supplies the labelled samples that feed every downstream stage; supports the per-system descriptions in Methods and Supplementary Note 2; the runtime traces underlie Supplementary Figure 14 (computational cost). |
| Scenario assets (e.g. `wu_5000_28.xml`, `*.net.xml`, `basic.vType.xml`, `problem.py`, `designer.py`, `airfoil_info/`, `benchmark_complete.tar`, `STL_Modeling.m`, `command_1.tcl`, ground-truth `true_*.csv` / `fieldData.txt`) | Provide the simulator inputs, network/geometry definitions, signal plans, vehicle types, calibration ground truth, or FEM/HyperMesh templates needed by `Running.py` to execute one black-box evaluation. | -- | Consumed only at runtime by `Running.py`; details are documented in each scenario's `Running/README.md`. | These define the physical/biological/mathematical setups portrayed in **Fig. 2** and in the Methods/Supplementary Note 2 system-description sections (e.g., **Supplementary Figures 6 and 7** for the Junction and Corridor networks). |

#### 2.2 `MSG/` — metamodel training and benchmarking

This subfolder trains the MSG surrogate at progressively increasing training-sample sizes and
benchmarks it against mainstream sample-efficient ML baselines (GP, SVR, RF, LightGBM, CatBoost,
XGBoost) and against alternative ensemble configurations (data-heterogeneous, model-heterogeneous,
fully heterogeneous).

| Code | Functionality | Inputs                                                                                | Outputs | Serves                                                                 |
| :--- | :--- |:--------------------------------------------------------------------------------------| :--- |:-----------------------------------------------------------------------|
| `MSG.py` | Defines `BaggingMeta` and trains the canonical MSG metamodel; reports per-objective WMAPE. | `Training_set.xlsx`, `y_Training_set.xlsx`, `Testing_set.xlsx`, `y_Testing_set.xlsx`  | `MSG.pkl`, `PRED-MSG-{n}.npy` | **Fig. 3 a1-f1** (predictive accuracy vs. ML baselines).               |
| `MSG_Model.py`, `MSG_Model_Data.py`, `MSG_CatBoost.py` | Train model-heterogeneous, fully-heterogeneous, and CatBoost-based MSG variants. | Same labelled training/testing sets                                                   | `PRED-MSG_Model-{n}.npy`, `PRED-MSG_Model_Data-{n}.npy`, `PRED-MSG_CatBoost-{n}.npy` | **Fig. 3 a2-f2** (ablation of base-learner configurations).            |
| Baseline scripts (implicit in `PRED-{GP,SVR,RF,LGB,XGB,CatBoost}-{n}.npy`) | Train and evaluate the corresponding mainstream ML baselines at the same sample sizes. | Same labelled sets                                                                    | `PRED-<model>-{n}.npy` | **Fig. 3 a1-f1**.                                                      |
| `Comparison.py` | Obtain predicted values on the test set across training-sample sizes for MSG/baselines. | `Training_set.xlsx`, `y_Training_set.xlsx`, `Testing_set.xlsx`, `y_Testing_set.xlsx`  | `PRED-*.npy` | **Fig. 3**.                                                            |
| `WMAPE for single.py`, `WMAPE for MSG.py` | Aggregates and plots the per-system MSG-vs.-baseline and MSG-variant comparisons. | All `PRED-*.npy`, `y_Testing_set.npy`                                                 | `MSG VS Single <System>.svg`, `MSG comparison <System>.svg` | **Fig. 3** and **Extended Data Fig. 2** (sublinear sample complexity). |

#### 2.3 `Learning/` — asymmetry-importance learning

This subfolder computes the multi-output **total-effect sensitivity indices** $P_i^{\text{total}}$
via Monte Carlo sampling on the trained MSG, evaluates Monte Carlo convergence, and runs the Jenks
Natural Breaks clustering that automatically identifies the representative factors.

| Code | Functionality | Inputs | Outputs | Serves |
| :--- | :--- | :--- | :--- | :--- |
| `Test_MC_convergence.py`, `Monte Carlo sample convergence curve.py` | Sweep the MC sample size and trace the WMAPE convergence of the estimated indices. | `MSG.pkl` (and analytical ground truth for WFG 3) | `PT-convergence.npy`, `Monte Carlo <System> 3Dball.svg` | **Fig. 4 a1-f1** (MC convergence; the asterisk marks the adopted MC sample size). |
| `Sensitivity_indices.py` | Runs `Multioutputs_Sensitivity_Analysis` (see `XLO/Learning.py`) at the chosen MC sample size; performs Natural-Breaks clustering. | `MSG.pkl`, `num_vars_all`, `bounds_all`, `resample_size`, `min_gvf` | `PT.npy`, `whether_to_optimize-Natural-Breaks.npy`, `Time-Sensitivity-Analysis.npy`, `Time-Natural-Breaks.npy` | **Fig. 4 a2-f2** (total-effect indices, cluster labels, representative subsets). |
| `Compute_cum_PT.py` | Sorts the total-effect indices in descending order and computes their cumulative-variance contribution. | `PT.npy` | `cum_PT_descending.npy` | The lavender cumulative-variance curve in **Fig. 4 a2-f2**. |
| `Sensitivity Index Bar Chart.py` | Plots the bar chart of total-effect indices with the representative-factor highlighting frame. | `PT.npy`, `cum_PT_descending.npy`, `whether_to_optimize-Natural-Breaks.npy` | `Sensitivity <System>.svg` | The **Fig. 4 a2-f2** rendering. |

In the case of `Material/`, this subfolder serves Supplementary Note 6 and the corresponding
discussion of the USED boundary condition rather than Fig. 4.

#### 2.4 `Optimization/` — black-box optimization in full vs. effective space

For each system, the optimization subfolder is organized as a set of parallel algorithm subdirectories
(`XLO/`, `MOBO<d>/`, `MOBO<|R|>/`, `NSGA<d>/`, `NSGA<|R|>/`) so that the same optimization budget is
executed (i) in the *original full-dimensional* space by vanilla NSGA-II and MOBO and (ii) in the
*low-dimensional effective space* identified by Learning. Aggregation scripts then assemble the
convergence curves and Pareto fronts.

| Code | Functionality                                                                                                                            | Inputs | Outputs | Serves                                                                                            |
| :--- |:-----------------------------------------------------------------------------------------------------------------------------------------| :--- | :--- |:--------------------------------------------------------------------------------------------------|
| `XLO/XLO.py` (and its scenario-specific `Running.py`, `MSG.py`, `MSG.pkl`, `whether_to_optimize-Natural-Breaks.npy`) | Runs the EIM-modified-hypervolume Bayesian loop in the latent effective space using the trained MSG and the natural-breaks labels.       | Trained `MSG.pkl`, labelled samples, reference points `r`, `r_good`, GA settings | `PF-repetition-{j}.npy`, `PS-repetition-{j}.npy`, `RHD-repetition-{j}.npy`, `Time-repetition-{j}.npy`, `All_PF-repetition-{j}.pkl` | **Fig. 5 a1-f1** (RHR trajectories) and **Fig. 5 a2-f2** (Pareto-front distributions).            |
| `MOBO<d>/MOBO<d>.py` and `NSGA<d>/NSGA<d>.py` | Vanilla multi-objective Bayesian optimization / NSGA-II in the **full** $d$-dimensional space.                                           | Initial samples, full-space bounds | `PF-`, `PS-`, `RHD-`, `Time-repetition-{j}.npy` | Baseline curves in **Fig. 5**.                                                              |
| `MOBO<| R                                                                                                                                        |>/` and `NSGA<|R| >/`                                                                                               | The same MOBO and NSGA-II algorithms but restricted to the **representative** factor subset (low-dimensional effective space), with redundant factors fixed at the same defaults used by XLO. | Same as above plus `whether_to_optimize-Natural-Breaks.npy` and `defaults` | Same set of repetition files | **Fig. 5** USED-guided baselines (demonstrating that the speed-up follows the discovered effective space, not the specific optimizer). |
| `Algorithm convergence curve.py` | Aggregates the `RHD-repetition-*.npy` files of every algorithm into the mean-and-shaded-band convergence plot.                           | `RHD-repetition-{j}.npy` from every algorithm folder | `RHR <System>.svg` | **Fig. 5 a1-f1** rendering.                                                                       |
| `Pareto front.py` | Pools the final `PF-repetition-{j}.npy` files of every algorithm and renders the comparative Pareto-front scatter / distribution figure. | All `PF-repetition-{j}.npy` | `PF <System>.svg` | **Fig. 5 a2-f2** rendering.                                                                       |
| `Time-repetition-{j}.npy` time traces (per algorithm) | Provide the wall-clock budgets used to verify the **29.5%-98.6%** runtime reduction.                                                     | -- | -- | **Supplementary Figure 14** (computational cost) and the runtime numbers quoted in the main text. |

Together, the four subfolders of every system reproduce the full pipeline summarized in Fig. 1 and
produce all per-system panels of Figs. 3-5.

---

## 3. Cross-domain systemic insights

The `Cross_domain_insights/` folder consolidates the optimization outputs from all six complex
systems and asks a structural question: across disparate domains, **do representative factors and
redundant factors exhibit different dispersion patterns within the Pareto-optimal solution space?**
This analysis underpins **Fig. 6** of the main text.

```text
Cross_domain_insights/
├── cNDI.py                    # Main analysis and plotting script
├── README.md                  # Detailed documentation of the analysis
├── Results/                   # PF/PS .npy files for each system, algorithm and repetition
├── cNDI.svg                   # Final Fig. 6 vector graphic
├── cNDI_raw_cNDI.csv          # Raw factor-level cNDI records
├── cNDI_summary.csv           # Aggregated cNDI statistics per system / algorithm / group
├── cNDI_rep_vs_red_tests.csv  # Mann-Whitney U test results (representative vs. redundant)
└── cNDI_run_info.csv          # Run-level diagnostics
```

`cNDI.py` loads the paired Pareto-front (PF) and Pareto-set (PS) `.npy` files exported by every
optimization run, clusters each Pareto front into local trade-off regions, and computes a
**Conditional Normalized Dispersion Index (cNDI)** for every input factor *within* comparable
trade-off regions. Conditioning on PF clusters is the key methodological choice — it prevents
legitimate trade-off-driven movement in representative factors from being misread as random drift, as
would happen with a global NDI. A **Mann-Whitney U test** then compares the cNDI distributions of
representative vs. redundant factors.

The recovered cross-domain pattern is consistent and statistically significant: **redundant factors
drift broadly across their feasible domains (high cNDI, near the uniform reference of 1), whereas
representative factors converge tightly within narrow bands (low cNDI)**. This dichotomy persists
across entirely different optimization oracles (vanilla NSGA-II/MOBO and XLO), supporting the
interpretation that the latent effective dimensionality discovered by XLO is an **intrinsic
structural property of complex systems** rather than a methodological artefact.

A detailed walkthrough of the cNDI definition, the PF-clustering step, file naming conventions for the
`Results/` folder, and the precise interpretation of every output CSV is provided in
[`Cross_domain_insights/README.md`](Cross_domain_insights/README.md).

---

## Reproducing the experiments

A typical end-to-end reproduction of one scenario is:

```bash
# 1. Initial sampling and system evaluation (writes Training/Testing sets + objectives)
cd <System>/Running        && python Running.py

# 2. Train and benchmark the MSG metamodel (writes MSG.pkl + PRED-*.npy)
cd ../MSG                  && python MSG.py
                              # plus the baseline / variant scripts for Fig. 3

# 3. Asymmetry-importance learning (writes PT.npy + whether_to_optimize-Natural-Breaks.npy)
cd ../Learning             && python Sensitivity_indices.py
                              # plus Test_MC_convergence.py for Fig. 4 a1-f1

# 4. Optimization in the effective space and full-dimensional baselines
cd ../Optimization/XLO     && python XLO.py
cd ../MOBO<d>              && python MOBO<d>.py
cd ../NSGA<d>              && python NSGA<d>.py
                              # repeat for the |R|-dimensional baselines

# 5. Aggregate the per-algorithm outputs into Fig. 5 panels
cd ..                      && python "Algorithm convergence curve.py"
                           && python "Pareto front.py"

# 6. Cross-domain analysis (after all six systems are complete)
cd ../../Cross_domain_insights && python cNDI.py
```

Each scenario's `Running/README.md` documents the external simulators (OpenMalaria, SUMO, OpenFOAM
inside Docker, Abaqus + HyperMesh + MATLAB, the HPA303 analytical evaluator) and the operating-system
specific configuration paths required to run that scenario's `Running.py`.

---

## Hardware

All experiments were conducted on a Dell Precision 7875 Tower equipped with an AMD Ryzen Threadripper
PRO 7995WX 96-core CPU at 2.50 GHz, 64 GB of RAM, and an NVIDIA GeForce RTX 4090 GPU.
