
# Explainable Learn-then-Optimize (XLO) Framework  
  
## 1. Algorithm Workflow  
  
![Extended Data Fig. 1](Extended%20Data%20Fig1.svg)  
  
**Detailed algorithmic and mathematical workflow of the XLO computational framework.**  
* **a**, A $d$-dimensional input vector $\mathbf{x}$ drives the evolution of a complex black-box system, yielding an $m$-dimensional system response $\mathbf{f}$, which is used to calculate the optimization objectives $\mathbf{y}$.  
* **b**, Architecture of the MSG metamodel. This surrogate model consists of two layers: the first layer deploys multiple XGBoost base learners to map high-dimensional input features to low-dimensional response surfaces; the second layer integrates the base predictions, utilizing Gaussian Process (GP) meta-learners to output the predicted mean and variance for each objective.  
* **c1**, Within the established affine coordinate system, multi-objective variances are statistically decoupled and aggregated into vectors based on their correlations.  
* **c2**, Monte Carlo (MC) methods are employed to compute the total sensitivity index for each factor.  
* **c3**, Factors are automatically clustered according to the estimated indices; larger label values indicate representative factors, whereas a label of 0 denotes stripped redundant factors.  
* **d1**, Redundant factors are fixed to default or empirically observed values, thereby delineating a low-dimensional effective space within the original high-dimensional space.  
* **d2**, Leveraging the predictive distribution of the MSG metamodel, the expected improvement matrix of the modified hypervolume for candidate points relative to the current Pareto front is calculated. By maximizing this acquisition function, the next evaluation point $\mathbf{x}^*$ is determined, driving the continuous evolution of the solution set.  
  
## 2. Code Functionality  
  
The algorithmic implementation consists of four main Python scripts. The functions of each script are introduced below:  
  
* **`Running.py` (Data Sampling & System Evaluation)**
  * This script is responsible for the initial data sampling within the high-dimensional parameter space.  
  * It utilizes Sobol quasi-random sequences to generate uniformly distributed training and testing sets.  
  * By calling the black-box system's objective function (`objective_function`), it computes the sample responses and exports the results as `.npy` and `.xlsx` files for subsequent model training.  
* **`MSG.py` (Metamodel Construction)** 
  * This script defines the Multi-output Stacked Generalization (MSG) metamodel, encapsulated in the `BaggingMeta` class.  
  * It is built on a two-layer ensemble learning architecture: the first layer trains multiple XGBoost models on bootstrap-resampled datasets with replacement (Bagging).  
  * The second layer treats the predictions of the base learners as meta-features and employs Gaussian Process Regressors (Stacking) to output the predicted mean and standard deviation for new samples.  
* **`Learning.py` (Sensitivity Analysis & Factor Screening)** 
  * This script calculates multi-output global sensitivity indices, using Monte Carlo sampling to efficiently estimate the total-effect indices ($P_i^{\text{total}}$) for each variable.  
  * It incorporates the `Natural_Breaks` clustering method based on the Jenks Natural Breaks optimization algorithm.  
  * By automatically identifying natural break points and evaluating the minimum Goodness of Variance Fit (`min_gvf`), it intelligently groups the parameters into representative variables (to be optimized) and redundant variables (to be held at default values).  
* **`Optimization.py` (Low-Dimensional Space Optimization)**
  * Executes the main iterative loop of multi-objective Bayesian black-box optimization in the reduced dimension.  
  * Constructs and evaluates the Expected Improvement Matrix (EIM) acquisition function based on the modified hypervolume.  
  * Applies a Genetic Algorithm (GA) to maximize the acquisition function and find the optimal sampling points for the next generation. 
  * It dynamically updates the Pareto front using fast non-dominated sorting on newly added sample points and calculates the relative hypervolume performance.  
  
## 3. Input Variables Dictionary  
  
The table below summarizes the key input variables required during the initialization and execution of each script, along with their physical meanings:  
  
| Variable | Script | Description |  
| :--- | :--- | :--- |  
| `num_vars_all` | Running, Learning, Optimization | Total number of decision variables/input parameters in the original black-box system space. |  
| `names_all` / `names` | Running, Learning, Optimization | List of strings representing the names of the parameters, used for dimension identification. |  
| `bounds_all` / `bounds` | All scripts | Array containing the lower and upper bounds for each parameter to define the search and sampling domain. |  
| `sample_size` | Running | Number of samples generated for the initial training set. |  
| `test_size` | Running | Number of independent samples generated for the testing set to evaluate model predictive accuracy. |  
| `obj` | Running, MSG, Learning | Number of optimization objectives evaluated by the complex system (output dimensionality). |  
| `n_estimators` | MSG | Number of XGBoost base learners in the first-layer ensemble of the MSG metamodel. |  
| `resample_size` | Learning | Monte Carlo resampling size used to construct the input mixing matrices when estimating sensitivity indices. |  
| `min_gvf` | Learning | The minimum Goodness of Variance Fit threshold used for Natural Breaks clustering. |  
| `whether_to_optimize` | Optimization | Array of clustering labels; values = 1 denote representative factors (included in optimization), and 0 denotes redundant factors. |  
| `defaults` | Optimization | Array of default or empirical values used to fix the redundant parameters, keeping them constant during optimization. |  
| `r`, `r_good` | Optimization | Reference point (`r`) for hypervolume calculation and the utopian reference point (`r_good`) for relative hypervolume ratio evaluation. |  
| `ga_pop`, `ga_iter` | Optimization | Population size and maximum iterations for the Genetic Algorithm (GA) used to optimize the acquisition function. |  
| `ga_mutation_prob` | Optimization | Specified mutation probability for the GA during acquisition function maximization. |  
| `max_iteration` | Optimization | Maximum number of iterations permitted for the Bayesian black-box optimization main loop. |  

  
## 4. Output Files Inventory 
  
Each script in the XLO framework generates specific data files, models, or performance logs. Below is a detailed breakdown of the output files produced by each `.py` file and their respective information:  
  
### `Running.py` (Data Sampling & System Evaluation)  
* **`Training_set.npy` / `Training_set.xlsx`**: The generated input variables (decision factors) for the initial training samples, mapped within their respective bounds.  
* **`y_Training_set.npy` / `y_Training_set.xlsx`**: The multi-objective evaluation results (responses) obtained by feeding the training set into the complex system's objective function.  
* **`Testing_set.npy` / `Testing_set.xlsx`**: The independent testing set inputs used strictly for evaluating the metamodel's predictive accuracy.  
* **`y_Testing_set.npy` / `y_Testing_set.xlsx`**: The system evaluation responses corresponding to the independent testing set.  
* **`Training-set.npy`**: An array logging the actual computational time consumed for evaluating each sample in the training set.  
  
### `MSG.py` (Metamodel Construction)  
* **`MSG.pkl`**: The fully trained Multi-output Stacked Generalization (MSG) metamodel. This serialized `joblib` object encapsulates all XGBoost base learners and Gaussian Process meta-learners. It is exported for rapid loading in the subsequent sensitivity analysis and optimization scripts, bypassing the need to retrain.  
  
### `Learning.py` (Sensitivity Analysis & Factor Screening)  
* **`PT.npy`**: A 1D array containing the calculated total-effect sensitivity indices ($P_i^{\text{total}}$) for all input factors, quantifying their variance contribution to the multi-output system.  
* **`whether_to_optimize.npy`**: The clustering labels derived from the Natural Breaks method. A value `=1` indicates a representative factor (to be optimized), while `0` indicates a redundant factor (to be fixed at default).  
* **`time-Sensitivity-Analysis.npy`**: The runtime logged for calculating the high-dimensional Monte Carlo sensitivity indices.  
* **`time-Natural-Breaks.npy`**: The runtime logged for executing the Jenks Natural Breaks clustering algorithm.  
  
### `Optimization.py` (Low-Dimensional Space Optimization)  
*(Note: `{j}` represents the index of the independent repetition/run, e.g., 0, 1, 2...)*  
* **`PF-repetition-{j}.npy`**: The final **Pareto Front** (objective function values) obtained at the end of the `j`-th optimization run.  
* **`PS-repetition-{j}.npy`**: The final **Pareto Set** (the optimal decision variable configurations in the reduced space) corresponding to the Pareto front of the `j`-th run.  
* **`All_PF-repetition-{j}.pkl`**: A serialized Python list tracking the historical evolution of the Pareto front at *each iteration* of the `j`-th run.   
* **`RHD-repetition-{j}.npy`**: An array recording the Relative Hypervolume Ratio (RHR) at each iteration, used to plot the algorithmic convergence curve.  
* **`Time-repetition-{j}.npy`**: An array logging the cumulative computational time elapsed at each iteration of the `j`-th run.