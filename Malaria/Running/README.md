
# Rural Malaria Transmission Scenario  
  
This folder provides the input files and Python wrapper used to run the rural malaria transmission case in the XLO framework. The case follows scenario 28 from Reiker et al., *Emulator-based Bayesian optimization for efficient multi-objective calibration of an individual-based model of malaria*, and uses OpenMalaria as the black-box simulator.  
  
  
## Scenario description  
  
We adopted a well-established benchmark (scenario 28) as the biological  experiment, constructing a simulation of malaria transmission in Rafin Marke village in northern Nigeria using the open-source platform OpenMalaria. The scenario models the natural progression of malaria infection and disease in a population of 5000 individuals over 1,275 days, with an age-structured demographic distribution and a maximum life expectancy of 99 years. The OpenMalaria simulator integrates three interconnected modules—within-host infection and immunity, pathogenesis, and indirect pathogenesis with comorbidity—comprising a total of 23 parameters. At selected survey time points, simulation outputs were recorded to compute the log-likelihood of malaria prevalence and parasite density across different age groups, which served as the objective function.
    
## Directory layout  
  
Place the following files in the same directory:  
  
```text  
malaria/  
├── Running.py              # Python wrapper for sampling, XML editing, OpenMalaria calls, and objective calculation  
├── OpenMalaria.exe         # OpenMalaria executable; not included in this repository  
├── wu_5000_28.xml          # OpenMalaria scenario file for Reiker et al. scenario 28  
├── scenario_35.xsd         # XML schema used by the scenario file  
├── fieldData.txt           # Field observations used to calculate calibration objectives  
├── densities.csv           # Auxiliary parasite-density reference table from the original calibration resources  
└── output.txt              # OpenMalaria output file; generated/overwritten during simulation  
```  
  
## Environment  
  
### Python dependencies  
  
The Python wrapper uses the following packages:  
  
```bash  
A typical Windows setup is:  
  
### OpenMalaria and Cygwin on Windows  
  
`Running.py` launches OpenMalaria through Cygwin Bash. This was necessary in the original local setup because the executable was run through a Unix-like command-line environment on Windows.  
  
Install Cygwin64 and make sure that Bash is available. The current script assumes the following Bash path:  
  
```python  
D:\cygwin64\bin\bash.exe  
```  
  
If your Cygwin installation is in another location, edit this line in `Running.py`:  
  
```python  
subprocess.run(  
 [r"D:\cygwin64\bin\bash.exe", "-l", "-c", bash_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  
```  
  
For example, if Cygwin is installed under `C:\cygwin64`, replace it with:  
  
```python  
[r"C:\cygwin64\bin\bash.exe", "-l", "-c", bash_command]  
```  
  
## Running OpenMalaria manually  
  
Before running the Python wrapper, first check whether OpenMalaria can run directly.  
  
Open **Cygwin64 Terminal**, move to this folder, and run:  
  
```bash  
cd /cygdrive/d/path/to/malaria./OpenMalaria.exe --scenario wu_5000_28.xml --output output.txt```  
  
Replace `/cygdrive/d/path/to/malaria` with your actual folder. For example, a Windows directory such as:  
  
```text  
D:\GitHub\XLO\examples\malaria  
```  
  
corresponds to:  
  
```bash  
/cygdrive/d/GitHub/XLO/examples/malaria  
```  
  
If the command succeeds, OpenMalaria will generate or overwrite `output.txt`.  
  
## Running the Python workflow  
  
After confirming that OpenMalaria works, run:  
  
```bash  
python Running.py```  
  
The script will:  
  
1. Load `fieldData.txt` and retain only scenario 28.  
2. Generate a 23-dimensional Sobol training set.  
3. Generate a 23-dimensional random testing set.  
4. Update the 23 calibrated parameters in `wu_5000_28.xml`.  
5. Update the OpenMalaria random seed.  
6. Call OpenMalaria through Cygwin Bash.  
7. Read `output.txt`.  
8. Compute two calibration objectives:  
   - parasite prevalence loss, based on `nPatent / nHost`;  
   - parasite density loss, based on `sumlogDens / nPatent`.  
9. Save the sampled inputs, objective values, and runtime records.   
```  
  
This means that a full run requires 1,200 OpenMalaria simulations. 
  
## Generated files  
  
Running `Running.py` produces:  
  
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
output.txt  
```  
  
`Training_set.*` and `Testing_set.*` store the sampled 23-dimensional parameter vectors.    
`y_Training_set.*` and `y_Testing_set.*` store the corresponding two-objective outputs.    
`Time-Training-set.npy` stores the runtime of each training simulation.    
`output.txt` is overwritten at each OpenMalaria call and therefore normally corresponds only to the most recent simulation.  
  
## Input and output files  
  
### `Running.py`  
  
Main Python entry point. It performs four tasks:  
  
- edits the OpenMalaria XML scenario file;  
- calls the OpenMalaria executable;  
- parses the simulator output;  
- computes the two calibration losses.  
  
The calibrated parameter names are:  
  
```text  
Simm, Xp, Yp, sigma2_i, Xy_star, Xh_star,  
ln_1_minus_alpha_m, alpha_m_decay, sigma2_0, Xv_star,  
Y2_star, alpha, v1, log_phi1, Q_D, Q_n,  
v0, YB1_star, F0, log2_over_omega,  
Y1_star, Y0_star, alpha_F_star  
```  
  
### `wu_5000_28.xml`  
  
OpenMalaria scenario file for the Rafin Marke, Nigeria benchmark. It defines the demographic structure, monitoring schedule, transmission model, intervention settings, and parameter block. `Running.py` modifies the 23 calibrated parameter values and the random seed before each simulator call.  
  
The monitoring section requests three output quantities:  
  
```text  
nHost  
nPatent  
sumlogDens  
```  
  
### `scenario_35.xsd`  
  
XML schema file for OpenMalaria scenario format version 35. Keep this file in the same directory as `wu_5000_28.xml` so that the scenario can be validated and interpreted correctly.  
  
### `fieldData.txt`  
  
Observed field data used for objective calculation. It is read as a whitespace-separated table with five columns:  
  
```text  
Scen  Survey  Age  Type  Val  
```  
  
Only records with `Scen == 28` are used.  
  
After filtering, the columns are interpreted as:  
  
```text  
Survey  Age  Type  Val  
```  
  
where:  
  
- `Survey` is the survey index;  
- `Age` is the age-group index;  
- `Type` identifies the measured quantity;  
- `Val` is the observed value.  
  
The relevant `Type` codes are:  
  
```text  
0 = nHost       # number of hosts in the age group  
3 = nPatent     # number of patent / parasite-positive hosts  
5 = sumlogDens  # sum of log parasite densities  
```  
  
### `densities.csv`  
  
Auxiliary parasite-density reference table inherited from the original malaria calibration resources. It contains columns such as:  
  
```text  
5day, Duration, Meanlog(Georgia,self-cleared subset)  
```  
  
  
### `output.txt`  
  
OpenMalaria simulation output file. It is generated by:  
  
```bash  
./OpenMalaria.exe --scenario wu_5000_28.xml --output output.txt```  
  
The file is parsed by `Running.py` as a whitespace-separated table with four columns:  
  
```text  
Survey  Age  Type  Val  
```  
  
The same `Type` codes are used:  
  
```text  
0 = nHost  
3 = nPatent  
5 = sumlogDens  
```  
  
For each simulation, `Running.py` compares `output.txt` against `fieldData.txt` to compute the two objective values.  
  
## Objective convention  
  
The manuscript formulates the malaria calibration as maximizing the likelihood of prevalence and parasite-density observations. In this executable wrapper, the objectives are returned as losses, so lower values are better.  
  
The two returned objectives are:  
  
```python  
[obj_prevalence_loss, obj_density_loss]  
```  
  
The prevalence loss is the negative binomial log-likelihood computed from simulated prevalence:  
  
```text  
p = nPatent / nHost  
```  
  
The density loss is the negative log-normal likelihood computed from the mean log-density term:  
  
```text  
sumlogDens / nPatent  
```  
  
with a density-bias correction of `4.8`.  
  
  
## References  
  
1. Reiker, T., Golumbeanu, M., Shattock, A., Burgert, L., Smith, T. A., Filippi, S., Cameron, E., & Penny, M. A. *Emulator-based Bayesian optimization for efficient multi-objective calibration of an individual-based model of malaria*. **Nature Communications** 12, 7212 (2021). https://doi.org/10.1038/s41467-021-27486-z  
  
2. Original calibration code: https://github.com/reikth/BayesOpt_Calibration  
  
3. OpenMalaria project: https://github.com/OpenMalaria-Org/openmalaria