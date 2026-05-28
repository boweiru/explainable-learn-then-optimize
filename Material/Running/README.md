# Architected Material

This directory provides the automated finite-element evaluation workflow for the architected-material benchmark. The benchmark is adapted from the gyroid scaffold design problem in *Machine learning-enabled constrained multi-objective design of architected materials* and *Deep active optimization for complex systems*.

The evaluation interface converts a 27-dimensional porosity array into a gyroid scaffold, generates a solid finite-element mesh, runs an Abaqus compression simulation, and extracts the elastic modulus and yield strength from the simulated force-displacement response.

## Scenario description

To optimize the mechanical compatibility and load-bearing capacity of 3D-printed bone scaffolds, we employed finite-element compression simulations to reconstruct their deformation and failure behavior under loading. The targets are porous scaffolds architected from gyroid unit cells. We adopted a 3 × 3 × 3 cubic lattice arrangement, wherein each scaffold is assembled from 27 constituent gyroid sub-units. The density of each individual sub-unit serves as a continuous decision variable, forming a 27-dimensional parameter space. To ensure smooth morphological transitions, spatial interpolation of the gyroid isosurface constants was applied across sub-units. Finite-element compression simulations were conducted using ABAQUS. Based on the extracted macroscopic compressive stress-strain response curves, the equivalent elastic modulus was calculated, and the yield strength was derived. The optimization task aims to simultaneously maximize the structural strength and minimize the deviation of the scaffold’s elastic modulus from a target value of 5000 MPa.

## Evaluation interface

The system evaluation function is:

```python
objective_function_sim(x, target_E=5000)
```

`x` is a porosity array for the 27 gyroid sub-units in the 3 × 3 × 3 scaffold:

```text
x = [x1, x2, ..., x27]
```

The input can be a NumPy array or any array-like object that can be flattened into 27 values. `objective_function_sim` writes these values into `Porosity.xlsx`, which is then read by the MATLAB geometry-generation script.

The returned objective vector is:

```text
[-Y, |E - target_E|]
```

where `E` is the equivalent elastic modulus and `Y` is the yield strength extracted from the simulated stress-strain curve. The first objective is written as `-Y` so that maximizing yield strength is expressed as a minimization objective.

## Example call

The following command demonstrates the expected function-call pattern. The input is a 27-dimensional porosity array corresponding to the 27 gyroid sub-units in the 3 × 3 × 3 scaffold.

Example call:

```bash
python -c "import numpy as np; from Running import objective_function_sim; print(objective_function_sim(np.full(27, 0.5)))"
```

This command evaluates a scaffold whose 27 sub-units are assigned the same porosity value of `0.5`. The returned value is the two-objective vector `[(E - target_E)^2, -Y]`.

## Directory contents

```text
.
├── Running.py
├── STL_Modeling.m
├── drawgyroid.m
├── createunitofv.m
├── createv_2.m
├── findneighbour.m
├── stlwrite.m
├── command_1.tcl
├── 3Dinp.py
├── BatchProcess_2.py
├── CurveProcess.py
└── README.md
```

`Running.py` defines `objective_function_sim(x, target_E=5000)` and coordinates the full evaluation workflow.


`STL_Modeling.m` is the MATLAB entry script for scaffold construction. It reads the 27 porosity values from `Porosity.xlsx`, arranges them into a 3 × 3 × 3 scaffold, maps porosity to gyroid isosurface constants, constructs the scaffold surface, and writes `1.stl`.

`drawgyroid.m` evaluates the gyroid field and extracts the scaffold isosurface.

`createv_2.m`, `createunitofv.m`, and `findneighbour.m` construct the spatially varying gyroid field and smooth the transition between neighboring sub-units.

`stlwrite.m` exports the MATLAB-generated scaffold surface to STL format.

`command_1.tcl` is the HyperMesh input script. It imports `1.stl`, applies shrink-wrap/solid meshing, sets the Abaqus Explicit export template, and exports an intermediate Abaqus input file named `adaqusModel1.inp`.

`3Dinp.py` edits `adaqusModel1.inp` in place. It removes the two-dimensional CPS3 element block inherited from the STL import/meshing process and keeps the three-dimensional C3D8R solid-element block used in the Abaqus compression simulation.

`BatchProcess_2.py` is the Abaqus model-configuration and job-submission script. It imports the processed `adaqusModel1.inp`, assigns material and section information, creates rigid compression platens, defines the Abaqus/Explicit compression step, defines contact and boundary conditions, submits the job, waits for completion, and writes the displacement-reaction force report `adaqusModel1.rpt`.

`CurveProcess.py` reads `adaqusModel1.rpt`, converts the displacement-force response into stress-strain data, estimates the equivalent elastic modulus and 0.2% offset yield strength, writes `E&Y.log`, and returns the objective vector used by `objective_function_sim`.

## Runtime files

The following files are generated or updated during evaluation:

```text
Porosity.xlsx
1.stl
1.txt
stl.msg
adaqusModel1.inp
adaqusModel1.odb
adaqusModel1.rpt
curve.mat
E&Y.log
command.tcl
```

`Porosity.xlsx` is generated by `Running.py` for each call to `objective_function_sim`.

`1.stl` is generated by MATLAB and imported by HyperMesh.

`1.txt` and `stl.msg` are auxiliary files produced during the STL generation/export process.

`adaqusModel1.inp` is first exported by HyperMesh and then modified by `3Dinp.py`. The file after `3Dinp.py` is the valid Abaqus input deck used by the simulation stage.

`adaqusModel1.odb` is the Abaqus output database generated by the completed compression simulation.

`adaqusModel1.rpt` is exported by `BatchProcess_2.py` from the Abaqus ODB and contains the displacement-reaction force curve.

`curve.mat` stores the converted force-displacement data during mechanical-property extraction.

`E&Y.log` records the extracted modulus and yield strength for completed evaluations and also records failed stages when the automatic retry routine is triggered.

`command.tcl` is a HyperMesh command-history/session artifact that may be written automatically in the start-in directory. It is not an input file for this workflow. The HyperMesh input script is `command_1.tcl`.

## Software requirements

This workflow was configured with:

```text
MATLAB R2024b
Altair HyperMesh 2019
Abaqus 2021
```

The three commercial software packages are installed and licensed independently. They do not depend on each other through shared libraries in this workflow. The coupling is file-based:

```text
MATLAB:      Porosity.xlsx → 1.stl
HyperMesh:  1.stl → intermediate adaqusModel1.inp
3Dinp.py:   intermediate adaqusModel1.inp → valid C3D8R Abaqus input deck
Abaqus:     valid adaqusModel1.inp → adaqusModel1.odb + adaqusModel1.rpt
CurveProcess.py: adaqusModel1.rpt → [-Y, |E - target_E|]
```

For setup verification, check the software in the same order as the evaluation pipeline:

```text
MATLAB R2024b → HyperMesh 2019 → Abaqus 2021
```

This is a workflow-order requirement, not an installation dependency.

## Software installation and configuration

### MATLAB R2024b

Install MATLAB R2024b with a valid license. Confirm that the executable path in `Running.py` is correct:

```python
matlabExe = r"D:\Matlab2024b\bin\matlab.exe"
```

If MATLAB is installed elsewhere, update this path.

PowerShell verification:

```powershell
& "D:\Matlab2024b\bin\matlab.exe" -batch "disp(version)"
```

A standalone geometry-generation check, after a 27-row `Porosity.xlsx` exists in the scenario directory, is:

```powershell
& "D:\Matlab2024b\bin\matlab.exe" -batch "run('STL_Modeling.m')"
```

The MATLAB scripts use relative paths. The working directory must be the scenario directory containing `Running.py`, `STL_Modeling.m`, and the helper `.m` files. A successful MATLAB stage generates `1.stl`.

### Altair HyperMesh 2019

Install HyperMesh 2019 with a valid Altair license. Confirm that the HyperMesh executable path in `Running.py` is correct:

```python
hypermeshExe = r"D:\Hyperworks2019\hm\bin\win64\hmopengl.exe"
```

If HyperMesh is installed elsewhere, update this path.

`Running.py` launches HyperMesh as:

```python
subprocess.run([hypermeshExe, "-batch", "-noconsole", "-tcl", "command_1.tcl"])
```

The HyperMesh batch stage requires a valid license. `Running.py` sets:

```python
os.environ['ALTAIR_LICENSE_PATH'] = os.getenv('LICENSE_FILE', '')
```

`command_1.tcl` contains HyperMesh 2019-specific Abaqus Explicit template paths:

```tcl
*templatefileset "D:/Hyperworks2019/templates/feoutput/abaqus/explicit"
*feoutputwithdata "D:/Hyperworks2019/templates/feoutput/abaqus/explicit" "adaqusModel1.inp" 0 0 1 1 3
```

If HyperMesh is installed in another directory, update these template paths.

A standalone HyperMesh check, after `1.stl` has been generated, is:

```powershell
& "D:\Hyperworks2019\hm\bin\win64\hmopengl.exe" -batch -noconsole -tcl ".\command_1.tcl"
```

This step generates an intermediate `adaqusModel1.inp`. Then run:

```powershell
python .\3Dinp.py
```

`3Dinp.py` removes the two-dimensional CPS3 element block and keeps the three-dimensional C3D8R solid elements. The processed `adaqusModel1.inp` is the input file used by the Abaqus simulation stage.

### Abaqus 2021

Install Abaqus 2021 / SIMULIA Established Products with a valid license. Confirm that the Abaqus command path in `Running.py` is correct:

```python
abaqusExe = r"C:\SIMULIA\Commands\abaqus.bat"
```

If Abaqus is installed elsewhere, update this path.

Abaqus is launched through Abaqus/CAE in no-GUI mode:

```python
subprocess.run([abaqusExe, 'cae', 'noGUI=' + 'BatchProcess_2.py'])
```

`BatchProcess_2.py` imports Abaqus-specific modules such as `abaqus`, `abaqusConstants`, `caeModules`, `part`, `assembly`, `step`, `interaction`, and `job`; therefore it must be executed by Abaqus/CAE, not by a normal Python interpreter.

PowerShell verification:

```powershell
& "C:\SIMULIA\Commands\abaqus.bat" information=release
& "C:\SIMULIA\Commands\abaqus.bat" verify -all
```

A standalone Abaqus check, after `adaqusModel1.inp` has been processed by `3Dinp.py`, is:

```powershell
& "C:\SIMULIA\Commands\abaqus.bat" cae noGUI=BatchProcess_2.py
```

A successful Abaqus stage generates `adaqusModel1.odb` and `adaqusModel1.rpt`.

Key configuration points:

- Abaqus must be licensed and callable from the same Windows terminal used by Python.
- The `Commands` directory must contain `abaqus.bat`.
- `BatchProcess_2.py` sets `numCpus=8` and `numDomains=8`; reduce these values if the local workstation or license does not support 8-core Abaqus/Explicit execution.
- The scenario directory must be writable because Abaqus writes job, lock, message, ODB, and report files in place.

## FEM evaluation workflow

`objective_function_sim(x, target_E=5000)` executes the following sequence.

### 1. Porosity transfer

The 27-dimensional porosity array is flattened and written to `Porosity.xlsx`. Each row corresponds to one gyroid sub-unit.

### 2. MATLAB geometry construction

MATLAB runs:

```text
STL_Modeling.m
```

This stage:

1. creates the 3 × 3 × 3 sub-unit coordinate table;
2. reads the porosity values from `Porosity.xlsx`;
3. maps the porosity values to gyroid isosurface constants using `r2 = -2.8218 * r2 + 2.352`;
4. constructs a spatially varying gyroid field with smooth transitions between neighboring units;
5. extracts the gyroid scaffold surface;
6. writes the scaffold geometry to `1.stl`.

Expected output:

```text
1.stl
```

### 3. HyperMesh solid meshing

HyperMesh runs:

```text
command_1.tcl
```

This stage:

1. imports `1.stl`;
2. sets the Abaqus Explicit export template;
3. performs shrink-wrap/solid meshing;
4. exports the intermediate Abaqus input file `adaqusModel1.inp`.

Expected intermediate output:

```text
adaqusModel1.inp
```

### 4. Abaqus input cleanup

Python runs:

```text
3Dinp.py
```

This script edits `adaqusModel1.inp` in place. It deletes the block starting from the CPS3 element definition with `ELSET=misc1` and stops before the C3D8R element definition with `ELSET=auto`. The retained C3D8R elements form the three-dimensional solid mesh used in the compression simulation.

### 5. Abaqus configuration and simulation

Abaqus/CAE runs:

```text
BatchProcess_2.py
```

This script:

1. imports the processed `adaqusModel1.inp` as the scaffold part;
2. defines the scaffold material, density, elasticity, and plastic response;
3. assigns the homogeneous solid section named `Gyroid`;
4. creates two rigid cylindrical compression platens;
5. places the scaffold and platens in the assembly;
6. defines the Abaqus/Explicit compression step;
7. requests displacement and reaction-force outputs;
8. defines general contact by creating `ContactProperty('IntProp-1')`, setting isotropic tangential `PENALTY` behavior with coefficient `0.125`, creating `ContactExp(name='Int-1')`, enabling all contact pairs through `useAllstar=ON`, and assigning the property to `(GLOBAL, SELF, 'IntProp-1')`;
9. fixes the lower platen and applies downward displacement to the upper platen;
10. submits job `adaqusModel1` with ODB output;
11. waits until the job is complete;
12. opens `adaqusModel1.odb` and writes the displacement-reaction force curve to `adaqusModel1.rpt`.

Expected outputs:

```text
adaqusModel1.odb
adaqusModel1.rpt
```

### 6. Mechanical-property extraction

`CurveProcess.calculate_EY(length=6, point1=350, point2=50, target_E=5000)` reads the Abaqus report and computes the two objectives.

This stage:

1. reads `adaqusModel1.rpt`;
2. saves the displacement-force data as `curve.mat`;
3. converts displacement to strain using `length = 6`;
4. converts reaction force to stress using `length^2`;
5. densifies the curve by repeatedly inserting midpoint samples;
6. identifies the steepest linear window to estimate `E`;
7. computes the 0.2% offset yield strength `Y`;
8. appends `E` and `Y` to `E&Y.log`;
9. returns:

```text
[-Y, |E - target_E|]
```

## References

1. Peng, B., Wei, Y., Qin, Y. et al. *Machine learning-enabled constrained multi-objective design of architected materials*. Nature Communications 14, 6630 (2023). https://doi.org/10.1038/s41467-023-42415-y

2. Wei, Y., Peng, B., Xie, R. et al. *Deep active optimization for complex systems*. Nature Computational Science 5, 801–812 (2025). https://doi.org/10.1038/s43588-025-00858-x

3. GAD-MALL source code: https://github.com/Bop2000/GAD-MALL

4. DANTE source code: https://github.com/Bop2000/DANTE/
