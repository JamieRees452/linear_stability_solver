# Linear Stability Solver

The broad aim of this solver is to investigate tropical instability waves (TIWs) in the equatorial oceans by means of a linear stability analysis.

Our linear stability solver solves the two-dimensional linearised equations of motion for an inviscid Buossinesq fluid on an equatorial beta plane. The equations of motion are discretised using a finite difference method and the resulting generalised eigenvalue problem is solved with scipy using an implicitly restarted Arnoldi method. The aim of this project is transparency in the results of my thesis since, with this repository, one can reproduce the results in the linear stability analysis section of my thesis.
 
## Table of Contents
* [General info](#general-info)
* [Setup](#setup)
* [Procedure](#procedure)
* [Code Examples](#codeexamples)
* [To Do](#todo)

## General info

Lorem ipsum

## Setup

The file structure should be as follows

```bash
/
├───linear_figures
│   ├───debug
│   └───thesis
├───original_data
└───saved_data
    ├───interpolated_data
    ├───NEMO_12
    ├───NEMO_25
    ├───Proehl_1
    ├───Proehl_2
    ├───Proehl_3
    ├───Proehl_4
    ├───Proehl_5
    ├───Proehl_6
    ├───Proehl_7
    └───Proehl_8
```

Here we 

## Procedure

As a complete example with the code, we investigate the linear stability of an eastward flowing Gaussian Jet centered on the equator and in depth (Proehl_1). The mean fields are visualised via

```
python linear_plot_mean_fields.py 100 50 Proehl_1 00 00
```

The mean fields are saved to /linear_figures/debug. The figures of importance are shown below

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![](/images/U_Proehl_1.png)|  ![](/images/r_Proehl_1.png)

## Code Examples

To find the most unstable eigenvalue/eigenvector pair for a functional mean flow

```
python calculate_evals_evecs_single.py 100 50 6.67e-6 Proehl_1 00 00
```

This produces the following output

![Figure](/images/example_screenshot.png)

To analyse the outputs we can plot the eigenvector, eddy energy budget, and GEOMETRIC diagnostics
```
python linear_plot_eigenvectors.py 100 50 6.67e-6 Proehl_1 00 00
python linear_plot_eddy_energy_budget.py 100 50 6.67e-6 Proehl_1 00 00
python linear_plot_geometric_diagnostics.py 100 50 6.67e-6 Proehl_1 00 00
```
These output figures are saved to the /saved_data/debug directory

## To Do

* Solve the linear stability problem using an additional independent numerical method
  - We have solved the linear stability problem using a finite difference method. However, we might have concerns over the reliability of the results obtained using only one method. To this extent, we aim to solve the linear stability problem using a Chebyshev collocation method.

* Pathlib
  - Files are specific to Unix. We should use pathlib so that the file structures also agree with Windows and Mac.


