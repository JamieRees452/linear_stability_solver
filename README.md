# Linear Stability Solver

The broad aim of this solver is to investigate tropical instability waves (TIWs) in the equatorial oceans by means of a linear stability analysis.

Our linear stability solver solves the two-dimensional linearised equations of motion for an inviscid Buossinesq fluid on an equatorial beta plane. The equations of motion are discretised using a finite difference method and the resulting generalised eigenvalue problem is solved with scipy using an implicitly restarted Arnoldi method. The aim of this project is transparency in the results of my thesis since, with this repository, one can reproduce the results in the linear stability analysis section of my thesis.
 
## Table of Contents
* [General info](#general-info)
* [Setup](#setup)
* [Code Examples](#codeexamples)
* [To Do](#todo)

## General info

Lorem ipsum

## Setup

The file structure should be as follows

```bash
C:.
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

## Code Examples

Here we 

## To Do

* Solve using a separate numerical method


