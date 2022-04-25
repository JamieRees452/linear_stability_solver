# linear_stability_solver
Solving the two dimensional linear stability problem for an equatorial ocean current in thermal wind balance


Files of interest are:

sparse_solver.py - Sparse eigensolver (scipy.linalg.eigs) for the generalised eigenvalue problem obtained from the discrete linear stability problem

dense_solver.py - Dense eigensolver (scipy.linalg.eig) for the generalised eigenvalue problem for the 1D and 2D Rayleigh-Kuo, Eady and Stone problems

sparse_solver calls upon the following module:

mean_fields.py - Calculate the mean zonal velocity and mean density fields for the Proehl and NEMO cases

mean_fields.py calls upon the following modules:

Calculate_NEMO_fields.py - Extract the data from the coupled integration for the mean zonal velocity and buoyancy 
Calculate_Proehl_fields.py - Calculate the mean zonal velocity fields for Proehls test cases
