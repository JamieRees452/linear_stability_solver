#!/usr/bin/env bash

echo Meridional Resolution
read ny

echo Vertical Resolution
read nz

echo Wavenumber
read k

echo Case
read mean_flow

echo Initial Guess
read init_guess

echo --------------------------------------------------------------------------
python calculate_evals_evecs.py $ny $nz $k $mean_flow $init_guess

echo --------------------------------------------------------------------------
python calculate_eddy_energy_budget.py $ny $nz $k $mean_flow

#echo --------------------------------------------------------------------------
#python calculate_geometric_diagnostics.py $ny $nz $k $mean_flow

echo --------------------------------------------------------------------------
python plot_eigenvectors.py $ny $nz $k $mean_flow

echo --------------------------------------------------------------------------
python plot_eddy_energy_budget.py $ny $nz $k $mean_flow

#echo --------------------------------------------------------------------------
#python plot_geometric_diagnostics.py $ny $nz $k $mean_flow
