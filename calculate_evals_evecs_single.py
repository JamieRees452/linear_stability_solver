"""
Find the most unstable eigenvalue and corresponding eigenvectors of the square matrices A and B.

Solve A @ x[i] = c[i] * B @ x[i], the generalised eigenvalue problem for c[i] eigenvalues with 
corresponding eigenvectors x[i]

Example
-------
python calculate_evals_evecs.py 50 50 6e-6 -0.2 NEMO_25 01 02
"""

import argparse
import numpy as np
import os

import eigensolver_single

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('k'          , type=float, help='Zonal wavenumber')
parser.add_argument('init_guess' , type=float, help='intial guess for the eigenvalue')
parser.add_argument('case'       , type=str  , help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
parser.add_argument('month0'     , type=str  , help='Data from month0 e.g. Jan=01')
parser.add_argument('month1'     , type=str  , help='Data from month1 e.g. Dec=12')
args = parser.parse_args()

ny, nz, k, init_guess, case, month0, month1 = args.ny, args.nz, args.k, args.init_guess, args.case, args.month0, args.month1

WORK_DIR = '/home/rees/lsa' 

print(f'\n    ------------------------------------------------------------------')
print(f'    |           Solving the Generalised Eigenvalue Problem           |')
print(f'    |                            Ax=cBx                              |')
print(f'    |           using an implicitly restarted Arnoldi method         |')
print(f'    |           and with coefficient matrices A,B = {int((3*ny-4)*(nz-1))}x{int((3*ny-4)*(nz-1))}      |')
print(f'    ------------------------------------------------------------------')
print(f'    | Inputs:                                                        |')
print(f'    | Grid Resolution   = ({ny:02},{nz:02})                                  |')
print(f'    | Wavenumber        = {k}                                      |')
print(f'    | Eigenvalue Search = {init_guess}                                       |')
print(f'    |                                                                |')
print(f'    | Progress:                                                      |')

# List of filenames (fname) where the outputs (eigenvalue, eigenvector, and maxdiff) are to be saved
fname = [f'{WORK_DIR}/saved_data/{case}/evals_{case}_{month0}_{month1}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
        f'{WORK_DIR}/saved_data/{case}/evecs_{case}_{month0}_{month1}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
        f'{WORK_DIR}/saved_data/{case}/maxdiffs_{case}_{month0}_{month1}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
        
# If the files (fname) already exist then we decide whether we want to overwrite the existing result
if (os.path.exists(fname[0]) and os.path.exists(fname[1]) and os.path.exists(fname[2])):
    user_input = input(f'    | Overwrite existing saved data?                                 |')
    
    # Overwrite the existing saved data
    if user_input == 'y':
        print(f'    | Overwriting existing saved data                                |')
        evals, evecs, maxdiffs = eigensolver_single.gep(ny, nz, k, init_guess, case, month0, month1)
        
        np.savetxt(fname[0], evals.view(float).reshape(-1, 2))
        np.savetxt(fname[1], evecs.view(float).reshape(-1, 2))
        np.savetxt(fname[2], [maxdiffs])
        
    # Do not overwrite the existing saved data
    else:
        print(f'    | Loading existing saved data                                    |')

        evals = np.loadtxt(fname[0]).view(complex).reshape(-1) 
        evecs = np.loadtxt(fname[1]).view(complex).reshape(-1) 
        maxdiffs = [np.loadtxt(fname[2])]
            
# If the files (fname) do not already exist then calculate and save the outputs
else: 
    evals, evecs, maxdiffs = eigensolver_single.gep(ny, nz, k, init_guess, case, month0, month1)
    
    np.savetxt(fname[0], evals.view(float).reshape(-1, 2))
    np.savetxt(fname[1], evecs.view(float).reshape(-1, 2))
    np.savetxt(fname[2], [maxdiffs])
    
cs = evals[np.argmax(evals.imag)]

print(f'    |                                                                |')
print(f'    | Outputs:                                                       |')
print(f'    | Eigenvalue = {cs.real:.16f}+{cs.imag:.16f}i           |')
print(f'    | Error      = {maxdiffs[0]}                             |')
print(f'    ------------------------------------------------------------------')
