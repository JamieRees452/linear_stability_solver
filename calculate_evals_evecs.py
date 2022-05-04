"""
Calculate the most unstable eigenvalue and eigenvector of the linear stability problem
for a given wavenumber
"""

import numpy as np
import os
import sys
import datetime
import time

import sparse_solver

# Inputs from the command terminal
ny, nz, k, case, init_guess, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4]), complex(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7]), str(sys.argv[8])

print(f'\nConfiguration:\nCase            = {case}\nGrid Resolution = ({ny:02},{nz:02})\nWavenumber      = {k}\n')
    
# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
            f'/home/rees/lsa/eigenvectors/evecs_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
else:
    fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
            f'/home/rees/lsa/eigenvectors/evecs_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
        
if os.path.exists(fname[0]) and os.path.exists(fname[1]): # If the files (fname) exist then we must decide whether to overwrite existing saved eigenvalues or not
    user_input = input(f'{fname[0]}\n{fname[1]}\n\nFiles already exist\nOverwrite existing saved data? (y/n)')
    if user_input == 'y':
        print(f'Overwriting existing saved data. Calculating eigenvalues and eigenvectors...')
        start_time = time.time()
        evals, evecs = sparse_solver.gep(ny, nz, k, case, integration, stability, assume, init_guess, values=1, tol_input=0, iter_input=1e6)
        end_time = time.time()
        
        np.savetxt(fname[0], evals.view(float).reshape(-1, 2))
        np.savetxt(fname[1], evecs.view(float).reshape(-1, 2))
    else:
        print(f'Loading existing saved data...')
        start_time = time.time()
        
        if not (os.path.exists(fname[0]) and os.path.exists(fname[1])):
            raise ValueError(f'The specified files do not exist\n{fname[0]}\n{fname[1]}')
        else:
            evals = np.loadtxt(fname[0]).view(complex).reshape(-1) 
            evecs = np.loadtxt(fname[1]).view(complex).reshape(-1) 
            
        end_time = time.time()
else: # If the files (fname) do not exist then calculate the eigenvalues and eigenvectors
    print(f'Calculating eigenvalues and eigenvectors...\n\nSaving eigenvalues  to {fname[0]}\nSaving eigenvectors to {fname[1]}')
    start_time = time.time()
    evals, evecs = sparse_solver.gep(ny, nz, k, case, integration, stability, assume, init_guess, values=1, tol_input=0, iter_input=1e6)
    end_time = time.time()
    
    np.savetxt(fname[0], evals.view(float).reshape(-1, 2))
    np.savetxt(fname[1], evecs.view(float).reshape(-1, 2))
    
cs = evals[np.argmax(evals.imag)]
print(f'\nWavenumber: {k}   Eigenvalue: {cs}   Time: {datetime.timedelta(seconds=int(end_time-start_time))}')
