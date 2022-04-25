"""
Calculate the growth rates over a range of wavenumbers
"""

import numpy as np
import os
import sys
from tqdm import tqdm

import sparse_solver

# Inputs from the command terminal
ny, nz, case, init_guess, values, integration, stability = int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), complex(sys.argv[4]), int(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7])

# Labels used for the initial guess eigenvalue for the filename
[guess_re, guess_im] = [str(init_guess.real*100).replace('.','').replace('-','m'), str(init_guess.imag*100).replace('.','')]

# Set up the range of wavenumbers to calculate growth rates over
k_start, k_end, k_num = 1e-8, 1.5e-5, 150; k_wavenum = np.linspace(k_start, k_end, k_num)
cs = np.zeros((values, k_num), dtype=complex) # Initialise the storage of the eigenvalues

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = f'/home/rees/lsa/growth_rate/growth_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{guess_re}_{guess_im}.txt'
else:
    fname = f'/home/rees/lsa/growth_rate/growth_{case}_{ny:02}_{nz:02}_{guess_re}_{guess_im}.txt'
    
if os.path.exists(fname):
    user_input = input(f'{fname}\n\nFile already exists\nOverwrite existing saved data? (y/n)')
    if user_input == 'y':
        print(f'Overwriting existing saved data. Calculating eigenvalues over a range of wavenumbers...')
        for count, k in enumerate(tqdm(k_wavenum, position=0, leave=True)):
            cs[:, count] = sparse_solver.gep(ny, nz, k, case, integration, stability, init_guess, values, tol_input=1e-6, iter_input=1e5)[0]
            
        np.savetxt(fname, cs.view(float).reshape(-1, 2))
        
    else:
        print(f'Previously calculated eigenvalues are located at {fname}')
else:
    print(f'Calculating eigenvalues over a range of wavenumbers...\n\nSaving eigenvalues to {fname}')
    for count, k in enumerate(tqdm(k_wavenum, position=0, leave=True)):
        cs[:, count] = sparse_solver.gep(ny, nz, k, case, integration, stability, init_guess, values, tol_input=1e-6, iter_input=1e5)[0]
        
    np.savetxt(fname, cs.view(float).reshape(-1, 2))
