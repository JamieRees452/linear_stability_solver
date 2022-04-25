import matplotlib.pyplot as plt
import numpy as np
import sys
import dense_solver
from tqdm import tqdm

ny, nz, case, dim, Ri = int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), float(sys.argv[5])

if case == 'RK':
    k_end = 3.0
    sigma_max = 0.16
    
if case == 'EADY' or case == 'STONE':
    if Ri == 0.5:
        k_end  = 3.0
        sigma_max = 0.9
        mu_list = np.asarray([0, 2])
        
    elif Ri == 1.0:
        k_end = 2.5
        sigma_max = 0.25
        
    elif Ri == 2.0:
        k_end = 1.6
        sigma_max = 0.2
        
mu = 0.0

k_start, k_num = 0.01, 100; k_wavenum = np.linspace(k_start, k_end, k_num)

#cs = np.empty((3*ny-2, len(k_wavenum)), dtype=complex); cs[:]=np.nan
#cs = np.empty(((3*ny-2)*(nz-1), len(k_wavenum)), dtype=complex); cs[:]=np.nan
#cs = np.empty((3*(nz-1), len(k_wavenum)), dtype=complex); cs[:]=np.nan
cs = np.empty((3*nz*ny, len(k_wavenum)), dtype=complex); cs[:]=np.nan

for count, k in enumerate(tqdm(k_wavenum, position=0, leave=True)):
    cs[:dense_solver.gep(ny, nz, k, mu, case, dim, Ri)[0].shape[0], count] = dense_solver.gep(ny, nz, k, mu, case, dim, Ri)[0]
    
sigma = np.asarray([k_wavenum[i]*cs[:, i].imag for i in range(k_num)])
phase = np.asarray([cs[:, i].real for i in range(k_num)])
phase_most = np.asarray([cs[np.argmax(cs[:,i].imag), i].real for i in range(k_num)])

fig, axes=plt.subplots(figsize=(6, 4), dpi=300)

axes.plot(k_wavenum, sigma, '.', ms=3, color='k')

axes.set_xlabel(r'$k$', fontsize=16)
axes.set_ylabel(r'$kc_{i}$', fontsize=16)

axes.set_xlim([0, k_end])
axes.set_ylim([0, sigma_max])

plt.tight_layout()
plt.savefig(f'/home/rees/lsa/test_cases/test_figures/Growth_{case}_{dim}_{ny:02}_{nz:02}_{Ri}.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes=plt.subplots(figsize=(6, 4), dpi=300)

axes.plot(k_wavenum, phase, '.', ms=1, color='k')
axes.plot(k_wavenum, phase_most, '.', ms=3, color='r')

axes.set_xlabel(r'$k$', fontsize=16)
axes.set_ylabel(r'$c_{r}$', fontsize=16)

axes.set_xlim([0, k_end])
axes.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f'/home/rees/lsa/test_cases/test_figures/Phase_{case}_{dim}_{ny:02}_{nz:02}_{Ri}.png', dpi=300, bbox_inches='tight')
plt.close()
