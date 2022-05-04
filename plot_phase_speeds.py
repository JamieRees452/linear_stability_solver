"""
Plot phase speeds
"""
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Inputs from the command terminal
ny, nz, case, values, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7])

print(f'\nSaving figures to:\n')

k_start, k_end, k_num = 1e-8, 1.5e-5, 150; k_wavenum = np.linspace(k_start, k_end, k_num)

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = f'/home/rees/lsa/growth_rate/growth_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_*.txt'
else:
    fname = f'/home/rees/lsa/growth_rate/growth_{case}_{ny:02}_{nz:02}_*.txt'

files = glob.glob(fname)

cs = np.array([np.loadtxt(filename).view(complex).reshape(values, k_num) for filename in files])
cs = cs.flatten().reshape(len(files)*values, k_num)

sigma = np.asarray([k_wavenum[i]*cs[:, i].imag for i in range(k_num)])
#frequency = np.asarray([k_wavenum[i]*cs[:, i].real for i in range(k_num)])
unstable_phase = np.array([cs[np.argmax(sigma[i,:]), i].real for i in range(sigma.shape[0])])
#unstable_frequency = np.array([k_wavenum[i]*cs[np.argmax(sigma[i,:]), i].real for i in range(sigma.shape[0])])

if case == 'NEMO':
    fname = f'/home/rees/lsa/figures/phase_speed/phase_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png'
else:
    fname = f'/home/rees/lsa/figures/phase_speed/phase_{case}_{ny:02}_{nz:02}.png'

fig, axes=plt.subplots(figsize=(6,4))

#axes.plot(k_wavenum, abs(frequency), '.', ms=3, color='k')
#axes.plot(k_wavenum, abs(unstable_frequency), '.', ms=3, color='r')

axes.plot(k_wavenum, cs.real.T, '.', ms=3, color='k')
axes.plot(k_wavenum, unstable_phase, '.', ms=3, color='r')

axes.set_xlabel(r'k [m$^{-1}$]')
axes.set_ylabel(r'Phase Speed [ms$^{-1}$]')
#axes.set_ylabel(r'Frequency [s$^{-1}$]')

#axes.set_xlim([1e-8, k_end])
#axes.set_xticks([0, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 1.25e-5, 1.5e-5])

#axes.set_ylim([0, 5e-6])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

axes.grid(alpha=.5)

plt.tight_layout()
plt.savefig(fname, dpi=300, bbox_inches='tight')
plt.close()

print(f'Phase Speeds : {fname}\n')
