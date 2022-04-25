"""
Plot growth rates
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Inputs from the command terminal
ny, nz, case, values, integration, stability = int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6])

print(f'\nSaving figures to:\n')

k_start, k_end, k_num = 1e-8, 1.5e-5, 150; k_wavenum = np.linspace(k_start, k_end, k_num)

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = f'/home/rees/lsa/growth_rate/growth_{case}_{integration}_{stability}_{ny:02}_{nz:02}_*.txt'
else:
    fname = f'/home/rees/lsa/growth_rate/growth_{case}_{ny:02}_{nz:02}_*.txt'
    
files = glob.glob(fname)

cs = np.array([np.loadtxt(filename).view(complex).reshape(values, k_num) for filename in files])
cs = cs.flatten().reshape(len(files)*values, k_num)

sigma = np.asarray([k_wavenum[i]*cs[:, i].imag for i in range(k_num)])

k_index    = np.unravel_index(np.argmax(sigma), np.array(sigma).shape)[0] # Argument for the wavenunber of the most unstable mode
eval_index = np.unravel_index(np.argmax(sigma), np.array(sigma).shape)[1] # 

phase_speed         = "{:.3f}".format(cs[eval_index, k_index].real)
growth_per_day      = "{:.3f}".format(sigma[k_index, eval_index]*86400)
unstable_wavelength = "{:.0f}".format(2*np.pi/(1000*k_wavenum[k_index]))

most_unstable  = np.array([np.amax(sigma[i,:]) for i in range(sigma.shape[0])])
unstable_phase = np.array([cs[np.argmax(sigma[i,:]), i].real for i in range(sigma.shape[0])])

#np.savetxt(fname, [k_wavenum[k_index]])

if case == 'NEMO':
    fname_png = f'/home/rees/lsa/figures/growth_rate/growth_{case}_{integration}_{stability}_{ny:02}_{nz:02}.png'
else:
    fname_png = f'/home/rees/lsa/figures/growth_rate/growth_{case}_{ny:02}_{nz:02}.png'

fig, axes=plt.subplots(figsize=(6,4))

axes2 = axes.twinx()
#axes3 = axes.twiny()

axes.plot(k_wavenum, sigma, '.', ms=3, color='k')
axes.plot(k_wavenum, most_unstable, '.', ms=3, color='r')

axes.set_xlabel(r'$k$ [m$^{-1}$]')
axes.set_ylabel(r'Growth Rate [s$^{-1}$]')
axes2.set_ylabel(r'Growth Rate [d$^{-1}$]', rotation=270, labelpad=20)

axes.set_xlim([1e-8, k_end])
axes.set_xticks([0, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 1.25e-5, 1.5e-5])
axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

growth_day = lambda sigma: sigma*86400
ymin, ymax = axes.get_ylim()
axes2.set_ylim((growth_day(ymin), growth_day(ymax)))
axes2.plot([],[])

xmin, xmax = axes.get_xlim()
#axes3.set_xlim((xmin, xmax))
#axes3.set_xticklabels(['','2513','1257','838','628','503','419'])
#axes3.plot([],[])
#axes3.set_xlabel(r'$\lambda$ [km]')

axes.grid(alpha=.5)

axes.text(0.02, 0.9, f'Phase Speed: {phase_speed} m/s\nGrowth Rate:  {growth_per_day}  /d\nWavelength :    {unstable_wavelength} km', transform=axes.transAxes, ha='left', va='center', family='monospace', fontsize=10, bbox=dict(facecolor='white'))

plt.tight_layout()
plt.savefig(fname_png, dpi=300, bbox_inches='tight')
plt.close()

print(f'Growth Rates : {fname_png}\n')
