"""
Plot growth rates
"""

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('case'       , type=str  , help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
parser.add_argument('month0'     , type=str  , help='Data from month0 e.g. Jan=01')
parser.add_argument('month1'     , type=str  , help='Data from month1 e.g. Dec=12')
parser.add_argument('values'     , type=int  , help='Number of output eigenvalues')
parser.add_argument('k_start'    , type=float, help='Starting wavenumber')
parser.add_argument('k_end'      , type=float, help='Ending wavenumber')
parser.add_argument('k_num'      , type=int  , help='Number of steps')
args = parser.parse_args()

ny, nz, init_guess, case, month0, month1, values, k_start, k_end, k_num = args.ny, args.nz, args.init_guess, args.case, args.month0, args.month1, args.values, args.k_start, args.k_end, args.k_num

WORK_DIR = os.getcwd() 

k_wavenum = np.linspace(k_start, k_end, k_num)
    
fname = f'{WORK_DIR}/saved_data/{case}/growth_{case}_{month0}_{month1}_{values}_{ny:02}_{nz:02}_*.txt'
    
files = glob.glob(fname)

cs = np.array([np.loadtxt(filename).view(complex).reshape(values, k_num) for filename in files])
cs = cs.flatten().reshape(len(files)*values, k_num)

sigma = np.asarray([abs(k_wavenum[i])*cs[:, i].imag for i in range(k_num)])

k_index    = np.unravel_index(np.argmax(sigma), np.array(sigma).shape)[0] # Argument for the wavenunber of the most unstable mode
eval_index = np.unravel_index(np.argmax(sigma), np.array(sigma).shape)[1] # 

phase_speed         = cs[eval_index, k_index].real
growth_per_day      = sigma[k_index, eval_index]*86400
unstable_wavelength = 2*np.pi/(1000*k_wavenum[k_index])

most_unstable  = np.array([np.amax(sigma[i,:]) for i in range(sigma.shape[0])])
stability_characteristics = [phase_speed, growth_per_day, unstable_wavelength, k_wavenum[k_index]]

stability_char = f'{WORK_DIR}/saved_data/{case}/stability_characteristics_{case}_{month0}_{month1}_{values}_{ny:02}_{nz:02}.txt'
np.savetxt(stability_char, stability_characteristics)

fig, axes=plt.subplots(figsize=(6,4))

axes2 = axes.twinx()

axes.plot(k_wavenum, sigma, '.', ms=3, color='k')
axes.plot(k_wavenum, most_unstable, '.', ms=3, color='r')

axes.set_xlabel(r'$k$ [m$^{-1}$]', fontsize=18)
axes.set_ylabel(r'Growth Rate [s$^{-1}$]', fontsize=18)

axes.set_xlim([0, 1e-5])
axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=16)
axes2.tick_params(axis='both', which='major', labelsize=16)


growth_day = lambda sigma: sigma*86400
ymin, ymax = axes.get_ylim()
axes2.set_ylim((growth_day(ymin), growth_day(ymax)))
axes2.plot([],[])

axes.grid(alpha=.5)

plt.tight_layout()
plt.savefig(f'{WORK_DIR}/linear_figures/debug/growth_{case}_{month0}_{month1}_{values}_{ny:02}_{nz:02}.png', dpi=300, bbox_inches='tight')
plt.close()
