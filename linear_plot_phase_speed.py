"""
Plot phase speeds
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

WORK_DIR = '/home/rees/lsa' 

k_wavenum = np.linspace(k_start, k_end, k_num)
    
fname = f'{WORK_DIR}/saved_data/{case}/growth_{case}_{month0}_{month1}_{values}_{ny:02}_{nz:02}_*.txt'
    
files = glob.glob(fname)

cs = np.array([np.loadtxt(filename).view(complex).reshape(values, k_num) for filename in files])
cs = cs.flatten().reshape(len(files)*values, k_num)

phase = np.asarray([cs[:, i].real for i in range(k_num)])

fig, axes=plt.subplots(figsize=(6,4))

axes.plot(k_wavenum, phase, '.', ms=3, color='k')
#axes.plot(k_wavenum, most_unstable, '.', ms=3, color='r')

axes.set_xlabel(r'$k$ [m$^{-1}$]', fontsize=18)
axes.set_ylabel(r'Phase Speed [ms$^{-1}$]', fontsize=18)

axes.set_xlim([0, 1e-5])
axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=16)

axes.grid(alpha=.5)

plt.tight_layout()
plt.savefig(f'{WORK_DIR}/linear_figures/debug/phase_{case}_{month0}_{month1}_{values}_{ny:02}_{nz:02}.png', dpi=300, bbox_inches='tight')
plt.close()
