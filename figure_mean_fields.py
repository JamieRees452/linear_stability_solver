"""
Plot the mean fields
"""

import argparse
import matplotlib.pyplot   as plt
import numpy               as np
import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable
from lib import domain
from lib import mean_fields

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int, help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int, help='Number of vertical gridpoints')
parser.add_argument('case'       , type=str, help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
parser.add_argument('integration', type=str, help='Integration: u-bx950 (1/4 deg) u-by430 (1/12 deg)')
parser.add_argument('stability'  , type=str, help='Stability: stable unstable')
parser.add_argument('assume'     , type=str, help='Assume: RAW TWB')
args = parser.parse_args()

ny, nz, case, integration, stability, assume = args.ny, args.nz, args.case, args.integration, args.stability, args.assume

#ny, nz, case, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6])

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case, integration)

# Calculate the mean fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case, integration, stability, assume)

fig, axes=plt.subplots(figsize=(12,4), nrows=1, ncols=2, sharex=True, sharey=True)

CS0 = axes[0].contourf(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), cmap='RdBu_r')
axes[0].contour(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes[0])
cax  = divider.append_axes("right", size="5%", pad = 0.05)
cbar = plt.colorbar(CS0, cax=cax)
cbar.ax.tick_params(labelsize=16)

CS1 = axes[1].contourf(Y, Z, r, levels = np.linspace(1021.5, 1027.5, 13), cmap='viridis_r')
axes[1].contour(Y, Z, r, levels = np.linspace(1021.5, 1027.5, 13), colors='k', linewidths=0.75)

#CS1 = axes[1].contourf(Y, Z, r, levels = np.linspace(1016.5, 1026.5, 21), cmap='viridis_r')
#axes[1].contour(Y, Z, r, levels = np.linspace(1016.5, 1026.5, 21), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS1, cax=cax)
cbar.ax.tick_params(labelsize=16)

if case == 'Proehl_1' or case == 'Proehl_2':    
    axes[0].set_xlim([-3e5, 0])
    axes[0].set_ylim([-800, -200])

    axes[0].set_xticks([-3e5, -2e5, -1e5, 0])
    axes[0].set_yticks([-800, -500, -200])
    
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
elif case == 'Proehl_3':
    axes[0].set_xlim([-1e6, 0])
    axes[0].set_ylim([-300, 0])

    axes[0].set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes[0].set_yticks([-300, -150, 0])
    
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
elif (case == 'Proehl_4' or case == 'Proehl_5' or case == 'Proehl_6' or
     case == 'Proehl_7' or case == 'Proehl_8'):
     
    axes[0].set_xlim([-8e5, 8e5])
    axes[0].set_ylim([-250, 0])

    axes[0].set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes[0].set_yticks([-250, -200, -150, -100, -50, 0])
    
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
else:
    axes[0].set_xlim([-111.12*15*1000, 111.12*15*1000])
    axes[0].set_ylim([-250, 0])

    axes[0].set_xticks([-15*111.12*1000, -10*111.12*1000, -5*111.12*1000, 0, 5*111.12*1000, 10*111.12*1000, 15*111.12*1000])
    axes[0].set_yticks([-250, -200, -150, -100, -50, 0])
    
    axes[0].set_xticklabels(['-15', '-10', '-5', '0', '5', '10', '15'])

axes[0].tick_params(axis='both', which='major', labelsize=16)
axes[1].tick_params(axis='both', which='major', labelsize=16)

axes[0].set_xlabel(f'Latitude [deg N]', fontsize=18)
axes[1].set_xlabel(f'Latitude [deg N]', fontsize=18)
axes[0].set_ylabel(f'Depth [m]', fontsize=18)
axes[0].set_title(r'$U$', fontsize=22)
axes[1].set_title(r'$\rho_{0}+\overline{\rho}$', fontsize=22)

plt.tight_layout()
plt.savefig(f'/home/rees/lsa/thesis_figures/{case}/{integration}/{stability}/{assume}/mean_fields/mean_fields_{ny:02}_{nz:02}.png', dpi=300, bbox_inches='tight')
plt.close()
