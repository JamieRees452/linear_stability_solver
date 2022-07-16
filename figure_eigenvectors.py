"""
Plot the amplitude and phase of an eigenvector
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib import domain
from lib import perturbations

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('k'          , type=float, help='Zonal wavenumber')
parser.add_argument('case'       , type=str  , help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
parser.add_argument('integration', type=str  , help='Integration: u-bx950 (1/4 deg) u-by430 (1/12 deg)')
parser.add_argument('stability'  , type=str  , help='Stability: stable unstable')
parser.add_argument('assume'     , type=str  , help='Assume: RAW TWB')
args = parser.parse_args()

ny, nz, k, case, integration, stability, assume = args.ny, args.nz, args.k, args.case, args.integration, args.stability, args.assume

# Inputs from the command terminal
#ny, nz, k, case, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7])

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case, integration)

# Calculate the perturbations
u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p = perturbations.on_each_grid(ny, nz, k, case, integration, stability, assume)

# Obtain the eigenvalues and eigenvectors from files

fname = [f'/home/rees/lsa/saved_data/{case}/{integration}/{stability}/{assume}/eigenvalues/evals_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
        f'/home/rees/lsa/saved_data/{case}/{integration}/{stability}/{assume}/eigenvectors/evecs_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
         
if os.path.exists(fname[0]) and os.path.exists(fname[1]):
    evals = np.loadtxt(fname[0]).view(complex).reshape(-1) 
    evecs = np.loadtxt(fname[1]).view(complex).reshape(-1) 
else:
    raise ValueError(f'The specified files do not exist\n{fname[0]}\n{fname[1]}')  

# State the files where the figures will be saved

fname = f'/home/rees/lsa/thesis_figures/{case}/{integration}/{stability}/{assume}/eigenvectors/eigenvectors_{ny:02}_{nz:02}_{str(int(k*1e8))}.png'
             
# Calculate the amplitudes and phases
             
u_amplitude   = abs(u);       u_amplitude_norm = u_amplitude/np.amax(abs(u));         u_phase = np.angle(u, deg=True)
v_amplitude   = abs(v_p);     v_amplitude_norm = v_amplitude/np.amax(abs(v_p));       v_phase = np.angle(v_p, deg=True)

#for i in range(ny-1):
#    for j in range(nz-1):
#        if u_amplitude[j, i] <= 0.01*np.amax(u_amplitude):
#            u_phase[j,i] = np.nan
#        else:
#            pass

# Plot figure

fig, axes = plt.subplots(figsize=(12,8), nrows=2, ncols=2, sharex=True, sharey=True)

axes[0,0].contourf(Y_mid, Z_mid, u_amplitude_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes[0,0].contour(Y_mid, Z_mid, u_amplitude_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

CS = axes[0,1].contourf(Y_mid, Z_mid, u_phase, levels = np.asarray([-180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180]), cmap='RdBu_r')
axes[0,1].contour(Y_mid, Z_mid, u_phase, levels = np.asarray([-180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180]), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes[0,1])
cax = divider.append_axes("right", size="5%", pad = 0.05)
cbar = plt.colorbar(CS, cax=cax, ticks=np.asarray([-180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180]))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_yticklabels([r'$-\pi$',r'$-5\pi/6$',r'$-2\pi/3$',r'$-\pi/2$',r'$-\pi/3$',r'$-\pi/6$',r'$\pi/6$',r'$\pi/3$',r'$\pi/2$',r'$2\pi/3$',r'$5\pi/6$',r'$\pi$'])

axes[1,0].contourf(Y_mid, Z_mid, v_amplitude_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes[1,0].contour(Y_mid, Z_mid, v_amplitude_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

CS = axes[1,1].contourf(Y_mid, Z_mid, v_phase, levels = np.asarray([-180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180]), cmap='RdBu_r')
axes[1,1].contour(Y_mid, Z_mid, v_phase, levels = np.asarray([-180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180]), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes[1,1])
cax = divider.append_axes("right", size="5%", pad = 0.05)
cbar = plt.colorbar(CS, cax=cax, ticks=np.asarray([-180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180]))
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_yticklabels([r'$-\pi$',r'$-5\pi/6$',r'$-2\pi/3$',r'$-\pi/2$',r'$-\pi/3$',r'$-\pi/6$',r'$\pi/6$',r'$\pi/3$',r'$\pi/2$',r'$2\pi/3$',r'$5\pi/6$',r'$\pi$'])

if case == 'Proehl_1' or case == 'Proehl_2':    
    axes[0,0].set_xlim([-3e5, 0])
    axes[0,0].set_ylim([-800, -200])

    axes[0,0].set_xticks([-3e5, -2e5, -1e5, 0])
    axes[0,0].set_yticks([-800, -500, -200])
    
    axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
elif case == 'Proehl_3':
    axes[0,0].set_xlim([-1e6, 0])
    axes[0,0].set_ylim([-300, 0])

    axes[0,0].set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes[0,0].set_yticks([-300, -150, 0])
    
    axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
elif (case == 'Proehl_4' or case == 'Proehl_5' or case == 'Proehl_6' or
     case == 'Proehl_7' or case == 'Proehl_8'):
     
    axes[0,0].set_xlim([-8e5, 8e5])
    axes[0,0].set_ylim([-250, 0])

    axes[0,0].set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes[0,0].set_yticks([-250, -200, -150, -100, -50, 0])
    
    axes[0,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
else:
    axes[0,0].set_xlim([-111.12*15*1000, 111.12*15*1000])
    axes[0,0].set_ylim([-250, 0])

    axes[0,0].set_xticks([-15*111.12*1000, -10*111.12*1000, -5*111.12*1000, 0, 5*111.12*1000, 10*111.12*1000, 15*111.12*1000])
    axes[0,0].set_yticks([-250, -200, -150, -100, -50, 0])
    
    axes[0,0].set_xticklabels(['-15', '-10', '-5', '0', '5', '10', '15'])
    
axes[0,0].tick_params(axis='both', which='major', labelsize=16)
axes[1,0].tick_params(axis='both', which='major', labelsize=16)
axes[0,1].tick_params(axis='both', which='major', labelsize=16)
axes[1,1].tick_params(axis='both', which='major', labelsize=16)

u_maximum = "{:.2e}".format(np.amax(u_amplitude))
axes[0,0].text(0.225, 0.075, f'Max: {u_maximum}', transform=axes[0,0].transAxes, ha='center', va='center', family='monospace', fontsize=18, bbox=dict(facecolor='white'))

v_maximum = "{:.2e}".format(np.amax(v_amplitude))
axes[1,0].text(0.225, 0.075, f'Max: {v_maximum}', transform=axes[1,0].transAxes, ha='center', va='center', family='monospace', fontsize=18, bbox=dict(facecolor='white'))

axes[1,0].set_xlabel(f'Latitude [deg N]', fontsize=18)
axes[1,1].set_xlabel(f'Latitude [deg N]', fontsize=18)
axes[0,0].set_ylabel(f'Depth [m]', fontsize=18)
axes[1,0].set_ylabel(f'Depth [m]', fontsize=18)

axes[0,0].set_title(r'$|u^{\prime}|$', fontsize=22)
axes[1,0].set_title(r'$|v^{\prime}|$', fontsize=22)
axes[0,1].set_title(r'Phase($u^{\prime}$)', fontsize=22)
axes[1,1].set_title(r'Phase($v^{\prime}$)', fontsize=22)

plt.tight_layout()
plt.savefig(fname, dpi=300, bbox_inches='tight')
plt.show()
