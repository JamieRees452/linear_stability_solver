"""
Plot the mean fields U, Uy, Uz, r, ry, rz
"""

import argparse
import matplotlib
import matplotlib.pyplot   as plt
import numpy               as np
import os 
import sys
from   mpl_toolkits.axes_grid1 import make_axes_locatable

import calculate_NEMO_fields
import domain
import mean_fields

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('case'       , type=str  , help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
parser.add_argument('month0'     , type=str  , help='Data from month0 e.g. Jan=01')
parser.add_argument('month1'     , type=str  , help='Data from month1 e.g. Dec=12')
args = parser.parse_args()

ny, nz, case, month0, month1 = args.ny, args.nz, args.case, args.month0, args.month1

WORK_DIR = os.getcwd() 

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case)

# Calculate the mean fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case, month0, month1)

# Set the dimensional parameters
g, r0, beta = 9.81, 1026, 2.29e-11    

fname = [f'{WORK_DIR}/linear_figures/debug/U_mean_field_{case}_{month0}_{month1}_{ny:02}_{nz:02}.png',
         f'{WORK_DIR}/linear_figures/debug/Uy_mean_field_{case}_{month0}_{month1}_{ny:02}_{nz:02}.png',
         f'{WORK_DIR}/linear_figures/debug/Uz_mean_field_{case}_{month0}_{month1}_{ny:02}_{nz:02}.png',
         f'{WORK_DIR}/linear_figures/debug/r_mean_field_{case}_{month0}_{month1}_{ny:02}_{nz:02}.png',
         f'{WORK_DIR}/linear_figures/debug/ry_mean_field_{case}_{month0}_{month1}_{ny:02}_{nz:02}.png',
         f'{WORK_DIR}/linear_figures/debug/rz_mean_field_{case}_{month0}_{month1}_{ny:02}_{nz:02}.png']
         
def plot_mean_field(case, mean_data, levels, cmap, title, save_to):
    """
    Plot the mean fields U, Uy, Uz, r, ry, rz for a given case
                            
    Parameters
    ----------
    case : str
         Mean fields about which to perform the linear stability analysis
         e.g. Proehl_[1-8] - Proehls test cases (Proehl (1996) and Proehl (1998))
              NEMO_25      - Data from the 1/4deg coupled AOGCM
              NEMO_12      - Data from the 1/12deg coupled AOGCM
    
    mean_data : array
         Mean field data to plot
         
    levels : array-like
         Determines the number and positions of the contour lines/regions
    
    cmap : str
         The Colormap name used to map scalar data to colors
    
    title : str
         Axes title
         
    save_to : str
         Location for where to save the figure
    """

    fig, axes=plt.subplots(figsize=(6,4))

    CS = axes.contourf(Y, Z, mean_data, levels = levels, cmap=cmap)
    axes.contour(Y, Z, mean_data, levels = levels, colors='k', linewidths=0.75)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad = 0.05)
    
    cbar = plt.colorbar(CS, cax=cax)
    cbar.ax.tick_params(labelsize=16)

    if case == 'Proehl_1' or case == 'Proehl_2':
        axes.set_xlim([-3e5, 0])
        axes.set_ylim([-800, -200])

        axes.set_xticks([-3e5, -2e5, -1e5, 0])
        axes.set_yticks([-800, -500, -200])
        
        axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    elif case == 'Proehl_3':
        axes.set_xlim([-1e6, 0])
        axes.set_ylim([-300, 0])

        axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
        axes.set_yticks([-300, -150, 0])
        
        axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    elif (case == 'Proehl_4' or case == 'Proehl_5' or case == 'Proehl_6' or
         case == 'Proehl_7' or case == 'Proehl_8'):
         
        axes.set_xlim([-8e5, 8e5])
        axes.set_ylim([-250, 0])

        axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
        axes.set_yticks([-250, -200, -150, -100, -50, 0])
        
        axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    else:
        axes.set_xlim([-111.12*15*1000, 111.12*15*1000])
        axes.set_ylim([-250, 0])

        axes.set_xticks([-15*111.12*1000, -10*111.12*1000, -5*111.12*1000, 0, 5*111.12*1000, 10*111.12*1000, 15*111.12*1000])
        axes.set_yticks([-250, -200, -150, -100, -50, 0])
        
        axes.set_xticklabels(['-15', '-10', '-5', '0', '5', '10', '15'])

    axes.tick_params(axis='both', which='major', labelsize=16)

    axes.set_xlabel(f'Latitude [deg N]', fontsize=18)
    axes.set_ylabel(f'Depth [m]', fontsize=18)
    axes.set_title(title, fontsize=22)

    plt.tight_layout()
    plt.savefig(save_to, dpi=300, bbox_inches='tight')
    plt.close()

mean_data = [U, Uy, Uz, r, ry, rz]

levels = [np.delete(np.linspace(-1.5, 1.5, 31), 15), 
np.linspace(-np.amax(abs(Uy)), np.amax(abs(Uy)), 21),
np.linspace(-np.amax(abs(Uz)), np.amax(abs(Uz)), 21),
np.linspace(1016, 1028, 25),
np.linspace(-np.amax(abs(ry)), np.amax(abs(ry)), 21),
np.linspace(-np.amax(abs(rz)), np.amax(abs(rz)), 21)]

cmap = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'viridis_r', 'RdBu_r', 'RdBu_r']

title = [r'$U$',r'$U_{y}$',r'$U_{z}$',r'$\rho_{0}+\overline{\rho}$',r'$\rho_{y}$',r'$\rho_{z}$']

for i in range(6):
    plot_mean_field(case, mean_data[i], levels[i], cmap[i], title[i], fname[i])
