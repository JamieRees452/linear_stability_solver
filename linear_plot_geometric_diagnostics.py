"""
Plot GEOMETRIC diagnostics
"""

import argparse
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sys
from   mpl_toolkits.axes_grid1 import make_axes_locatable

import domain
import mean_fields
import perturbations

WORK_DIR = '/home/rees/lsa'

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('k'          , type=float, help='Zonal wavenumber')
parser.add_argument('case'       , type=str  , help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
parser.add_argument('month0'     , type=str  , help='Data from month0 e.g. Jan=01')
parser.add_argument('month1'     , type=str  , help='Data from month1 e.g. Dec=12')
args = parser.parse_args()

ny, nz, k, case, month0, month1 = args.ny, args.nz, args.k, args.case, args.month0, args.month1

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case)

# Dimensional values
g, r0, beta = 9.81, 1026, 2.29e-11  

# Calculate the mean zonal velocity and density fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf  = mean_fields.on_each_grid(ny, nz, case, month0, month1)

# Calculate perturbations
u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p = perturbations.on_each_grid(ny, nz, k, case, month0, month1)

# Zonal average of the product of two quantities
def za(data1, data2):
    return 0.25*(np.conj(data1)*data2+data1*np.conj(data2))
    
K = 0.5*(za(u, u)+za(v_p, v_p)).real              # Eddy Kinetic Energy
P = 0.5*(-g/r0)*(1/rz_mid)*za(rho_p, rho_p).real  # Eddy Potential Energy

M = 0.5*(za(v_p, v_p) - za(u, u)).real            #
N = za(u, v_p).real                               # Reynolds Stress

gamma_m = np.sqrt(M**2 + N**2)/K             # Horizontal eddy anisotropy
phi_m = 0.5*np.angle(-N/M)                   # Horizonal eddy tilt


fname = [f'{WORK_DIR}/linear_figures/debug/geo_m_{case}_{month0}_{month1}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
         f'{WORK_DIR}/linear_figures/debug/geo_n_{case}_{month0}_{month1}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
         f'{WORK_DIR}/linear_figures/debug/geo_gamma_m_{case}_{month0}_{month1}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
         f'{WORK_DIR}/linear_figures/debug/geo_phi_m_{case}_{month0}_{month1}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']
         
def plot_geometric(geometric_data, levels, cmap, title, save_to):
    """
    Plot GEOMETRIC diagnostics
                            
    Parameters
    ----------
    geometric_data : array
         GEOMETRIC data to plot
         
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

    CS = axes.contourf(Y_mid, Z_mid, geometric_data, levels = levels, cmap=cmap)
    #axes.contour(Y_mid, Z_mid, geometric_data, levels = levels, colors='k', linewidths=0.75)
    
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
    
geometric_data = [M, N, gamma_m, phi_m]

levels = [np.linspace(-np.amax(abs(M)), np.amax(abs(M)), 21), 
np.linspace(-np.amax(abs(N)), np.amax(abs(N)), 21),
np.linspace(0, 1, 50),
np.linspace(-np.pi, np.pi, 50)]

cmap = ['RdBu_r', 'RdBu_r', 'viridis', 'RdBu_r']

title = [r'$(\overline{v^{\prime^{2}}} - \overline{u^{\prime^{2}}})/2$',
r'$\overline{u^{\prime} v^{\prime}}$',
r'$\gamma_{m}$',
r'$\phi_{m}$']

for i in range(4):
    plot_geometric(geometric_data[i], levels[i], cmap[i], title[i], fname[i])
             
