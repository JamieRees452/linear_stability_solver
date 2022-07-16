"""
Plot the terms in the eddy energy budget
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib import domain
from lib import mean_fields
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

#ny, nz, k, case, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7])

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case, integration)

# Calculate the mean zonal velocity and density fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf  = mean_fields.on_each_grid(ny, nz, case, integration, stability, assume)

# Calculate the perturbations on each grid
u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p = perturbations.on_each_grid(ny, nz, k, case, integration, stability, assume)

# Calculate the eddy pressure fluxes for the quiver plots
def za(data1, data2):
    return 0.25*(np.conj(data1)*data2+data1*np.conj(data2))
    
g = 9.81; r0 = 1026; beta = 2.29e-11
alpha = (-g/r0)*(ry_mid/rz_mid)
    
BTC = -za(u, v_p)*Uy_mid                       # Barotropic Conversion
BCC = -alpha*za(v_p, rho_p)                    # Baroclinic Conversion
KHC = -za(u, w_p)*Uz_mid                       # Kelvin-Helmholtz Conversion
PEKE = (g/r0)*za(w_p, rho_p)                   # Conversion between EKE and EPE
EKE = 0.5*(za(u, u)+za(v_p, v_p))              # Eddy Kinetic Energy
EPE = 0.5*(-g/r0)*(1/rz_mid)*za(rho_p, rho_p)  # Eddy Potential Energy
MEPFD = za(v_p, p).real
VEPFD = za(w_p, p).real

BTC = BTC.real; BCC = BCC.real; KHC = KHC.real; PEKE = PEKE.real; EKE = EKE.real; EPE = EPE.real

def domain_average(variable):
    return np.mean(np.mean(variable))

BTC_da  = domain_average(BTC.real)
KHC_da  = domain_average(KHC.real)
BCC_da  = domain_average(BCC.real)
PEKE_da = domain_average(PEKE.real)
EKE_da  = domain_average(EKE.real)
EPE_da  = domain_average(EPE.real)

print(f'\nDomain Average:\n')
print(f'BTC  = {BTC_da} ({100*BTC_da/(abs(BTC_da)+abs(KHC_da)+abs(BCC_da)):.1f}%)')
print(f'KHC  = {KHC_da} ({100*KHC_da/(abs(BTC_da)+abs(KHC_da)+abs(BCC_da)):.1f}%)')
print(f'BCC  = {BCC_da} ({100*BCC_da/(abs(BTC_da)+abs(KHC_da)+abs(BCC_da)):.1f}%)')
print(f'PEKE = {PEKE_da}')
print(f'EKE  = {EKE_da} ({100*EKE_da/(EKE_da+EPE_da):.1f}%)')
print(f'EPE  = {EPE_da} ({100*EPE_da/(EKE_da+EPE_da):.1f}%)\n')

MEPFD_norm = MEPFD/np.sqrt(MEPFD**2+VEPFD**2); VEPFD_norm = VEPFD/np.sqrt(MEPFD**2+VEPFD**2)

BTC_norm = BTC/np.amax(abs(BTC)); KHC_norm = KHC/np.amax(abs(KHC)); BCC_norm = BCC/np.amax(abs(BCC)); PEKE_norm = PEKE/np.amax(abs(PEKE))

######## PLOT FIGURES ###########################################################################################################

fig, axes=plt.subplots(figsize=(12,8), nrows=2, ncols=2, sharex=True, sharey=True)

axes[0,0].contourf(Y_mid, Z_mid, BTC_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes[0,0].contour(Y_mid, Z_mid, BTC_norm, levels = np.delete(np.linspace(-1, 1, 21),10), colors='k', linewidths=0.75)
axes[0,0].contour(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), colors='k', linewidths=0.75, alpha=.75)

axes[0,1].contourf(Y_mid, Z_mid, BCC_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes[0,1].contour(Y_mid, Z_mid, BCC_norm, levels = np.delete(np.linspace(-1, 1, 21),10), colors='k', linewidths=0.75)
axes[0,1].contour(Y, Z, r, levels = 20, colors='k', linewidths=0.75, alpha=.75)

axes[1,0].contourf(Y_mid, Z_mid, KHC_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes[1,0].contour(Y_mid, Z_mid, KHC_norm, levels = np.delete(np.linspace(-1, 1, 21),10), colors='k', linewidths=0.75)
axes[1,0].contour(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), colors='k', linewidths=0.75, alpha=.75)

axes[1,1].contourf(Y_mid, Z_mid, PEKE_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes[1,1].contour(Y_mid, Z_mid, PEKE_norm, levels = np.delete(np.linspace(-1, 1, 21),10), colors='k', linewidths=0.75)

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes[0,0].set_xlim([-3e5, 0])
    axes[0,0].set_ylim([-800, -200])

    axes[0,0].set_xticks([-3e5, -2e5, -1e5, 0])
    axes[0,0].set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
    axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
elif case == 'Proehl_3':
    axes[0,0].set_xlim([-1e6, 0])
    axes[0,0].set_ylim([-300, 0])

    axes[0,0].set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes[0,0].set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
    axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
elif (case == 'Proehl_4' or case == 'Proehl_5' or case == 'Proehl_6' or
     case == 'Proehl_7' or case == 'Proehl_8'):
    axes[0,0].set_xlim([-8e5, 8e5])
    axes[0,0].set_ylim([-250, 0])

    axes[0,0].set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes[0,0].set_yticks([-250, -200, -150, -100, -50, 0])
    
    axes[0,0].quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    axes[0,1].quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    axes[1,0].quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    axes[1,1].quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
    axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
else:
    #axes[0,0].set_xlim([-111.12*15*1000, 111.12*15*1000])
    #axes[0,0].set_ylim([-250, 0])
    
    axes[0,0].set_xlim([-111.12*10*1000, 111.12*10*1000])
    axes[0,0].set_ylim([-150, 0])

    #axes[0,0].set_xticks([-15*111.12*1000, -10*111.12*1000, -5*111.12*1000, 0, 5*111.12*1000, 10*111.12*1000, 15*111.12*1000])
    #axes[0,0].set_yticks([-250, -200, -150, -100, -50, 0])
    
    axes[0,0].set_xticks([-10*111.12*1000, -5*111.12*1000, 0, 5*111.12*1000, 10*111.12*1000])
    axes[0,0].set_yticks([-150, -100, -50, 0])
    
    #axes[0,0].set_xticklabels(['-15', '-10', '-5', '0', '5', '10', '15'])
    axes[0,0].set_xticklabels(['-10', '-5', '0', '5', '10'])
    
    axes[0,0].quiver(Y_mid[::2, :], Z_mid[::2, :], MEPFD_norm[::2, :], VEPFD_norm[::2, :], angles='xy', scale=40)
    axes[0,1].quiver(Y_mid[::2, :], Z_mid[::2, :], MEPFD_norm[::2, :], VEPFD_norm[::2, :], angles='xy', scale=40)
    axes[1,0].quiver(Y_mid[::2, :], Z_mid[::2, :], MEPFD_norm[::2, :], VEPFD_norm[::2, :], angles='xy', scale=40)
    axes[1,1].quiver(Y_mid[::2, :], Z_mid[::2, :], MEPFD_norm[::2, :], VEPFD_norm[::2, :], angles='xy', scale=40)

axes[0,0].tick_params(axis='both', which='major', labelsize=16)
axes[1,0].tick_params(axis='both', which='major', labelsize=16)
axes[1,1].tick_params(axis='both', which='major', labelsize=16)

BTC_min = "{:.2e}".format(np.amin(BTC)); BTC_max = "{:.2e}".format(np.amax(BTC))
BCC_min = "{:.2e}".format(np.amin(BCC)); BCC_max = "{:.2e}".format(np.amax(BCC))
KHC_min = "{:.2e}".format(np.amin(KHC)); KHC_max = "{:.2e}".format(np.amax(KHC))
PEKE_min = "{:.2e}".format(np.amin(PEKE)); PEKE_max = "{:.2e}".format(np.amax(PEKE))

axes[0,0].text(0.21, 0.12, f'Max: {BTC_max}\nMin:{BTC_min}', transform=axes[0,0].transAxes, ha='center', va='center', family='monospace', fontsize=18, bbox=dict(facecolor='white'))
axes[0,1].text(0.21, 0.12, f'Max: {BCC_max}\nMin:{BCC_min}', transform=axes[0,1].transAxes, ha='center', va='center', family='monospace', fontsize=18, bbox=dict(facecolor='white'))
axes[1,0].text(0.21, 0.12, f'Max: {KHC_max}\nMin:{KHC_min}', transform=axes[1,0].transAxes, ha='center', va='center', family='monospace', fontsize=18, bbox=dict(facecolor='white'))
axes[1,1].text(0.21, 0.12, f'Max: {PEKE_max}\nMin:{PEKE_min}', transform=axes[1,1].transAxes, ha='center', va='center', family='monospace', fontsize=18, bbox=dict(facecolor='white'))

axes[1,0].set_xlabel(f'Latitude [deg N]', fontsize=18)
axes[1,1].set_xlabel(f'Latitude [deg N]', fontsize=18)

axes[0,0].set_ylabel(f'Depth [m]', fontsize=18)
axes[1,0].set_ylabel(f'Depth [m]', fontsize=18)

axes[0,0].set_title(r'$-\overline{u^{\prime} v^{\prime}}U_{y}$', fontsize=22)
axes[0,1].set_title(r'$-\alpha \overline{v^{\prime} \rho^{\prime}}$', fontsize=22)
axes[1,0].set_title(r'$-\overline{u^{\prime} w^{\prime}}U_{z}$', fontsize=22)
axes[1,1].set_title(r'$-(g/\rho_{0})\overline{w^{\prime} \rho^{\prime}}$', fontsize=22)

plt.tight_layout()
plt.savefig(f'/home/rees/lsa/thesis_figures/{case}/{integration}/{stability}/{assume}/eddy_conversions/eddy_conversions_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', dpi=300, bbox_inches='tight')
plt.close()
