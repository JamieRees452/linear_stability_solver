"""
Plot the terms in the eddy energy budget
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

import domain
import mean_fields
import perturbations

ny, nz, k, case, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7])

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case, integration)

# Dimensional values
g, r0, beta = 9.81, 1026, 2.29e-11 

# Calculate the mean zonal velocity and density fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf  = mean_fields.on_each_grid(ny, nz, case, integration, stability, assume)

Q  = -(1/r0)*(ry*Uz + (beta*Y-Uy)*rz); Qy = np.gradient(Q, y, axis=1)

# Calculate the perturbations on each grid
u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p = perturbations.on_each_grid(ny, nz, k, case, integration, stability, assume)

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/eigenvectors/evecs_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
else:
    fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/eigenvectors/evecs_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
         
if os.path.exists(fname[0]) and os.path.exists(fname[1]):
    print(f'Loading eigenvalues and eigenvectors...')
    evals = np.loadtxt(fname[0]).view(complex).reshape(-1) 
    evecs = np.loadtxt(fname[1]).view(complex).reshape(-1) 
    cs    = evals[np.argmax(evals.imag)]
else:
    raise ValueError(f'The specified files do not exist\n{fname[0]}\n{fname[1]}')  

[m,n] = np.where(U-cs.real<0); critical = np.zeros(U.shape); critical[m,n] = 99 # Critical Levels
EIL   = beta*y*(beta*y-Uy)-(k**2)*((U-cs.real)**2) # Effective Inertial Latitude

# Calculate the eddy pressure fluxes for the quiver plots
def za(data1, data2):
    return 0.25*(np.conj(data1)*data2+data1*np.conj(data2))

MEPFD = za(v_p, p).real; VEPFD = za(w_p, p).real

MEPFD_norm = MEPFD/np.sqrt(MEPFD**2+VEPFD**2)
VEPFD_norm = VEPFD/np.sqrt(MEPFD**2+VEPFD**2)

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = [f'/home/rees/lsa/eddy_energy_budget/BTC/BTC_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/KHC/KHC_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/BCC/BCC_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/MEPFD/MEPFD_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/VEPFD/VEPFD_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/EKE/EKE_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/EPE/EPE_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/LHS/LHS_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/RHS/RHS_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
else:
    fname = [f'/home/rees/lsa/eddy_energy_budget/BTC/BTC_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/KHC/KHC_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/BCC/BCC_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/MEPFD/MEPFD_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/VEPFD/VEPFD_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/EKE/EKE_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/EPE/EPE_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/LHS/LHS_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/RHS/RHS_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']

# Load the eddy energy budget terms
BTC   = np.loadtxt(fname[0]).reshape(int(nz-1), int(ny-1))
KHC   = np.loadtxt(fname[1]).reshape(int(nz-1), int(ny-1))
BCC   = np.loadtxt(fname[2]).reshape(int(nz-1), int(ny-1))
#MEPFD = np.loadtxt(fname[3]).reshape(int(nz-1), int(ny))
#VEPFD = np.loadtxt(fname[4]).reshape(int(nz), int(ny-1))
EKE   = np.loadtxt(fname[5]).reshape(int(nz-1), int(ny-1))
EPE   = np.loadtxt(fname[6]).reshape(int(nz-1), int(ny-1))
LHS   = np.loadtxt(fname[7]).reshape(int(nz-1), int(ny-1))
RHS   = np.loadtxt(fname[8]).reshape(int(nz-1), int(ny-1))

BTC_norm   = BTC/np.amax(abs(BTC)); KHC_norm = KHC/np.amax(abs(KHC)); BCC_norm = BCC/np.amax(abs(BCC))
MEPFD_normal = MEPFD/np.amax(abs(MEPFD)); VEPFD_normal = VEPFD/np.amax(abs(VEPFD))
EKE_norm   = EKE/np.amax(abs(EKE)); EPE_norm = EPE/np.amax(abs(EPE))
LHS_norm   = LHS/np.amax(abs(LHS)); RHS_norm = RHS/np.amax(abs(RHS))

if case == 'NEMO':
    fname = [f'/home/rees/lsa/figures/eddy_energy_budget/BTC/BTC_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/KHC/KHC_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/BCC/BCC_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/MEPFD/MEPFD_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/VEPFD/VEPFD_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/EKE/EKE_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/EPE/EPE_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/LHS/LHS_{case}_{integration}_{stability}_{assume}{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/RHS/RHS_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']
else:
    fname = [f'/home/rees/lsa/figures/eddy_energy_budget/BTC/BTC_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/KHC/KHC_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/BCC/BCC_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/MEPFD/MEPFD_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/VEPFD/VEPFD_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/EKE/EKE_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/EPE/EPE_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/LHS/LHS_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eddy_energy_budget/RHS/RHS_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']

######## PLOT FIGURES ###########################################################################################################

print(f'\nSaving figures to:\n')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, BTC, levels = np.linspace(-np.amax(abs(BTC)), np.amax(abs(BTC)), 20), cmap='RdBu_r')
#axes.contour(Y_mid, Z_mid, BTC, levels = np.linspace(-np.amax(abs(BTC)), np.amax(abs(BTC)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, BTC_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, BTC_norm, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

BTC_min = "{:.2e}".format(np.amin(BTC)); BTC_max = "{:.2e}".format(np.amax(BTC))
axes.text(0.15, 0.1, f'Max: {BTC_max}\nMin:{BTC_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$-\overline{u^{\prime} v^{\prime}}U_{y}$')

plt.tight_layout()
plt.savefig(fname[0], dpi=300, bbox_inches='tight')
plt.close()

print(f'Barotropic Conversion                    : {fname[0]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, KHC, levels = np.linspace(-np.amax(abs(KHC)), np.amax(abs(KHC)), 20), cmap='RdBu_r')
#axes.contour(Y_mid, Z_mid, KHC, levels = np.linspace(-np.amax(abs(KHC)), np.amax(abs(KHC)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, KHC_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, KHC_norm, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

KHC_min = "{:.2e}".format(np.amin(KHC)); KHC_max = "{:.2e}".format(np.amax(KHC))
axes.text(0.15, 0.1, f'Max: {KHC_max}\nMin:{KHC_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$-\overline{u^{\prime} w^{\prime}}U_{z}$')

plt.tight_layout()
plt.savefig(fname[1], dpi=300, bbox_inches='tight')
plt.close()

print(f'Kelvin-Helmholtz Conversion              : {fname[1]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, BCC, levels = np.linspace(-np.amax(abs(BCC)), np.amax(abs(BCC)), 20), cmap='RdBu_r')
#axes.contour(Y_mid, Z_mid, BCC, levels = np.linspace(-np.amax(abs(BCC)), np.amax(abs(BCC)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, BCC_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, BCC_norm, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

BCC_min = "{:.2e}".format(np.amin(BCC)); BCC_max = "{:.2e}".format(np.amax(BCC))
axes.text(0.15, 0.1, f'Max: {BCC_max}\nMin:{BCC_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$-\alpha\overline{v^{\prime} \rho^{\prime}}$')

plt.tight_layout()
plt.savefig(fname[2], dpi=300, bbox_inches='tight')
plt.close()

print(f'Baroclinic Conversion                    : {fname[2]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_full, Z_half, MEPFD, levels = np.linspace(-np.amax(abs(MEPFD)), np.amax(abs(MEPFD)), 20), cmap='RdBu_r')
#axes.contour(Y_full, Z_half, MEPFD, levels = np.linspace(-np.amax(abs(MEPFD)), np.amax(abs(MEPFD)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, MEPFD_normal, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, MEPFD_normal, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

MEPFD_min = "{:.2e}".format(np.amin(MEPFD)); MEPFD_max = "{:.2e}".format(np.amax(MEPFD))
axes.text(0.15, 0.1, f'Max: {MEPFD_max}\nMin:{MEPFD_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{v^{\prime} p^{\prime}}$')

plt.tight_layout()
plt.savefig(fname[3], dpi=300, bbox_inches='tight')
plt.close()

print(f'Meridional Eddy Pressure Flux Divergence : {fname[3]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_half, Z_full, VEPFD, levels = np.linspace(-np.amax(abs(VEPFD)), np.amax(abs(VEPFD)), 20), cmap='RdBu_r')
#axes.contour(Y_half, Z_full, VEPFD, levels = np.linspace(-np.amax(abs(VEPFD)), np.amax(abs(VEPFD)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, VEPFD_normal, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, VEPFD_normal, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

VEPFD_min = "{:.2e}".format(np.amin(VEPFD)); VEPFD_max = "{:.2e}".format(np.amax(VEPFD))
axes.text(0.15, 0.1, f'Max: {VEPFD_max}\nMin:{VEPFD_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{w^{\prime} p^{\prime}}$')

plt.tight_layout()
plt.savefig(fname[4], dpi=300, bbox_inches='tight')
plt.close()

print(f'Vertical Eddy Pressure Flux Divergence   : {fname[4]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, EKE, levels = np.linspace(0, np.amax(abs(EKE)), 10), cmap='Reds')
#axes.contour(Y_mid, Z_mid, EKE, levels = np.linspace(0, np.amax(abs(EKE)), 10), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, EKE_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y_mid, Z_mid, EKE_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

EKE_max = "{:.2e}".format(np.amax(EKE))
axes.text(0.15, 0.05, f'Max: {EKE_max}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\dfrac{1}{2}(\overline{u^{\prime^{2}}} + \overline{v^{\prime^{2}}})$')

plt.tight_layout()
plt.savefig(fname[5], dpi=300, bbox_inches='tight')
plt.close()

print(f'Eddy Kinetic Energy                      : {fname[5]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, EPE, levels = np.linspace(0, np.amax(abs(EPE)), 10), cmap='Reds')
#axes.contour(Y_mid, Z_mid, EPE, levels = np.linspace(0, np.amax(abs(EPE)), 10), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, EPE_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y_mid, Z_mid, EPE_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

EPE_max = "{:.2e}".format(np.amax(EPE))
axes.text(0.15, 0.05, f'Max: {EPE_max}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{b^{\prime^{2}}}/2N^{2}$')

plt.tight_layout()
plt.savefig(fname[6], dpi=300, bbox_inches='tight')
plt.close()

print(f'Eddy Potential Energy                    : {fname[6]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, LHS, levels = np.linspace(0, np.amax(abs(LHS)), 10), cmap='Reds')
#axes.contour(Y_mid, Z_mid, LHS, levels = np.linspace(0, np.amax(abs(LHS)), 10), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, LHS_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y_mid, Z_mid, LHS_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

LHS_max = "{:.2e}".format(np.amax(LHS))
axes.text(0.15, 0.05, f'Max: {LHS_max}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'Total Eddy Energy')

plt.tight_layout()
plt.savefig(fname[7], dpi=300, bbox_inches='tight')
plt.close()

print(f'Total Eddy Energy (LHS)                  : {fname[7]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, RHS, levels = np.linspace(-np.amax(abs(RHS)), np.amax(abs(RHS)), 20), cmap='RdBu_r')
#axes.contour(Y_mid, Z_mid, RHS, levels = np.linspace(-np.amax(abs(RHS)), np.amax(abs(RHS)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, RHS_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y_mid, Z_mid, RHS_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
    q = axes.quiver(Y_mid[::2,:], Z_mid[::2,:], MEPFD_norm[::2,:], VEPFD_norm[::2,:], angles='xy', scale=30)
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
    q = axes.quiver(Y_mid, Z_mid, MEPFD_norm, VEPFD_norm, angles='xy', scale=40)

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

RHS_min = "{:.2e}".format(np.amin(RHS)); RHS_max = "{:.2e}".format(np.amax(RHS))
axes.text(0.15, 0.1, f'Max: {RHS_max}\nMin:{RHS_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'Total Eddy Energy (RHS)')

plt.tight_layout()
plt.savefig(fname[8], dpi=300, bbox_inches='tight')
plt.close()

print(f'Total Eddy Energy (RHS)                  : {fname[8]}\n')
