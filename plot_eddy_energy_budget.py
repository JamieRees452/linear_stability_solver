"""
Plot the terms in the eddy energy budget
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mean_fields

ny, nz, k, case = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4])

if case == 'NEMO':
    if integration == 'u-by430':
        lat = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/latitude_12.txt')
    else:
        lat = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/latitude_25.txt')
        
    depth = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/depth.txt'); depth = -depth[::-1]
    
    L = abs(lat[0])*111.12*1000
    D = abs(depth[0])
    
else:
    L = (10*111.12)*1000 # Meridional half-width of the domain (m)
    D = 1000             # Depth of the domain (m)

y = np.linspace(-L, L, ny); z = np.linspace(-D, 0, nz) 

dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]

Y,Z         = np.meshgrid(y, z);         Y_full,Z_half = np.meshgrid(y, z_mid) 
Y_mid,Z_mid = np.meshgrid(y_mid, z_mid); Y_half,Z_full = np.meshgrid(y_mid, z)

# Dimensional values
g, r0, beta = 9.81, 1026, 2.29e-11 

# Calculate the mean zonal velocity and density fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf  = mean_fields.on_each_grid(ny, nz, case)

Q  = -(1/r0)*(ry*Uz + (beta*Y-Uy)*rz); Qy = np.gradient(Q, y, axis=1)

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/eigenvectors/evecs_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
else:
    fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/eigenvectors/evecs_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
         
if os.path.exists(fname[0]) and os.path.exists(fname[1]):
    print(f'Loading eigenvalues and eigenvectors...')
    evals = np.loadtxt(fname[0]).view(complex).reshape(-1) 
    evecs = np.loadtxt(fname[1]).view(complex).reshape(-1) 
    cs = evals[np.argmax(evals.imag)].imag
else:
    raise ValueError(f'The specified files do not exist\n{fname[0]}\n{fname[1]}')  

[m,n] = np.where(U-cs.real<0); critical = np.zeros(U.shape); critical[m,n] = 99 # Critical Levels
EIL   = beta*y*(beta*y-Uy)-(k**2)*((U-cs.real)**2) # Effective Inertial Latitude

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = [f'/home/rees/lsa/eddy_energy_budget/BTC/BTC_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/KHC/KHC_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/BCC/BCC_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/MEPFD/MEPFD_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/VEPFD/VEPFD_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/EKE/EKE_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/EPE/EPE_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/LHS/LHS_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/eddy_energy_budget/RHS/RHS_{integration}_{stability}_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
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
MEPFD = np.loadtxt(fname[3]).reshape(int(nz-1), int(ny))
VEPFD = np.loadtxt(fname[4]).reshape(int(nz), int(ny-1))
EKE   = np.loadtxt(fname[5]).reshape(int(nz-1), int(ny-1))
EPE   = np.loadtxt(fname[6]).reshape(int(nz-1), int(ny-1))
LHS   = np.loadtxt(fname[7]).reshape(int(nz-1), int(ny-1))
RHS   = np.loadtxt(fname[8]).reshape(int(nz-1), int(ny-1))

######## PLOT FIGURES ###########################################################################################################

print(f'\nSaving figures to:\n')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, BTC, levels = np.linspace(-np.amax(abs(BTC)), np.amax(abs(BTC)), 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, BTC, levels = np.linspace(-np.amax(abs(BTC)), np.amax(abs(BTC)), 20), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':   
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$-\overline{u^{\prime} v^{\prime}}U_{y}$')

plt.tight_layout()
plt.savefig(fname[0], dpi=300, bbox_inches='tight')
plt.close()

print(f'Barotropic Conversion                    : {fname[0]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, KHC, levels = np.linspace(-np.amax(abs(KHC)), np.amax(abs(KHC)), 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, KHC, levels = np.linspace(-np.amax(abs(KHC)), np.amax(abs(KHC)), 20), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$-\overline{u^{\prime} w^{\prime}}U_{z}$')

plt.tight_layout()
plt.savefig(fname[1], dpi=300, bbox_inches='tight')
plt.close()

print(f'Kelvin-Helmholtz Conversion              : {fname[1]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, BCC, levels = np.linspace(-np.amax(abs(BCC)), np.amax(abs(BCC)), 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, BCC, levels = np.linspace(-np.amax(abs(BCC)), np.amax(abs(BCC)), 20), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$-\alpha\overline{v^{\prime} \rho^{\prime}}$')

plt.tight_layout()
plt.savefig(fname[2], dpi=300, bbox_inches='tight')
plt.close()

print(f'Baroclinic Conversion                    : {fname[2]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_full, Z_half, MEPFD, levels = np.linspace(-np.amax(abs(MEPFD)), np.amax(abs(MEPFD)), 20), cmap='RdBu_r')
axes.contour(Y_full, Z_half, MEPFD, levels = np.linspace(-np.amax(abs(MEPFD)), np.amax(abs(MEPFD)), 20), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':    
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{v^{\prime} p^{\prime}}$')

plt.tight_layout()
plt.savefig(fname[3], dpi=300, bbox_inches='tight')
plt.close()

print(f'Meridional Eddy Pressure Flux Divergence : {fname[3]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_half, Z_full, VEPFD, levels = np.linspace(-np.amax(abs(VEPFD)), np.amax(abs(VEPFD)), 20), cmap='RdBu_r')
axes.contour(Y_half, Z_full, VEPFD, levels = np.linspace(-np.amax(abs(VEPFD)), np.amax(abs(VEPFD)), 20), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':    
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{w^{\prime} p^{\prime}}$')

plt.tight_layout()
plt.savefig(fname[4], dpi=300, bbox_inches='tight')
plt.close()

print(f'Vertical Eddy Pressure Flux Divergence   : {fname[4]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, EKE, levels = np.linspace(0, np.amax(abs(EKE)), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, EKE, levels = np.linspace(0, np.amax(abs(EKE)), 10), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':    
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\dfrac{1}{2}(\overline{u^{\prime^{2}}} + \overline{v^{\prime^{2}}})$')

plt.tight_layout()
plt.savefig(fname[5], dpi=300, bbox_inches='tight')
plt.close()

print(f'Eddy Kinetic Energy                      : {fname[5]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, EPE, levels = np.linspace(0, np.amax(abs(EPE)), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, EPE, levels = np.linspace(0, np.amax(abs(EPE)), 10), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':    
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{b^{\prime^{2}}}/2N^{2}$')

plt.tight_layout()
plt.savefig(fname[6], dpi=300, bbox_inches='tight')
plt.close()

print(f'Eddy Potential Energy                    : {fname[6]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, LHS, levels = np.linspace(0, np.amax(abs(LHS)), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, LHS, levels = np.linspace(0, np.amax(abs(LHS)), 10), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':    
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'Total Eddy Energy')

plt.tight_layout()
plt.savefig(fname[7], dpi=300, bbox_inches='tight')
plt.close()

print(f'Total Eddy Energy (LHS)                  : {fname[7]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, RHS, levels = np.linspace(-np.amax(abs(RHS)), np.amax(abs(RHS)), 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, RHS, levels = np.linspace(-np.amax(abs(RHS)), np.amax(abs(RHS)), 20), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':
    axes.set_xlim([-3e5, 0])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-1e6, 0])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes.set_yticks([-300, -150, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'Total Eddy Energy (RHS)')

plt.tight_layout()
plt.savefig(fname[8], dpi=300, bbox_inches='tight')
plt.close()

print(f'Total Eddy Energy (RHS)                  : {fname[8]}\n')
