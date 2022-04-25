"""
Plot the amplitude and phase of an eigenvector
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mean_fields
import perturbations

# Inputs from the command terminal
ny, nz, k, case = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4])

# Dimensional values
g, r0 = 9.81, 1026

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
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf  = mean_fields.on_each_grid(ny, nz, case, integration, stability)
  
Q  = -(1/r0)*(ry*Uz + (beta*Y-Uy)*rz); Qy = np.gradient(Q, y, axis=1)

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

[m,n] = np.where(U-cs.real<0); critical = np.zeros(U.shape); critical[m,n] = 99 # Critical levels
EIL   = beta*y*(beta*y-Uy)-(k**2)*((U-cs.real)**2)                              # Effective Inertial Latitude

u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p = perturbations.on_each_grid(ny, nz, k, case, integration, stability)

## PLOT AMPLITUDES ############################################################################################################################

u_amplitude = abs(u); v_amplitude = abs(v_p); p_amplitude = abs(p); w_amplitude = abs(w_p); rho_amplitude = abs(rho_p)

if case == 'NEMO':
    fname = [f'/home/rees/lsa/figures/eigenvectors/amplitude/u/u_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/amplitude/v/v_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/amplitude/p/p_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/amplitude/w/w_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/amplitude/rho/rho_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']
else:
    fname = [f'/home/rees/lsa/figures/eigenvectors/amplitude/u/u_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/amplitude/v/v_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/amplitude/p/p_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/amplitude/w/w_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/amplitude/rho/rho_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']
         
print(f'\nSaving figures to:\n')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, u_amplitude, levels = np.linspace(0, np.amax(u_amplitude), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, u_amplitude, levels = np.linspace(0, np.amax(u_amplitude), 10), colors='k', linewidths=0.75)

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
axes.set_title(r'$|u^{\prime}|$')

plt.tight_layout()
plt.savefig(fname[0], dpi=300, bbox_inches='tight')
plt.close()

print(f'u   : {fname[0]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, v_amplitude, levels = np.linspace(0, np.amax(v_amplitude), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, v_amplitude, levels = np.linspace(0, np.amax(v_amplitude), 10), colors='k', linewidths=0.75)

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
axes.set_title(r'$|v^{\prime}|$')

plt.tight_layout()
plt.savefig(fname[1], dpi=300, bbox_inches='tight')
plt.close()

print(f'v   : {fname[1]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, p_amplitude, levels = np.linspace(0, np.amax(p_amplitude), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, p_amplitude, levels = np.linspace(0, np.amax(p_amplitude), 10), colors='k', linewidths=0.75)

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
axes.set_title(r'$|p^{\prime}|$')

plt.tight_layout()
plt.savefig(fname[2], dpi=300, bbox_inches='tight')
plt.close()

print(f'p   : {fname[2]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, w_amplitude, levels = np.linspace(0, np.amax(w_amplitude), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, w_amplitude, levels = np.linspace(0, np.amax(w_amplitude), 10), colors='k', linewidths=0.75)

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
axes.set_title(r'$|w^{\prime}|$')

plt.tight_layout()
plt.savefig(fname[3], dpi=300, bbox_inches='tight')
plt.close()

print(f'w   : {fname[3]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, rho_amplitude, levels = np.linspace(0, np.amax(rho_amplitude), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, rho_amplitude, levels = np.linspace(0, np.amax(rho_amplitude), 10), colors='k', linewidths=0.75)

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
axes.set_title(r'$|\rho^{\prime}|$')

plt.tight_layout()
plt.savefig(fname[4], dpi=300, bbox_inches='tight')
plt.close()

print(f'rho : {fname[4]}\n')

## PLOT PHASES ##############################################################################################################################

u_phase = np.angle(u, deg=True); v_phase = np.angle(v_p, deg=True); w_phase = np.angle(w_p, deg=True)
p_phase = np.angle(p, deg=True); rho_phase = np.angle(rho_p, deg=True)

if case == 'NEMO':
    fname = [f'/home/rees/lsa/figures/eigenvectors/phase/u/u_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/phase/v/v_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/phase/p/p_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/phase/w/w_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/phase/rho/rho_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']
else:
    fname = [f'/home/rees/lsa/figures/eigenvectors/phase/u/u_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/phase/v/v_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/phase/p/p_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/phase/w/w_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/eigenvectors/phase/rho/rho_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']
         
print(f'\nSaving figures to:\n')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, u_phase, levels = np.linspace(-180, 180, 13), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, u_phase, levels = np.linspace(-180, 180, 13), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)

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
axes.set_title(r'Phase($u^{\prime}$) [deg]')

plt.tight_layout()
plt.savefig(fname[0], dpi=300, bbox_inches='tight')
plt.close()

print(f'u   : {fname[0]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, v_phase, levels = np.linspace(-180, 180, 13), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, v_phase, levels = np.linspace(-180, 180, 13), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)

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
axes.set_title(r'Phase($v^{\prime}$) [deg]')

plt.tight_layout()
plt.savefig(fname[1], dpi=300, bbox_inches='tight')
plt.close()

print(f'v   : {fname[1]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, p_phase, levels = np.linspace(-180, 180, 13), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, p_phase, levels = np.linspace(-180, 180, 13), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)

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
axes.set_title(r'Phase($p^{\prime}$) [deg]')

plt.tight_layout()
plt.savefig(fname[2], dpi=300, bbox_inches='tight')
plt.close()

print(f'p   : {fname[2]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, w_phase, levels = np.linspace(-180, 180, 13), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, w_phase, levels = np.linspace(-180, 180, 13), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)

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
axes.set_title(r'Phase($w^{\prime}$)')

plt.tight_layout()
plt.savefig(fname[3], dpi=300, bbox_inches='tight')
plt.close()

print(f'w   : {fname[3]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, rho_phase, levels = np.linspace(-180, 180, 13), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, rho_phase, levels = np.linspace(-180, 180, 13), colors='k', linewidths=0.75)

axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)
axes.contour(Y, Z, EIL, levels = 0, colors='k', linestyles='--', linewidths=2)
critical_level = axes.contourf(Y, Z, critical, 1, hatches=['', '......'], colors='none')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)

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
axes.set_title(r'Phase($\rho^{\prime}$) [deg]')

plt.tight_layout()
plt.savefig(fname[4], dpi=300, bbox_inches='tight')
plt.close()

print(f'rho : {fname[4]}\n')
