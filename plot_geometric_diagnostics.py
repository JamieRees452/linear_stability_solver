"""
Plot GEOMETRIC diagnostics
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Inputs from the command terminal
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

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = [f'/home/rees/lsa/geometric/K/K_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/P/P_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/E/E_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/M/M_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/N/N_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/R/R_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/S/S_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/gamma_m/gamma_m_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/gamma_b/gamma_b_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/phi_m/phi_m_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/phi_b/phi_b_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/lambda/lambda_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/geometric/phi_t/phi_t_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/geometric/gamma_t/gamma_t_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
else:
    fname = [f'/home/rees/lsa/geometric/K/K_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/P/P_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/E/E_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/M/M_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/N/N_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/R/R_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/S/S_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/gamma_m/gamma_m_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/gamma_b/gamma_b_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/phi_m/phi_m_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/phi_b/phi_b_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
             f'/home/rees/lsa/geometric/lambda/lambda_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/geometric/phi_t/phi_t_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/geometric/gamma_t/gamma_t_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']

# Load GEOMETRIC diagnostics
K       = np.loadtxt(fname[0]).reshape(int(nz-1), int(ny-1))
P       = np.loadtxt(fname[1]).reshape(int(nz-1), int(ny-1))
E       = np.loadtxt(fname[2]).reshape(int(nz-1), int(ny-1))
M       = np.loadtxt(fname[3]).reshape(int(nz-1), int(ny-1))
N       = np.loadtxt(fname[4]).reshape(int(nz-1), int(ny-1))
R       = np.loadtxt(fname[5]).reshape(int(nz-1), int(ny-1))
S       = np.loadtxt(fname[6]).reshape(int(nz-1), int(ny-1))
gamma_m = np.loadtxt(fname[7]).reshape(int(nz-1), int(ny-1))
gamma_b = np.loadtxt(fname[8]).reshape(int(nz-1), int(ny-1))
phi_m   = np.loadtxt(fname[9]).reshape(int(nz-1), int(ny-1))
phi_b   = np.loadtxt(fname[10]).reshape(int(nz-1), int(ny-1))
lam     = np.loadtxt(fname[11]).reshape(int(nz-1), int(ny-1))
phi_t   = np.loadtxt(fname[12]).reshape(int(nz-1), int(ny-1))
gamma_t = np.loadtxt(fname[13]).reshape(int(nz-1), int(ny-1))

## PLOT GEOMETRIC DIAGNOSTICS #######################################################################################

print(f'\nSaving figures to:\n')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, K, levels = np.linspace(0, np.amax(abs(K)), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, K, levels = np.linspace(0, np.amax(abs(K)), 10), colors='k', linewidths=0.75)

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
axes.set_title(r'$K$')

plt.tight_layout()
plt.savefig(fname[0], dpi=300, bbox_inches='tight')
plt.close()

print(f'K       : {fname[0]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, P, levels = np.linspace(0, np.amax(abs(P)), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, P, levels = np.linspace(0, np.amax(abs(P)), 10), colors='k', linewidths=0.75)

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
axes.set_title(r'$P$')

plt.tight_layout()
plt.savefig(fname[1], dpi=300, bbox_inches='tight')
plt.close()

print(f'P       : {fname[1]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, E, levels = np.linspace(0, np.amax(abs(E)), 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, E, levels = np.linspace(0, np.amax(abs(E)), 10), colors='k', linewidths=0.75)

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
axes.set_title(r'$E$')

plt.tight_layout()
plt.savefig(fname[2], dpi=300, bbox_inches='tight')
plt.close()

print(f'E       : {fname[2]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, M, levels = np.linspace(-np.amax(abs(M)), np.amax(abs(M)), 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, M, levels = np.linspace(-np.amax(abs(M)), np.amax(abs(M)), 20), colors='k', linewidths=0.75)

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
axes.set_title(r'$M$')

plt.tight_layout()
plt.savefig(fname[3], dpi=300, bbox_inches='tight')
plt.close()

print(f'M       : {fname[3]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, N, levels = np.linspace(-np.amax(abs(N)), np.amax(abs(N)), 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, N, levels = np.linspace(-np.amax(abs(N)), np.amax(abs(N)), 20), colors='k', linewidths=0.75)

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
axes.set_title(r'$N$')

plt.tight_layout()
plt.savefig(fname[4], dpi=300, bbox_inches='tight')
plt.close()

print(f'N       : {fname[4]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, R, levels = np.linspace(-np.amax(abs(R)), np.amax(abs(R)), 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, R, levels = np.linspace(-np.amax(abs(R)), np.amax(abs(R)), 20), colors='k', linewidths=0.75)

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
axes.set_title(r'$R$')

plt.tight_layout()
plt.savefig(fname[5], dpi=300, bbox_inches='tight')
plt.close()

print(f'R       : {fname[5]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, S, levels = np.linspace(-np.amax(abs(S)), np.amax(abs(S)), 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, S, levels = np.linspace(-np.amax(abs(S)), np.amax(abs(S)), 20), colors='k', linewidths=0.75)

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
axes.set_title(r'$S$')

plt.tight_layout()
plt.savefig(fname[6], dpi=300, bbox_inches='tight')
plt.close()

print(f'S       : {fname[6]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, gamma_m, levels = np.linspace(0, 1, 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, gamma_m, levels = np.linspace(0, 1, 10), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

plt.colorbar(CS, cax=cax)

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
axes.set_title(r'$\gamma_{m}$')

plt.tight_layout()
plt.savefig(fname[7], dpi=300, bbox_inches='tight')
plt.close()

print(f'gamma_m : {fname[7]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, gamma_b, levels = np.linspace(0, 1, 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, gamma_b, levels = np.linspace(0, 1, 10), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

plt.colorbar(CS, cax=cax)

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
axes.set_title(r'$\gamma_{b}$')

plt.tight_layout()
plt.savefig(fname[8], dpi=300, bbox_inches='tight')
plt.close()

print(f'gamma_b : {fname[8]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, phi_m, levels = np.linspace(-np.pi/2, np.pi/2, 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, phi_m, levels = np.linspace(-np.pi/2, np.pi/2, 20), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.set_ticks([-np.pi/2, 0, np.pi/2])
cbar.set_ticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$'])
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
axes.set_title(r'$\phi_{m}$')

plt.tight_layout()
plt.savefig(fname[9], dpi=300, bbox_inches='tight')
plt.close()

print(f'phi_m   : {fname[9]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, phi_b, levels = np.linspace(-np.pi/2, np.pi/2, 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, phi_b, levels = np.linspace(-np.pi/2, np.pi/2, 20), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.set_ticks([-np.pi/2, 0, np.pi/2])
cbar.set_ticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$'])
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
axes.set_title(r'$\phi_{b}$')

plt.tight_layout()
plt.savefig(fname[10], dpi=300, bbox_inches='tight')
plt.close()

print(f'phi_b   : {fname[10]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, lam, levels = np.linspace(-np.pi/2, np.pi/2, 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, lam, levels = np.linspace(-np.pi/2, np.pi/2, 20), colors='k', linewidths=0.75)

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
axes.set_title(r'$\lambda$')

plt.tight_layout()
plt.savefig(fname[11], dpi=300, bbox_inches='tight')
plt.close()

print(f'lambda  : {fname[11]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, phi_t, levels = np.linspace(-np.pi/2, np.pi/2, 20), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, phi_t, levels = np.linspace(-np.pi/2, np.pi/2, 20), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.set_ticks([-np.pi/2, 0, np.pi/2])
cbar.set_ticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$'])
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
axes.set_title(r'$\phi_{t}$')

plt.tight_layout()
plt.savefig(fname[12], dpi=300, bbox_inches='tight')
plt.close()

print(f'phi_t   : {fname[12]}')
fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, gamma_t, levels = np.linspace(0, 1, 10), cmap='Reds')
axes.contour(Y_mid, Z_mid, gamma_t, levels = np.linspace(0, 1, 10), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

plt.colorbar(CS, cax=cax)

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
axes.set_title(r'$\gamma_{t}$')

plt.tight_layout()
plt.savefig(fname[13], dpi=300, bbox_inches='tight')
plt.close()

print(f'gamma_t : {fname[13]}\n')
