"""
Plot the mean fields
"""

import matplotlib.pyplot   as plt
import numpy               as np
from scipy.integrate import trapz
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

import domain
import mean_fields

ny, nz, case, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6])

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case, integration)

U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case, integration, stability, assume)
g, r0, beta = 9.81, 1026, 2.29e-11    

Q  = -(1/r0)*(ry*Uz + (beta*Y-Uy)*rz)
Qy = np.gradient(Q, y, axis=1)

if case == 'NEMO':
    fname = [f'/home/rees/lsa/figures/mean_fields/U/U_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/r/r_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/ryrz/ryrz_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/Qy/Qy_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/Uy/Uy_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/Uz/Uz_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/ry/ry_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/rz/rz_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png']
else:
    fname = [f'/home/rees/lsa/figures/mean_fields/U/U_{case}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/r/r_{case}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/ryrz/ryrz_{case}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/Qy/Qy_{case}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/Uy/Uy_{case}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/Uz/Uz_{case}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/ry/ry_{case}_{ny:02}_{nz:02}.png',
             f'/home/rees/lsa/figures/mean_fields/rz/rz_{case}_{ny:02}_{nz:02}.png']
    
print(f'\nSaving figures to:\n')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), cmap='RdBu_r')
axes.contour(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), colors='k', linewidths=0.75)

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
    
elif case == 'NEMO':
    axes.set_ylim([-250, 0])

    axes.set_xticks([-1.5e6, -1e6, -5e5, 0, 5e5, 1e6, 1.5e6])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])


axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$U$')

plt.tight_layout()
plt.savefig(fname[0], dpi=300, bbox_inches='tight')
plt.close()

print(f'U    : {fname[0]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, r, levels = 20, cmap='viridis_r')
axes.contour(Y, Z, r, levels = 20, colors='k', linewidths=0.75)

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
    
elif case == 'NEMO':
    axes.set_ylim([-250, 0])

    axes.set_xticks([-1.5e6, -1e6, -5e5, 0, 5e5, 1e6, 1.5e6])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])

axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{\rho}$')

plt.tight_layout()
plt.savefig(fname[1], dpi=300, bbox_inches='tight')
plt.close()

print(f'r    : {fname[1]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, ry/rz, levels = np.linspace(-np.amax(abs(ry/rz)), np.amax(abs(ry/rz)), 20), cmap='RdBu_r')
axes.contour(Y, Z, ry/rz, levels = np.linspace(-np.amax(abs(ry/rz)), np.amax(abs(ry/rz)), 20), colors='k', linewidths=0.75)

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
    
elif case == 'NEMO':
    axes.set_ylim([-250, 0])

    axes.set_xticks([-1.5e6, -1e6, -5e5, 0, 5e5, 1e6, 1.5e6])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])


axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{\rho}_{y}/\overline{\rho}_{z}$')

plt.tight_layout()
plt.savefig(fname[2], dpi=300, bbox_inches='tight')
plt.close()

print(f'ryrz : {fname[2]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, Qy, levels = np.linspace(-np.amax(abs(Qy)), np.amax(abs(Qy)), 21), cmap='RdBu_r')
axes.contour(Y, Z, Qy, levels = np.linspace(-np.amax(abs(Qy)), np.amax(abs(Qy)), 21), colors='k', linewidths=0.75)
axes.contour(Y, Z, Qy, levels = 0, colors='k', linewidths=2)

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
    
elif case == 'NEMO':
    axes.set_ylim([-250, 0])

    axes.set_xticks([-1.5e6, -1e6, -5e5, 0, 5e5, 1e6, 1.5e6])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])


axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$Q_{y}$')

plt.tight_layout()
plt.savefig(fname[3], dpi=300, bbox_inches='tight')
plt.close()

print(f'Qy   : {fname[3]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, Uy, levels = np.linspace(-np.amax(abs(Uy)), np.amax(abs(Uy)), 21), cmap='RdBu_r')
axes.contour(Y, Z, Uy, levels = np.linspace(-np.amax(abs(Uy)), np.amax(abs(Uy)), 21), colors='k', linewidths=0.75)

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
    
elif case == 'NEMO':
    axes.set_ylim([-250, 0])

    axes.set_xticks([-1.5e6, -1e6, -5e5, 0, 5e5, 1e6, 1.5e6])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])


axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$U_{y}$')

plt.tight_layout()
plt.savefig(fname[4], dpi=300, bbox_inches='tight')
plt.close()

print(f'Uy   : {fname[4]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, Uz, levels = np.linspace(-np.amax(abs(Uz)), np.amax(abs(Uz)), 21), cmap='RdBu_r')
axes.contour(Y, Z, Uz, levels = np.linspace(-np.amax(abs(Uz)), np.amax(abs(Uz)), 21), colors='k', linewidths=0.75)

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
    
elif case == 'NEMO':
    axes.set_ylim([-250, 0])

    axes.set_xticks([-1.5e6, -1e6, -5e5, 0, 5e5, 1e6, 1.5e6])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])


axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$U_{z}$')

plt.tight_layout()
plt.savefig(fname[5], dpi=300, bbox_inches='tight')
plt.close()

print(f'Uz   : {fname[5]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, ry, levels = np.linspace(-np.amax(abs(ry)), np.amax(abs(ry)), 21), cmap='RdBu_r')
axes.contour(Y, Z, ry, levels = np.linspace(-np.amax(abs(ry)), np.amax(abs(ry)), 21), colors='k', linewidths=0.75)

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
    
elif case == 'NEMO':
    axes.set_ylim([-250, 0])

    axes.set_xticks([-1.5e6, -1e6, -5e5, 0, 5e5, 1e6, 1.5e6])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])


axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{\rho}_{y}$')

plt.tight_layout()
plt.savefig(fname[6], dpi=300, bbox_inches='tight')
plt.close()

print(f'ry   : {fname[6]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, rz, levels = np.linspace(-np.amax(abs(rz)), np.amax(abs(rz)), 21), cmap='RdBu_r')
axes.contour(Y, Z, rz, levels = np.linspace(-np.amax(abs(rz)), np.amax(abs(rz)), 21), colors='k', linewidths=0.75)

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
    
elif case == 'NEMO':
    axes.set_ylim([-250, 0])

    axes.set_xticks([-1.5e6, -1e6, -5e5, 0, 5e5, 1e6, 1.5e6])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])
    
else:
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-250, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes.set_yticks([-250, -200, -150, -100, -50, 0])


axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$\overline{\rho}_{z}$')

plt.tight_layout()
plt.savefig(fname[7], dpi=300, bbox_inches='tight')
plt.close()

print(f'rz   : {fname[7]}\n')
