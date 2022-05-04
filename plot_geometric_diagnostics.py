"""
Plot GEOMETRIC diagnostics
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

import domain
import mean_fields

def ellipse(center, eccentricity, tilt, magnitude, Ny):
    
    tilt_deg = tilt*(180/np.pi)
    
    major_ax = magnitude
    minor_ax = (major_ax**2)*(1-eccentricity**2)
    
    if Ny<0: # with the shear
        patch = mpatches.Ellipse(center, major_ax, minor_ax, tilt_deg, fc='none', ls='solid', ec='k', lw=1.)
        
    elif Ny>0: # against the shear
        patch = mpatches.Ellipse(center, major_ax, minor_ax, tilt_deg, fc='none', ls='solid', ec='b', lw=1.)
        
    else:
        patch = mpatches.Ellipse(center, major_ax, minor_ax, tilt_deg, fc='none', ls='solid', ec='k', lw=1.)
    
    return patch
    
def domain_ellipse(ny, nz, L, D):

    y = np.linspace(-L, L, ny); z = np.linspace(-D, 0, nz) 

    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]

    Y,Z         = np.meshgrid(y, z);         Y_full,Z_half = np.meshgrid(y, z_mid) 
    Y_mid,Z_mid = np.meshgrid(y_mid, z_mid); Y_half,Z_full = np.meshgrid(y_mid, z)
    
    return y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D

# Inputs from the command terminal
ny, nz, k, case, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7])

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case, integration)

# Calculate the mean fields
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case, integration, stability, assume)

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = [f'/home/rees/lsa/geometric/K/K_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/P/P_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/E/E_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/M/M_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/N/N_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/R/R_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/S/S_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/gamma_m/gamma_m_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/gamma_b/gamma_b_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/phi_m/phi_m_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/phi_b/phi_b_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/lambda/lambda_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
                 f'/home/rees/lsa/geometric/phi_t/phi_t_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
                 f'/home/rees/lsa/geometric/gamma_t/gamma_t_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
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

if case == 'NEMO':
    fname = [f'/home/rees/lsa/figures/geometric/K/K_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/P/P_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/E/E_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/M/M_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/N/N_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/R/R_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/S/S_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/gamma_m/gamma_m_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/gamma_b/gamma_b_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/phi_m/phi_m_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/phi_b/phi_b_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/lambda/lambda_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png',
             f'/home/rees/lsa/figures/geometric/phi_t/phi_t_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png',
             f'/home/rees/lsa/figures/geometric/gamma_t/gamma_t_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']
else:
    fname = [f'/home/rees/lsa/figures/geometric/K/K_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/P/P_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/E/E_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/M/M_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/N/N_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/R/R_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/S/S_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/gamma_m/gamma_m_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/gamma_b/gamma_b_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/phi_m/phi_m_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/phi_b/phi_b_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png', 
             f'/home/rees/lsa/figures/geometric/lambda/lambda_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png',
             f'/home/rees/lsa/figures/geometric/phi_t/phi_t_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png',
             f'/home/rees/lsa/figures/geometric/gamma_t/gamma_t_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.png']
             
K_norm   = K/np.amax(abs(K)); P_norm = P/np.amax(abs(P)); E_norm = E/np.amax(abs(E))
M_norm = M/np.amax(abs(M)); N_norm = N/np.amax(abs(N)); R_norm = R/np.amax(abs(R)); S_norm = S/np.amax(abs(S)) 

eccentricity = gamma_m
tilt         = phi_m
magnitude    = np.ones_like(tilt) #EKE[i,j]/np.amax(abs(EKE))

y0, y0_mid, dy0, Y0, Y0_mid, Y0_half, Y0_full, z0, z0_mid, dz0, Z0, Z0_mid, Z0_half, Z0_full, L0, D0 = domain_ellipse(ny, nz, (10*111.12)*1000, 1000) # depends on grid spacing
y1, y1_mid, dy1, Y1, Y1_mid, Y1_half, Y1_full, z1, z1_mid, dz1, Z1, Z1_mid, Z1_half, Z1_full, L1, D1 = domain_ellipse(ny, nz, 25, 50) # depends on grid spacing

Ny = np.gradient(N, Y_mid[0,:], axis=1)

y_step       = 2
Y2_mid       = Y1_mid[:,::y_step]; 
Z2_mid       = Z1_mid[:,::y_step]
eccentricity = gamma_m[:,::y_step]
tilt         = phi_m[:,::y_step]
magnitude    = np.ones_like(tilt)[:,::y_step] #EKE[i,j]/np.amax(abs(EKE))
Ny2          = Ny[:,::y_step]
K2           = K[:,::y_step]; K2_norm = K2/np.amax(K2)

print(f'\nSaving figures to:\n')

fig, axes=plt.subplots(figsize=(6,6))

#CS = axes.contourf(Y_mid, Z_mid, K, levels = np.linspace(0, np.amax(abs(K)), 10), cmap='Reds')
#axes.contour(Y_mid, Z_mid, K, levels = np.linspace(0, np.amax(abs(K)), 10), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y1_mid, Z1_mid, K_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y1_mid, Z1_mid, K_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

for i in range(Y2_mid.shape[0]):
    for j in range(Y2_mid.shape[1]):
        #if K2_norm[i,j]>1e-2:
        ell = ellipse((Y2_mid[i,j], Z2_mid[i,j]), eccentricity[i,j], tilt[i,j], 1, Ny2[i,j])
        axes.add_patch(ell)

if case == 'Proehl_1':
    axes.set_xlim([-10,10])
    axes.set_ylim([-35,-15])
    
    axes.set_xticks([-9, -4.5, 0, 4.5, 9])
    axes.set_yticks([-35, -30, -25, -20, -15])
    
    axes.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    axes.set_yticklabels([r'$-700$', r'$-600$', r'$-500$', r'$-400$', r'$-300$'])
    
elif case == 'Proehl_2':
    axes.set_xlim([-5,5])
    axes.set_ylim([-30,-20])
    
    axes.set_xticks([-4.5, -2.25, 0, 2.25, 4.5])
    axes.set_yticks([-30, -25, -20])
    
    axes.set_xticklabels([r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$'])
    axes.set_yticklabels([r'$-600$', r'$-500$', r'$-400$'])
    
elif case == 'Proehl_3':
    axes.set_xlim([-20,20])
    axes.set_ylim([-40,0])
    
    axes.set_xticks([-18, -9, 0, 9, 18])
    axes.set_yticks([-40, -30, -20, -10, 0])
    
    axes.set_xticklabels([r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$'])
    axes.set_yticklabels([r'$-800$', r'$-600$', r'$-400$', r'$-200$', r'$0$'])
    
elif case == 'Proehl_4':
    axes.set_xlim([-15,15])
    axes.set_ylim([-30,0])
    
    axes.set_xticks([-13.5, -9, -4.5, 0, 4.5, 9, 13.5])
    axes.set_yticks([-40, -30, -20, -10, 0])
    
    axes.set_xticklabels([r'$-3$', r'$-2$', r'$1$', r'$0$', r'$1$', r'$2$', r'$3$'])
    axes.set_yticklabels([r'$-800$', r'$-600$', r'$-400$', r'$-200$', r'$0$'])
    
elif case == 'Proehl_5':
    axes.set_xlim([-5,5])
    axes.set_ylim([-10,0])
    
    axes.set_xticks([-4.5, -2.25, 0, 2.25, 4.5])
    axes.set_yticks([-10, -5, 0])
    
    axes.set_xticklabels([r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$'])
    axes.set_yticklabels([r'$-200$', r'$-100$', r'$0$'])
    
else:
    axes.set_xlim([-10,10])
    axes.set_ylim([-20,0])
    
    axes.set_xticks([-9, -4.5, 0, 4.5, 9])
    axes.set_yticks([-20, -15, -10, -5, 0])
    
    axes.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    axes.set_yticklabels([r'$-400$', r'$-300$', r'$-200$', r'$-100$', r'$0$'])

K_max = "{:.2e}".format(np.amax(K))
axes.text(0.15, 0.05, f'Max: {K_max}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(r'Latitude [$10^{5}$m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$K$')

plt.tight_layout()
plt.savefig(fname[0], dpi=300, bbox_inches='tight')
plt.close()

print(f'K       : {fname[0]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, P, levels = np.linspace(0, np.amax(abs(P)), 10), cmap='Reds')
#axes.contour(Y_mid, Z_mid, P, levels = np.linspace(0, np.amax(abs(P)), 10), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, P_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y_mid, Z_mid, P_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

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

P_max = "{:.2e}".format(np.amax(P))
axes.text(0.15, 0.05, f'Max: {P_max}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$P$')

plt.tight_layout()
plt.savefig(fname[1], dpi=300, bbox_inches='tight')
plt.close()

print(f'P       : {fname[1]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, E, levels = np.linspace(0, np.amax(abs(E)), 10), cmap='Reds')
#axes.contour(Y_mid, Z_mid, E, levels = np.linspace(0, np.amax(abs(E)), 10), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, E_norm, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y_mid, Z_mid, E_norm, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

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

E_max = "{:.2e}".format(np.amax(E))
axes.text(0.15, 0.05, f'Max: {E_max}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$E$')

plt.tight_layout()
plt.savefig(fname[2], dpi=300, bbox_inches='tight')
plt.close()

print(f'E       : {fname[2]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, M, levels = np.linspace(-np.amax(abs(M)), np.amax(abs(M)), 20), cmap='RdBu_r')
#axes.contour(Y_mid, Z_mid, M, levels = np.linspace(-np.amax(abs(M)), np.amax(abs(M)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, M_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, M_norm, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)
axes.contour(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), colors='k', alpha=0.3, linewidths=0.75)

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

M_min = "{:.2e}".format(np.amin(M)); M_max = "{:.2e}".format(np.amax(M))
axes.text(0.15, 0.1, f'Max: {M_max}\nMin:{M_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$M$')

plt.tight_layout()
plt.savefig(fname[3], dpi=300, bbox_inches='tight')
plt.close()

print(f'M       : {fname[3]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, N, levels = np.linspace(-np.amax(abs(N)), np.amax(abs(N)), 20), cmap='RdBu_r')
#axes.contour(Y_mid, Z_mid, N, levels = np.linspace(-np.amax(abs(N)), np.amax(abs(N)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, N_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, N_norm, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)
axes.contour(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), colors='k', alpha=0.3, linewidths=0.75)

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

N_min = "{:.2e}".format(np.amin(N)); N_max = "{:.2e}".format(np.amax(N))
axes.text(0.15, 0.1, f'Max: {N_max}\nMin:{N_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$N$')

plt.tight_layout()
plt.savefig(fname[4], dpi=300, bbox_inches='tight')
plt.close()

print(f'N       : {fname[4]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, R, levels = np.linspace(-np.amax(abs(R)), np.amax(abs(R)), 20), cmap='RdBu_r')
#axes.contour(Y_mid, Z_mid, R, levels = np.linspace(-np.amax(abs(R)), np.amax(abs(R)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, R_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, R_norm, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)

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

R_min = "{:.2e}".format(np.amin(R)); R_max = "{:.2e}".format(np.amax(R))
axes.text(0.15, 0.1, f'Max: {R_max}\nMin:{R_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$R$')

plt.tight_layout()
plt.savefig(fname[5], dpi=300, bbox_inches='tight')
plt.close()

print(f'R       : {fname[5]}')

fig, axes=plt.subplots(figsize=(6,4))

#CS = axes.contourf(Y_mid, Z_mid, S, levels = np.linspace(-np.amax(abs(S)), np.amax(abs(S)), 20), cmap='RdBu_r')
#axes.contour(Y_mid, Z_mid, S, levels = np.linspace(-np.amax(abs(S)), np.amax(abs(S)), 20), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#cbar = plt.colorbar(CS, cax=cax)
#cbar.formatter.set_powerlimits((0,0))
#cbar.update_ticks()

CS = axes.contourf(Y_mid, Z_mid, S_norm, levels = np.linspace(-1, 1, 21), cmap='RdBu_r')
axes.contour(Y_mid, Z_mid, S_norm, levels = np.linspace(-1, 1, 21), colors='k', linewidths=0.75)

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

S_min = "{:.2e}".format(np.amin(S)); S_max = "{:.2e}".format(np.amax(S))
axes.text(0.15, 0.1, f'Max: {S_max}\nMin:{S_min}', transform=axes.transAxes, ha='center', va='center', family='monospace', fontsize=12, bbox=dict(facecolor='white'))

axes.set_xlabel(f'Latitude [m]', fontsize=14)
axes.set_ylabel(f'Depth [m]', fontsize=14)
axes.set_title(r'$S$')

plt.tight_layout()
plt.savefig(fname[6], dpi=300, bbox_inches='tight')
plt.close()

print(f'S       : {fname[6]}')

fig, axes=plt.subplots(figsize=(6,6))

CS = axes.contourf(Y1_mid, Z1_mid, gamma_m, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y1_mid, Z1_mid, gamma_m, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#plt.colorbar(CS, cax=cax)

for i in range(Y2_mid.shape[0]):
    for j in range(Y2_mid.shape[1]):
        #if K2_norm[i,j]>1e-2:
        ell = ellipse((Y2_mid[i,j], Z2_mid[i,j]), eccentricity[i,j], tilt[i,j], 1, Ny2[i,j])
        axes.add_patch(ell)

if case == 'Proehl_1':
    axes.set_xlim([-10,10])
    axes.set_ylim([-35,-15])
    
    axes.set_xticks([-9, -4.5, 0, 4.5, 9])
    axes.set_yticks([-35, -30, -25, -20, -15])
    
    axes.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    axes.set_yticklabels([r'$-700$', r'$-600$', r'$-500$', r'$-400$', r'$-300$'])
    
elif case == 'Proehl_2':
    axes.set_xlim([-5,5])
    axes.set_ylim([-30,-20])
    
    axes.set_xticks([-4.5, -2.25, 0, 2.25, 4.5])
    axes.set_yticks([-30, -25, -20])
    
    axes.set_xticklabels([r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$'])
    axes.set_yticklabels([r'$-600$', r'$-500$', r'$-400$'])
    
elif case == 'Proehl_3':
    axes.set_xlim([-20,20])
    axes.set_ylim([-40,0])
    
    axes.set_xticks([-18, -9, 0, 9, 18])
    axes.set_yticks([-40, -30, -20, -10, 0])
    
    axes.set_xticklabels([r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$'])
    axes.set_yticklabels([r'$-800$', r'$-600$', r'$-400$', r'$-200$', r'$0$'])
    
elif case == 'Proehl_4':
    axes.set_xlim([-15,15])
    axes.set_ylim([-30,0])
    
    axes.set_xticks([-13.5, -9, -4.5, 0, 4.5, 9, 13.5])
    axes.set_yticks([-40, -30, -20, -10, 0])
    
    axes.set_xticklabels([r'$-3$', r'$-2$', r'$1$', r'$0$', r'$1$', r'$2$', r'$3$'])
    axes.set_yticklabels([r'$-800$', r'$-600$', r'$-400$', r'$-200$', r'$0$'])
    
elif case == 'Proehl_5':
    axes.set_xlim([-5,5])
    axes.set_ylim([-10,0])
    
    axes.set_xticks([-4.5, -2.25, 0, 2.25, 4.5])
    axes.set_yticks([-10, -5, 0])
    
    axes.set_xticklabels([r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$'])
    axes.set_yticklabels([r'$-200$', r'$-100$', r'$0$'])
    
else:
    axes.set_xlim([-10,10])
    axes.set_ylim([-20,0])
    
    axes.set_xticks([-9, -4.5, 0, 4.5, 9])
    axes.set_yticks([-20, -15, -10, -5, 0])
    
    axes.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    axes.set_yticklabels([r'$-400$', r'$-300$', r'$-200$', r'$-100$', r'$0$'])
    
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(r'Latitude [$10^{5}$m]', fontsize=14)
axes.set_ylabel(r'Depth [m]', fontsize=14)
axes.set_title(r'$\gamma_{m}$')

plt.tight_layout()
plt.savefig(fname[7], dpi=300, bbox_inches='tight')
plt.close()

print(f'gamma_m : {fname[7]}')

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y_mid, Z_mid, gamma_b, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y_mid, Z_mid, gamma_b, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#plt.colorbar(CS, cax=cax)

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

fig, axes=plt.subplots(figsize=(6,6))

CS = axes.contourf(Y1_mid, Z1_mid, phi_m, levels = np.linspace(-np.pi/2, np.pi/2, 20), cmap='RdBu_r')
axes.contour(Y1_mid, Z1_mid, phi_m, levels = np.linspace(-np.pi/2, np.pi/2, 20), colors='k', linewidths=0.75)

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.set_ticks([-np.pi/2, 0, np.pi/2])
cbar.set_ticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$'])
cbar.update_ticks()

for i in range(Y2_mid.shape[0]):
    for j in range(Y2_mid.shape[1]):
        #if K2_norm[i,j]>1e-2:
        ell = ellipse((Y2_mid[i,j], Z2_mid[i,j]), eccentricity[i,j], tilt[i,j], 1, Ny2[i,j])
        axes.add_patch(ell)

if case == 'Proehl_1':
    axes.set_xlim([-10,10])
    axes.set_ylim([-35,-15])
    
    axes.set_xticks([-9, -4.5, 0, 4.5, 9])
    axes.set_yticks([-35, -30, -25, -20, -15])
    
    axes.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    axes.set_yticklabels([r'$-700$', r'$-600$', r'$-500$', r'$-400$', r'$-300$'])
    
elif case == 'Proehl_2':
    axes.set_xlim([-5,5])
    axes.set_ylim([-30,-20])
    
    axes.set_xticks([-4.5, -2.25, 0, 2.25, 4.5])
    axes.set_yticks([-30, -25, -20])
    
    axes.set_xticklabels([r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$'])
    axes.set_yticklabels([r'$-600$', r'$-500$', r'$-400$'])
    
elif case == 'Proehl_3':
    axes.set_xlim([-20,20])
    axes.set_ylim([-40,0])
    
    axes.set_xticks([-18, -9, 0, 9, 18])
    axes.set_yticks([-40, -30, -20, -10, 0])
    
    axes.set_xticklabels([r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$'])
    axes.set_yticklabels([r'$-800$', r'$-600$', r'$-400$', r'$-200$', r'$0$'])
    
elif case == 'Proehl_4':
    axes.set_xlim([-15,15])
    axes.set_ylim([-30,0])
    
    axes.set_xticks([-13.5, -9, -4.5, 0, 4.5, 9, 13.5])
    axes.set_yticks([-40, -30, -20, -10, 0])
    
    axes.set_xticklabels([r'$-3$', r'$-2$', r'$1$', r'$0$', r'$1$', r'$2$', r'$3$'])
    axes.set_yticklabels([r'$-800$', r'$-600$', r'$-400$', r'$-200$', r'$0$'])
    
elif case == 'Proehl_5':
    axes.set_xlim([-5,5])
    axes.set_ylim([-10,0])
    
    axes.set_xticks([-4.5, -2.25, 0, 2.25, 4.5])
    axes.set_yticks([-10, -5, 0])
    
    axes.set_xticklabels([r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$'])
    axes.set_yticklabels([r'$-200$', r'$-100$', r'$0$'])
    
else:
    axes.set_xlim([-10,10])
    axes.set_ylim([-20,0])
    
    axes.set_xticks([-9, -4.5, 0, 4.5, 9])
    axes.set_yticks([-20, -15, -10, -5, 0])
    
    axes.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    axes.set_yticklabels([r'$-400$', r'$-300$', r'$-200$', r'$-100$', r'$0$'])
    
axes.tick_params(axis='both', which='major', labelsize=14)

axes.set_xlabel(r'Latitude [$10^{5}$m]', fontsize=14)
axes.set_ylabel(r'Depth [m]', fontsize=14)
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

CS = axes.contourf(Y_mid, Z_mid, gamma_t, levels = np.linspace(0, 1, 11), cmap='Reds')
axes.contour(Y_mid, Z_mid, gamma_t, levels = np.linspace(0, 1, 11), colors='k', linewidths=0.75)

#divider = make_axes_locatable(axes)
#cax = divider.append_axes("right", size="5%", pad = 0.05)
#plt.colorbar(CS, cax=cax)

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
