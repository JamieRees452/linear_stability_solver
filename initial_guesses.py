import matplotlib.pyplot   as plt
import numpy               as np
from scipy.integrate import trapz
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mean_fields

ny, nz, case = int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3])

if case == 'NEMO':
    lat   = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/latitude_12.txt')
    depth = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/depth.txt'); depth = -depth[::-1]
    
    L = abs(lat[0])*111.12*1000
    D = 1000
    
else:
    L = (10*111.12)*1000 # Meridional half-width of the domain (m)
    D = 1000             # Depth of the domain (m)

y = np.linspace(-L, L, ny); z = np.linspace(-D, 0, nz) 

dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]

Y,Z         = np.meshgrid(y, z);         Y_full,Z_half = np.meshgrid(y, z_mid) 
Y_mid,Z_mid = np.meshgrid(y_mid, z_mid); Y_half,Z_full = np.meshgrid(y_mid, z)

U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry,ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case)
g, r0, beta = 9.81, 1026, 2.29e-11    

Q  = (1/r0)*(Uz_hf*ry_hf + (beta*Y_half-Uy_hf)*rz_hf)
Qy = np.gradient(Q, y_mid, axis=1)

Qy_sign_change = np.zeros((Z.shape[0], 2), dtype=float)
Y_points = np.zeros((Z.shape[0], 2), dtype=float)
Z_points = np.zeros((Z.shape[0], 2), dtype=float)
for i in range(Z.shape[0]):
    args = np.where(np.sign(Qy[i,:-1]) != np.sign(Qy[i,1:]))[0]+1
    if len(args)==0:
        Qy_sign_change[i,:] = np.nan, np.nan
        Y_points[i,:] = np.nan, np.nan
        Z_points[i,:] = np.nan, np.nan
    else:
        Qy_sign_change[i,:] = U[i, args[0]], U[i, args[1]]
        Y_points[i,:] = Y_half[i, args[0]], Y_half[i, args[1]]
        Z_points[i,:] = Z_full[i, args[0]], Z_full[i, args[1]]

np.savetxt(f'/home/rees/lsa/initial_guesses/init_guess_{case}_{ny:02}_{nz:02}.txt', [np.nanmin(Qy_sign_change), np.nanmax(Qy_sign_change)])

fig, axes=plt.subplots(figsize=(6,4))

CS = axes.contourf(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15))
axes.contour(Y_half, Z_full, Qy, levels = 0, colors='k', linewidths=2)
axes.plot(Y_points, Z_points, '.', ms=1, color='r')

divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad = 0.05)

cbar = plt.colorbar(CS, cax=cax)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()

if case == 'Proehl_1' or case == 'Proehl_2':
    axes.set_xlim([-3e5, 3e5])
    axes.set_ylim([-800, -200])

    axes.set_xticks([-3e5, -2e5, -1e5, 0, 1e5, 2e5, 3e5])
    axes.set_yticks([-800, -500, -200])
    
elif case == 'Proehl_3':
    axes.set_xlim([-8e5, 8e5])
    axes.set_ylim([-300, 0])

    axes.set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
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
axes.set_title(r'')

plt.tight_layout()
plt.savefig(f'/home/rees/lsa/figures/initial_guesses/init_guess_{case}_{ny:02}_{nz:02}.png', dpi=300, bbox_inches='tight')
plt.close()
