"""
Plot the mean buoyancy frequency at the equator (to be used for integration thermal wind balance)
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

import calculate_NEMO_fields
import domain

ny, nz, case, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6])

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case, integration)

if case == 'NEMO':
    N2, N2_mid = calculate_NEMO_fields.mean_buoyancy(nz, integration, stability, assume) 
else:
    N2     = 8.883e-5*np.ones(Z.shape[0])
    N2_mid = 8.883e-5*np.ones(Z_mid.shape[0])
    
fname = f'/home/rees/lsa/figures/mean_fields/N2/N2_{case}_{integration}_{stability}_{assume}_{ny:02}_{nz:02}.png'
print(N2)
print(f'\nSaving figures to:\n')
    
fig, axes = plt.subplots(figsize=(6, 4))

axes.plot(N2, z)

axes.ticklabel_format(style="sci", axis='x', scilimits=((0,0)))
axes.tick_params(axis="both", which="major", labelsize=14)

axes.set_xlabel(r'$N^{2}$ [s$^{-2}$]')
axes.set_ylabel(f'Depth [m]')
axes.set_title(f'Buoyancy Frequency')

axes.set_xlim([0, 3.5e-4])
axes.set_ylim([-1000, 0])

plt.tight_layout()
plt.savefig(fname, dpi=300, bbox_inches='tight')
plt.close()

print(f'N2 : {fname}\n')
