"""
Calculate each term in the eddy energy budget
"""

import numpy as np
import sys

import domain
import mean_fields
import perturbations

# Inputs from the command terminal
ny, nz, k, case, integration, stability, assume = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7])

# Dimensional values
g, r0 = 9.81, 1026

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case, integration)

# Calculate the mean zonal velocity and density fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry,ry_mid, ry_hf, rz, rz_mid, rz_hf  = mean_fields.on_each_grid(ny, nz, case, integration, stability, assume)

# Calculate the perturbations on each grid
u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p = perturbations.on_each_grid(ny, nz, k, case, integration, stability, assume)

# Zonal average of the product of two quantities
def za(data1, data2):
    return 0.25*(np.conj(data1)*data2+data1*np.conj(data2))
    
alpha = (-g/r0)*(ry_mid/rz_mid)

BTC = -za(u, v_p)*Uy_mid                       # Barotropic Conversion
BCC = -alpha*za(v_p, rho_p)                    # Baroclinic Conversion
KHC = -za(u, w_p)*Uz_mid                       # Kelvin-Helmholtz Conversion
MEPFD = za(v, p_v)                             # Meridional Eddy Pressure Flux
VEPFD = za(w, p_w)                             # Vertical Eddy Pressure Flux
EKE = 0.5*(za(u, u)+za(v_p, v_p))              # Eddy Kinetic Energy
EPE = 0.5*(-g/r0)*(1/rz_mid)*za(rho_p, rho_p)  # Eddy Potential Energy

ddyMEPFD = np.zeros((nz-1, ny-1), dtype=complex)
for j in range(ny-1):
    for l in range(nz-1):
        ddyMEPFD[l, j] = (1/dy)*(MEPFD[l, j+1] - MEPFD[l, j])
        
ddzVEPFD = np.zeros((nz-1, ny-1), dtype=complex)
for j in range(ny-1):
    for l in range(nz-1):
        ddzVEPFD[l, j] = (1/dz)*(VEPFD[l+1, j] - VEPFD[l, j])
        
LHS = EKE + EPE                                # Total Eddy Energy
RHS = BTC + KHC + BCC + ddyMEPFD + ddzVEPFD

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

# Save each term in the eddy energy budget
np.savetxt(fname[0], BTC.real.flatten())
np.savetxt(fname[1], KHC.real.flatten())
np.savetxt(fname[2], BCC.real.flatten())
np.savetxt(fname[3], MEPFD.real.flatten())
np.savetxt(fname[4], VEPFD.real.flatten())
np.savetxt(fname[5], EKE.real.flatten())
np.savetxt(fname[6], EPE.real.flatten())
np.savetxt(fname[7], LHS.real.flatten())
np.savetxt(fname[8], RHS.real.flatten()) 
