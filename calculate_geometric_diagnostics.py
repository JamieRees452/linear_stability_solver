"""
Calculate GEOMETRIC diagnostics
"""

import numpy as np
import sys

import mean_fields
import perturbations

# Inputs from the command terminal
ny, nz, k, case = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4])

# Dimensional values
g, r0, f0, N2 = 9.81, 1026, 1e-4, 1e-4

# Calculate the mean zonal velocity and density fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry,ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case, integration, stability)

# Calculate the perturbations on each grid
u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p = perturbations.on_each_grid(ny, nz, k, case, integration, stability)

# Zonal average of the product of two quantities
def za(data1, data2):
    return 0.25*np.conj(data1)*data2+data1*np.conj(data2)
    
K = 0.5*(za(u, u)+za(v_p, v_p))              # Eddy Kinetic Energy
P = 0.5*(-g/r0)*(1/rz_mid)*za(rho_p, rho_p)  # Eddy Potential Energy
E = K+P                                      # Total Eddy Energy

M = 0.5*(za(v_p, v_p) - za(u, u))            #
N = za(u, v_p)                               # Reynolds Stress
R = (f0/N2)*(g/r0)*za(rho_p, u)              #
S = (f0/N2)*(g/r0)*za(rho_p, v_p)            #

gamma_m = np.sqrt(M**2 + N**2)/K             # Horizontal eddy anisotropy
gamma_b = gamma_m                            # Vertical eddy anisotropy
#gamma_b = (np.sqrt(N2)/(2*f0))*np.sqrt((R**2 + S**2)/(K*P))

phi_m = 0.5*np.angle(-N/M)                   # Horizonal eddy tilt
phi_b = phi_m                                # Vertical eddy tilt
#phi_b = np.arccos(R/(np.sqrt(R**2+S**2)))

lam = np.arccos(np.sqrt(K/E))

phi_t = 0.5*np.angle(gamma_b*np.tan(2*lam))  #
gamma_t = np.cos(2*lam)/np.cos(2*phi_t)      #

# File names for NEMO profiles should contain the integration and stability
if case == 'NEMO':
    fname = [f'/home/rees/lsa/geometric/K/K_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/P/P_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/E/E_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/M/M_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/N/N_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/R/R_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/S/S_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/gamma_m/gamma_m_{integration}_{stability}_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/gamma_b/gamma_b_{integration}_{stability}_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/phi_m/phi_m_{integration}_{stability}_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/phi_b/phi_b_{integration}_{stability}_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt', 
                 f'/home/rees/lsa/geometric/lambda/lambda_{integration}_{stability}_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
                 f'/home/rees/lsa/geometric/phi_t/phi_t_{integration}_{stability}_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
                 f'/home/rees/lsa/geometric/gamma_t/gamma_t_{integration}_{stability}_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
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

# Save the diagnostics
np.savetxt(fname[0], K.real.flatten())
np.savetxt(fname[1], P.real.flatten())
np.savetxt(fname[2], E.real.flatten())
np.savetxt(fname[3], M.real.flatten())
np.savetxt(fname[4], N.real.flatten())
np.savetxt(fname[5], R.real.flatten())
np.savetxt(fname[6], S.real.flatten())
np.savetxt(fname[7], gamma_m.real.flatten())
np.savetxt(fname[8], gamma_b.real.flatten()) 
np.savetxt(fname[9], phi_m.real.flatten()) 
np.savetxt(fname[10], phi_b.real.flatten()) 
np.savetxt(fname[11], lam.real.flatten()) 
np.savetxt(fname[12], phi_t.real.flatten()) 
np.savetxt(fname[13], gamma_t.real.flatten()) 
