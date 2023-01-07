"""
Calculate/obtain the mean zonal velocity and density fields, from NEMO data or from Proehls test cases, on each grid
e.g. Zonal velocity (U) at (h)alf points in y, (f)ull points in z is given by U_hf
"""

import numpy as np
import os
from   scipy import integrate
from   scipy.integrate import trapz
import sys

import calculate_NEMO_fields
import calculate_Proehl_fields
import domain

def on_each_grid(ny, nz, case, month0, month1):
    """
    Calculate the mean fields on each grid
                            
    Parameters
    ----------
    ny : int
         Meridional grid resolution
    
    nz : int
         Vertical grid resolution
     
    k : float
         Zonal wavenumber
    
    init_guess : float
         Initial eigenvalue guess used to search a region with the Arnoldi method
         This guess corresponds to the real part of the phase speed 
         
    case : str
         Mean fields about which to perform the linear stability analysis
         e.g. Proehl_[1-8] - Proehls test cases (Proehl (1996) and Proehl (1998))
              NEMO_25      - Data from the 1/4deg coupled AOGCM
              NEMO_12      - Data from the 1/12deg coupled AOGCM
    
    month0 : str
         Month of data from the NEMO coupled model
         e.g. Jan = 01, Feb = 02, ..., Dec = 12
    
    month1 : str
         Month of data from the NEMO coupled model
         e.g. Jan = 01, Feb = 02, ..., Dec = 12
         
         month0 and month1 will be averaged to obtain the mean field we investigate
        
    Returns
    -------
    U : (nz, ny) ndarray
         Mean zonal velocity
    
    r : (nz, ny) ndarray
         Mean density field
        
    U_mid, U_hf : ndarray
         Mean zonal velocity calculated at different points on the staggered grid
         
    r_mid, r_hf, r_fh : ndarray
         Mean density calculated at different points on the staggered grid
         
    Uy, Uy_mid, Uy_hf, ry, ry_mid, ry_hf : ndarray
         Meridional gradients of mean zonal velocity and density at different points on the staggered grid
         
    Uz, Uz_mid, Uz_hf, rz, rz_mid, rz_hf : ndarray
         Vertical gradients of mean zonal velocity and density at different points on the staggered grid
    """

    # Calculate the grid for a given case and integration
    y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case)
    
    beta   = 2.29e-11          # Meridional gradient of the Coriolis parameter (m^{-1}s^{-1})
    r0     = 1026              # Background density (kg m^{3})
    g      = 9.81              # Gravitational acceleration (ms^{-2})
    
    # Obtain the mean zonal velocity fields
    
    if (case == 'Proehl_1' or case == 'Proehl_2' or case == 'Proehl_3' or case == 'Proehl_4' or
        case == 'Proehl_5' or case == 'Proehl_6' or case == 'Proehl_7' or case == 'Proehl_8'):
        
        U    , Uy    , Uz     = calculate_Proehl_fields.mean_velocity(Y     , Z     , case)
        U_mid, Uy_mid, Uz_mid = calculate_Proehl_fields.mean_velocity(Y_mid , Z_mid , case)
        U_hf , Uy_hf , Uz_hf  = calculate_Proehl_fields.mean_velocity(Y_half, Z_full, case)
        U_fh , Uy_fh , Uz_fh  = calculate_Proehl_fields.mean_velocity(Y_full, Z_half, case)

        N2     = 8.883e-5*np.ones(Z.shape[0])
        N2_mid = 8.883e-5*np.ones(Z_mid.shape[0])

        r = (r0/g)*(beta*integrate.cumtrapz(Y*Uz, y, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y), 1)).T) + r0
        ry = (beta*r0/g)*Y*Uz; rz = np.gradient(r, z, axis=0); 

        r_hf = (r0/g)*(beta*integrate.cumtrapz(Y_half*Uz_hf, y_mid, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y)-1, 1)).T) + r0
        ry_hf = (beta*r0/g)*Y_half*Uz_hf; rz_hf = np.gradient(r_hf, z, axis=0)

        r_mid = (r0/g)*(beta*integrate.cumtrapz(Y_mid*Uz_mid, y_mid, initial=0) - np.tile(integrate.cumtrapz(N2_mid, z_mid, initial=0), (len(y)-1, 1)).T) + r0
        ry_mid = (beta*r0/g)*Y_mid*Uz_mid; rz_mid = np.gradient(r_mid, z_mid, axis=0)
        
    elif case == 'NEMO_25' or case == 'NEMO_12':
    
        U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf = calculate_NEMO_fields.load_mean_velocity(ny, nz, case, month0, month1)
        r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf       = calculate_NEMO_fields.load_mean_density(ny, nz, case, month0, month1)
        
    else:
        print(f'{case} in not a valid case')
        
    return U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry,ry_mid, ry_hf, rz, rz_mid, rz_hf 
    
    #if assume == 'TWB':
    
    #    if (case == 'Proehl_1' or case == 'Proehl_2' or case == 'Proehl_3' or case == 'Proehl_4' or
    #   case == 'Proehl_5' or case == 'Proehl_6' or case == 'Proehl_7' or case == 'Proehl_8'):
       
    #        U    , Uy    , Uz     = calculate_Proehl_fields.mean_velocity(Y     , Z     , case)
    #        U_mid, Uy_mid, Uz_mid = calculate_Proehl_fields.mean_velocity(Y_mid , Z_mid , case)
    #        U_hf , Uy_hf , Uz_hf  = calculate_Proehl_fields.mean_velocity(Y_half, Z_full, case)
    #        U_fh , Uy_fh , Uz_fh  = calculate_Proehl_fields.mean_velocity(Y_full, Z_half, case)

    #        N2     = 8.883e-5*np.ones(Z.shape[0])
    #        N2_mid = 8.883e-5*np.ones(Z_mid.shape[0])

    #        r = (r0/g)*(beta*integrate.cumtrapz(Y*Uz, y, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y), 1)).T) + r0
    #        ry = (beta*r0/g)*Y*Uz; rz = np.gradient(r, z, axis=0); 

    #        r_hf = (r0/g)*(beta*integrate.cumtrapz(Y_half*Uz_hf, y_mid, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y)-1, 1)).T) + r0
    #        ry_hf = (beta*r0/g)*Y_half*Uz_hf; rz_hf = np.gradient(r_hf, z, axis=0)

    #        r_mid = (r0/g)*(beta*integrate.cumtrapz(Y_mid*Uz_mid, y_mid, initial=0) - np.tile(integrate.cumtrapz(N2_mid, z_mid, initial=0), (len(y)-1, 1)).T) + r0
    #        ry_mid = (beta*r0/g)*Y_mid*Uz_mid; rz_mid = np.gradient(r_mid, z_mid, axis=0)
        
    #    else:
    #        U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf = calculate_NEMO_fields.load_mean_velocity(ny, nz, case, integration, stability, assume)
            
    #        N2, N2_mid = calculate_NEMO_fields.mean_buoyancy(nz, case, integration, stability, assume) 
                
    #        eq = ny//2

    #        y_north  = y[eq:]; y_south      = y[:eq][::-1] 
    #        Y_north  = Y[:, eq:]; Y_south   = Y[:, :eq][:, ::-1]
    #        Z_north  = Z[:, eq:]; Z_south   = Z[:, :eq][:, ::-1]
    #        Uz_north = Uz[:, eq:]; Uz_south = Uz[:, :eq][:, ::-1]

    #        y_mid_north = y_mid[eq:]; y_mid_south    = y_mid[:eq][::-1] 
    #        Y_hf_north  = Y_half[:, eq:]; Y_hf_south = Y_half[:, :eq][:, ::-1]
    #        Z_hf_north  = Z_full[:, eq:]; Z_hf_south = Z_full[:, :eq][:, ::-1]
    #        Uz_hf_north = Uz_hf[:, eq:]; Uz_hf_south = Uz_hf[:, :eq][:, ::-1]

    #        Y_mid_north  = Y_mid[:, eq:]; Y_mid_south   = Y_mid[:, :eq][:, ::-1]
    #        Z_mid_north  = Z_mid[:, eq:]; Z_mid_south   = Z_mid[:, :eq][:, ::-1]
    #        Uz_mid_north = Uz_mid[:, eq:]; Uz_mid_south = Uz_mid[:, :eq][:, ::-1]

    #        r_north = (r0/g)*(beta*integrate.cumtrapz(Y_north*Uz_north, y_north, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y_north), 1)).T) + r0
    #        r_south = (r0/g)*(beta*integrate.cumtrapz(Y_south*Uz_south, y_south, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y_south), 1)).T) + r0
    #        r       = np.hstack([r_south[:, ::-1], r_north]); ry = (beta*r0/g)*Y*Uz; rz = np.gradient(r, z, axis=0)

    #        r_hf_north = (r0/g)*(beta*integrate.cumtrapz(Y_hf_north*Uz_hf_north, y_mid_north, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y_mid_north), 1)).T) + r0
    #        r_hf_south = (r0/g)*(beta*integrate.cumtrapz(Y_hf_south*Uz_hf_south, y_mid_south, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y_mid_south), 1)).T) + r0
    #        r_hf       = np.hstack([r_hf_south[:, ::-1], r_hf_north]); ry_hf = (beta*r0/g)*Y_half*Uz_hf; rz_hf = np.gradient(r_hf, z, axis=0)

    #        r_mid_north = (r0/g)*(beta*integrate.cumtrapz(Y_mid_north*Uz_mid_north, y_mid_north, initial=0) - np.tile(integrate.cumtrapz(N2_mid, z_mid, initial=0), (len(y_mid_north), 1)).T) + r0
    #        r_mid_south = (r0/g)*(beta*integrate.cumtrapz(Y_mid_south*Uz_mid_south, y_mid_south, initial=0) - np.tile(integrate.cumtrapz(N2_mid, z_mid, initial=0), (len(y_mid_south), 1)).T) + r0
    #        r_mid       = np.hstack([r_mid_south[:, ::-1], r_mid_north]); ry_mid = (beta*r0/g)*Y_mid*Uz_mid; rz_mid = np.gradient(r_mid, z_mid, axis=0)
            
    #elif assume == 'RAW':
    
    #    if (case == 'Proehl_1' or case == 'Proehl_2' or case == 'Proehl_3' or case == 'Proehl_4' or
    #   case == 'Proehl_5' or case == 'Proehl_6' or case == 'Proehl_7' or case == 'Proehl_8'):
       
    #       print(f'Use TWB for case={case}')
       
    #    else:

    #        U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf = calculate_NEMO_fields.load_mean_velocity(ny, nz, case, integration, stability, assume)
    #        r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf       = calculate_NEMO_fields.load_mean_density(ny, nz, case, integration, stability, assume)
