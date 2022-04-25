"""
Calculate the mean zonal velocity and density fields, from the NEMO data or from Proehls test cases, on each grid
e.g. Zonal velocity (U) at (h)alf points in y, (f)ull points in z is given by U_hf
"""

import numpy as np
import os
from scipy import integrate
from scipy.integrate import trapz
import sys

import calculate_NEMO_fields
import calculate_Proehl_fields

def on_each_grid(ny, nz, case, integration, stability):
    """
    Calculate the mean fields on each grid
                            
    Parameters
    ----------
    
    ny : int
        Number of meridional grid points
    
    nz : int
        Number of vertical grid points
        
    integration : str
        Specify the coupled integration
        u-by430 = 1/12 deg; u-bx950 = 1/4 deg
        
    stability : str
        Specify whether the mean profile for the specified integration
        is stable or unstable
        
    Returns
    -------
    
    U : (nz, ny) ndarray
        Mean zonal velocity
    
    r : (nz, ny) ndarray
        Mean density field calculated using thermal wind balance
    """

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

    if case == 'NEMO':
        U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf = calculate_NEMO_fields.load_mean_velocity(ny, nz, integration, stability)
    else:
        U    , Uy    , Uz     = calculate_Proehl_fields.mean_velocity(Y     , Z     , case)
        U_mid, Uy_mid, Uz_mid = calculate_Proehl_fields.mean_velocity(Y_mid , Z_mid , case)
        U_hf , Uy_hf , Uz_hf  = calculate_Proehl_fields.mean_velocity(Y_half, Z_full, case)
        U_fh , Uy_fh , Uz_fh  = calculate_Proehl_fields.mean_velocity(Y_full, Z_half, case)

    # Typical values for the equatorial ocean 
    beta   = 2.29e-11          # Meridional gradient of the Coriolis parameter (m^{-1}s^{-1})
    r0     = 1026              # Background density (kg m^{3})
    g      = 9.81              # Gravitational acceleration (ms^{-2})

    # Calculate the mean density profile (by thermal wind balance)
    
    if case == 'NEMO':
        N2, N2_mid = calculate_NEMO_fields.load_mean_buoyancy(nz, integration, stability) 
    else:
        N2     = 8.883e-5*np.ones(Z.shape[0])
        N2_mid = 8.883e-5*np.ones(Z_mid.shape[0])

    r = (r0/g)*(beta*integrate.cumtrapz(Y*Uz, y, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y), 1)).T) + r0
    ry = (beta*r0/g)*Y*Uz; rz = np.gradient(r, z, axis=0); 
    
    # For the NEMO case we artificially change rz such that rz<0 everywhere in the domain. This allows us to preliminarily calculate growth rates
    # We do still need to check where the error (or even if it is an error) in rz. We comment this change out for the Proehl cases.
    rz=rz-np.amax(rz)*1.5 
    
    r_hf = (r0/g)*(beta*integrate.cumtrapz(Y_half*Uz_hf, y_mid, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y)-1, 1)).T) + r0
    ry_hf = (beta*r0/g)*Y_half*Uz_hf; rz_hf = np.gradient(r_hf, z, axis=0); rz_hf=rz_hf-np.amax(rz_hf)*1.5 # artificial changes

    r_mid = (r0/g)*(beta*integrate.cumtrapz(Y_mid*Uz_mid, y_mid, initial=0) - np.tile(integrate.cumtrapz(N2_mid, z_mid, initial=0), (len(y)-1, 1)).T) + r0
    ry_mid = (beta*r0/g)*Y_mid*Uz_mid; rz_mid = np.gradient(r_mid, z_mid, axis=0); rz_mid=rz_mid-np.amax(rz_mid)*1.5 # artificial changes
    
    return U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry,ry_mid, ry_hf, rz, rz_mid, rz_hf 
