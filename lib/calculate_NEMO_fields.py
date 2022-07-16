"""
Calculate the mean velocity and buoyancy fields averaged over a period of stability/instability
and interpolate onto a grid of specified resolution (ny, nz)
"""

import numpy as np
import os
from scipy.interpolate import interp1d
import scipy.interpolate as interp

def mean_velocity(ny, nz, case, integration, stability, assume):
    """
    Load the mean velocity and interpolate them onto a new grid of shape (ny, nz)
    then save the mean velocity
                            
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
        Mean zonal velocity interpolated onto a (ny, nz) grid
    """

    if stability == 'unstable':
        months = ['07', '08'] # Unstable period is given by July-August
    else:
        months = ['03', '04'] # Stable period is given by March-October
        
    U_fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/U_{integration}_{months[0]}.txt',
               f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/U_{integration}_{months[1]}.txt']
               
    if integration == 'u-by430':
        U_0 = np.loadtxt(U_fname[0]).reshape(47, 365)[::-1]; U_1 = np.loadtxt(U_fname[1]).reshape(47, 365)[::-1]
    else:
        U_0 = np.loadtxt(U_fname[0]).reshape(47, 123)[::-1]; U_1 = np.loadtxt(U_fname[1]).reshape(47, 123)[::-1]

    U = (U_0 + U_1)/2 # Average the mean zonal velocity over the two months

    if integration == 'u-by430':
        lat = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/latitude_12.txt') # 1/12 deg
    else:
        lat = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/latitude_25.txt') # 1/4 deg
        
    depth = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/depth.txt'); depth = -depth[::-1]
    
    y = lat*111.12*1000; z = depth; Y, Z = np.meshgrid(y, z)
    
    # make changes to the mean current structure here !!!!!!!!!!!!
    
    #SECS = np.where((-1.5e6<Y) & (Y<-1e5) & (U<0))
    #EUC  = np.where((-3e5<Y) & (Y<3e5) & (U>0))
    #SECN = np.where((5e4<Y) & (Y<6e5) & (U<0))
    #NECC = np.where((Y>4e5) & (U>0))

    #U[SECS] = 0.0*U[SECS]
    #U[EUC] = 1.0*U[EUC]
    #U[SECN] = 0.0*U[SECN]
    #U[NECC] = 0.0*U[NECC]
    
    Uy = np.gradient(U, y, axis=1); Uz = np.gradient(U, z, axis=0)
    
    # Interpolate onto a grid of (ny, nz) points (explanation of this section coming soon, but by inspection of plots it seems correct) ######
    
    y_old = lat/np.amax(abs(lat)); z_old = depth/np.amax(abs(depth)); Y_old, Z_old = np.meshgrid(y_old, z_old)
    
    if case == 'NEMO':
        y = np.linspace(y_old[0], y_old[-1], ny); z = np.linspace(-1000, -0, nz)/np.amax(abs(depth))
        
    elif case == 'NEMO_rigid_lid':
        y = np.linspace(y_old[0], y_old[-1], ny); z = np.linspace(-1000, -75, nz)/np.amax(abs(depth))
        
    else:
        print(f'{case} is not a valid choice of case')
        
    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]
    
    Y, Z = np.meshgrid(y, z);                 Y_mid, Z_mid = np.meshgrid(y_mid, z_mid)
    Y_full, Z_half = np.meshgrid(y, z_mid); Y_half, Z_full = np.meshgrid(y_mid, z)
    
    U_grid = np.stack([Y_old.ravel(), Z_old.ravel()], -1)
    
    U_interpolator  = interp.RBFInterpolator(U_grid, U.ravel(), smoothing=0, kernel='cubic')
    Uy_interpolator = interp.RBFInterpolator(U_grid, Uy.ravel(), smoothing=0, kernel='cubic')
    Uz_interpolator = interp.RBFInterpolator(U_grid, Uz.ravel(), smoothing=0, kernel='cubic')
    
    grid = np.stack([Y.ravel(), Z.ravel()], -1); grid_mid = np.stack([Y_mid.ravel(), Z_mid.ravel()], -1)
    grid_fh = np.stack([Y_full.ravel(), Z_half.ravel()], -1); grid_hf = np.stack([Y_half.ravel(), Z_full.ravel()], -1)
    
    U     = U_interpolator(grid).reshape(Y.shape)
    U_mid = U_interpolator(grid_mid).reshape(Y_mid.shape)
    U_hf  = U_interpolator(grid_hf).reshape(Y_half.shape)
    U_fh  = U_interpolator(grid_fh).reshape(Y_full.shape)
    
    Uy     = Uy_interpolator(grid).reshape(Y.shape)
    Uy_mid = Uy_interpolator(grid_mid).reshape(Y_mid.shape) 
    Uy_hf  = Uy_interpolator(grid_hf).reshape(Y_half.shape)
    Uz     = Uz_interpolator(grid).reshape(Y.shape)
    Uz_mid = Uz_interpolator(grid_mid).reshape(Y_mid.shape)
    Uz_hf  = Uz_interpolator(grid_hf).reshape(Y_half.shape)
    
    ######################################################################################################################
    
    # The interpolation procedure can be expensive for high resolutions hence we should instead save the data
    # and load it in future when we need it
    
    # File names for NEMO profiles should contain the integration and stability
    fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/U_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/U_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/U_hf_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/U_fh_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uy_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uy_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uy_hf_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uz_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uz_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uz_hf_{ny:02}_{nz:02}.txt']
             
    np.savetxt(fname[0], U.flatten()); np.savetxt(fname[1], U_mid.flatten()); np.savetxt(fname[2], U_hf.flatten()); np.savetxt(fname[3], U_fh.flatten())
    np.savetxt(fname[4], Uy.flatten()); np.savetxt(fname[5], Uy_mid.flatten()); np.savetxt(fname[6], Uy_hf.flatten())
    np.savetxt(fname[7], Uz.flatten()); np.savetxt(fname[8], Uz_mid.flatten()); np.savetxt(fname[9], Uz_hf.flatten())
    
    return U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf
    
def load_mean_velocity(ny, nz, case, integration, stability, assume):
    """
    Load the previously saved mean zonal velocity data
                            
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
        Mean zonal velocity interpolated onto a (ny, nz) grid
    """

    # File names for NEMO profiles should contain the integration and stability
    fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/U_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/U_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/U_hf_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/U_fh_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uy_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uy_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uy_hf_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uz_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uz_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/Uz_hf_{ny:02}_{nz:02}.txt']
             
    if os.path.exists(fname[0]):
        # If the interpolated data exists, then load it rather than recalculating
        U      = np.loadtxt(fname[0]).reshape(nz, ny)
        U_mid  = np.loadtxt(fname[1]).reshape(nz-1, ny-1)
        U_hf   = np.loadtxt(fname[2]).reshape(nz, ny-1)
        U_fh   = np.loadtxt(fname[3]).reshape(nz-1, ny)
        Uy     = np.loadtxt(fname[4]).reshape(nz, ny)
        Uy_mid = np.loadtxt(fname[5]).reshape(nz-1, ny-1)
        Uy_hf  = np.loadtxt(fname[6]).reshape(nz, ny-1)
        Uz     = np.loadtxt(fname[7]).reshape(nz, ny)
        Uz_mid = np.loadtxt(fname[8]).reshape(nz-1, ny-1)
        Uz_hf  = np.loadtxt(fname[9]).reshape(nz, ny-1)
        
    else:
        # If the mean velocity fields have not already been calculated, then do so now
        print(f'NEMO velocity field files do not exist. Calculating NEMO fields')
        U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf = mean_velocity(ny, nz, case, integration, stability, assume)
        
    return U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf
    
def mean_density(ny, nz, case, integration, stability, assume):
    """
    Load the mean velocity and interpolate them onto a new grid of shape (ny, nz)
    then save the mean velocity
                            
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
        Mean zonal velocity interpolated onto a (ny, nz) grid
    """

    if stability == 'unstable':
        months = ['07', '08'] # Unstable period is given by July-August
    else:
        months = ['03', '04'] # Stable period is given by March-October
        
    U_fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/r_{integration}_{months[0]}.txt',
               f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/r_{integration}_{months[1]}.txt']
               
    if integration == 'u-by430':
        r_0 = np.loadtxt(U_fname[0]).reshape(47, 365)[::-1]; r_1 = np.loadtxt(U_fname[1]).reshape(47, 365)[::-1]
    else:
        r_0 = np.loadtxt(U_fname[0]).reshape(47, 123)[::-1]; r_1 = np.loadtxt(U_fname[1]).reshape(47, 123)[::-1]

    r = (r_0 + r_1)/2 # Average the mean zonal velocity over the two months

    if integration == 'u-by430':
        lat = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/latitude_12.txt') # 1/12 deg
    else:
        lat = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/latitude_25.txt') # 1/4 deg
        
    depth = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/depth.txt'); depth = -depth[::-1]
    
    y = lat*111.12*1000; z = depth; Y, Z = np.meshgrid(y, z)
    
    ry = np.gradient(r, y, axis=1); rz = np.gradient(r, z, axis=0)
    
    # Interpolate onto a grid of (ny, nz) points (explanation of this section coming soon, but by inspection of plots it seems correct) ######
    
    y_old = lat/np.amax(abs(lat)); z_old = depth/np.amax(abs(depth)); Y_old, Z_old = np.meshgrid(y_old, z_old)
    
    if case == 'NEMO':
        y = np.linspace(y_old[0], y_old[-1], ny); z = np.linspace(-1000, -0, nz)/np.amax(abs(depth))
        
    elif case == 'NEMO_rigid_lid':
        y = np.linspace(y_old[0], y_old[-1], ny); z = np.linspace(-1000, -75, nz)/np.amax(abs(depth))
        
    else:
        print(f'{case} is not a valid choice of case')
    
    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]
    
    Y, Z = np.meshgrid(y, z);                 Y_mid, Z_mid = np.meshgrid(y_mid, z_mid)
    Y_full, Z_half = np.meshgrid(y, z_mid); Y_half, Z_full = np.meshgrid(y_mid, z)
    
    r_grid = np.stack([Y_old.ravel(), Z_old.ravel()], -1)
    
    r_interpolator  = interp.RBFInterpolator(r_grid, r.ravel(), smoothing=0, kernel='cubic')
    ry_interpolator = interp.RBFInterpolator(r_grid, ry.ravel(), smoothing=0, kernel='cubic')
    rz_interpolator = interp.RBFInterpolator(r_grid, rz.ravel(), smoothing=0, kernel='cubic')
    
    grid = np.stack([Y.ravel(), Z.ravel()], -1); grid_mid = np.stack([Y_mid.ravel(), Z_mid.ravel()], -1)
    grid_fh = np.stack([Y_full.ravel(), Z_half.ravel()], -1); grid_hf = np.stack([Y_half.ravel(), Z_full.ravel()], -1)
    
    r     = r_interpolator(grid).reshape(Y.shape)
    r_mid = r_interpolator(grid_mid).reshape(Y_mid.shape)
    r_hf  = r_interpolator(grid_hf).reshape(Y_half.shape)
    
    ry     = ry_interpolator(grid).reshape(Y.shape)
    ry_mid = ry_interpolator(grid_mid).reshape(Y_mid.shape) 
    ry_hf  = ry_interpolator(grid_hf).reshape(Y_half.shape)
    rz     = rz_interpolator(grid).reshape(Y.shape)
    rz_mid = rz_interpolator(grid_mid).reshape(Y_mid.shape)
    rz_hf  = rz_interpolator(grid_hf).reshape(Y_half.shape)
    
    ######################################################################################################################
    
    # The interpolation procedure can be expensive for high resolutions hence we should instead save the data
    # and load it in future when we need it
    
    # File names for NEMO profiles should contain the integration and stability
    fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/r_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/r_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/r_hf_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/ry_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/ry_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/ry_hf_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/rz_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/rz_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/rz_hf_{ny:02}_{nz:02}.txt']
             
    np.savetxt(fname[0], r.flatten()); np.savetxt(fname[1], r_mid.flatten()); np.savetxt(fname[2], r_hf.flatten())
    np.savetxt(fname[3], ry.flatten()); np.savetxt(fname[4], ry_mid.flatten()); np.savetxt(fname[5], ry_hf.flatten())
    np.savetxt(fname[6], rz.flatten()); np.savetxt(fname[7], rz_mid.flatten()); np.savetxt(fname[8], rz_hf.flatten())
    
    return r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf
    
def load_mean_density(ny, nz, case, integration, stability, assume):
    """
    Load the previously saved mean zonal velocity data
                            
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
    
    r : (nz, ny) ndarray
        Mean density interpolated onto a (ny, nz) grid
    """

    # File names for NEMO profiles should contain the integration and stability
    fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/r_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/r_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/r_hf_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/ry_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/ry_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/ry_hf_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/rz_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/rz_mid_{ny:02}_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/rz_hf_{ny:02}_{nz:02}.txt']
             
    if os.path.exists(fname[0]):
        # If the interpolated data exists, then load it rather than recalculating
        r      = np.loadtxt(fname[0]).reshape(nz, ny)
        r_mid  = np.loadtxt(fname[1]).reshape(nz-1, ny-1)
        r_hf   = np.loadtxt(fname[2]).reshape(nz, ny-1)
        ry     = np.loadtxt(fname[3]).reshape(nz, ny)
        ry_mid = np.loadtxt(fname[4]).reshape(nz-1, ny-1)
        ry_hf  = np.loadtxt(fname[5]).reshape(nz, ny-1)
        rz     = np.loadtxt(fname[6]).reshape(nz, ny)
        rz_mid = np.loadtxt(fname[7]).reshape(nz-1, ny-1)
        rz_hf  = np.loadtxt(fname[8]).reshape(nz, ny-1)
        
    else:
        # If the mean velocity fields have not already been calculated, then do so now
        print(f'NEMO density field files do not exist. Calculating NEMO fields')
        r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_density(ny, nz, case, integration, stability, assume)
        
    return r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf

def mean_buoyancy(nz, case, integration, stability, assume):
    """
    Load the mean buoyancy and interpolate onto a new grid of shape (nz, )
    then save the data
                            
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
    
    N2 : (nz, ) ndarray
        Mean buoyancy frequency interpolated onto a (nz, ) grid
    """

    if stability == 'unstable':
        months = ['07', '08'] # Unstable period is given by July-August
    else:
        months = ['03', '04'] # Stable period is given by March-April
        
    N2_fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/N2_{integration}_{months[0]}.txt',
                f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/N2_{integration}_{months[1]}.txt']
               
    if integration == 'u-by430':
        N2_0 = np.loadtxt(N2_fname[0]).reshape(47, 365)[::-1]; N2_1 = np.loadtxt(N2_fname[1]).reshape(47, 365)[::-1]
    else:
        N2_0 = np.loadtxt(N2_fname[0]).reshape(47, 123)[::-1]; N2_1 = np.loadtxt(N2_fname[1]).reshape(47, 123)[::-1]

    N2_mean = (N2_0 + N2_1)/2; N2_mean[0,:] = N2_mean[1,:]-1e-6 # so that N2 is non-zero at the bottom of the domain
    
    if integration == 'u-by430':
        N2_mean = N2_mean[:, 182]
    else:
        N2_mean = N2_mean[:, 61]
    
    #N2 = np.mean(N2, axis=1) # Take the average over the two months, and then average over the meridional extent of the domain

    if integration == 'u-by430':
        lat = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/latitude_12.txt')
    else:
        lat = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/latitude_25.txt')

    depth = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/depth.txt'); depth = -depth[::-1]
    
    # Interpolate the buoyancy frequency onto a uniformly spaced grid of nz points
    
    f = interp1d(depth, N2_mean, kind='cubic', fill_value="extrapolate")
    
    if case == 'NEMO':
        depth_linear = np.linspace(-1000, -0, nz)
    elif case == 'NEMO_rigid_lid':
        depth_linear = np.linspace(-1000, -75, nz)
    else:
        print(f'{case} is not a valid case')
        
    d_depth = abs(depth_linear[1]-depth_linear[0])
    
    depth_mid = (depth_linear[:depth_linear.size]+0.5*d_depth)[:-1]
    
    
    N2 = f(depth_linear); N2_mid = f(depth_mid)
    
    # Save the interpolated buoyancy frequency
    fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/N2_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/N2_mid_{nz:02}.txt']
             
    np.savetxt(fname[0], N2); np.savetxt(fname[1], N2_mid.flatten())
    
    return N2, N2_mid
    
def load_mean_buoyancy(nz, case, integration, stability, assume):
    """
    Load the previously saved mean buoyancy data
                            
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
        Mean zonal velocity interpolated onto a (ny, nz) grid
    """

    fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/N2_{nz:02}.txt',
             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/N2_mid_{nz:02}.txt']
             
    if os.path.exists(fname[0]):
        # If the mean buoyancy fields have already been calculated, then load the data
        N2      = np.loadtxt(fname[0])
        N2_mid  = np.loadtxt(fname[1])
        
    else:
        # If the mean buoyancy fields have not already been calculated, then do so now
        print(f'NEMO density field files do not exist. Calculating density fields')
        N2, N2_mid  = mean_buoyancy(nz, case, integration, stability, assume)
        
    return N2, N2_mid
