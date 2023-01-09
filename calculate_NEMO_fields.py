"""
Calculate the mean zonal velocity and density fields averaged over two months
and interpolate onto a grid of specified resolution (ny, nz)
"""

import numpy as np
import os
from   scipy.interpolate import interp1d
import scipy.interpolate as interp

WORK_DIR = os.getcwd()

def mean_velocity(ny, nz, case, month0, month1):
    """
    Load the raw mean velocity data from NEMO and interpolate onto a new grid of shape (ny, nz)
                            
    Parameters
    ----------
    ny : int
         Meridional grid resolution
    
    nz : int
         Vertical grid resolution
         
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
    U, U_mid, U_hf : ndarray
        Mean zonal velocity interpolated onto staggered grid
        
    Uy, Uy_mid, Uy_hf : ndarray
        Meridional gradient of mean zonal velocity interpolated onto staggered grid
        
    Uz, Uz_mid, Uz_hf : ndarray
        Vertical gradient of mean zonal velocity interpolated onto staggered grid
    """
        
    U_fname = [f'{WORK_DIR}/original_data/U_{case[-2:]}_{month0}.txt',
               f'{WORK_DIR}/original_data/U_{case[-2:]}_{month1}.txt']
               
    if case == 'NEMO_25':
        U_0 = np.loadtxt(U_fname[0]).reshape(47, 123)[::-1]; U_1 = np.loadtxt(U_fname[1]).reshape(47, 123)[::-1]
        lat = np.loadtxt(f'{WORK_DIR}/original_data/latitude_25.txt')
        
    elif case == 'NEMO_12':
        U_0 = np.loadtxt(U_fname[0]).reshape(47, 365)[::-1]; U_1 = np.loadtxt(U_fname[1]).reshape(47, 365)[::-1]
        lat = np.loadtxt(f'{WORK_DIR}/original_data/latitude_12.txt')
        
    else:
        print(f'Error with case choice')

    U = (U_0 + U_1)/2 # Average the mean zonal velocity over the two months
        
    depth = np.loadtxt(f'{WORK_DIR}/original_data/depth.txt'); depth = -depth[::-1]
    
    y = lat*111.12*1000; z = depth; Y, Z = np.meshgrid(y, z)
       
    #SECS = np.where((-1.5e6<Y) & (Y<-1e5) & (U<0))
    #EUC  = np.where((-3e5<Y) & (Y<3e5) & (U>0))
    #SECN = np.where((5e4<Y) & (Y<6e5) & (U<0))
    #NECC = np.where((Y>4e5) & (U>0))

    #U[SECS] = 0.0*U[SECS]
    #U[EUC] = 1.0*U[EUC]
    #U[SECN] = 0.0*U[SECN]
    #U[NECC] = 0.0*U[NECC]
    
    # Interpolate onto a grid of (ny, nz) points 
    
    y_old = lat/np.amax(abs(lat)); z_old = depth/np.amax(abs(depth)); Y_old, Z_old = np.meshgrid(y_old, z_old)
    
    y = np.linspace(y_old[0], y_old[-1], ny); z = np.linspace(-1000, -0, nz)/np.amax(abs(depth))
        
    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]
    
    Y, Z = np.meshgrid(y, z);                 Y_mid, Z_mid = np.meshgrid(y_mid, z_mid)
    Y_full, Z_half = np.meshgrid(y, z_mid); Y_half, Z_full = np.meshgrid(y_mid, z)
    
    U_grid = np.stack([Y_old.ravel(), Z_old.ravel()], -1)
    
    U_interpolator  = interp.RBFInterpolator(U_grid, U.ravel(), smoothing=0, kernel='cubic')
    
    grid = np.stack([Y.ravel(), Z.ravel()], -1); grid_mid = np.stack([Y_mid.ravel(), Z_mid.ravel()], -1)
    grid_fh = np.stack([Y_full.ravel(), Z_half.ravel()], -1); grid_hf = np.stack([Y_half.ravel(), Z_full.ravel()], -1)
    
    U     = U_interpolator(grid).reshape(Y.shape)
    U_mid = U_interpolator(grid_mid).reshape(Y_mid.shape)
    U_hf  = U_interpolator(grid_hf).reshape(Y_half.shape)
    U_fh  = U_interpolator(grid_fh).reshape(Y_full.shape)
    
    y = np.linspace(y_old[0], y_old[-1], ny)*np.amax(abs(lat))*111.12*1000; z = np.linspace(-1000, -0, nz)
    
    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]

    Y,Z         = np.meshgrid(y, z);         Y_full,Z_half = np.meshgrid(y, z_mid) 
    Y_mid,Z_mid = np.meshgrid(y_mid, z_mid); Y_half,Z_full = np.meshgrid(y_mid, z)
    
    Uy     = np.gradient(U, Y[0,:], axis=1); 
    Uy_mid = np.gradient(U_mid, Y_mid[0,:], axis=1); 
    Uy_hf  = np.gradient(U_hf, Y_half[0,:], axis=1); 
    
    Uz     = np.gradient(U, Z[:,0], axis=0)
    Uz_mid = np.gradient(U_mid, Z_mid[:,0], axis=0)
    Uz_hf  = np.gradient(U_hf, Z_full[:,0], axis=0)
    
    fname = [f'{WORK_DIR}/saved_data/interpolated_data/U_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/U_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/U_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/U_fh_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uy_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uy_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uy_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uz_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uz_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uz_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt']
             
    np.savetxt(fname[0], U.flatten()); np.savetxt(fname[1], U_mid.flatten()); np.savetxt(fname[2], U_hf.flatten()); np.savetxt(fname[3], U_fh.flatten())
    np.savetxt(fname[4], Uy.flatten()); np.savetxt(fname[5], Uy_mid.flatten()); np.savetxt(fname[6], Uy_hf.flatten())
    np.savetxt(fname[7], Uz.flatten()); np.savetxt(fname[8], Uz_mid.flatten()); np.savetxt(fname[9], Uz_hf.flatten())
    
    return U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf
    
def load_mean_velocity(ny, nz, case, month0, month1):
    """
    Load the previously saved interpolated mean zonal velocity data
                            
    Parameters
    ----------
    ny : int
         Meridional grid resolution
    
    nz : int
         Vertical grid resolution
         
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
    U, U_mid, U_hf : ndarray
        Mean zonal velocity interpolated onto staggered grid
        
    Uy, Uy_mid, Uy_hf : ndarray
        Meridional gradient of mean zonal velocity interpolated onto staggered grid
        
    Uz, Uz_mid, Uz_hf : ndarray
        Vertical gradient of mean zonal velocity interpolated onto staggered grid
    """
    
    fname = [f'{WORK_DIR}/saved_data/interpolated_data/U_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/U_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/U_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/U_fh_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uy_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uy_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uy_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uz_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uz_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/Uz_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt'] 
             
    if (os.path.exists(fname[0]) and os.path.exists(fname[1]) and os.path.exists(fname[2]) and os.path.exists(fname[3]) and
        os.path.exists(fname[4]) and os.path.exists(fname[5]) and os.path.exists(fname[6]) and os.path.exists(fname[7]) and os.path.exists(fname[8])):
        
        # If the interpolated data exists then load it rather than recalculating
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
        print(f'    | Interpolating NEMO velocity fields onto specified uniform grid |')
        U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf = mean_velocity(ny, nz, case, month0, month1)
        
    return U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf
    
def mean_density(ny, nz, case, month0, month1):
    """
    Load the raw mean density data from NEMO and interpolate onto a new grid of shape (ny, nz)
                            
    Parameters
    ----------
    ny : int
         Meridional grid resolution
    
    nz : int
         Vertical grid resolution
         
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
    r, r_mid, r_hf : ndarray
        Mean zonal velocity interpolated onto staggered grid
        
    ry, ry_mid, ry_hf : ndarray
        Meridional gradient of mean zonal velocity interpolated onto staggered grid
        
    rz, rz_mid, rz_hf : ndarray
        Vertical gradient of mean zonal velocity interpolated onto staggered grid
    """
        
    r_fname = [f'{WORK_DIR}/original_data/r_{case[-2:]}_{month0}.txt',
               f'{WORK_DIR}/original_data/r_{case[-2:]}_{month1}.txt']
               
    if case == 'NEMO_25':
        r_0 = np.loadtxt(r_fname[0]).reshape(47, 123)[::-1]; r_1 = np.loadtxt(r_fname[1]).reshape(47, 123)[::-1]
        lat = np.loadtxt(f'{WORK_DIR}/original_data/latitude_25.txt')
        
    elif case == 'NEMO_12':
        r_0 = np.loadtxt(r_fname[0]).reshape(47, 365)[::-1]; r_1 = np.loadtxt(r_fname[1]).reshape(47, 365)[::-1]
        lat = np.loadtxt(f'{WORK_DIR}/original_data/latitude_12.txt')
        
    else:
        print(f'Error with case choice')

    r = (r_0 + r_1)/2 # Average the mean density over the two months
        
    depth = np.loadtxt(f'{WORK_DIR}/original_data/depth.txt'); depth = -depth[::-1]
    
    y = lat*111.12*1000; z = depth; Y, Z = np.meshgrid(y, z)
    
    # Interpolate onto a grid of (ny, nz) points
    
    y_old = lat/np.amax(abs(lat)); z_old = depth/np.amax(abs(depth)); Y_old, Z_old = np.meshgrid(y_old, z_old)
    
    y = np.linspace(y_old[0], y_old[-1], ny); z = np.linspace(-1000, -0, nz)/np.amax(abs(depth))
    
    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]
    
    Y, Z = np.meshgrid(y, z);                 Y_mid, Z_mid = np.meshgrid(y_mid, z_mid)
    Y_full, Z_half = np.meshgrid(y, z_mid); Y_half, Z_full = np.meshgrid(y_mid, z)
    
    r_grid = np.stack([Y_old.ravel(), Z_old.ravel()], -1)
    
    r_interpolator  = interp.RBFInterpolator(r_grid, r.ravel(), smoothing=0, kernel='cubic')
    
    grid = np.stack([Y.ravel(), Z.ravel()], -1); grid_mid = np.stack([Y_mid.ravel(), Z_mid.ravel()], -1)
    grid_fh = np.stack([Y_full.ravel(), Z_half.ravel()], -1); grid_hf = np.stack([Y_half.ravel(), Z_full.ravel()], -1)
    
    r     = r_interpolator(grid).reshape(Y.shape)
    r_mid = r_interpolator(grid_mid).reshape(Y_mid.shape)
    r_hf  = r_interpolator(grid_hf).reshape(Y_half.shape)
    
    y = np.linspace(y_old[0], y_old[-1], ny)*np.amax(abs(lat))*111.12*1000; z = np.linspace(-1000, -0, nz)
    
    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]

    Y,Z         = np.meshgrid(y, z);         Y_full,Z_half = np.meshgrid(y, z_mid) 
    Y_mid,Z_mid = np.meshgrid(y_mid, z_mid); Y_half,Z_full = np.meshgrid(y_mid, z)
    
    ry     = np.gradient(r, Y[0,:], axis=1); 
    ry_mid = np.gradient(r_mid, Y_mid[0,:], axis=1); 
    ry_hf  = np.gradient(r_hf, Y_half[0,:], axis=1); 
    
    rz     = np.gradient(r, Z[:,0], axis=0)
    rz_mid = np.gradient(r_mid, Z_mid[:,0], axis=0)
    rz_hf  = np.gradient(r_hf, Z_full[:,0], axis=0)
    
    fname = [f'{WORK_DIR}/saved_data/interpolated_data/r_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/r_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/r_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/r_fh_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/ry_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/ry_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/ry_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/rz_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/rz_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/rz_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt'] 
             
    np.savetxt(fname[0], r.flatten()); np.savetxt(fname[1], r_mid.flatten()); np.savetxt(fname[2], r_hf.flatten())
    np.savetxt(fname[3], ry.flatten()); np.savetxt(fname[4], ry_mid.flatten()); np.savetxt(fname[5], ry_hf.flatten())
    np.savetxt(fname[6], rz.flatten()); np.savetxt(fname[7], rz_mid.flatten()); np.savetxt(fname[8], rz_hf.flatten())
    
    return r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf
    
def load_mean_density(ny, nz, case, month0, month1):
    """
    Load the previously saved interpolated mean density data
                            
    Parameters
    ----------
    ny : int
         Meridional grid resolution
    
    nz : int
         Vertical grid resolution
         
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
    r, r_mid, r_hf : ndarray
        Mean density interpolated onto staggered grid
        
    ry, ry_mid, ry_hf : ndarray
        Meridional gradient of mean density interpolated onto staggered grid
        
    rz, rz_mid, rz_hf : ndarray
        Vertical gradient of mean density interpolated onto staggered grid
    """
    
    fname = [f'{WORK_DIR}/saved_data/interpolated_data/r_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/r_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/r_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/r_fh_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/ry_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/ry_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/ry_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/rz_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/rz_mid_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt',
             f'{WORK_DIR}/saved_data/interpolated_data/rz_hf_{case}_{month0}_{month1}_{ny:02}_{nz:02}.txt'] 
             
    if (os.path.exists(fname[0]) and os.path.exists(fname[1]) and os.path.exists(fname[2]) and os.path.exists(fname[3]) and
        os.path.exists(fname[4]) and os.path.exists(fname[5]) and os.path.exists(fname[6]) and os.path.exists(fname[7]) and os.path.exists(fname[8])):
        
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
        print(f'    | Interpolating NEMO density fields onto specified uniform grid  |')
        r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_density(ny, nz, case, month0, month1)
        
    return r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf

#def mean_buoyancy(nz, case, integration, stability, assume):
  

#    if stability == 'unstable':
#        months = ['07', '08'] # Unstable period is given by July-August
#    else:
#        months = ['03', '04'] # Stable period is given by March-April
        
#    N2_fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/N2_{integration}_{months[0]}.txt',
#                f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/N2_{integration}_{months[1]}.txt']
               
#    if integration == 'u-by430':
#        N2_0 = np.loadtxt(N2_fname[0]).reshape(47, 365)[::-1]; N2_1 = np.loadtxt(N2_fname[1]).reshape(47, 365)[::-1]
#    else:
#        N2_0 = np.loadtxt(N2_fname[0]).reshape(47, 123)[::-1]; N2_1 = np.loadtxt(N2_fname[1]).reshape(47, 123)[::-1]

#    N2_mean = (N2_0 + N2_1)/2; N2_mean[0,:] = N2_mean[1,:]-1e-6 # so that N2 is non-zero at the bottom of the domain
    
#    if integration == 'u-by430':
#        N2_mean = N2_mean[:, 182]
#    else:
#        N2_mean = N2_mean[:, 61]
    
    #N2 = np.mean(N2, axis=1) # Take the average over the two months, and then average over the meridional extent of the domain

#    if integration == 'u-by430':
#        lat = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/latitude_12.txt')
#    else:
#        lat = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/latitude_25.txt')

#    depth = np.loadtxt(f'/home/rees/lsa/saved_data/NEMO_mean_fields/RAW/depth.txt'); depth = -depth[::-1]
    
    # Interpolate the buoyancy frequency onto a uniformly spaced grid of nz points
    
#    f = interp1d(depth, N2_mean, kind='cubic', fill_value="extrapolate")
    
#    if case == 'NEMO':
#        depth_linear = np.linspace(-1000, -0, nz)
#    elif case == 'NEMO_rigid_lid':
#        depth_linear = np.linspace(-1000, -75, nz)
#    else:
#        print(f'{case} is not a valid case')
        
#    d_depth = abs(depth_linear[1]-depth_linear[0])
    
#    depth_mid = (depth_linear[:depth_linear.size]+0.5*d_depth)[:-1]
    
    
#    N2 = f(depth_linear); N2_mid = f(depth_mid)
    
    # Save the interpolated buoyancy frequency
#    fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/N2_{nz:02}.txt',
#             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/N2_mid_{nz:02}.txt']
             
#    np.savetxt(fname[0], N2); np.savetxt(fname[1], N2_mid.flatten())
    
#    return N2, N2_mid
    
#def load_mean_buoyancy(nz, case, integration, stability, assume):

#    fname = [f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/N2_{nz:02}.txt',
#             f'/home/rees/lsa/saved_data/NEMO_mean_fields/{case}/{integration}/{stability}/{assume}/N2_mid_{nz:02}.txt']
             
#    if os.path.exists(fname[0]):
        # If the mean buoyancy fields have already been calculated, then load the data
#        N2      = np.loadtxt(fname[0])
#        N2_mid  = np.loadtxt(fname[1])
        
#    else:
        # If the mean buoyancy fields have not already been calculated, then do so now
#        print(f'NEMO density field files do not exist. Calculating density fields')
#        N2, N2_mid  = mean_buoyancy(nz, case, integration, stability, assume)
        
#    return N2, N2_mid
