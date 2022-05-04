"""
Calculate grids
"""
import numpy as np

def grid(ny, nz, case, integration):
    """
    Calculate the meshgrids for the domain on each grid
 
    Parameters
    ----------
    
    ny : int
        Meridional grid resolution
        
    nz : int
        Vertical grid resolution
        
    case : str
        
        
    integration : str
        
        
    Returns
    -------
    
    y, y_mid : (ny, ) array
        Linearly spaced points in the meridional direction
    
    dy : float
        Meridional grid spacing
        
    Y, Y_mid, Y_half, Y_full : 2D array
        Meshgrid with Z, Z_mid, Z_half, Z_full
        
    L : float
        Meridional half-width of the domain
        
    D : float 
        Depth of the domain
    """
    
    if case == 'NEMO':
        if integration == 'u-by430':
            lat = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/RAW/latitude_12.txt')
        else:
            lat = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/RAW/latitude_25.txt')

        depth = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/RAW/depth.txt'); depth = -depth[::-1]
        
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
    
    return y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D
