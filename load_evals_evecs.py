import numpy as np
import os

def load_data(ny, nz, k, case):
    """
    Load the eigenvalues and most unstable eigenvector for the linear stability problem. 
    Reshape the most unstable eigenvector into u, v and p.
    
    Parameters
    ----------
    
    ny : int
        Number of grid points in the meridional direction
        
    nz : int
        Number of grid points in the vertical direction
        
    k : float
        Non-dimensional zonal wavenumber
       
    case : str
        Specifies the mean zonal flow and density profile for the linear stability problem
    """
    fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
             f'/home/rees/lsa/eigenvectors/evecs_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
             
    if not (os.path.exists(fname[0]) and os.path.exists(fname[1])):
        raise ValueError(f'The specified files do not exist\n{fname[0]}\n{fname[1]}')
    else:
        evals = np.loadtxt(fname[0]).view(complex).reshape(-1) 
        evecs = np.loadtxt(fname[1]).view(complex).reshape(-1) 
    
    return evals, evecs
