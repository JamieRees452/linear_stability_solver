"""
Numerical solver for the linear stability problem of two dimensional mean zonal flows with a corresponding mean density field in thermal wind balance 
"""

import matplotlib.pyplot   as plt
import numpy               as np
from   scipy.integrate     import trapz
from   scipy.linalg        import eig, block_diag
from   scipy.sparse        import diags
from   scipy.sparse.linalg import eigs, ArpackNoConvergence
from   scipy               import sparse

import mean_fields 

def tridiag(subdiag, diag, supdiag, bc):
    """
    Create a tridiagonal matrix of the form
    
    |  bc[0]           bc[1]           0              0           0            ...               .                   0          |
    |subdiag[0]      diag[1]       supdiag[1]         0           0            ...               .                   0          |
    |    0          subdiag[1]      diag[2]       supdiag[2]      0            ...               .                   0          |
    |    0               0         subdiag[2]      diag[3]    supdiag[3]       ...               .                   0          |
    |    .               .             .              .           .            ...               .                   .          |
    |    .               .             .              .           .            ...               .                   .          |
    |    .               .             .              .           .            ...               .                   0          |
    |    0               .             .              .           .       subdiag[-2]       diag[-2]           supdiag[-1]      |
    |    0               0             0              0           0             0              bc[2]               bc[3]        |
    
    This allows us to easily implement the vertical boundary conditions
                        
    Parameters
    ----------
    
    subdiag : float
        Elements on the subdiagonal of the tridagonal matrix
        
    diag : float
        Elements on the diagonal of the tridiagonal matrix
        
    supdiag : float
        Elements on the superdiagonal of the tridiagonal matrix
        
    bc : list of floats
        Entries corresponding to boundary conditions
        
    Returns
    -------
    
    tri_matrix : matrix
        Tridiagonal matrix 
    """
    
    # Create a diagonal matrix with tridiagonal entries and then convert to a regular numpy array
    tri_matrix = diags([subdiag, diag, supdiag],[-1, 0, 1]).toarray()
    
    # Add entries corresponding to boundary conditions
    tri_matrix[0, 0]   = bc[0]; tri_matrix[0, 1]   = bc[1]
    tri_matrix[-1, -2] = bc[2]; tri_matrix[-1, -1] = bc[3]
    
    return tri_matrix
 
def gep(ny, nz, k, case, integration, stability, init_guess, values, tol_input, iter_input): 
    """
    Solve the generalised eigenvalue problem (gep) corresponding to a discretised linear stability problem
                            
    Parameters
    ----------
    
    ny : int
        Number of meridional grid points
    
    nz : int
        Number of vertical grid points
        
    k : float
        Non-dimensional zonal wavenumber
        
    case : str
        Specify the problem to solve 
        
    init_guess : complex
        Initial guess to start the search for eigenvalues
        
    values : int
        Number of eigenvalues to obtain 
        
    tol_input : int
        Relative accuracy for eigenvalues (stopping criterion). A value of 0 implies machine precision
        
    iter_input : int
        Maximum number of Arnoldi update iterations allowed
        
    integration : str
        Specify the coupled integration
        u-by430 = 1/12 deg; u-bx950 = 1/4 deg
        
    stability : str
        Specify whether the mean profile for the specified integration
        is stable or unstable
        
    Returns
    -------
    
    evals : (values, ) ndarray
        The first (values) eigenvalues ordered by largest imaginary part
    
    evecs : complex
        Most unstable (right) eigenvector
    """
    
    ########################################################################################################################################################################################################
    # (i) Set up the domain
    ########################################################################################################################################################################################################

    if case == 'NEMO':
        if integration == 'u-by430':
            lat = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/latitude_12.txt')
        else:
            lat = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/latitude_25.txt')
            
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
        
    ########################################################################################################################################################################################################
    # (ii) Specify the mean zonal flow and denisty profiles
    ########################################################################################################################################################################################################

    U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case, integration, stability)

    ########################################################################################################################################################################################################
    # (iii) Typical dimensional parameters 
    ########################################################################################################################################################################################################
           
    # Typical values for the equatorial ocean 
    beta   = 2.29e-11          # Meridional gradient of the Coriolis parameter (m^{-1}s^{-1})
    r0     = 1026              # Background density (kg m^{3})
    g      = 9.81              # Gravitational acceleration (ms^{-2})

    ########################################################################################################################################################################################################
    # (vi) Build the zonal momentum equation
    ########################################################################################################################################################################################################
                
    ZLU = k*np.diag(U_mid.flatten(order='F'))

    ZLV0 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 

    for i in range((ny-1)*(nz-1)):                
        ZLV0[i, i]        = beta*Y_mid.flatten(order='F')[i]                          
        ZLV0[i, i+(nz-1)] = beta*Y_mid.flatten(order='F')[i] 

    ZLV1 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
    for i in range((ny-1)*(nz-1)):                
        ZLV1[i, i]        = (Uy_mid.flatten(order='F'))[i]                            
        ZLV1[i, i+(nz-1)] = (Uy_mid.flatten(order='F'))[i] 

    ZLV2 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1))))  
    for i in range(ny-1):                        
        ZLV2[i*(nz-1):(i+1)*(nz-1), i*(nz-1):(i+1)*(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]+(ry_hf/rz_hf)[1:, i]), (ry_hf/rz_hf)[1:-1, i], [(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 
        ZLV2[i*(nz-1):(i+1)*(nz-1), i*(nz-1)+(nz-1):(i+1)*(nz-1)+(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]+(ry_hf/rz_hf)[1:, i]), (ry_hf/rz_hf)[1:-1, i], [(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 

    ZLV2 = (Uz_mid.flatten(order='F')*(ZLV2.T)).T 

    ZLV = (1j/2)*ZLV0 - (1j/2)*ZLV1 + (1j/8)*ZLV2 

    ZLP0 = (k/r0)*np.eye((ny-1)*(nz-1))

    ZLP1 = np.asarray([tridiag((U_hf/rz_hf)[1:-1, i], -((U_hf/rz_hf)[:-1, i]-(U_hf/rz_hf)[1:, i]), -(U_hf/rz_hf)[1:-1, i], [(U_hf/rz_hf)[1, i], -(U_hf/rz_hf)[1, i], (U_hf/rz_hf)[-2, i], -(U_hf/rz_hf)[-2, i]]) for i in range(ny-1)])
    ZLP1 = block_diag(*ZLP1) 
    ZLP1 = (Uz_mid.flatten(order='F')*(ZLP1.T)).T

    ZLP = ZLP0 - (1/g)*(k/(2*dz))*ZLP1

    ZRU = k*np.eye((ny-1)*(nz-1))
    ZRV = np.zeros_like(ZLV)

    ZRP1 = np.asarray([tridiag((1/rz_hf)[1:-1, i], -((1/rz_hf)[:-1, i]-(1/rz_hf)[1:, i]), -(1/rz_hf)[1:-1, i], [(1/rz_hf)[1, i], -(1/rz_hf)[1, i], (1/rz_hf)[-2, i], -(1/rz_hf)[-2, i]]) for i in range(ny-1)])
    ZRP1 = block_diag(*ZRP1) 
    ZRP1 = (Uz_mid.flatten(order='F')*(ZRP1.T)).T

    ZRP = np.zeros_like(ZLP0) - (1/g)*(k/(2*dz))*ZRP1

    ########################################################################################################################################################################################################
    # (vii) Build the meridional momentum equation
    ########################################################################################################################################################################################################

    MLU0 = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))
    for i in range((ny-2)*(nz-1)):                
        MLU0[i+(nz-1), i]        = beta*Y_full.flatten(order='F')[i+(nz-1)]                 
        MLU0[i+(nz-1), i+(nz-1)] = beta*Y_full.flatten(order='F')[i+(nz-1)] 
    MLU = (-1j/2)*MLU0
        
    MLV = k*np.diag(U_fh.flatten(order='F'))

    MLP0 = np.zeros((ny*(nz-1), (ny-1)*(nz-1))) 
    for i in range((ny-2)*(nz-1)):                
        MLP0[i+(nz-1), i]        = 1                              
        MLP0[i+(nz-1), i+(nz-1)] = -1 
    MLP = (1/r0)*(1j/dy)*MLP0
    
    MRU = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))
    MRV = k*np.eye(ny*(nz-1)) 
    MRP = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))

    ########################################################################################################################################################################################################
    # (viii) Build the continuity equation
    ########################################################################################################################################################################################################   
    
    CLU = k*np.eye((ny-1)*(nz-1))

    CLV0 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
    for i in range((ny-1)*(nz-1)):               
        CLV0[i, i]        = 1                           
        CLV0[i, i+(nz-1)] = -1

    CLV1 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
    for i in range(ny-1):                        
        CLV1[i*(nz-1):(i+1)*(nz-1), i*(nz-1):(i+1)*(nz-1)]               = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]-(ry_hf/rz_hf)[1:, i]), -(ry_hf/rz_hf)[1:-1, i], [-(ry_hf/rz_hf)[1, i], -(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 
        CLV1[i*(nz-1):(i+1)*(nz-1), i*(nz-1)+(nz-1):(i+1)*(nz-1)+(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]-(ry_hf/rz_hf)[1:, i]), -(ry_hf/rz_hf)[1:-1, i], [-(ry_hf/rz_hf)[1, i], -(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 

    CLV = (1j/dy)*CLV0 - (1j/(4*dz))*CLV1 

    CLP_blocks = np.asarray([tridiag((U_hf/rz_hf)[1:-1, i], -((U_hf/rz_hf)[:-1, i]+(U_hf/rz_hf)[1:, i]), (U_hf/rz_hf)[1:-1, i], [-(U_hf/rz_hf)[1, i], (U_hf/rz_hf)[1, i], (U_hf/rz_hf)[-2, i], -(U_hf/rz_hf)[-2, i]]) for i in range(ny-1)])
    CLP        = (1/g)*(k/(dz**2))*block_diag(*CLP_blocks) 

    CRU = np.zeros_like(CLU)
    CRV = np.zeros_like(CLV)

    CRP_blocks = np.asarray([tridiag((1/rz_hf)[1:-1, i], -((1/rz_hf)[:-1, i]+(1/rz_hf)[1:, i]), (1/rz_hf)[1:-1, i], [-(1/rz_hf)[1, i], (1/rz_hf)[1, i], (1/rz_hf)[-2, i], -(1/rz_hf)[-2, i]]) for i in range(ny-1)])
    CRP        = (1/g)*(k/(dz**2))*block_diag(*CRP_blocks) 
        
    ########################################################################################################################################################################################################
    # (iX) Apply boundary conditions
    ########################################################################################################################################################################################################

    MLV[:(nz-1),:(nz-1)]   = np.diag(np.ones(nz-1)) 
    MLV[-(nz-1):,-(nz-1):] = np.diag(np.ones(nz-1))

    MRV[:(nz-1),:]  = 0
    MRV[-(nz-1):,:] = 0

    ########################################################################################################################################################################################################
    # (X) Build the coefficient matrices A and B of the generalised eigenvalue problem
    ########################################################################################################################################################################################################
    
    # Form the LHS and RHS of each equation of motion as matrices
    ZLE = np.hstack([ZLU, ZLV, ZLP]); MLE = np.hstack([MLU, MLV, MLP]); CLE = np.hstack([CLU, CLV, CLP])
    ZRE = np.hstack([ZRU, ZRV, ZRP]); MRE = np.hstack([MRU, MRV, MRP]); CRE = np.hstack([CRU, CRV, CRP])
    
    # Build the coefficient matrices 
    A = np.vstack([ZLE, MLE, CLE]); B = np.vstack([ZRE, MRE, CRE])
    
    ########################################################################################################################################################################################################
    # (Xi)  Solve the (sparse) generalised eigenvalue problem
    ########################################################################################################################################################################################################
    
    sA = sparse.csr_matrix(A); sB = sparse.csr_matrix(B)
    
    try:
        evals, evecs = eigs(sA, k=values, M=sB, which='LI', sigma=init_guess, tol=tol_input, maxiter=iter_input)
    except ArpackNoConvergence as err:
        print(f'ARPACK failed to converge at wavenumber {k}. Eigenvalue at the final iteration is {err.eigenvalues}.\n\nReducing tolerance to tol={tol_input*1e2}\nIncreasing maximum iterations to maxiter={iter_input*10}')
        evals, evecs = eigs(sA, k=values, M=sB, which='LI', sigma=init_guess, tol=tol_input*1e2, maxiter=iter_input*10)
    
    evecs = evecs[:, np.argmax(evals.imag)]; evecs = evecs/np.linalg.norm(evecs)
    
    return evals, evecs
