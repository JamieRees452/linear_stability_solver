"""
Numerical solver for the (dense) linear stability problem of two dimensional mean zonal flows with a corresponding mean density field in thermal wind balance 
"""

from   itertools           import compress 
import matplotlib.pyplot   as plt
import numpy               as np
from   scipy               import integrate
from   scipy.integrate     import trapz
from   scipy.linalg        import eig, block_diag
from   scipy.sparse        import diags
from   scipy.sparse.linalg import eigs
import sys

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
    
def clean_evals(A, B, evals, vl, evecs, suppress):
    """
    Clean the eigenvalue spectrum by deleting eigenvalues that:
    (i)   Have infinite real part
    (ii)  Are ill-conditioned wrt some tolerance
    (iii) Do not satisfy the generalised eigenvalue problem
                        
    Parameters
    ----------
    
    A : (N, N) matrix
        LHS matrix of the generalised eigenvalue problem
        
    B : (N, N) matrix
        RHS matrix of the generalised eigenvalue problem

    evals : (N, ) array
        Eigenvalue spectrum of the generalised eigenvalue problem
        
    vl : (N, N) array
        Left eigenvectors of the generlaised eigenvalue problem
        
    evecs : (N, N) array
        Right eigenvectors of the generalised eigenvalue problem
        
    suppress : boolean
        if False then output print statements 
        
    Returns
    -------
    
    evals : complex array
        Cleaned eigenvalue spectrum
    
    cs : complex float
        Most unstable eigenvalue of evals
    
    evecs : complex array
        Eigenvector corresponding to the most unstable eigenvalue
    
    """
    
    if suppress == False:
        print(f'\nTotal number of eigenvalues                  : {evals.shape[0]}')
    else:
        pass
    
    # Clean the eigenspectrum by checking (i) Finite eigenvalues (ii) Equality and (iii) Condition number ------------
    
    # (i)
    # Create a list of arguments of finite eigenvalues and retain only those eigenpairs which are finite
    finite_args = list(np.where(evals.real!=np.inf)[0])
    
    if suppress == False:
        print(f'Number of infinite eigenvalues               : {evals.shape[0]-len(finite_args)}')
    
    evals = evals[finite_args]; vl = vl[:, finite_args]; evecs = evecs[:, finite_args]
    
    # (ii)
    # For each eigenvalue/eigenvector pair, compare the LHS and RHS of the generalised eigenvalue problem and 
    # create a list of all those pairs which do satisfy the equality
    equality_check = [np.allclose(A@evecs[:, i], evals[i]*B@evecs[:, i]) for i in range(evecs.shape[1])] 
    args           = list(compress(range(len(equality_check)), equality_check))
    
    # Note that these print statements are not representative of the entire eigenvalue spectrum since
    # we have already neglected eigenvalues with an infinite real part
    if suppress == False:
        print(f'Number of eigenpairs not satisfying equality : {evals.shape[0]-len(args)}')

    # Retain only the eigenvalues/eigenvectors (and left eigenvectors) that satisfy this equality
    evals = evals[args]; vl = vl[:, args]; evecs = evecs[:, args]
    
    # (iii)
    # Calculate the condition number of the eigenvalues and create a list of all those eigenpairs which have 
    # a condition number less than the specified condition number tolerance (cond_num)
    eval_cond = abs(1/np.diag(np.dot(vl.T, evecs)))
    cond_args = list(np.where(eval_cond<1e5)[0])
    
    if suppress == False:
        print(f'Number of poorly conditioned eigenvalues     : {evals.shape[0]-len(cond_args)}')

    # Retain only the eigenvalue/eigenvectors (we do not need left eigenvectors anymore) that have a 
    # sufficiently small condition number
    evals = evals[cond_args]; evecs = evecs[:, cond_args]
    
    if suppress == False:
        print(f'Number of accurately calculated eigenvalues  : {evals.shape[0]}')
    
    # If evals is empty, then there are no finite well-conditioned eigenvalues
    # Output an error suggesting that we need to try a larger tolerance on the condition number
    if not len(evals):
        raise ValueError(f'There are no finite well-conditioned eigenvalues within the specified tolerance.'+ 
                         f'Try a larger condition number tolerance of at least {np.ceil(np.amin(eval_cond))}')
    
    return evals, evecs

def gep(ny, nz, k, mu, case, dim, Ri): 
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
        
    mu : float
        Non-dimensional meridional wavenumber
        
    case : str
        Specify the problem to solve 
        
    Returns
    -------
    
    evals : complex array
        Cleaned eigenvalue spectrum from the gep
    
    cs : complex
        Most unstable eigenvalue
    
    evecs : complex
        Most unstable (right) eigenvector
        
    dim : str
        The dimension of the linear stability problem (e.g. 2D = Two-dimensional)
    """
    
    ########################################################################################################################################################################################################
    # (I) Set up the domain
    ########################################################################################################################################################################################################
    
    # Using the grid resolution we separate into cases based on whether the problem is one or two dimensional
    if ny == 1 and nz > 1:
        dim = '1D_V'        # One-dimensional in the (V)ertical
        
    elif ny > 1 and nz == 1:
        dim = '1D_M'        # One-dimensional in the (M)eridional
        
    elif ny > 1 and nz > 1:
        dim = '2D'          # Two-dimensional
        
    else:
        raise ValueError(f'Grid resolution (ny,nz)=({ny},{nz}) is not a valid choice')
        
    if dim == '1D_M':
        if case == 'RK':
            d = 1e6; L = np.pi*d
            y = np.linspace(-L, L, ny); dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
            
        else:
            raise ValueError(f'{case} is not a valid case for a {dim} problem')    
    
    elif dim == '1D_V':
        if case == 'EADY' or case == 'STONE':
            L = 1
            z = np.linspace(0, L, nz); dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1] 
        
        else:
            raise ValueError(f'{case} is not a valid case for a {dim} problem')
    
    elif dim == '2D':        
        if case == 'RK':
            d = 1e6; L = np.pi*d
            D = 1000  
            
            y = np.linspace(-L, L, ny); z = np.linspace(-D, 0, nz) 
        
        elif case == 'EADY' or case == 'STONE':
            L = 1
            D = 1

            y = np.linspace(-L, L, ny);  z = np.linspace(0, D, nz)
            
        else:
            raise ValueError(f'{case} is not a valid case for a {dim} problem')

        dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
        dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]

        Y,Z         = np.meshgrid(y, z);         Y_full,Z_half = np.meshgrid(y, z_mid) 
        Y_mid,Z_mid = np.meshgrid(y_mid, z_mid); Y_half,Z_full = np.meshgrid(y_mid, z)
        
        
    ########################################################################################################################################################################################################
    # (II) Typical dimensional parameters 
    ########################################################################################################################################################################################################
    
    if dim == '1D_M':
        if case == 'RK':
            pass
            
    elif dim == '1D_V':
        if case == 'EADY' or case == 'STONE':
            # Typical values for a midlatitude atmosphere  
            f0  = 1.0e-4             # Coriolis frequency (s^{-1})
            H   = 10*1000.0          # Height of tropopause (m)
            N   = 1.0e-2             # Buoyancy frequency (s^{-1})
            V   = N*H/(np.sqrt(Ri))  # Velocity (ms^{-1})
            L   = V/f0               # Length Scale (m)
            eps = V/(f0*L)           # Rossby Number  
            S   = ((N*H)/(f0*L))**2  # Stratification Parameter
    
    elif dim == '2D':            
        if case == 'RK':
            beta   = 2.29e-11          # Meridional gradient of the Coriolis parameter (m^{-1}s^{-1})
            N2     = 8.883e-5;         # Buoyancy frequency (s^{-2}) (as in Proehl 96/98)
            N      = np.sqrt(N2)       #
            r0     = 1026              # Background density (kg m^{3})
            g      = 9.81              # Gravitational acceleration (ms^{-2})
            eps    = V/(beta*(L**2))   # Equatorial Rossby number 

        elif case == 'EADY' or case == 'STONE':
            # Typical values for a midlatitude atmosphere 
            f0  = 1.0e-4             # Coriolis frequency
            H   = 10*1000.0          # Height of tropopause (m)
            N   = 1.0e-2             # Buoyancy frequency
            V   = N*H/(np.sqrt(Ri))  # Velocity
            L   = V/f0               # Length Scale
            eps = V/(f0*L)           # Rossby Number  
            S   = ((N*H)/(f0*L))**2  # Stratification Parameter
        
    ########################################################################################################################################################################################################
    # (III) Specify the mean zonal flow and density profiles
    ########################################################################################################################################################################################################
    
    if dim == '1D_M':
        if case == 'RK':
            beta = 2.29e-11; V = (beta*(L**2))
            
            U  = (V/2)*(1+np.cos((y/d)));  U_mid = (V/2)*(1+np.cos((y_mid/d)))
            Uy = np.gradient(U,y);        Uy_mid = np.gradient(U_mid,y_mid)
    
    elif dim == '1D_V':
        if case == 'EADY' or case == 'STONE':
            U  = z;                U_mid = z_mid
            Uz = np.ones_like(z); Uz_mid = np.ones_like(z_mid)
            
            ry = Uz; rz = (-S/eps)*np.ones(nz)
    
    elif dim == '2D':        
        if case == 'EADY' or case == 'STONE':
            U_mid = Z_mid; U_fh = Z_half; U_hf = Z_full
            Uy_hf = np.zeros_like(U_hf); Uy_mid = np.zeros_like(U_mid)
            Uz_hf = np.ones_like(U_hf);  Uz_mid = np.ones_like(U_mid)
            
            ry_hf = Uz_hf; rz_hf = (-S/eps)*np.ones_like(U_hf)
            
        elif case == 'RK':
            beta = 2.29e-11; V = (beta*(L**2))
            
            U_mid = (V/2)*(1+np.cos(np.pi*Y_mid/L)); Uy_mid = np.gradient(U_mid, y_mid, axis=1); Uz_mid = np.gradient(U_mid, z_mid, axis=0)
            U_hf  = (V/2)*(1+np.cos(np.pi*Y_half/L)); Uz_hf = np.gradient(U_hf, z, axis=0)
            U_fh  = (V/2)*(1+np.cos(np.pi*Y_full/L)); 
            
            r_hf  = (beta*r0/g)*integrate.cumtrapz(Y_half*Uz_hf, y_mid, initial=0) - np.tile((N2*r0/g)*z, (len(y)-1, 1)).T
            ry_hf = (beta*r0/g)*Y_half*Uz_hf; rz_hf = np.gradient(r_hf, z, axis=0)
            
    ########################################################################################################################################################################################################
    # (IV) Non-dimensionalise the equations of motion for the Rayleigh-Kuo problem (EADY/STONE are already non-dimensional)
    ########################################################################################################################################################################################################
    
    if dim == '1D_M':
        if case == 'RK':
            y = y/L; dy = abs(y[1]-y[0]); y_mid = y_mid/L; 
            U = U/V; U_mid = U_mid/V; Uy = (L/V)*Uy; Uy_mid = (L/V)*Uy_mid
    
    elif dim == '2D':            
        if case == 'RK':
            y = y/L; y_mid = y_mid/L; dy = abs(y[1]-y[0])
            z = z/D; z_mid = z_mid/D; dz = abs(z[1]-z[0])

            U_hf  = U_hf/V; U_fh = U_fh/V

            U_mid = U_mid/V; Uy_mid = (L/V)*Uy_mid; Uz_mid = (D/V)*Uz_mid  

            ry_hf = (g*D/(r0*beta*L*V))*ry_hf; rz_hf = (g*(D**2)/(r0*beta*(L**2)*V))*rz_hf  

            Y_mid,Z_mid = np.meshgrid(y_mid,z_mid); Y_full,Z_half = np.meshgrid(y,z_mid)
            
        elif case == 'EADY' or case == 'STONE':
            pass
    
    # Form matrices from the discretised equation of motion ------------------------------------------
    # The names for each of the matrices is explained by the following examples:
    # ZLU = ((Z)onal momentum equation)((L)HS of the equation)((u) perturbation)
    # CRP = ((C)ontinuity equation)((R)HS of the equation)((p) perturbation)

    ########################################################################################################################################################################################################
    # (V) Build the zonal momentum equation
    ########################################################################################################################################################################################################
    
    if dim == '1D_M':
        if case == 'RK':
            ZLU = k * np.diag(U_mid) 
            
            ZLV0 = np.zeros((ny-1, ny))
            for i in range(ny-1):
                ZLV0[i][i]   = (y_mid[i]-Uy_mid[i])
                ZLV0[i][i+1] = (y_mid[i]-Uy_mid[i])
            ZLV = (1j/2)*ZLV0
            
            ZLP = k*np.eye(ny-1)

            ZRU = k*np.eye(ny-1)
            ZRV = np.zeros_like(ZLV)
            ZRP = np.zeros_like(ZLP) 
    
    elif dim == '1D_V':
        if case == 'EADY' or case == 'STONE':
            ZLU = eps*k*np.diag(U_mid)
            
            if case == 'EADY':
                ZLV1 = ZLP1 = ZRP1 = 0
                
            elif case == 'STONE':
                ZLV1 = tridiag((ry/rz)[1:-1], ((ry/rz)[:-1]+(ry/rz)[1:]), (ry/rz)[1:-1], [(ry/rz)[1], (ry/rz)[1], (ry/rz)[-2], (ry/rz)[-2]])
                ZLP1 = tridiag((U/rz)[1:-1] ,  -((U/rz)[:-1]-(U/rz)[1:]), -(U/rz)[1:-1], [ (U/rz)[1], -(U/rz)[1],  (U/rz)[-2], -(U/rz)[-2]])
                ZRP1 = tridiag((1/rz)[1:-1] ,  -((1/rz)[:-1]-(1/rz)[1:]), -(1/rz)[1:-1], [ (1/rz)[1], -(1/rz)[1],  (1/rz)[-2], -(1/rz)[-2]])
                
            ZLV = 1j*np.eye(nz-1) + (1j*eps/4)*ZLV1
            ZLP = k*np.eye(nz-1) - (k*eps/(2*dz))*ZLP1
            
            ZRU = eps*k*np.eye(nz-1)  
            ZRV = np.zeros_like(ZLV)
            ZRP = np.zeros_like(ZLP) - (k*eps/(2*dz))*ZRP1
                
    elif dim == '2D':
        ZLU = eps*k*np.diag(U_mid.flatten(order='F'))

        ZLV0 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
        if case == 'EADY' or case == 'STONE':
            for i in range((ny-1)*(nz-1)):                
                ZLV0[i, i]        = 1 # f plane
                ZLV0[i, i+(nz-1)] = 1 # f plane
        
        elif case == 'RK':
            for i in range((ny-1)*(nz-1)):                
                ZLV0[i, i]        = Y_mid.flatten(order='F')[i] # Equatorial beta plane                          
                ZLV0[i, i+(nz-1)] = Y_mid.flatten(order='F')[i] # Equatorial beta plane

        ZLV1 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
        for i in range((ny-1)*(nz-1)):                
            ZLV1[i, i]        = (Uy_mid.flatten(order='F'))[i]                            
            ZLV1[i, i+(nz-1)] = (Uy_mid.flatten(order='F'))[i] 

        ZLV2 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
        if case == 'EADY':
            pass
        
        elif case == 'RK' or case == 'STONE':
            for i in range(ny-1):                        
                ZLV2[i*(nz-1):(i+1)*(nz-1), i*(nz-1):(i+1)*(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]+(ry_hf/rz_hf)[1:, i]), (ry_hf/rz_hf)[1:-1, i], [(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 
                ZLV2[i*(nz-1):(i+1)*(nz-1), i*(nz-1)+(nz-1):(i+1)*(nz-1)+(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]+(ry_hf/rz_hf)[1:, i]), (ry_hf/rz_hf)[1:-1, i], [(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 

        ZLV2 = (Uz_mid.flatten(order='F')*(ZLV2.T)).T 

        ZLV = (1j/2)*ZLV0 - (1j*eps/2)*ZLV1 + (1j*eps/8)*ZLV2 

        ZLP0 = k*np.eye((ny-1)*(nz-1))
        
        if case == 'EADY':
            ZLP1 = np.zeros_like(ZLP0)

        elif case == 'RK' or case == 'STONE':
            ZLP1 = np.asarray([tridiag((U_hf/rz_hf)[1:-1, i], -((U_hf/rz_hf)[:-1, i]-(U_hf/rz_hf)[1:, i]), -(U_hf/rz_hf)[1:-1, i], [(U_hf/rz_hf)[1, i], -(U_hf/rz_hf)[1, i], (U_hf/rz_hf)[-2, i], -(U_hf/rz_hf)[-2, i]]) for i in range(ny-1)])
            ZLP1 = block_diag(*ZLP1) 
            ZLP1 = (Uz_mid.flatten(order='F')*(ZLP1.T)).T

        ZLP = ZLP0 - ((eps*k)/(2*dz))*ZLP1

        ZRU = k*np.eye((ny-1)*(nz-1))
        ZRV = np.zeros_like(ZLV)
        
        if case == 'EADY':
            ZRP1 = np.zeros_like(ZLP0)

        elif case == 'RK' or case == 'STONE':
            ZRP1 = np.asarray([tridiag((1/rz_hf)[1:-1, i], -((1/rz_hf)[:-1, i]-(1/rz_hf)[1:, i]), -(1/rz_hf)[1:-1, i], [(1/rz_hf)[1, i], -(1/rz_hf)[1, i], (1/rz_hf)[-2, i], -(1/rz_hf)[-2, i]]) for i in range(ny-1)])
            ZRP1 = block_diag(*ZRP1) 
            ZRP1 = (Uz_mid.flatten(order='F')*(ZRP1.T)).T

        ZRP = np.zeros_like(ZLP0) - ((eps*k)/(2*dz))*ZRP1

    ########################################################################################################################################################################################################
    # (VI) Build the meridional momentum equation
    ########################################################################################################################################################################################################
    
    if dim == '1D_M':
        if case == 'RK':
            MLU0 = np.zeros((ny, ny-1))
            for i in range(1, ny-1):
                MLU0[i][i-1] = y[i]
                MLU0[i][i]   = y[i]
            MLU = (-1j/2)*MLU0

            MLV = k*np.diag(U)

            MLP0= np.zeros((ny ,ny-1))
            for i in range(1, ny-1):
                MLP0[i][i-1] = 1
                MLP0[i][i]   = -1
            MLP = (1j/dy)*MLP0

            MRU = np.zeros_like(MLU)
            MRV = k*np.eye(ny)  
            MRP = np.zeros_like(MLP)
    
    elif dim == '1D_V':
        if case == 'EADY' or case == 'STONE':
            MLU = -1j*np.eye(nz-1)
            MLV = eps*k*np.diag(U_mid) 
            MLP = mu*np.eye(nz-1) 

            MRU = np.zeros_like(MLU)
            MRV = eps*k*np.eye(nz-1)
            MRP = np.zeros_like(MLP)

    elif dim == '2D':
        if case == 'EADY' or case == 'STONE':
            MLU0 = np.zeros((ny*(nz-1),ny*(nz-1))) 
            for i in range((ny-1)*(nz-1)):                 
                MLU0[i+(nz-1), i]        = 1 # f plane                   
                MLU0[i+(nz-1), i+(nz-1)] = 1 # f plane
            MLU = (-1j/2)*MLU0
                
        elif case == 'RK':
            MLU0 = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))
            for i in range((ny-2)*(nz-1)):                
                MLU0[i+(nz-1), i]        = Y_full.flatten(order='F')[i+(nz-1)] # Equatorial beta plane (beta=1 due to non-dimensionalisation)                 
                MLU0[i+(nz-1), i+(nz-1)] = Y_full.flatten(order='F')[i+(nz-1)] # Equatorial beta plane (beta=1 due to non-dimensionalisation)  
            MLU = (-1j/2)*MLU0
        
        MLV = eps*k*np.diag(U_fh.flatten(order='F'))
        
        if case == 'EADY' or case == 'STONE':
            MLP0 = np.zeros((ny*(nz-1),ny*(nz-1))) 
            for i in range((ny-1)*(nz-1)):                
                MLP0[i+(nz-1), i]         = 1                              
                MLP0[i+(nz-1), i+(nz-1)] = -1 
            MLP = (1j/dy)*MLP0
                
        elif case == 'RK':
            MLP0 = np.zeros((ny*(nz-1), (ny-1)*(nz-1))) 
            for i in range((ny-2)*(nz-1)):                
                MLP0[i+(nz-1), i]        = 1                              
                MLP0[i+(nz-1), i+(nz-1)] = -1 
            MLP = (1j/dy)*MLP0
        
        MRU = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))
        MRV = k*np.eye(ny*(nz-1)) 
        MRP = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))

    ########################################################################################################################################################################################################
    # (VII) Build the continuity equation
    ########################################################################################################################################################################################################
    
    if dim == '1D_M':
        if case == 'RK':
            CLU = k * np.eye(ny-1)

            CLV0 = np.zeros((ny-1, ny))
            for i in range(ny-1):
                CLV0[i][i]   = 1
                CLV0[i][i+1] = -1
            CLV = (1j/dy)*CLV0

            CLP = np.zeros(((ny-1), (ny-1))) 

            CRU = np.zeros_like(CLU)
            CRV = np.zeros_like(CLV)
            CRP = np.zeros_like(CLP)
    
    elif dim == '1D_V':
        if case == 'EADY' or case == 'STONE':
            CLU = k*np.eye(nz-1)
            
            CLV1 = tridiag(-(ry/rz)[1:-1], ((ry/rz)[1:]-(ry/rz)[:-1]), (ry/rz)[1:-1], [(ry/rz)[1], (ry/rz)[1], -(ry/rz)[-2], -(ry/rz)[-2]]) 
            CLV  = mu*np.eye(nz-1) + (1j/(2*dz))*CLV1
            
            CLP1 = tridiag((U/rz)[1:-1], -((U/rz)[:-1]+(U/rz)[1:]), (U/rz)[1:-1], [-(U/rz)[1], (U/rz)[1], (U/rz)[-2], -(U/rz)[-2]])
            CLP  = (k/(dz**2))*CLP1

            CRU = np.zeros_like(CLU) 
            CRV = np.zeros_like(CLV)
            
            CRP1 = tridiag((1/rz)[1:-1], -((1/rz)[:-1]+(1/rz)[1:]), (1/rz)[1:-1], [-(1/rz)[1], (1/rz)[1], (1/rz)[-2], -(1/rz)[-2]])
            CRP  = (k/(dz**2))*CRP1    
    
    elif dim == '2D':
        CLU = k*np.eye((ny-1)*(nz-1))

        CLV0 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
        for i in range((ny-1)*(nz-1)):               
            CLV0[i, i]        = 1                           
            CLV0[i, i+(nz-1)] = -1

        CLV1 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
        for i in range(ny-1):                        
            CLV1[i*(nz-1):(i+1)*(nz-1), i*(nz-1):(i+1)*(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]-(ry_hf/rz_hf)[1:, i]), -(ry_hf/rz_hf)[1:-1, i], [-(ry_hf/rz_hf)[1, i], -(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 
            CLV1[i*(nz-1):(i+1)*(nz-1), i*(nz-1)+(nz-1):(i+1)*(nz-1)+(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]-(ry_hf/rz_hf)[1:, i]), -(ry_hf/rz_hf)[1:-1, i], [-(ry_hf/rz_hf)[1, i], -(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 

        CLV = (1j/dy)*CLV0 - (1j/(4*dz))*CLV1 

        CLP_blocks = np.asarray([tridiag((U_hf/rz_hf)[1:-1, i], -((U_hf/rz_hf)[:-1, i]+(U_hf/rz_hf)[1:, i]), (U_hf/rz_hf)[1:-1, i], [-(U_hf/rz_hf)[1, i], (U_hf/rz_hf)[1, i], (U_hf/rz_hf)[-2, i], -(U_hf/rz_hf)[-2, i]]) for i in range(ny-1)])
        CLP = (k/(dz**2))*block_diag(*CLP_blocks) 

        CRU = np.zeros_like(CLU)
        CRV = np.zeros_like(CLV)

        CRP_blocks = np.asarray([tridiag((1/rz_hf)[1:-1, i], -((1/rz_hf)[:-1, i]+(1/rz_hf)[1:, i]), (1/rz_hf)[1:-1, i], [-(1/rz_hf)[1, i], (1/rz_hf)[1, i], (1/rz_hf)[-2, i], -(1/rz_hf)[-2, i]]) for i in range(ny-1)])
        CRP = (k/(dz**2))*block_diag(*CRP_blocks) 
        
    ########################################################################################################################################################################################################
    # (VIII) Apply boundary conditions
    ########################################################################################################################################################################################################
    
    if dim == '2D':
    
        # No-normal flow at the meridional walls of the domain
        if case == 'RK':
            MLV[:(nz-1),:(nz-1)]   = np.diag(np.ones(nz-1)) 
            MLV[-(nz-1):,-(nz-1):] = np.diag(np.ones(nz-1))

            MRV[:(nz-1),:]  = 0
            MRV[-(nz-1):,:] = 0
        
        # Periodic flow at the meridional walls of the domain
        if case == 'EADY' or case == 'STONE':
            ZLU = np.hstack([ZLU, np.zeros((ZLU.shape[0], (nz-1)))]) 
            ZLU = np.vstack([ZLU, np.zeros(((nz-1), ZLU.shape[1]))])

            ZLU[-(nz-1):,-(nz-1):] = np.diag(np.ones(nz-1)); 
            ZLU[-(nz-1):,:(nz-1)] = -np.diag(np.ones(nz-1))

            ZLV = np.vstack([ZLV, np.zeros(((nz-1), ZLV.shape[1]))]) 

            ZLP = np.hstack([ZLP, np.zeros((ZLP.shape[0], (nz-1)))])
            ZLP = np.vstack([ZLP, np.zeros(((nz-1), ZLP.shape[1]))])

            ZRU = np.hstack([ZRU, np.zeros((ZRU.shape[0], (nz-1)))])
            ZRU = np.vstack([ZRU, np.zeros(((nz-1), ZRU.shape[1]))])

            ZRV = np.zeros((ny*(nz-1), (ny*(nz-1)))) 

            ZRP = np.hstack([ZRP, np.zeros((ZRP.shape[0], (nz-1)))])
            ZRP = np.vstack([ZRP, np.zeros(((nz-1), ZRP.shape[1]))])

            MLV[:(nz-1),:]        = 0
            MLV[:(nz-1),:(nz-1)]  = np.diag(np.ones(nz-1))
            MLV[:(nz-1),-(nz-1):] = -np.diag(np.ones(nz-1))

            MRU = np.zeros((ny*(nz-1),ny*(nz-1)))
            MRP = np.zeros((ny*(nz-1),ny*(nz-1)))

            MRV[:(nz-1),:] = 0

            CLU = np.hstack([CLU, np.zeros((CLU.shape[0], (nz-1)))])
            CLU = np.vstack([CLU, np.zeros(((nz-1), CLU.shape[1]))])

            CLV = np.vstack([CLV, np.zeros(((nz-1), CLV.shape[1]))]) 

            CLP = np.hstack([CLP, np.zeros((CLP.shape[0], (nz-1)))])
            CLP = np.vstack([CLP, np.zeros(((nz-1), CLP.shape[1]))])

            CLP[-(nz-1):,-(nz-1):] = np.diag(np.ones(nz-1))
            CLP[-(nz-1):,:(nz-1)] = -np.diag(np.ones(nz-1))

            CRU = np.zeros((ny*(nz-1),ny*(nz-1)))
            CRV = np.zeros((ny*(nz-1),ny*(nz-1)))

            CRP = np.hstack([CRP, np.zeros((CRP.shape[0], (nz-1)))])
            CRP = np.vstack([CRP, np.zeros(((nz-1), CRP.shape[1]))])

    ########################################################################################################################################################################################################
    # (IX) Build the coefficient matrices A and B of the generalised eigenvalue problem
    ########################################################################################################################################################################################################
    
    # Form the LHS and RHS of each equation of motion as matrices
    ZLE = np.hstack([ZLU, ZLV, ZLP]); MLE = np.hstack([MLU, MLV, MLP]); CLE = np.hstack([CLU, CLV, CLP])
    ZRE = np.hstack([ZRU, ZRV, ZRP]); MRE = np.hstack([MRU, MRV, MRP]); CRE = np.hstack([CRU, CRV, CRP])
    
    # Build the coefficient matrices 
    A = np.vstack([ZLE, MLE, CLE]); B = np.vstack([ZRE, MRE, CRE])
    
    ########################################################################################################################################################################################################
    # (X)  Solve the generalised eigenvalue problem and clean the eigenvalue spectrum
    ########################################################################################################################################################################################################
    evals, vl, evecs = eig(A, B, left=True, right=True)
    evals, evecs     = clean_evals(A, B, evals, vl, evecs, True)
    evecs = evecs[:, np.argmax(evals.imag)]
    
    return evals, evecs
