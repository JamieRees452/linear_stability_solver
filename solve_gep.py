"""
Find the most unstable eigenvalue and corresponding eigenvectors of the square matrices A and B.

Solve A @ x[i] = c[i] * B @ x[i], the generalised eigenvalue problem for c[i] eigenvalues with 
corresponding eigenvectors x[i]
"""

import numpy               as np
from   scipy.sparse.linalg import eigs, ArpackNoConvergence
from   scipy               import sparse

def evals(A, B, init_guess):
    """
    Find the most unstable eigenvalue and eigenvector ofthe square matrices A and B.
    
    Solve the generalised eigenvalue problem, A @ x[i] = c[i] * B @ x[i], for c[i] eigenvalues
    and corresponding eigenvectors x[i]
    
    Parameters
    ----------
    A : complex matrix
         Left coefficient matrix of the generalised eigenvalue problem Ax=cBx
    
    B : complex matrix
         Right coefficient matrix of the generalised eigenvalue problem Ax=cBx
     
    init_guess : float
         Initial eigenvalue guess used to search a region with the Arnoldi method
         This guess corresponds to the real part of the phase speed 
    
    Returns 
    -------
    evals : complex
            Eigenvalue of the most unstable mode
    
    evecs : ndarray
            Corresponding eigenvector of the most unstable mode
    
    maxdiffs : float
            Error of the eigenvalue/eigenvector pair
    """
    try:
        evals, evecs = eigs(A, k=1, M=B, which='SI', sigma=init_guess, v0=np.ones(A.shape[0]))
    except ArpackNoConvergence as err:
        print(f'ARPACK failed to converge at wavenumber {k}. Eigenvalue at the final iteration is {err.eigenvalues}.\n')
    
    diffs = A.dot(evecs)-B.dot(evecs)*evals
    maxdiffs = np.linalg.norm(diffs, axis=0, ord=np.inf)
    
    evecs = evecs[:, np.argmax(evals.imag)]
    
    return evals, evecs, maxdiffs
