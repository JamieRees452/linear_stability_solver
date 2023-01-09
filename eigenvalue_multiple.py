import build_matrices
import solve_gep

def gep(ny, nz, k, case, integration, stability, assume, init_guess, values, tol_input):
    
    A, B = build_matrices.gep(ny, nz, k, case, integration, stability, assume)
    
    evals, evecs, maxdiffs = solve_gep.evals(A, B, init_guess, values, tol_input)
      
    return evals, evecs, maxdiffs
