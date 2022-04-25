import numpy as np
import os
import sys

import mean_fields

def on_each_grid(ny, nz, k, case, integration, stability):
    
    # Calculate the domain used for contour plotting---------------------------------------------------
    
    if case == 'NEMO':
        if integration == 'u-by430':
            lat = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/latitude_12.txt')
        else:
            lat = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/latitude_25.txt')
            
        depth = np.loadtxt(f'/home/rees/lsa/NEMO_mean_fields/depth.txt'); depth = -depth[::-1]
        
        L = abs(lat[0])*111.12*1000
        D = abs(depth[0])
        
    else:
        L = (10*111.32)*1000 # Meridional half-width of the domain (m)
        D = 1000             # Depth of the domain (m)

    y = np.linspace(-L, L, ny); z = np.linspace(-D, 0, nz) 

    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]

    Y,Z         = np.meshgrid(y, z);         Y_full,Z_half = np.meshgrid(y, z_mid) 
    Y_mid,Z_mid = np.meshgrid(y_mid, z_mid); Y_half,Z_full = np.meshgrid(y_mid, z)
    
    g, r0 = 9.81, 1026
    
    # Specify the filenames of the eigenvector that we aim to plot ----------------------------------
    
    if case == 'NEMO':
        fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
                f'/home/rees/lsa/eigenvectors/evecs_{case}_{integration}_{stability}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']    
    else:
        fname = [f'/home/rees/lsa/eigenvalues/evals_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt',
                f'/home/rees/lsa/eigenvectors/evecs_{case}_{ny:02}_{nz:02}_{str(int(k*1e8))}.txt']
    
    # If the eigenvector has not been previously calculated then we do so now
    if os.path.exists(fname[0]) and os.path.exists(fname[1]):
        print(f'Loading eigenvalues and eigenvectors...')
        evals = np.loadtxt(fname[0]).view(complex).reshape(-1) 
        evecs = np.loadtxt(fname[1]).view(complex).reshape(-1) 
        cs = evals[np.argmax(evals.imag)].imag
    else:
        raise ValueError(f'The specified files do not exist\n{fname[0]}\n{fname[1]}')  
       
    U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry,ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case, integration, stability)
       
    u = np.reshape(evecs[:(ny-1)*(nz-1)], (ny-1, nz-1)).T
    v = np.reshape(evecs[(ny-1)*(nz-1):(2*ny-1)*(nz-1)], (ny, nz-1)).T
    p = np.reshape(evecs[(2*ny-1)*(nz-1):], (ny-1, nz-1)).T     
            
    u_v = np.zeros((nz-1, ny), dtype=complex)
    for j in range(1,ny-1):
        for l in range(nz-1):
            u_v[l, j] = 0.5*(u[l, j] + u[l, j-1])

    u_w = np.zeros((nz, ny-1), dtype=complex)
    for j in range(ny-1):
        for l in range(1,nz-1):
            u_w[l, j] = 0.5*(u[l, j] + u[l-1, j])

    v_p = np.zeros((nz-1, ny-1), dtype=complex)
    for j in range(ny-1):
        for l in range(nz-1):
            v_p[l, j] = 0.5*(v[l, j+1] + v[l, j])

    v_w = np.zeros((nz, ny-1), dtype=complex)
    for j in range(ny-1):
        for l in range(1,nz-1):
            v_w[l, j] = 0.25*(v[l-1, j]+v[l-1,j+1]+v[l, j]+v[l, j+1])

    p_v = np.zeros((nz-1, ny), dtype=complex)
    for j in range(1,ny-1):
        for l in range(nz-1):
            p_v[l, j] = 0.5*(p[l, j] + p[l, j-1])

    p_w = np.zeros((nz, ny-1), dtype=complex)
    for j in range(ny-1):
        for l in range(1,nz-1):
            p_w[l, j] = 0.5*(p[l, j] + p[l-1, j])  
            
    # Calculate the density perturbation 
            
    dpdz_w = np.zeros((nz, ny-1), dtype=complex)
    for j in range(ny-1):
        for l in range(1,nz-1):
            dpdz_w[l, j] = 0.5*(dpdz[l, j] + dpdz[l-1, j])
            
    rho = -(1/g)*dpdz_w 
    
    #rho = np.zeros((nz, ny-1), dtype=complex)
    #for j in range(ny-1):
    #    for l in range(nz-2):
    #        rho[l, j] = (1j/(k*(U_hf[l,j]-cs)))*v_w[l, j]*ry_hf[l,j]+w[l,j]*rz_hf[l,j])        

    rho_v = np.zeros((nz-1, ny), dtype=complex)
    for j in range(1,ny-1):
        for l in range(nz-1):
            rho_v[l, j] = 0.25*(rho[l, j-1]+rho[l,j]+rho[l+1, j-1]+rho[l+1, j])

    rho_p = np.zeros((nz-1, ny-1), dtype=complex)
    for j in range(ny-1):
        for l in range(nz-1):
            rho_p[l, j] = 0.5*(rho[l+1, j]+rho[l, j])    
            
    # Calculate the vertical velocity perturbation        
    
    w = np.zeros((nz, ny-1), dtype=complex)
    for j in range(ny-1):
        for l in range(1,nz-1):
            w[l, j] = (1j*k/g)*(1/rz_hf[l, j])*(U_hf[l, j]-cs)*dpdz_w[l,j]-v_w[l, j]*(ry_hf/rz_hf)[l, j]

    w_v = np.zeros((nz-1, ny), dtype=complex)
    for j in range(1,ny-1):
        for l in range(nz-1):
            w_v[l, j] = 0.25*(w[l, j-1]+w[l,j]+w[l+1, j-1]+w[l+1, j])

    w_p = np.zeros((nz-1, ny-1), dtype=complex)
    for j in range(ny-1):
        for l in range(nz-1):
            w_p[l, j] = 0.5*(w[l+1, j]+w[l, j])

    return u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p 
