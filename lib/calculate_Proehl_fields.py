"""
Calculate U, Uy and Uz for Proehls cases
"""

import numpy as np
import os

def mean_velocity(Y, Z, mean_flow): 
    """
    Calculate the mean fields for a specified Proehl case
                            
    Parameters
    ----------
    
    Y : int
        Number of meridional grid points
    
    Z : int
        Number of vertical grid points
        
    mean_flow : str
        Specify the coupled integration
        u-by430 = 1/12 deg; u-bx950 = 1/4 deg
        
    Returns
    -------
    
    U : 2D ndarray
        Mean zonal velocity
    
    Uy : 2D ndarray
        Meridional gradient of the mean zonal velocity
        
    Uz : 2D ndarray
        Vertical gradient of the mean zonal velocity
    """  
    if mean_flow == 'Proehl_1':
        y_center = 0; z_center = -500; maximum = 1.5; y_decay = 100*1000; z_decay = 100 
        
        U  = maximum*np.exp(-((Y-y_center)**2)/(2*(y_decay**2))-((Z-z_center)**2)/(2*(z_decay**2)))  
        Uy = -(maximum*(Y-y_center)/(y_decay**2))*np.exp(-((Y-y_center)**2)/(2*(y_decay**2))-((Z-z_center)**2)/(2*(z_decay**2)))  
        Uz = -(maximum*(Z-z_center)/(z_decay**2))*np.exp(-((Y-y_center)**2)/(2*(y_decay**2))-((Z-z_center)**2)/(2*(z_decay**2)))  
        
    elif mean_flow == 'Proehl_2':
        y_center = 0; z_center = -500; maximum = -1.5; y_decay = 100*1000; z_decay = 100 
        
        U  = maximum*np.exp(-((Y-y_center)**2)/(2*(y_decay**2))-((Z-z_center)**2)/(2*(z_decay**2)))  
        Uy = -(maximum*(Y-y_center)/(y_decay**2))*np.exp(-((Y-y_center)**2)/(2*(y_decay**2))-((Z-z_center)**2)/(2*(z_decay**2)))  
        Uz = -(maximum*(Z-z_center)/(z_decay**2))*np.exp(-((Y-y_center)**2)/(2*(y_decay**2))-((Z-z_center)**2)/(2*(z_decay**2)))  
        
    elif mean_flow == 'Proehl_3':            
        secs_y_center = -(4*111)*1000; secs_z_center = 0;  secs_maximum = -0.5; secs_decay_y = 100*1000; secs_decay_z = 100
        secn_y_center = (4*111)*1000;  secn_z_center = 0;  secn_maximum = -0.5; secn_decay_y = 100*1000; secn_decay_z = 100

        SECS = secs_maximum*np.exp(-(((Y-secs_y_center)**2)/(2*(secs_decay_y**2)))-(((Z-secs_z_center)**2)/(2*(secs_decay_z**2))))
        SECN = secn_maximum*np.exp(-(((Y-secn_y_center)**2)/(2*(secn_decay_y**2)))-(((Z-secn_z_center)**2)/(2*(secn_decay_z**2))))
        U    = SECS + SECN; Uy = np.gradient(U, Y[0,:], axis=1); Uz = np.gradient(U, Z[:,0], axis=0)
        
    elif mean_flow == 'Proehl_4':
        secs_y_center = -200*1000; secs_z_center = 0;  secs_maximum = -0.5; secs_decay_y = 100*1000; secs_decay_z = 200
        secn_y_center = 200*1000;  secn_z_center = 0;  secn_maximum = -0.5; secn_decay_y = 100*1000; secn_decay_z = 200

        SECS = secs_maximum*np.exp(-(((Y-secs_y_center)**2)/(2*(secs_decay_y**2)))-(((Z-secs_z_center)**2)/(2*(secs_decay_z**2))))
        SECN = secn_maximum*np.exp(-(((Y-secn_y_center)**2)/(2*(secn_decay_y**2)))-(((Z-secn_z_center)**2)/(2*(secn_decay_z**2))))
        U    = SECS + SECN; Uy = np.gradient(U, Y[0,:], axis=1); Uz = np.gradient(U, Z[:,0], axis=0)
        
    elif mean_flow == 'Proehl_5':
        secs_y_center = -200*1000; secs_z_center = 0;     secs_maximum = -0.5; secs_decay_y = 100*1000; secs_decay_z = 200 
        euc_y_center  = 0;          euc_z_center = -120;   euc_maximum = 1.0;   euc_decay_y = 125*1000;  euc_decay_z = 75
        secn_y_center = 200*1000;  secn_z_center = 0;     secn_maximum = -0.5; secn_decay_y = 100*1000; secn_decay_z = 200 

        SECS = secs_maximum*np.exp(-(((Y-secs_y_center)**2)/(2*(secs_decay_y**2)))-(((Z-secs_z_center)**2)/(2*(secs_decay_z**2))))
        EUC  = euc_maximum*np.exp(-(((Y-euc_y_center)**2)/(2*(euc_decay_y**2)))-(((Z-euc_z_center)**2)/(2*(euc_decay_z**2))))
        SECN = secn_maximum*np.exp(-(((Y-secn_y_center)**2)/(2*(secn_decay_y**2)))-(((Z-secn_z_center)**2)/(2*(secn_decay_z**2)))) 
        U    = SECS + EUC + SECN; Uy = np.gradient(U, Y[0,:], axis=1); Uz = np.gradient(U, Z[:,0], axis=0)
        
    elif mean_flow == 'Proehl_6':
        secs_y_center = -200*1000; secs_z_center = 0;     secs_maximum = -0.3; secs_decay_y = 100*1000; secs_decay_z = 200 
        euc_y_center  = 0;          euc_z_center = -120;   euc_maximum = 1.0;   euc_decay_y = 125*1000;  euc_decay_z = 75
        secn_y_center = 200*1000;  secn_z_center = 0;     secn_maximum = -0.7; secn_decay_y = 100*1000; secn_decay_z = 200 
        
        SECS = secs_maximum*np.exp(-(((Y-secs_y_center)**2)/(2*(secs_decay_y**2)))-(((Z-secs_z_center)**2)/(2*(secs_decay_z**2))))
        EUC  = euc_maximum*np.exp(-(((Y-euc_y_center)**2)/(2*(euc_decay_y**2)))-(((Z-euc_z_center)**2)/(2*(euc_decay_z**2))))
        SECN = secn_maximum*np.exp(-(((Y-secn_y_center)**2)/(2*(secn_decay_y**2)))-(((Z-secn_z_center)**2)/(2*(secn_decay_z**2)))) 
        U    = SECS + EUC + SECN; Uy = np.gradient(U, Y[0,:], axis=1); Uz = np.gradient(U, Z[:,0], axis=0)
        
    elif mean_flow == 'Proehl_7':
        secs_y_center = -300*1000; secs_z_center = 0;     secs_maximum = -0.3; secs_decay_y = 200*1000; secs_decay_z = 250 
        euc_y_center  = 0;          euc_z_center = -120;   euc_maximum = 1.0;   euc_decay_y = 125*1000; euc_decay_z  = 75
        secn_y_center = 200*1000;  secn_z_center = 0;     secn_maximum = -0.7; secn_decay_y = 100*1000; secn_decay_z = 150 

        SECS = secs_maximum*np.exp(-(((Y-secs_y_center)**2)/(2*(secs_decay_y**2)))-(((Z-secs_z_center)**2)/(2*(secs_decay_z**2))))
        EUC  = euc_maximum*np.exp(-(((Y-euc_y_center)**2)/(2*(euc_decay_y**2)))-(((Z-euc_z_center)**2)/(2*(euc_decay_z**2))))
        SECN = secn_maximum*np.exp(-(((Y-secn_y_center)**2)/(2*(secn_decay_y**2)))-(((Z-secn_z_center)**2)/(2*(secn_decay_z**2)))) 
        U    = SECS + EUC + SECN; Uy = np.gradient(U, Y[0,:], axis=1); Uz = np.gradient(U, Z[:,0], axis=0)
        
    elif mean_flow == 'Proehl_8':
        secs_y_center = -300*1000; secs_z_center = 0;     secs_maximum = -0.3; secs_decay_y = 200*1000; secs_decay_z = 250 
        euc_y_center  = 0;          euc_z_center = -120;   euc_maximum = 1.0;   euc_decay_y = 125*1000; euc_decay_z  = 75
        secn_y_center = 200*1000;  secn_z_center = 0;     secn_maximum = -0.7; secn_decay_y = 100*1000; secn_decay_z = 150 
        necc_y_center = 600*1000;  necc_z_center = 0;     necc_maximum = 0.5;  necc_decay_y = 150*1000; necc_decay_z = 250

        SECS = secs_maximum*np.exp(-(((Y-secs_y_center)**2)/(2*(secs_decay_y**2)))-(((Z-secs_z_center)**2)/(2*(secs_decay_z**2))))
        SECN = secn_maximum*np.exp(-(((Y-secn_y_center)**2)/(2*(secn_decay_y**2)))-(((Z-secn_z_center)**2)/(2*(secn_decay_z**2))))
        EUC  = euc_maximum*np.exp(-(((Y-euc_y_center)**2)/(2*(euc_decay_y**2)))-(((Z-euc_z_center)**2)/(2*(euc_decay_z**2))))
        NECC = necc_maximum*np.exp(-(((Y-necc_y_center)**2)/(2*(necc_decay_y**2)))-(((Z-euc_z_center)**2)/(2*(necc_decay_z**2))))
        U    = SECS + EUC + SECN + NECC; Uy = np.gradient(U, Y[0,:], axis=1); Uz = np.gradient(U, Z[:,0], axis=0)
        
    else:
        raise ValueError(f'{mean_flow} is not a valid option for a mean flow')
    
    return U, Uy, Uz
