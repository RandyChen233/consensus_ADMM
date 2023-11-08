import numpy as np
from scipy.sparse import csc_array, lil_matrix
import casadi as cs

def linear_kinodynamics(dt,n_agent):
    #Decision vector is a = [a_x, a_y, a_z]
    #State vector is X = [p_x, p_y, p_z, v_x, v_y, v_z]
    #Sampling interval is dt
    A_tot = np.zeros((6*n_agent, 6*n_agent))
    B_tot = np.zeros((6*n_agent, 3*n_agent))
    A=np.zeros((6,6))
    B=np.zeros((6,3))
    A[0:3,3:6]=np.eye(3)
    B[3:6,0:3]=np.eye(3)
    m = A.shape[0]
        
    """Reference: https://github.com/PKU-MACDLab/IMPC-DR/blob/main/3D/uav.py """
    m = 6
    h = dt
    A=np.dot(np.linalg.inv(np.eye(m)-h/2*A),(np.eye(m)+h/2*A))
    B=np.dot(np.linalg.inv(np.eye(m)-h/2*A)*h,B)
    
    for i in range(n_agent):
        A_tot[i*6:(i+1)*6,i*6:(i+1)*6] = A
        B_tot[i*6:(i+1)*6,i*3:(i+1)*3] = B
        
    
    return A_tot, B_tot
