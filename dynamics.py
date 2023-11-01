import numpy as np
from scipy.sparse import csc_array, lil_matrix
import casadi as cs

def linear_kinodynamics(dt,n_agent):
    #Decision vector is a = [a_x, a_y, a_z]
    #State vector is X = [p_x, p_y, p_z, v_x, v_y, v_z]
    #Sampling interval is dt
    A_tot = np.zeros((6*n_agent, 6*n_agent))
    B_tot = np.zeros((6*n_agent, 3*n_agent))
    A = np.array([[1, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0 , dt ,0],\
                           [0, 0, 1, 0, 0 , dt],\
                           [0, 0, 0, 1, 0 ,0],\
                           [0, 0, 0, 0, 1 ,0],\
                           [0, 0, 0, 0, 0, 1]])
    B = np.array([[dt**2/2, 0, 0],\
                [0, dt**2/2, 0],\
                [0, 0, dt**2/2],\
                [dt, 0, 0 ],\
                [0, dt , 0],\
                [0, 0, dt]])

    for i in range(n_agent):
        A_tot[i*6:(i+1)*6,i*6:(i+1)*6] = A
        B_tot[i*6:(i+1)*6,i*3:(i+1)*3] = B
        
    
    return A_tot, B_tot

# def linear_kinodynamics(dt,n_agents):
#     #The following is a double integrator model in 3D Euclidean space
    
#     #Input vector is u = [a_x, a_y, a_z]
#     #State vector is x = [p_x, p_y, p_z, v_x, v_y, v_z]
#     #Sampling interval is dt
    
#     #x(t+1) = Ax(t) + Bu(t), LTI model
    
#     A_tot = lil_matrix((6*n_agents, 6*n_agents))
#     B_tot = lil_matrix((6*n_agents, 3*n_agents))
    
#     A = csc_array([[1, 0, 0, dt, 0, 0],
#                            [0, 1, 0, 0 , dt ,0],\
#                            [0, 0, 1, 0, 0 , dt],\
#                            [0, 0, 0, 1, 0 ,0],\
#                            [0, 0, 0, 0, 1 ,0],\
#                            [0, 0, 0, 0, 0, 1]])
    
#     B = csc_array([[dt**2/2, 0, 0],\
#                 [0, dt**2/2, 0],\
#                 [0, 0, dt**2/2],\
#                 [dt, 0, 0 ],\
#                 [0, dt , 0],\
#                 [0, 0, dt]])
    
#     for i in range(n_agents):
#         A_tot[i*6:(i+1)*6,i*6:(i+1)*6] = A
#         B_tot[i*6:(i+1)*6,i*3:(i+1)*3] = B
        
#     return cs.DM_from_csc(A_tot), cs.DM_from_csc(B_tot)