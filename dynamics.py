import numpy as np

def linear_kinodynamics(dt):
    #The following is a double integrator model in 3D Euclidean space
    
    #Input vector is u = [a_x, a_y, a_z]
    #State vector is x = [p_x, p_y, p_z, v_x, v_y, v_z]
    #Sampling interval is dt
    
    #x(t+1) = Ax(t) + Bu(t), LTI model
    
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

        
    return A, B