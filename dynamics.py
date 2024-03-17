import numpy as np
from scipy.sparse import csc_array, lil_matrix
import casadi as cs

"""For linear kinodynamics:"""

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


def forward_pass(A, B, horizon, x_curr, u_curr):
	
	x_prev = x_curr
	
	x_rollout = np.zeros((x_curr.shape[0], horizon+1))
	x_rollout[:,0] = x_prev.flatten()
	
	for t in range(1,horizon+1):
		x_next = A@x_prev + B@u_curr
		x_prev = x_next
		x_rollout[:,t] = x_prev.flatten()
	
	return x_rollout   


"""For 12-DOF nonlinear quadrotor dynamics (for single drone):"""

def generate_f_12DOF(x, u):

	# # NOTE: Assume homogeneity of agents.
	# n_agents = len(x_dims_local) #e.g., [12,12,12]
	# n_states = x_dims_local[0] # no. of states for each drone
	# n_controls = 4 #no. of control inputs
	
	# def f(x, u):
		
		# x_dot = cs.MX.zeros(x.numel())

	# x_dot = cs.MX.zeros(n_states)

	# for i_agent in range(n_agents):
	# 	i_xstart = i_agent * n_states
	# 	i_ustart = i_agent * n_controls

	#12-DOF quadrotor model (constant parameters assumed already), see derivation in notebooks/DeriveEOM.ipynb
	# x_dot[i_xstart:i_xstart + n_states] = cs.vertcat(
	# 	x[i_xstart + 6: i_xstart + 9],
	# 	(x[i_xstart + 9] * cs.cos(x[i_xstart+5]) - x[i_xstart + 10]*cs.sin(x[i_xstart+5]))/cs.cos(x[i_xstart+4]),
	# 	x[i_xstart + 9] * cs.sin(x[i_xstart+5]) + x[i_xstart + 10]*cs.cos(x[i_xstart+5]),
	# 	-x[i_xstart + 9] * cs.cos(x[i_xstart+5]) * cs.tan(x[i_xstart+4]) + x[i_xstart + 10]*cs.sin(x[i_xstart+5])*cs.tan(x[i_xstart+4]) + x[i_xstart+11], 
	# 	2*u[i_ustart+3] * cs.sin(x[i_xstart+4]), 
	# 	-2*u[i_ustart+3] * cs.sin(x[i_xstart+3]) * cs.cos(x[i_xstart+4]), 
	# 	2*u[i_ustart+3] * cs.cos(x[i_xstart+3]) * cs.cos(x[i_xstart+4]) - 981/100, 
	# 	10000*u[i_ustart]/23 - 17*x[i_xstart+10] * x[i_xstart+11]/23, 
	# 	10000*u[i_ustart+1]/23 + 17*x[i_xstart+9] * x[i_xstart+11]/23, 
	# 	250*u[i_ustart+2],
	# 	)
 
	p_x = x[0]
	p_y = x[1]
	p_z = x[2]
	phi = x[3]
	theta = x[4]
	psi = x[5]
	v_x = x[6]
	v_y = x[7]
	v_z = x[8]
	w_x = x[9]
	w_y = x[10]
	w_z = x[11]
	tau_x = u[0]
	tau_y = u[1]
	tau_z = u[2]
	f_z = u[3]

	x_dot = cs.vertcat(
				v_x,
				v_y,
				v_z,
				(w_x*cs.cos(psi) - w_y*cs.sin(psi))/cs.cos(theta),
				w_x*cs.sin(psi) + w_y*cs.cos(psi),
				-w_x*cs.cos(psi)*cs.tan(theta) + w_y*cs.sin(psi)*cs.tan(theta) + w_z, 
				2*f_z*cs.sin(theta), 
				-2*f_z*cs.sin(phi)*cs.cos(theta), 
				2*f_z*cs.cos(phi)*cs.cos(theta) - 981/100, 
				10000*tau_x/23 - 17*w_y*w_z/23, 
				10000*tau_y/23 + 17*w_x*w_z/23, 
				250*tau_z,
				)
			
	return x_dot
	
	# return f


# """"For simplified 6 DOF quadrotor model"""
def generate_f_6DOF_single(x, u):
	g = 9.8
	# NOTE: Assume homogeneity of agents.
	px = x[0]
	py = x[1]
	pz = x[2]
	vx = x[3]
	vy = x[4]
	vz = x[5]
	theta = u[0]
	phi = u[1]
	tau = u[2]
	g = 9.81
 
	x_dot = cs.vertcat(
		vx,
		vy,
		vz,
		g*cs.tan(theta),
		-g*cs.tan(phi),
		tau-g,
		)
		
	return x_dot



def generate_f(x_dims_local):
    g = 9.8
    # NOTE: Assume homogeneity of agents.
    n_agents = len(x_dims_local)
    n_states = x_dims_local[0]
    n_controls = 3
    
    def f(x, u):
        x_dot = cs.MX.zeros(x.numel())
        for i_agent in range(n_agents):
            i_xstart = i_agent * n_states
            i_ustart = i_agent * n_controls
            x_dot[i_xstart:i_xstart + n_states] = cs.vertcat(
                x[i_xstart + 3: i_xstart + 6],
                g*cs.tan(u[i_ustart]), -g*cs.tan(u[i_ustart+1]), u[i_ustart+2] - g
                )
            
        return x_dot
    
    return f