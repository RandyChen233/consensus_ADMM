import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import casadi as cs
import dpilqr
from time import perf_counter
import sys
from logger import setup_logger_admm

import logging
from pathlib import Path
import multiprocessing as mp
from os import getpid
import os
from time import strftime

from dynamics import *
import util
from multiprocessing import Process, Pipe


opts = {'error_on_fail':False}

"""To keep the code simple, peer-to-peer communication is not used"""

# def solve_distributed_rhc(ids, n_states, n_inputs, 
# 						  n_agents, x0, xr, T, 
# 						  radius, Q, R, Qf, ADMM_ITER, 
# 						  convex_problem = True, n_trial=None):
	
# 	n_dims = [3]*n_agents
# 	u_ref = np.array([0, 0, 0]*n_agents)
# 	# u_ref = np.zeros([n_inputs]*n_agents)
# 	x_dims = [n_states]*n_agents
# 	u_dims = [n_inputs]*n_agents
	
# 	nx = n_states*n_agents
# 	nu = n_inputs*n_agents
# 	X_full = np.zeros((0, nx))
# 	U_full = np.zeros((0, nu))
# 	X_full = np.r_[X_full, x0.reshape(1,-1)]
	
# 	distributed_mpc_iters =0
# 	solve_times_mean = []   
# 	solve_times_std = []
	
# 	x_curr = x0
# 	u_curr = np.zeros((nu, 1))
# 	obj_history = [np.inf]
# 	t = 0
# 	dt = 0.1

# 	SOVA_admm = True

# 	while not np.all(np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), \
# 													n_agents, n_states) <= 0.1)):
# 		#Determine communication graph based on the current state (position vector) of all agents:
# 		graph = util.define_inter_graph_threshold(x_curr, radius, x_dims, ids, n_dims)
		
# 		split_states_curr = util.split_graph(x_curr.T, x_dims, graph)
# 		split_states_final = util.split_graph(xr.T, x_dims, graph)
# 		split_inputs_curr = util.split_graph(u_curr.T, u_dims ,graph)
		
		
# 		X_dec = np.zeros((nx, 1))
# 		U_dec = np.zeros((nu, 1))
		
# 		solve_times_per_problem = []
		
# 		#Looping through each sub-problem
# 		#Each sub-problem is then solved via consensus-ADMM (agents only negotiates with its immediate one-hop neighbors)
# 		for (x_curr_i, xf_i, u_curr_i,  (prob, ids_) , place_holder) in zip(split_states_curr, 
# 																		split_states_final,
# 																		split_inputs_curr,
# 																		graph.items(),
# 																		range(len(graph))):
# 			n_agents_local = x_curr_i.size//n_states
# 			print(f'Current sub-problem has {n_agents_local} agents \n')
			
# 			try:
# 				x_trj_converged_i, u_trj_converged_i, _, iter_time_i, cost_i = solve_consensus(n_states, 
# 																		n_inputs, 
# 																		n_agents_local,
# 																		x_curr_i.reshape(-1,1), 
# 																		u_curr_i.reshape(-1,1),
# 																		xf_i.reshape(-1,1), 
# 																		T, radius, 
# 																		Q[:x_curr_i.size,:x_curr_i.size], 
# 																		R[:u_curr_i.size,:u_curr_i.size], 
# 																		Qf[:x_curr_i.size,:x_curr_i.size], 
# 																		ADMM_ITER,
# 																		convex_problem)
# 				solve_times_per_problem.append(iter_time_i)
				
# 			except EOFError or RuntimeError:
# 				print(f'Current MPC iteration is {distributed_mpc_iters}!')
# 				converged = False
# 				obj_trj = np.inf
# 				logging.info(
# 				f'{n_trial},'
# 				f'{n_agents},{t},{converged},'
# 				f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times_mean)},{np.mean(solve_times_std)},{ADMM_ITER},'
# 				f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
# 				)
# 				return X_full, U_full, obj_trj, np.mean(solve_times_mean), obj_history
				
# 			i_prob = ids_.index(prob)
			
# 			#Collecting solutions from different potential game sub-problems at current time step t:
# 			X_dec[place_holder * n_states : (place_holder + 1) * n_states, :] = x_trj_converged_i[
# 				1, i_prob * n_states : (i_prob + 1) * n_states
# 				].reshape(-1,1)
			
# 			U_dec[place_holder * n_inputs : (place_holder + 1) * n_inputs, :] = u_trj_converged_i[
# 				0, i_prob * n_inputs : (i_prob + 1) * n_inputs
# 				].reshape(-1,1)
		

# 		obj_curr = float(util.objective(X_dec, U_dec, u_ref, xr, Q, R, Qf))
# 		obj_history.append(obj_curr)
		
# 		solve_times_mean.append(np.mean(solve_times_per_problem))  
# 		solve_times_std.append(np.std(solve_times_per_problem))
# 		del solve_times_per_problem
		
# 		t += dt
# 		u_curr = U_dec
# 		x_curr = X_dec
#   		#x_curr = Ad @ x_curr + Bd @ u_curr
		
# 		X_full = np.r_[X_full, X_dec.reshape(1,-1)] #Collect the overall trajectory from the "god's eye view"
# 		U_full = np.r_[U_full, U_dec.reshape(1,-1)]
# 		distributed_mpc_iters += 1
		
# 		if distributed_mpc_iters > 50:
# 			print(f'Max iters reached; exiting MPC loops')
# 			converged = False
# 			break
		
# 	if np.all(dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
# 		converged = True
		
# 	print(f'Final distance to goal is {dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')    
	
	
# 	obj_trj = float(util.objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf) )
	
# 	logging.info(
# 		f'{n_trial},'
# 		f'{n_agents},{t},{converged},'
# 		f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times_mean)},{np.mean(solve_times_std)},{ADMM_ITER},'
# 		f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
# 		)
	
# 	return X_full, U_full, obj_trj, np.mean(solve_times_mean), obj_history

# setup_logger_admm()
def solve_admm_mpc(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, MAX_ITER, ADMM_ITER, n_trial=None):
	SOVA_admm = False
	nx = n_states*n_agents
	nu = n_inputs*n_agents
	X_full = np.zeros((0, nx))
	U_full = np.zeros((0, nu))
	X_full = np.r_[X_full, x0.reshape(1,-1)]
	u_ref = np.array([0, 0, 0]*n_agents)
	
	mpc_iter = 0
	obj_history = [np.inf]
	solve_times = []
	t = 0
	dt = 0.1
 
	nx = n_states*n_agents
	nu = n_inputs*n_agents
	N = n_agents

	r_min = 2.0*radius
	x_curr = x0
	u_curr = np.zeros((nu, 1))
	
	dt = 0.1
	Ad, Bd = linear_kinodynamics(dt,N)
	A_i = Ad[0:6,0:6]
	B_i = Bd[0:6,0:3]
	
	while not np.all(util.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states) <= 0.1):
		try:
		
			#Solve current MPC via consensus ADMM:
			x_trj_converged, u_trj_converged, _, admm_time=solve_consensus_nonlinear(Ad, Bd,
																			A_i,B_i,
                                                                   			x_curr,
																			u_curr,
                                                                    		xr, Q, R, Qf,
																			T, nx, nu,r_min,
                                                                      		N, ADMM_ITER, convex_problem = True,
																			n_trial = None)
			solve_times.append(admm_time)
			
			
		except EOFError or RuntimeError:
			admm_time = np.inf
			solve_times.append(admm_time)
			print('Error encountered in ADMM iterations !! Exiting...')
			converged = False
			obj_trj = np.inf
			logging.info(
			f'{n_trial},'
			f'{n_agents},{t},{converged},'
			f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times)},{np.std(solve_times)}, {MAX_ITER},'
			f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
			)
			return X_full, U_full, obj_trj, np.mean(solve_times), obj_history
			
		obj_history.append(float(util.objective(x_trj_converged.T, u_trj_converged.T, u_ref, xr, Q, R, Qf)))
		
		x_curr = x_trj_converged[1]
		u_curr = u_trj_converged[0]
		
		X_full = np.r_[X_full, x_curr.reshape(1,-1)]
		U_full = np.r_[U_full, u_curr.reshape(1,-1)]
		
		mpc_iter += 1
		t += dt
		if mpc_iter > MAX_ITER:
			print('Max MPC iters reached!Exiting MPC loops...')
			converged = False
			break

	print(f'Final distance to goal is {dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')
	
	if np.all(dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
		converged = True

	obj_trj = float(util.objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf))
	
	logging.info(
	f'{n_trial},'
	f'{n_agents},{t},{converged},'
	f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times)},{np.std(solve_times)}, {MAX_ITER},'
	f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
	)
	
	return X_full, U_full, obj_trj, np.mean(solve_times), obj_history


def solve_consensus(Ad, Bd,
                    A_i,B_i,
                    x_curr,
                    u_curr, 
                    xr, Q, R, Qf, 
                    T, nx, nu, r_min,
                    N, ADMM_ITER, convex_problem, n_trial):

	n_agents = N
	n_states = 6
	n_inputs = 3
	Q = Q[0:n_states,0:n_states]
	R = R[0:n_inputs,0:n_inputs]
	Qf = Qf[0:n_states,0:n_states]
	def run_worker(agent_id, pipe, x_curr):
		opti = Opti('conic')
  
		"Reference https://www.cvxpy.org/examples/applications/consensus_opt.html"
		xbar = opti.parameter((T+1)*nx + T*nu)
		opti.set_value(xbar, cs.GenDM_zeros((T+1)*nx + T*nu,1))   
		states = opti.variable((T+1)*nx + T* nu)
		u = opti.parameter((T+1)*nx + T*nu)
		opti.set_value(u, cs.GenDM_zeros((T+1)*nx + T*nu,1))
		X0 = opti.parameter(x_curr[:n_states].shape[0],1)

		rho = 0.5
		f = 0 		#local cost
		#Local quadratic tracking cost:
		for t in range(T):
			state_curr = states[:(T+1)*nx][t*nx:(t+1)*nx][agent_id*n_states:(agent_id+1)*n_states]
			state_ref = xr[agent_id*n_states:(agent_id+1)*n_states]
			for idx in range(n_states):
				f += (state_curr[idx] - state_ref[idx]) * Q[idx,idx] * (state_curr[idx] - state_ref[idx]) 
				# print(f.is_scalar())
			input_curr = states[(T+1)*nx:][t*nu:(t+1)*nu][agent_id*n_inputs:(agent_id+1)*n_inputs]
			for idu in range(n_inputs):
				f += (input_curr[idu]) * R[idu,idu] * (input_curr[idu])
				# print(f.is_scalar())
		
		#Local quadratic terminal cost:
		for idf in range(n_states):
			states_T = states[T*nx:(T+1)*nx][agent_id*n_states:(agent_id+1)*n_states]
			state_ref = xr[agent_id*n_states:(agent_id+1)*n_states]
			f += (states_T[idf] - state_ref[idf]) * Qf[idf,idf] * (states_T[idf] - state_ref[idf])
			# print(f.is_scalar())
     
		"""Reference: https://www.cvxpy.org/examples/applications/consensus_opt.html"""
		#Augmented cost :
		f += (rho/2)*sumsqr(states - xbar + u) 

		#Local constraints for current MPC:
		opti.subject_to(states[0:nx][agent_id*n_states:(agent_id+1)*n_states] == X0)
		for k in range(T):
			X_next = states[:(T+1)*nx][(k+1)*nx:(k+2)*nx][agent_id*n_states:(agent_id+1)*n_states]
			X_curr = states[:(T+1)*nx][k*nx:(k+1)*nx][agent_id*n_states:(agent_id+1)*n_states]
			U_curr = states[(T+1)*nx:][k*nu:(k+1)*nu][agent_id*n_inputs:(agent_id+1)*n_inputs]

			opti.subject_to(X_next==A_i @ X_curr + B_i @ U_curr) # close the gaps
			
			#Constrain the acceleration vector
			opti.subject_to(U_curr <= np.array([2, 2, 2]))
			opti.subject_to(np.array([-2, -2, -2]) <= U_curr)
			
			#Constrain velocity:
			for i in range(n_agents):
				opti.subject_to(X_curr[3:6] <= np.array([1.5, 1.5, 1.5]))
				opti.subject_to(np.array([-1.5, -1.5, -1.5]) <= X_curr[3:6])
	
			# Collision avoidance constraints via BVCs

			i = agent_id
			agent_i_trj = forward_pass(A_i,
										B_i,
										T,
										x_curr[i *n_states:(i +1)*n_states],
										u_curr[i *n_inputs:(i +1)*n_inputs])
			
			p_i_next = X_curr[:3]
			for j in range(n_agents):
				if j != i:
					# continue
					agent_j_trj = forward_pass( A_i,
												B_i,
												T,
												x_curr[j*n_states:(j+1)*n_states],
												u_curr[j*n_inputs:(j+1)*n_inputs])

					a_ij  =  (agent_i_trj[:,k][:3] -agent_j_trj[:,k][:3])/cs.norm_2(agent_i_trj[:,k][:3] -agent_j_trj[:,k][:3])
					b_ij = cs.dot(a_ij, (agent_i_trj[:,k][:3] + agent_j_trj[:,k][:3])/2) + r_min/2
					opti.subject_to(cs.dot(a_ij, p_i_next) >= b_ij )


		# ADMM loop
		iters = 0
		opti.solver("osqp",opts)

		opti.minimize(f)
		opti.set_value(X0,x_curr[agent_id*n_states:(agent_id+1)*n_states])
  
		while True:
			try:    
				sol = opti.solve()
				# """Checking sparsity of constraint jacobian:"""
				# plt.spy(sol.value(jacobian(d[f"opti_{agent_id}"].g,d[f"opti_{agent_id}"].x)))
				# plt.show()
				pipe.send(sol.value(states))
				opti.set_value(xbar, pipe.recv()) #receive the averaged result from the main process.
				opti.set_value(u, sol.value( u + states - xbar))
				pipe.send(sol.value( u + states - xbar))
				iters += 1
				logging.info(
					f'{agent_id},{n_trial},'
					f'{n_agents},{iters},'
					f'{sol.value(f)},'
					)
    
			except EOFError:
				print("Connection closed.")
				break
				 
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_worker, args=(i, remote, x_curr))]
		procs[-1].start()

	solution_list = []
	# x_bar_history = [np.zeros((nx, 1))] #Recording how the averaged value x_bar evolves
	iter = 0
	t0 = perf_counter()
	"""Execute ADMM loop in parallel:"""
	converged = False
	# for _ in range(ADMM_ITER):
	residual_list = [np.inf]
	while not converged:
		# Gather and average Y_i
		xbar = sum(pipe.recv() for pipe in pipes)/N
		solution_list.append(xbar)

		# Scatter xbar
		for pipe in pipes:
			pipe.send(xbar)
		iter += 1

		residual = np.linalg.norm(sum(pipe.recv() for pipe in pipes)/N)
		residual_list.append(residual)
		print(f'Current residual is {residual}')
  
		#Specify a dual convergence criterion (||r_{k+1}-r_{k}|| <= eps)
		if abs(residual_list[iter]-residual_list[iter-1]) <= 0.005 and iter >=10:
			converged = True
			print(f'Current ADMM updates converged at the {iter}th iter\n')
			break

		if iter > ADMM_ITER:
			print(f'Max ADMM iterations reached!\n')
			break
		
	admm_iter_time = perf_counter() - t0  
	[p.terminate() for p in procs]

	x_trj_converged = solution_list[-1][:(T+1)*nx].reshape((T+1,nx))
	u_trj_converged = solution_list[-1][(T+1)*nx:].reshape((T,nu))
	
	return x_trj_converged, u_trj_converged, solution_list, admm_iter_time



def solve_consensus_nonlinear(Ad, Bd,
                    A_i,B_i,
                    x_curr,
                    u_curr, 
                    xr, Q, R, Qf, 
                    T, nx, nu, r_min,
                    N, ADMM_ITER, convex_problem, n_trial):

	n_agents = N
	n_states = 6
	n_inputs = 3
	Q = Q[0:n_states,0:n_states]
	R = R[0:n_inputs,0:n_inputs]
	Qf = Qf[0:n_states,0:n_states]
	x_dims = [n_states]*n_agents
	n_dims = [3]*n_agents
	def run_worker(agent_id, pipe, x_curr):
		opti = Opti()
		"Reference https://www.cvxpy.org/examples/applications/consensus_opt.html"
		xbar = opti.parameter((T+1)*nx + T*nu)
		opti.set_value(xbar, cs.GenDM_zeros((T+1)*nx + T*nu,1))   
		states = opti.variable((T+1)*nx + T* nu)
		u = opti.parameter((T+1)*nx + T*nu)
		opti.set_value(u, cs.GenDM_zeros((T+1)*nx + T*nu,1))
		X0 = opti.parameter(x_curr[:n_states].shape[0],1)

		rho = 0.5
		f = 0 		#local cost
		#Local quadratic tracking cost:
		for t in range(T):
			state_curr = states[:(T+1)*nx][t*nx:(t+1)*nx][agent_id*n_states:(agent_id+1)*n_states]
			state_ref = xr[agent_id*n_states:(agent_id+1)*n_states]
			for idx in range(n_states):
				f += (state_curr[idx] - state_ref[idx]) * Q[idx,idx] * (state_curr[idx] - state_ref[idx]) 
				# print(f.is_scalar())
			input_curr = states[(T+1)*nx:][t*nu:(t+1)*nu][agent_id*n_inputs:(agent_id+1)*n_inputs]
			for idu in range(n_inputs):
				f += (input_curr[idu]) * R[idu,idu] * (input_curr[idu])
				# print(f.is_scalar())
		
		#Local quadratic terminal cost:
		for idf in range(n_states):
			states_T = states[T*nx:(T+1)*nx][agent_id*n_states:(agent_id+1)*n_states]
			state_ref = xr[agent_id*n_states:(agent_id+1)*n_states]
			f += (states_T[idf] - state_ref[idf]) * Qf[idf,idf] * (states_T[idf] - state_ref[idf])
			# print(f.is_scalar())
     
		"""Reference: https://www.cvxpy.org/examples/applications/consensus_opt.html"""
		#Augmented cost :
		f += (rho/2)*sumsqr(states - xbar + u) 

		#Local constraints for current MPC:
		opti.subject_to(states[0:nx][agent_id*n_states:(agent_id+1)*n_states] == X0)
		for k in range(T):
			X_next = states[:(T+1)*nx][(k+1)*nx:(k+2)*nx][agent_id*n_states:(agent_id+1)*n_states]
			X_curr = states[:(T+1)*nx][k*nx:(k+1)*nx][agent_id*n_states:(agent_id+1)*n_states]
			U_curr = states[(T+1)*nx:][k*nu:(k+1)*nu][agent_id*n_inputs:(agent_id+1)*n_inputs]

			opti.subject_to(X_next==A_i @ X_curr + B_i @ U_curr) # close the gaps
			
			# Constrain the acceleration vector
			opti.subject_to(U_curr <= np.array([2, 2, 2]))
			opti.subject_to(np.array([-2, -2, -2]) <= U_curr)
			
			# #Constrain velocity:
			# for i in range(n_agents):
			# 	opti.subject_to(X_curr[3:6] <= np.array([1.5, 1.5, 1.5]))
			# 	opti.subject_to(np.array([-1.5, -1.5, -1.5]) <= X_curr[3:6])
	
			# Collision avoidance constraints
			if agent_id == 0:
				p_i = states[:(T+1)*nx][(k+1)*nx:(k+2)*nx][agent_id*n_states:(agent_id+1)*n_states][:3]
				for j in range(n_agents):
					if j != agent_id:
						p_j = states[:(T+1)*nx][(k+1)*nx:(k+2)*nx][j*n_states:(j+1)*n_states][:3]
						# opti.subject_to(cs.norm_2(p_j-p_i) >= 0 )
						p_ij = p_j - p_i
						opti.subject_to(sqrt(p_ij[0]**2 + p_ij[1]**2 + p_ij[2]**2 + 0.001) >= 2*r_min)

			# TODO: This probably should not be enforced in the primal update directly....

		# ADMM loop
		iters = 0
		opti.solver("ipopt",opts)

		opti.minimize(f)
		opti.set_value(X0,x_curr[agent_id*n_states:(agent_id+1)*n_states])
  
		while True:
			try:    
				sol = opti.solve()
				# """Checking sparsity of constraint jacobian:"""
				# plt.spy(sol.value(jacobian(d[f"opti_{agent_id}"].g,d[f"opti_{agent_id}"].x)))
				# plt.show()
				pipe.send(sol.value(states))
				opti.set_value(xbar, pipe.recv()) #receive the averaged result from the main process.
				opti.set_value(u, sol.value( u + states - xbar))
				pipe.send(sol.value( u + states - xbar))
				iters += 1
				logging.info(
					f'{agent_id},{n_trial},'
					f'{n_agents},{iters},'
					f'{sol.value(f)},'
					)
    
			except EOFError:
				print("Connection closed.")
				break
				 
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_worker, args=(i, remote, x_curr))]
		procs[-1].start()

	solution_list = []
	# x_bar_history = [np.zeros((nx, 1))] #Recording how the averaged value x_bar evolves
	iter = 0
	t0 = perf_counter()
	"""Execute ADMM loop in parallel:"""
	converged = False
	# for _ in range(ADMM_ITER):
	residual_list = [np.inf]
	while not converged:
		# Gather and average Y_i
		xbar = sum(pipe.recv() for pipe in pipes)/N
		solution_list.append(xbar)

		# Scatter xbar
		for pipe in pipes:
			pipe.send(xbar)
		iter += 1

		residual = np.linalg.norm(sum(pipe.recv() for pipe in pipes)/N)
		residual_list.append(residual)
		print(f'Current residual is {residual}')
		#Specify a dual convergence criterion (||r_{k+1}-r_{k}|| <= eps)
		if abs(residual_list[iter]-residual_list[iter-1]) <= 0.005 and iter >=10:
			converged = True
			print(f'Current ADMM updates converged at the {iter}th iter\n')
			break

		if iter > ADMM_ITER:
			print(f'Max ADMM iterations reached!\n')
			break
		
	admm_iter_time = perf_counter() - t0  
	[p.terminate() for p in procs]

	x_trj_converged = solution_list[-1][:(T+1)*nx].reshape((T+1,nx))
	u_trj_converged = solution_list[-1][(T+1)*nx:].reshape((T,nu))
	
	return x_trj_converged, u_trj_converged, solution_list, admm_iter_time

