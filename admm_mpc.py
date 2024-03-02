import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import casadi as cs
from time import perf_counter

from mpc import *
import logging
from dynamics import *
import util
from util import *
from multiprocessing import Process, Pipe


opts = {'error_on_fail':False}

"""To keep the code simple, peer-to-peer communication is not used"""
def solve_admm_mpc(ids, 
                   n_states, 
                   n_inputs, 
                   n_agents, 
                   x0, 
                   xr, 
                   T, 
                   radius, 
                   Q, 
                   R, Qf, MPC_ITER, ADMM_ITER, convex = False, potential_ADMM = True, n_trial=None):
	
	nx = n_states*n_agents
	nu = n_inputs*n_agents
	
	x_dims = [n_states]*n_agents
	u_dims = [n_inputs]*n_agents
	n_dims = [3]*n_agents

	X_full = np.zeros((0, nx))
	U_full = np.zeros((0, nu))
	X_full = np.r_[X_full, x0.reshape(1,-1)]
	u_ref = np.array([0, 0, 0]*n_agents)
	
	mpc_iter = 0
	obj_history = [np.inf]
	solve_times = []
	t = 0
	nx = n_states*n_agents
	nu = n_inputs*n_agents
	N = n_agents

	r_min = 2.0*radius
	x_curr = x0
	u_curr = np.zeros((nu, 1))
	
	dt = 0.1

	
	while not np.all(util.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states) < 0.1):
		X_dec = np.zeros((nx, 1))
		U_dec = np.zeros((nu, 1))
		try:

			#Solve current MPC via consensus ADMM:
			if convex:
				Ad, Bd = linear_kinodynamics(dt,N)
				A_i = Ad[0:6,0:6]
				B_i = Bd[0:6,0:3]
				state_curr, input_curr,  admm_time, obj_curr= solve_consensus(Ad, Bd,
															A_i,B_i,
															x_curr,
															u_curr,
															xr, Q, R, Qf,
															T, nx, nu,r_min,
															N, ADMM_ITER, mpc_iter, convex_problem = True,
															n_trial = n_trial)
			if potential_ADMM:
				graph = util.define_inter_graph_threshold(x_curr, radius, x_dims, ids, n_dims)
        
				split_states_initial = split_graph(x_curr.T, x_dims, graph)
				split_states_final = split_graph(xr.T, x_dims, graph)
				split_inputs_ref = split_graph(u_ref.reshape(-1, 1).T, u_dims, graph)

				solve_times_per_problem = []
				for (x0_i, xf_i, u_ref_i , (i_prob, ids_) , place_holder) in zip(split_states_initial, 
                                       split_states_final,
                                       split_inputs_ref,
                                       graph.items(),
                                       range(len(graph))):
					print(f'Current sub-problem has {x0_i.size//n_states} agents \n')
					N_i = x0_i.size//n_states
					Q_i, R_i, Qf_i = Q[:x0_i.size,:x0_i.size], R[:u_ref_i.size,:u_ref_i.size] ,Qf[:x0_i.size,:x0_i.size]

					state_curr, input_curr,  admm_time, obj_curr = solve_consensus_potential(
																						x0_i.reshape(-1,1),
																						xf_i.reshape(-1,1),
																						Q_i, 
																						R_i, 
																						Qf_i,
																						T, 
																						r_min,
																						N_i, 
																						ADMM_ITER, 
																						mpc_iter, 
																						convex_problem = False, 
																						n_trial = n_trial)
					solve_times_per_problem.append(admm_time)

					print(f'state_curr has shape {state_curr.shape}, input_curr has shape {input_curr.shape}')
     
					X_dec[place_holder * n_states : (place_holder + 1) * n_states, :] = state_curr[place_holder * n_states : (place_holder + 1) * n_states]
					U_dec[place_holder * n_inputs : (place_holder + 1) * n_inputs, :] = input_curr[place_holder * n_inputs : (place_holder + 1) * n_inputs]
     
			# 	state_curr, input_curr,  admm_time, obj_curr = solve_consensus_nonlinear(Ad, Bd,
			# 																	A_i,B_i,
			# 																	x_curr,
			# 																	u_curr,
			# 																	xr, Q, R, Qf,
			# 																	T, nx, nu,r_min,
			# 																	N, ADMM_ITER, mpc_iter, convex_problem = True,
			# 																	n_trial = n_trial)

			# solve_times.append(admm_time)
				
				
		except (EOFError, RuntimeError) as e:
			admm_time = np.inf
			solve_times.append(admm_time)
			print('Error encountered in ADMM iterations !! Exiting...')
			converged = False
			obj_trj = np.inf
			return X_full, U_full, np.mean(solve_times), obj_history
			# break
			
		obj_history.append(float(obj_curr))
  
		if potential_ADMM:
			X_full = np.r_[X_full, X_dec.reshape(1,-1)]
			U_full = np.r_[U_full, U_dec.reshape(1,-1)]
		else:
			X_full = np.r_[X_full, state_curr.reshape(1,-1)]
			U_full = np.r_[U_full, input_curr.reshape(1,-1)]
		
		mpc_iter += 1
		t += dt
  
		if mpc_iter > MPC_ITER:
			print('Max MPC iters reached !Exiting MPC loops...')
			converged = False
			break

		if abs(obj_history[mpc_iter] - obj_history[mpc_iter-1]) <=0.05:
			print('Problem objective can no longer decrease! Exiting MPC loops...')
			converged = False
			break


	print(f'Final distance to goal is {util.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states)}')
	
	if np.all(util.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states) <= 0.1):
		converged = True


	return X_full, U_full, np.mean(solve_times), obj_history


"""Consensus based linear-quadratic MPC: """

def solve_consensus(Ad, Bd,
					A_i,B_i,
					x_curr,
					u_curr, 
					xr, Q, R, Qf, 
					T, nx, nu, r_min,
					N, ADMM_ITER, mpc_iter, convex_problem, n_trial):
	convex = True
	n_agents = N
	n_states = 6
	n_inputs = 3
	Q = Q[0:n_states,0:n_states]
	R = R[0:n_inputs,0:n_inputs]
	Qf = Qf[0:n_states,0:n_states]
	def run_worker(agent_id, pipe, x_curr):
		# opti = Opti('conic')
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

			i = agent_id
		
			# Collision avoidance constraints via BVCs:
			agent_i_trj = forward_pass(A_i,
										B_i,
										T,
										x_curr[i *n_states:(i +1)*n_states],
										u_curr[i *n_inputs:(i +1)*n_inputs])
			
			p_i_next = X_curr[:3]
			for j in range(n_agents):
				if j != i:
					agent_j_trj = forward_pass( A_i,
												B_i,
												T,
												x_curr[j*n_states:(j+1)*n_states],
												u_curr[j*n_inputs:(j+1)*n_inputs])

					a_ij  =  (agent_i_trj[:,k][:3] -agent_j_trj[:,k][:3])/cs.norm_2(agent_i_trj[:,k][:3] -agent_j_trj[:,k][:3])
					b_ij = cs.dot(a_ij, (agent_i_trj[:,k][:3] + agent_j_trj[:,k][:3])/2) + r_min/2
					opti.subject_to(cs.dot(a_ij, p_i_next) >= b_ij )


		#Terminal input constraint for recursive feasibility:
		U_f = states[(T+1)*nx:][(T-1)*nu:T*nu][agent_id*n_inputs:(agent_id+1)*n_inputs]
		opti.subject_to(U_f[:3]== np.zeros(3))

		# ADMM loop
		iters = 0
		# opti.solver("osqp",opts)
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
				pipe.send(sol.value(states - xbar))
				pipe.send(sol.value(f-(rho/2)*sumsqr(sol.value(states - xbar + u)))) #This step is not part of the ADMM algorithm; only used for data logging!
				iters += 1
				# sol_prev = sol
				# if iters > 0:
					# opti.set_initial(sol_prev.value_variables())
			except EOFError:
				print("Connection closed.")
				# pass
				print('EOFError')
				break

			except RuntimeError:
				print('RuntimeError')
				break
				 
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_worker, args=(i, remote, x_curr))]
		procs[-1].start()

	# solution_list = []
	iter = 0
	t0 = perf_counter()
 
	"""Execute ADMM loop in parallel:"""
	converged = False
	residual_list = [np.inf] #dual residuals
	primal_residual = [np.inf] #primal residuals
	objective_val_list = []
	
	while not converged:
		iter += 1
		# Gather and average \theta^i
		x_all = [pipe.recv() for pipe in pipes]
		res = [np.linalg.norm(x_all[j] - x_all[i]) for i in range(len(x_all)) for j in range(i+1, len(x_all))]
		primal_residual.append(sum(res)/N) #Mean sqaured residuals 
				
		xbar = sum(x_all)/N
		# xbar = sum(pipe.recv() for pipe in pipes)/N
		# solution_list.append(xbar)
		# Scatter xbar
		for pipe in pipes:
			pipe.send(xbar)

		dual_res_all = [pipe.recv() for pipe in pipes]
		dual_res = [np.linalg.norm(dual_res_all[j]) for j in range(len(dual_res_all))]
		residual_list.append(sum(dual_res)/N) #Mean sqaured residuals 
		# print(f'Current residual is {residual}')
		objective_val = sum(pipe.recv() for pipe in pipes)
		objective_val_list.append(objective_val)	

		logging.info(
					f'{convex},{n_trial},'
					f'{n_agents},{iter},{mpc_iter},{objective_val},'
					f'{residual_list[iter]},{primal_residual[iter]},'
					)
		
		#Specify a dual convergence criterion (||r_{k+1}-r_{k}|| <= eps)
		# if abs(residual_list[iter]-residual_list[iter-1]) <= 0.005 and iter >=10:
		if abs(residual_list[iter]) <= 0.005 and primal_residual[iter] <= 0.005 and iter >=10:
			converged = True
			print(f'Current ADMM updates converged at the {iter}th iter\n')
			break

		if iter > ADMM_ITER:
			print(f'Max ADMM iterations reached!\n')
			break

	admm_iter_time = perf_counter() - t0  
	[p.terminate() for p in procs]
 
	state_trj_curr = [x_all[i][:(T+1)*nx].reshape((T+1, nx))[1,i*n_states:(i+1)*n_states] for i in range(N)]
	input_trj_curr = [x_all[i][(T+1)*nx:].reshape((T, nu))[0,i*n_inputs:(i+1)*n_inputs] for i in range(N)]

	state_curr = np.concatenate(state_trj_curr)
	input_curr = np.concatenate(input_trj_curr)

	return state_curr, input_curr,  admm_iter_time, objective_val_list[-1]


"""Consensus based nonlinear MPC"""

#Constraints are nonlinear:
def solve_consensus_nonlinear(Ad, Bd,
					A_i,B_i,
					x_curr,
					u_curr, 
					xr, Q, R, Qf, 
					T, nx, nu, r_min,
					N, ADMM_ITER, mpc_iter, convex_problem, n_trial):
	convex = False
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
   
			# opti.subject_to(X_next[:3] <= np.array([3.5, 3.5, 2.5]))
			# opti.subject_to(np.array([0, 0, 0]) <= X_next[:3])
			
			# #Constrain velocity:
			# for i in range(n_agents):
			# 	opti.subject_to(X_curr[3:6] <= np.array([1.5, 1.5, 1.5]))
			# 	opti.subject_to(np.array([-1.5, -1.5, -1.5]) <= X_curr[3:6])
	
			# Collision avoidance constraints
			p_i = states[:(T+1)*nx][(k+1)*nx:(k+2)*nx][agent_id*n_states:(agent_id+1)*n_states][:3]
			for j in range(n_agents):
				if j != agent_id:
					p_j = states[:(T+1)*nx][(k+1)*nx:(k+2)*nx][j*n_states:(j+1)*n_states][:3]
					p_ij = p_j - p_i
					opti.subject_to(sqrt(p_ij[0]**2 + p_ij[1]**2 + p_ij[2]**2 + 0.001) >= 2*r_min)

		U_f = states[(T+1)*nx:][(T-1)*nu:T*nu][agent_id*n_inputs:(agent_id+1)*n_inputs]
		opti.subject_to(U_f[:3]== np.zeros(3))
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
				pipe.send(sol.value(states - xbar))
				pipe.send(sol.value(f-(rho/2)*sumsqr(sol.value(states - xbar + u)))) #this step is not part of the algorithm; used solely for logging data !
				iters += 1
				
			except EOFError:
				print("Connection closed.")
				# pass
				print('EOFError')
				break

			except RuntimeError:
				print('RuntimeError')
				break
				 
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_worker, args=(i, remote, x_curr))]
		procs[-1].start()

	# solution_list = []
	iter = 0
	t0 = perf_counter()
	"""Execute ADMM loop in parallel:"""
	converged = False
	residual_list = [np.inf] #dual residuals
	primal_residual = [np.inf] #primal residuals
	objective_val_list = []
	while not converged:
		iter += 1
		# Gather and average \theta^i
		x_all = [pipe.recv() for pipe in pipes]
		res = [np.linalg.norm(x_all[j] - x_all[i]) for i in range(len(x_all)) for j in range(i+1, len(x_all))]
		primal_residual.append(sum(res)/N) #Mean sqaured residuals 
				
		xbar = sum(x_all)/N
		# xbar = sum(pipe.recv() for pipe in pipes)/N
		# solution_list.append(xbar)
		# Scatter xbar
		for pipe in pipes:
			pipe.send(xbar)

		dual_res_all = [pipe.recv() for pipe in pipes]
		dual_res = [np.linalg.norm(dual_res_all[j]) for j in range(len(dual_res_all))]
		residual_list.append(sum(dual_res)/N) #Mean sqaured residuals 

		objective_val = sum(pipe.recv() for pipe in pipes) #Global objective function (=sum of local objectives)
		objective_val_list.append(objective_val)	

		logging.info(
					f'{convex},{n_trial},'
					f'{n_agents},{iter},{mpc_iter},{objective_val},'
					f'{residual_list[iter]},{primal_residual[iter]},'
					)
		
		#Specify a dual convergence criterion (||r_{k+1}-r_{k}|| <= eps)
		# if abs(residual_list[iter]-residual_list[iter-1]) <= 0.005 and iter >=10:
		if abs(residual_list[iter]) <= 0.005 and primal_residual[iter] <= 0.005 and iter >=10:
			converged = True
			print(f'Current ADMM updates converged at the {iter}th iter\n')
			break

		if iter > ADMM_ITER:
			print(f'Max ADMM iterations reached!\n')
			break

	admm_iter_time = perf_counter() - t0  
	[p.terminate() for p in procs]
	# states = opti.variable((T+1)*nx + T* nu)
	state_trj_curr = [x_all[i][:(T+1)*nx].reshape((T+1, nx))[1,i*n_states:(i+1)*n_states] for i in range(N)]
	input_trj_curr = [x_all[i][(T+1)*nx:].reshape((T, nu))[0,i*n_inputs:(i+1)*n_inputs] for i in range(N)]

	state_curr = np.concatenate(state_trj_curr)
	input_curr = np.concatenate(input_trj_curr)
	# x_trj_converged = solution_list[-1][:(T+1)*nx].reshape((T+1,nx))
	# u_trj_converged = solution_list[-1][(T+1)*nx:].reshape((T,nu))
		
 
	return state_curr, input_curr,  admm_iter_time, objective_val_list[-1]

"""Consensus based potential game"""
def solve_consensus_potential(
					x_curr,
					xr, Q, R, Qf, 
					T, r_min,
					N, ADMM_ITER, 
					mpc_iter, 
					convex_problem, 
					n_trial):
	convex = False
	n_agents = N
	n_states = 12
	n_inputs = 4
	nx = n_agents * n_states
	nu = n_agents * n_inputs

	Q = Q[0:n_states,0:n_states]
	R = R[0:n_inputs,0:n_inputs]
	Qf = Qf[0:n_states,0:n_states]
	x_dims = [n_states]*n_agents
	n_dims = [3]*n_agents

	#If current neighborhood has only 1 agent, no consensus is needed.
	if n_agents == 1:
		state_curr, input_curr, solve_time_i, obj_val_i = solve_mpc_single_nonlinear(n_agents, x_curr, xr, T,  Q, R, Qf)
		
		return state_curr, input_curr ,solve_time_i, obj_val_i
	
	else:
		def run_worker(agent_id, pipe, x_curr):
			opti = Opti()
			"Reference https://www.cvxpy.org/examples/applications/consensus_opt.html"
			xbar = opti.parameter((T+1)*nx + T*nu)
			opti.set_value(xbar, cs.GenDM_zeros((T+1)*nx + T*nu,1))   
			states = opti.variable((T+1)*nx + T* nu)
			u = opti.parameter((T+1)*nx + T*nu)
			opti.set_value(u, cs.GenDM_zeros((T+1)*nx + T*nu,1))
			X0 = opti.parameter(x_curr[:n_states].shape[0], 1)
			dt = 0.1
			rho = 0.5
			cost_i = 0 		#initialize local cost for each agent i

			#Local quadratic tracking cost for each agent:
			for t in range(T):
				state_curr = states[:(T+1)*nx][t*nx:(t+1)*nx][agent_id*n_states:(agent_id+1)*n_states]
				state_ref = xr[agent_id*n_states:(agent_id+1)*n_states]
				for idx in range(n_states):
					cost_i += (state_curr[idx] - state_ref[idx]) * Q[idx,idx] * (state_curr[idx] - state_ref[idx]) 
					# print(f.is_scalar())
				input_curr = states[(T+1)*nx:][t*nu:(t+1)*nu][agent_id*n_inputs:(agent_id+1)*n_inputs]
				for idu in range(n_inputs):
					cost_i += (input_curr[idu]) * R[idu,idu] * (input_curr[idu])
				
			
			#Local quadratic terminal cost:
			for idf in range(n_states):
				states_T = states[T*nx:(T+1)*nx][agent_id*n_states:(agent_id+1)*n_states]
				state_ref = xr[agent_id*n_states:(agent_id+1)*n_states]
				cost_i += (states_T[idf] - state_ref[idf]) * Qf[idf,idf] * (states_T[idf] - state_ref[idf])
			
			#Local collision avoidance cost along the finite horizon:
			# coll_cost = 0 
			# for t in range(T):
			# 	distances = util.compute_pairwise_distance_nd_Sym(states[:(T+1)*nx][t*nx:(t+1)*nx], x_dims, n_dims)
			# 	for dist in distances:
			# 		coll_cost += fmin(0,(dist - 2.5 * r_min))**2 * 200
     
			# cost_i += coll_cost
   
			"""Reference: https://www.cvxpy.org/examples/applications/consensus_opt.html"""
			#Augmented cost :
			cost_i += (rho/2)*sumsqr(states - xbar + u) 

			#Local constraints for current MPC:
			opti.subject_to(states[0:nx][agent_id*n_states:(agent_id+1)*n_states] == X0)

			#Implement Runge Kutta's 4-th oder method for discretization:		
			f = generate_f(x_dims)
			# for k in range(T):
			# 	X_next = states[:(T+1)*nx][(k+1)*nx:(k+2)*nx]
			# 	X_curr = states[:(T+1)*nx][k*nx:(k+1)*nx]
			# 	U_curr = states[(T+1)*nx:][k*nu:(k+1)*nu]

			# 	k1 = f(X_curr,U_curr)
			# 	k2 = f(X_curr+dt/2*k1, U_curr)
			# 	k3 = f(X_curr+dt/2*k2, U_curr)
			# 	k4 = f(X_curr+dt*k3,   U_curr)

			# 	x_update = X_curr + dt/6*(k1+2*k2+2*k3+k4) 

			# 	opti.subject_to(X_next==x_update) # close the gap
				
				# Constrain the acceleration vector
				# opti.subject_to(U_curr <= np.array([2, 2, 2, 0.5*4*9.8] * n_agents)) #mass of the drone is 0.5 kg
				# opti.subject_to(np.array([-2, -2, -2, -0.5*4*9.8] * n_agents) <= U_curr) 
				
		
			# ADMM loop
			iters = 0
			opti.solver("ipopt",opts)

			opti.minimize(cost_i)
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
					pipe.send(sol.value(states - xbar))
					pipe.send(sol.value(cost_i-(rho/2)*sumsqr(sol.value(states - xbar + u)))) #this step is not part of the algorithm; used solely for logging data !
					iters += 1
					
				except EOFError:
					print("Connection closed.")
					# pass
					print('EOFError')
					break

				except RuntimeError:
					print('RuntimeError')
					break
					
		pipes = []
		procs = []
		for i in range(N):
			local, remote = Pipe()
			pipes += [local]
			procs += [Process(target=run_worker, args=(i, remote, x_curr))]
			procs[-1].start()

		# solution_list = []
		iter = 0
		t0 = perf_counter()
		"""Execute ADMM loop in parallel:"""
		converged = False
		residual_list = [np.inf] #dual residuals
		primal_residual = [np.inf] #primal residuals
		objective_val_list = []
		while not converged:
			iter += 1
			# Gather and average \theta^i
			x_all = [pipe.recv() for pipe in pipes]
			res = [np.linalg.norm(x_all[j] - x_all[i]) for i in range(len(x_all)) for j in range(i+1, len(x_all))]
			primal_residual.append(sum(res)/N) #Mean sqaured residuals 
					
			xbar = sum(x_all)/N
			# xbar = sum(pipe.recv() for pipe in pipes)/N
			# solution_list.append(xbar)
			# Scatter xbar
			for pipe in pipes:
				pipe.send(xbar)

			dual_res_all = [pipe.recv() for pipe in pipes]
			dual_res = [np.linalg.norm(dual_res_all[j]) for j in range(len(dual_res_all))]
			residual_list.append(sum(dual_res)/N) #Mean sqaured residuals 

			objective_val = sum(pipe.recv() for pipe in pipes) #Global objective function (=sum of local objectives)
			objective_val_list.append(objective_val)	

			logging.info(
						f'{convex},{n_trial},'
						f'{n_agents},{iter},{mpc_iter},{objective_val},'
						f'{residual_list[iter]},{primal_residual[iter]},'
						)
			
			#Specify a dual convergence criterion (||r_{k+1}-r_{k}|| <= eps)
			# if abs(residual_list[iter]-residual_list[iter-1]) <= 0.005 and iter >=10:
			if abs(residual_list[iter]) <= 0.005 and primal_residual[iter] <= 0.005 and iter >=10:
				converged = True
				print(f'Current ADMM updates converged at the {iter}th iter\n')
				break

			if iter > ADMM_ITER:
				print(f'Max ADMM iterations reached!\n')
				break

		admm_iter_time = perf_counter() - t0  
		[p.terminate() for p in procs]
		# states = opti.variable((T+1)*nx + T* nu)
		state_trj_curr = [x_all[i][:(T+1)*nx].reshape((T+1, nx))[1,i*n_states:(i+1)*n_states] for i in range(N)]
		input_trj_curr = [x_all[i][(T+1)*nx:].reshape((T, nu))[0,i*n_inputs:(i+1)*n_inputs] for i in range(N)]

		state_curr = np.concatenate(state_trj_curr)
		input_curr = np.concatenate(input_trj_curr)

		
	return state_curr, input_curr,  admm_iter_time, objective_val_list[-1]