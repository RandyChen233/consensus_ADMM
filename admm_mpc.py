import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import casadi as cs
import dpilqr
from time import perf_counter
import sys

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


def solve_distributed_rhc(ids, n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, ADMM_ITER, convex_problem = True, n_trial=None):
    
    n_dims = [3]*n_agents
    u_ref = np.array([0, 0, 0]*n_agents)
    # u_ref = np.zeros([n_inputs]*n_agents)
    x_dims = [n_states]*n_agents
    u_dims = [n_inputs]*n_agents
    
    nx = n_states*n_agents
    nu = n_inputs*n_agents
    X_full = np.zeros((0, nx))
    U_full = np.zeros((0, nu))
    X_full = np.r_[X_full, x0.reshape(1,-1)]
    
    distributed_mpc_iters =0
    solve_times_mean = []
    solve_times_std = []
    
    x_curr = x0
    obj_history = [np.inf]
    t = 0
    dt = 0.1

    SOVA_admm = True
    
    # t_kill = 5*n_agents * T * dt
    while not np.all(np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), \
                                                    n_agents, n_states, 3) <= 0.1)):
        # rel_dists = util.compute_pairwise_distance_nd_Sym(x0,x_dims,n_dims)
        graph = util.define_inter_graph_threshold(x_curr, radius, x_dims, ids, n_dims)
        
        split_states_curr = util.split_graph(x_curr.T, x_dims, graph)
        split_states_final = util.split_graph(xr.T, x_dims, graph)
        split_inputs_ref = util.split_graph(u_ref.reshape(-1, 1).T, u_dims, graph)
        
        X_dec = np.zeros((nx, 1))
        U_dec = np.zeros((nu, 1))
        
        solve_times_per_problem = []
        
        for (x_curr_i, xf_i, u_ref_i , (prob, ids_) , place_holder) in zip(split_states_curr, 
                                       split_states_final,
                                       split_inputs_ref,
                                       graph.items(),
                                       range(len(graph))):
            print(f'Current sub-problem has {x_curr_i.size//n_states} agents \n')
            
            try:
                x_trj_converged_i, u_trj_converged_i, _, iter_time_i = solve_consensus(n_states, \
                                                                        n_inputs, \
                                                                        x_curr_i.size//n_states,\
                                                                        x_curr_i.reshape(-1,1), \
                                                                        xf_i.reshape(-1,1), \
                                                                        T, radius, \
                                                                        Q[:x_curr_i.size,:x_curr_i.size], \
                                                                        R[:u_ref_i.size,:u_ref_i.size], \
                                                                        Qf[:x_curr_i.size,:x_curr_i.size], \
                                                                        ADMM_ITER,
                                                                        convex_problem)
                solve_times_per_problem.append(iter_time_i)
                
            except EOFError or RuntimeError:
                converged = False
                obj_trj = np.inf
                logging.info(
                f'{n_trial},'
                f'{n_agents},{t},{converged},'
                f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times_mean)},{np.mean(solve_times_std)},{ADMM_ITER},'
                f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
                )
                return X_full, U_full, obj_trj, np.mean(solve_times_mean), obj_history
                
            i_prob = ids_.index(prob)
            
            #Collecting solutions from different potential game sub-problems at current time step K:
            X_dec[place_holder * n_states : (place_holder + 1) * n_states, :] = x_trj_converged_i[
                1, i_prob * n_states : (i_prob + 1) * n_states
                ].reshape(-1,1)
            
            U_dec[place_holder * n_inputs : (place_holder + 1) * n_inputs, :] = u_trj_converged_i[
                0, i_prob * n_inputs : (i_prob + 1) * n_inputs
                ].reshape(-1,1)

        obj_curr = float(util.objective(X_dec, U_dec, u_ref, xr, Q, R, Qf))
        obj_history.append(obj_curr)
        
        solve_times_mean.append(np.mean(solve_times_per_problem))  
        solve_times_std.append(np.std(solve_times_per_problem))
        
        t += dt
        x_curr = X_dec
        
        X_full = np.r_[X_full, X_dec.reshape(1,-1)]
        U_full = np.r_[U_full, U_dec.reshape(1,-1)]
        
        distributed_mpc_iters += 1
        
        if distributed_mpc_iters > 50:
            print(f'Max iters reached; exiting MPC loops')
            converged = False
            break
        
    if np.all(dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        converged = True
        
    print(f'Final distance to goal is {dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')    
    
    
    obj_trj = float(util.objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf) )
    
    logging.info(
        f'{n_trial},'
        f'{n_agents},{t},{converged},'
        f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times_mean)},{np.mean(solve_times_std)},{ADMM_ITER},'
        f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
        )
    
    return X_full, U_full, obj_trj, np.mean(solve_times_mean), obj_history




# def solve_admm_mpc(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, MAX_ITER, n_trial=None):
#     SOVA_admm = False
#     nx = n_states*n_agents
#     nu = n_inputs*n_agents
#     X_full = np.zeros((0, nx))
#     U_full = np.zeros((0, nu))
#     X_full = np.r_[X_full, x0.reshape(1,-1)]
#     u_ref = np.array([0, 0, 0]*n_agents)
#     # u_ref = np.array([0, 0, 9.8]*n_agents)
    
#     x_curr = x0
#     mpc_iter = 0
#     obj_history = [np.inf]
#     solve_times = []
#     t = 0
#     dt = 0.1

#     while not np.all(dpilqr.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
        
#         try:
#             x_trj_converged, u_trj_converged, _, admm_time = solve_consensus(n_states, n_inputs, n_agents, x_curr, \
#                                                                  xr, T, radius, Q, R, Qf, MAX_ITER)
            
#             solve_times.append(admm_time)
            
#         except EOFError or RuntimeError:
#             admm_time = np.inf
#             solve_times.append(admm_time)
#             print('Error encountered in solve_iteration!! Exiting...')
#             converged = False
#             obj_trj = np.inf
#             logging.info(
#             f'{n_trial},'
#             f'{n_agents},{t},{converged},'
#             f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times)},{np.std(solve_times)}, {MAX_ITER},'
#             f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
#             )
#             return X_full, U_full, obj_trj, np.mean(solve_times), obj_history
            
#         obj_history.append(float(util.objective(x_trj_converged.T, u_trj_converged.T, u_ref, xr, Q, R, Qf)))
        
#         x_curr = x_trj_converged[1]
#         u_curr = u_trj_converged[0]
        
#         X_full = np.r_[X_full, x_curr.reshape(1,-1)]
#         U_full = np.r_[U_full, u_curr.reshape(1,-1)]
        
#         mpc_iter += 1
#         t += dt
#         if mpc_iter > 35:
#             print('Max MPC iters reached!Exiting MPC loops...')
#             converged = False
#             break

#     print(f'Final distance to goal is {dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)}')
    
#     if np.all(dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3) <= 0.1):
#         converged = True

#     obj_trj = float(util.objective(X_full.T, U_full.T, u_ref, xr, Q, R, Qf))
    
#     logging.info(
#     f'{n_trial},'
#     f'{n_agents},{t},{converged},'
#     f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times)},{np.std(solve_times)}, {MAX_ITER},'
#     f'{dpilqr.distance_to_goal(X_full[-1].flatten(), xr.flatten(), n_agents, n_states, 3)},'
#     )
    
    
#     return X_full, U_full, obj_trj, np.mean(solve_times), obj_history


def solve_consensus(n_states, n_inputs, n_agents, 
                    x0, xr, T, radius, Q, R, Qf, 
                    ADMM_ITER = 10, convex_problem=True):
    """Define constants"""
    #pos_neighbors: positions of neighboring agents computed at the previous time step 
    
    nx = n_states*n_agents
    nu = n_inputs*n_agents
    N = n_agents
    x_dims = [n_states]*n_agents
    n_dims = [3]*n_agents
    r_min = 1.8*radius
    
    """Creating empty dicts to hold Casadi variables for each sub-problem
    Note: our ADMM algorithm is still synchronous within each sub-problem, but since
    each sub-problem has different sizes at each time step t, we choose to iterate over each
    sub-problem in a sequential fashion.
    """
    f_list = {}
    d = {} 
    states = {}
    for id in range(N):
        #Initialize Opti() class:
        if convex_problem:
        
            d["opti_{0}".format(id)] = Opti('conic')
        
        else:
            d["opti_{0}".format(id)] = Opti()
        
        #Augmented ADMM state : Y = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        states["Y_{0}".format(id)] = d[f"opti_{id}"].variable((T+1)*nx + T* nu)
        cost = 0

        #Quadratic tracking cost:
        for t in range(T):
            for idx in range(nx):
                state_curr = states[f"Y_{id}"][:(T+1)*nx][t*nx:(t+1)*nx][idx]
                state_ref = xr[idx]
                cost += (state_curr - state_ref) * \
                            Q[idx,idx] * \
                            (state_curr - state_ref) 
            for idu in range(nu):
                cost += (states[f"Y_{id}"][(T+1)*nx:][t*nu:(t+1)*nu][idu]) *  \
                            R[idu,idu] * \
                            (states[f"Y_{id}"][(T+1)*nx:][t*nu:(t+1)*nu][idu])
        
        #Quadratic terminal cost:
        for idf in range(nx):
            cost += (states[f"Y_{id}"][:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf]) * \
                    Qf[idf,idf] * \
                    (states[f"Y_{id}"][:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf])

        f_list["cost_{0}".format(id)] = cost
        

    def run_worker(agent_id, cost, pipe):
        "Reference https://www.cvxpy.org/examples/applications/consensus_opt.html"
        
        xbar = d[f"opti_{agent_id}"].parameter((T+1)*nx + T*nu)
        d[f"opti_{agent_id}"].set_value(xbar, cs.GenDM_zeros((T+1)*nx + T*nu,1))    
        
        u = d[f"opti_{agent_id}"].parameter((T+1)*nx + T*nu)
        d[f"opti_{agent_id}"].set_value(u, cs.GenDM_zeros((T+1)*nx + T*nu,1))

        # rho = 1.0
        rho = 0.1
        cost += (rho/2)*sumsqr(states[f"Y_{agent_id}"] - xbar + u)
        dt = 0.1
        Ad, Bd = linear_kinodynamics(dt,N)
        # ADMM loop
        iter = 0
        while True:
            try:
                for k in range(T):
                    X_next = states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx]
                    X_curr = states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx]
                    U_curr = states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu]
                    # print(f'No. of agents is {n_agents}, shape of X_curr is {X_curr.shape}')
                    print(f'No. of agents is {n_agents}, shape of U_curr is {U_curr.shape}')
                    print(f'Shape of Bd is {Bd.shape}')
                
                    d[f"opti_{agent_id}"].subject_to(X_next==Ad @ X_curr + Bd @ U_curr) # close the gaps
                    
                    d[f"opti_{agent_id}"].subject_to(U_curr <= np.tile(np.array([2, 2, 2]),(N,)).reshape(-1,1))
                    d[f"opti_{agent_id}"].subject_to(np.tile(np.array([-2, -2, -2]),(N,)).reshape(-1,1) <= U_curr)

                    if convex_problem:
                        #Collision avoidance via Bufferd Voronoi Cells
                        """Reference: https://ieeexplore.ieee.org/document/7828016"""
                        if N > 1:
                            for i in range(N):
                                p_i_next =  X_next[i*n_states:(i+1)*n_states][:3]
                                p_i_prev = x0[i*n_states:(i+1)*n_states][:3]
                                for j in range(N):
                                    if j!= i:
                                        p_j_prev = x0[j*n_states:(j+1)*n_states][:3]
                                        p_ij = p_j_prev - p_i_prev
                                        bvc_i = p_ij.T @ p_i_next - (p_ij.T @ (p_i_prev + p_j_prev)/2 + \
                                                                    r_min*cs.norm_2(p_i_prev-p_j_prev))
                                        d[f"opti_{agent_id}"].subject_to(bvc_i <= 0)
                    else:
                        if N > 1:
                            if k <=2:
                                distances = util.compute_pairwise_distance_nd_Sym(states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx], x_dims, n_dims)
                                # Collision avoidance constraint
                                for dist in distances:
                                    d[f"opti_{agent_id}"].subject_to(dist >= r_min) 
                            else:
                                pass
                            
                # Terminal velocity constraint?
                # for i in range(N):
                #     d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][:(T+1)*nx][T*nx:(T+1)*nx][i*n_states:(i+1)*n_states][3:6]== np.zeros(3).reshape(-1,1))
                
                X0 = d[f"opti_{agent_id}"].parameter(x0.shape[0],1)
                d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][0:nx] == X0)
                cost_tot = cost 
                d[f"opti_{agent_id}"].minimize(cost_tot)
                
                if convex_problem:
                    d[f"opti_{agent_id}"].solver("osqp",opts) #Our problem is convex
                else:
                    d[f"opti_{agent_id}"].solver("ipopt",opts)
                
                if iter > 0:
                    d[f"opti_{agent_id}"].set_initial(sol_prev.value_variables())
                
                d[f"opti_{agent_id}"].set_value(X0,x0)
                sol = d[f"opti_{agent_id}"].solve()
                
                #Checking sparsity of constraint jacobian:
                #plt.spy(sol.value(jacobian(d[f"opti_{agent_id}"].g,d[f"opti_{agent_id}"].x)))
                #plt.show()
     
                sol_prev = sol
                pipe.send(sol.value(states[f"Y_{agent_id}"]))
                
                d[f"opti_{agent_id}"].set_value(xbar, pipe.recv()) #receive the averaged result from the main process.
                d[f"opti_{agent_id}"].set_value(u, sol.value( u + states[f"Y_{agent_id}"] - xbar))
                
                d[f"opti_{agent_id}"].subject_to()
                
                iter += 1                
                
            except EOFError:
                print("Connection closed.")
                break
                    
    pipes = []
    procs = []
    for i in range(N):
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_worker, args=(i, f_list[f"cost_{i}"], remote))]
        procs[-1].start()

    solution_list = []
        
    x_bar_history = [np.ones((nx, 1))*np.inf] #Recording how the consensus value x_bar evolves until consensus reached
    iter = 0
    
    t0 = perf_counter()
    """Execute ADMM loop in parallel:"""
    for i in range(ADMM_ITER):
        
        # Gather and average Y_i
        xbar = sum(pipe.recv() for pipe in pipes)/N
        
        x_bar_history.append(xbar)
        solution_list.append(xbar)
        
        # Scatter xbar
        for pipe in pipes:
            pipe.send(xbar)
        
        iter += 1
        
    admm_iter_time = perf_counter() - t0  
    [p.terminate() for p in procs]
        
    x_trj_converged = solution_list[-1][:(T+1)*nx].reshape((T+1,nx))
    u_trj_converged = solution_list[-1][(T+1)*nx:].reshape((T,nu))
    
    return x_trj_converged, u_trj_converged, x_bar_history, admm_iter_time