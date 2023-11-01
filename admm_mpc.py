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









def solve_iteration(pos_neighbors, n_states, n_inputs, n_agents, 
                    x0, xr, T, radius, Q, R, Qf, 
                    current_mpc_iter,MAX_ITER = 5):
    """Define constants"""
    #pos_neighbors: positions of neighboring agents computed at the previous time step 
    
    #T is the horizon
    nx = n_states*n_agents
    nu = n_inputs*n_agents
    N = n_agents

    x_dims = [n_states] * N
    n_dims = [3]*N

    u_ref = np.array([0, 0, 0]*N).reshape(-1,1)
    
    """Creating empty dicts to hold Casadi variables for each sub-problem
    Note: our ADMM algorithm is still synchronous within each sub-problem, but since
    each sub-problem has different sizes at each time step t, we choose to iterate over each
    sub-problem in a sequential fashion.
    """
    f_list = {}
    d = {} 
    states = {}
    dt = 0.1
    for id in range(N):
        #Initialize Opti() class
        d["opti_{0}".format(id)] = Opti()
        
        #Augmented ADMM state : Y = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        states["Y_{0}".format(id)] = d[f"opti_{id}"].variable((T+1)*nx + T* nu)
        cost = 0

        # Quadratic tracking cost:
        for t in range(T):
            for idx in range(nx):
                cost += (states[f"Y_{id}"][:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) *  \
                Q[idx,idx]* (states[f"Y_{id}"][:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) 
            for idu in range(nu):
                cost += (states[f"Y_{id}"][(T+1)*nx:][t*nu:(t+1)*nu][idu] - u_ref[idu]) *  \
                R[idu,idu] * (states[f"Y_{id}"][(T+1)*nx:][t*nu:(t+1)*nu][idu] - u_ref[idu])
        # Quadratic terminal cost:
        for idf in range(nx):
            cost += (states[f"Y_{id}"][:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf]) * \
            Qf[idf,idf] * (states[f"Y_{id}"][:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf])

        # f_list.append(cost)
        f_list["cost_{0}".format(id)] = cost
        

    def run_worker(agent_id, cost, pipe):
        "Reference https://www.cvxpy.org/examples/applications/consensus_opt.html"
        
        xbar = d[f"opti_{agent_id}"].parameter((T+1)*nx + T*nu)
        d[f"opti_{agent_id}"].set_value(xbar, cs.GenDM_zeros((T+1)*nx + T*nu,1))    
        
        u = d[f"opti_{agent_id}"].parameter((T+1)*nx + T*nu)
        d[f"opti_{agent_id}"].set_value(u, cs.GenDM_zeros((T+1)*nx + T*nu,1))

        # rho = 1.0
        rho = 0.5
        cost += (rho/2)*sumsqr(states[f"Y_{agent_id}"] - xbar + u)
        dt = 0.1
        Ad, Bd = linear_kinodynamics(dt)
        # ADMM loop
        iter = 0
        while True:
            try:
                coll_cost = 0
                smooth_trj_cost = 0
                for k in range(T):
                    
                    x_next = [states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx] \
                    == Ad @ states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx] \
                    + Bd @states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu]]
                    
                    d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx]==x_next) # close the gaps
                    
                    d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([np.pi/6, np.pi/6, 20]),(N,)).reshape(-1,1))
                    d[f"opti_{agent_id}"].subject_to(np.tile(np.array([-np.pi/6, -np.pi/6, 0]),(N,)).reshape(-1,1) <= states[f"Y_{agent_id}"][(T+1)*nx:][k*nu:(k+1)*nu])

                    #Collision avoidance via Bufferd Voronoi Cells
                    """Reference: https://ieeexplore.ieee.org/document/7828016"""
                    if n_agents > 1:
                        r_min = 2.5*radius
                        for i in range(n_agents):
                            p_i_next =  states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx]
                            p_i_prev = pos_neighbors[i]
                            for j in range(n_agents):
                                if j!= i:
                                    p_j_prev = pos_neighbors[j]
                                    p_ij = p_j_prev - p_i_prev
                                    bvc_i = p_ij.T @ p_i_next - (p_ij.T @ (p_i_prev + p_j_prev)/2 + \
                                                                r_min*cs.norm(p_i_prev-p_j_prev))
                                    d[f"opti_{agent_id}"].subject_to(bvc_i <= 0)
                                
                    #Trajectory smoothing term
                    for ind in range(nx):
                        smooth_trj_cost += (states[f"Y_{agent_id}"][:(T+1)*nx][(k+1)*nx:(k+2)*nx][ind]-\
                                            states[f"Y_{agent_id}"][:(T+1)*nx][k*nx:(k+1)*nx][ind])**2
                    
                X0 = d[f"opti_{agent_id}"].parameter(x0.shape[0],1)
 
                d[f"opti_{agent_id}"].subject_to(states[f"Y_{agent_id}"][0:nx] == X0)
                
                cost_tot = cost + coll_cost/N + smooth_trj_cost
                
                d[f"opti_{agent_id}"].minimize(cost_tot)
                d[f"opti_{agent_id}"].solver("ipopt")
                
                if iter > 0:
                    d[f"opti_{agent_id}"].set_initial(sol_prev.value_variables())
                
                d[f"opti_{agent_id}"].set_value(X0,x0)
                sol = d[f"opti_{agent_id}"].solve()
      
                sol_prev = sol
                pipe.send(sol.value(states[f"Y_{agent_id}"]))
                
                d[f"opti_{agent_id}"].set_value(xbar, pipe.recv()) #receive the averaged result from the main process.
                d[f"opti_{agent_id}"].set_value(u, sol.value( u + states[f"Y_{agent_id}"] - xbar))
                
                # print(f'Current iteration is {iter}')
                
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
    admm_iter_t0 = perf_counter()
    
    x_bar_history = [np.ones((nx, 1))*np.inf]
    iter = 0
    t0 = perf_counter()
    for i in range(MAX_ITER):
        
        # Gather and average Y_i
        xbar = sum(pipe.recv() for pipe in pipes)/N
        
        x_bar_history.append(xbar)
        solution_list.append(xbar)
        
        # Scatter xbar
        for pipe in pipes:
            pipe.send(xbar)
        
        iter += 1
        
    admm_iter_time = perf_counter() - admm_iter_t0    
    [p.terminate() for p in procs]
        
    x_trj_converged = solution_list[-1][:(T+1)*nx].reshape((T+1,nx))
    u_trj_converged = solution_list[-1][(T+1)*nx:].reshape((T,nu))
    
    return x_trj_converged, u_trj_converged, admm_iter_time