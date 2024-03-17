from casadi import *
import util
import numpy as np
from dynamics import *
from time import perf_counter
import logging
from dynamics import generate_f
"""This is a vanilla centralized MPC demo without C-ADMM"""



"""Linear quadratic MPC, centralized (with receding horizon implementation)"""
def solve_mpc_centralized(n_agents, x0, xr, T, radius, Q, R, Qf, MAX_ITER, n_trial = None):
    SOVA_admm = 'centralized_mpc'
    n_states = 6
    n_inputs = 3
    nx = n_agents * n_states
    nu = n_agents * n_inputs
    N = n_agents
    opti = Opti('conic')
    Y_state = opti.variable((T+1)*nx + T*nu)
    cost = 0
    
    u_ref = np.array([0, 0, 0] * N).reshape(-1,1)
    
    for t in range(T):
        for idx in range(nx):
            cost += (Y_state[:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) *  \
            Q[idx,idx]* (Y_state[:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) 
        for idu in range(nu):
            cost += (Y_state[(T+1)*nx:][t*nu:(t+1)*nu][idu]) *  \
            R[idu,idu] * (Y_state[(T+1)*nx:][t*nu:(t+1)*nu][idu])

    for idf in range(nx):
        cost += (Y_state[:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf]) * \
        Qf[idf,idf] * (Y_state[:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf])

    obj_hist = [np.inf]
    x_curr = x0
    u_curr = np.zeros((nu,1))
 
    X_trj = np.zeros((0, nx))
    U_trj = np.zeros((0, nu))
    X_trj = np.r_[X_trj, x0.T]
    iters = 0

    solve_times = []
    t = 0
    dt = 0.1
    
    x_dims = [6]*N
    n_dims = [3]*N
    r_min = 2*radius
    
    Ad,Bd = linear_kinodynamics(dt,n_agents)
    A_i = Ad[0:6,0:6]
    B_i = Bd[0:6,0:3]
    while not np.all(util.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states) <= 0.1):
        for k in range(T):
            X_next = Y_state[:(T+1)*nx][(k+1)*nx:(k+2)*nx]
            X_curr = Y_state[:(T+1)*nx][k*nx:(k+1)*nx]
            U_curr = Y_state[(T+1)*nx:][k*nu:(k+1)*nu]
        
            opti.subject_to(X_next==Ad @ X_curr + Bd @ U_curr) # close the gaps
            
            #Constrain acceleration (control input):
            opti.subject_to(Y_state[(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([2, 2, 2]),(N,)).reshape(-1,1))
            opti.subject_to(np.tile(np.array([-2, -2, -2]),(N,)).reshape(-1,1) <= Y_state[(T+1)*nx:][k*nu:(k+1)*nu])

            #Constrain velocity:
            for i in range(n_agents):
                opti.subject_to(Y_state[(T+1)*nx:][i*n_states:(i+1)*n_states][3:6] <= np.array([1.5, 1.5, 1.5]))
                opti.subject_to(np.array([-1.5, -1.5, -1.5]) <= Y_state[(T+1)*nx:][i*n_states:(i+1)*n_states][3:6])

            #Collision avoidance via BVCs
            for i in range(n_agents):
                agent_i_trj = forward_pass(	A_i,
                                               B_i,
                                            T,
                                            x_curr[i*n_states:(i+1)*n_states],
                                            u_curr[i*n_inputs:(i+1)*n_inputs])
                p_i_next = X_curr[i*n_states:(i+1)*n_states][:3]
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
      
        X0 = opti.parameter(x0.shape[0],1)     
        opti.subject_to(Y_state[0:nx] == X0)
        
        cost_tot = cost
        
        opti.minimize(cost_tot)

        opti.solver("osqp")
        opti.set_value(X0,x_curr)
        
        if iters > 0:
            opti.set_initial(sol_prev.value_variables())
            
        t0 = perf_counter()
        try:
            sol = opti.solve()
            
        except RuntimeError:
            converged=False
            break
            
        sol_prev = sol
        solve_times.append(perf_counter() - t0)
        obj_hist.append(sol.value(cost_tot))

        ctrl = sol.value(Y_state)[(T+1)*nx:].reshape((T, nu))[0]
        u_curr = ctrl
        x_curr = sol.value(Y_state)[:(T+1)*nx].reshape((T+1,nx))[1]
        X_trj = np.r_[X_trj, x_curr.reshape(1,-1)]
        U_trj = np.r_[U_trj, ctrl.reshape(1,-1)]
        
        opti.subject_to()
        
        iters += 1
        t += dt
        if iters > MAX_ITER:
            converged = False
            print(f'Max MPC iters reached; exiting MPC loops.....')
            break
    
    if np.all(util.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states) <= 0.1):
        converged = True
        
    
    obj_trj = float(util.objective(X_trj.T, U_trj.T, u_ref, xr, Q, R, Qf)) 
    logging.info(
    f'{n_trial},'
    f'{n_agents},{t},{converged},'
    f'{obj_trj},{T},{dt},{radius},{SOVA_admm},{np.mean(solve_times)}, {np.std(solve_times)}, {MAX_ITER},'
    f'{util.distance_to_goal(X_trj[-1].flatten(), xr.flatten(), n_agents, n_states)},'
    )
    
    print(f'Distance to goal is {util.distance_to_goal(X_trj[-1].flatten(), xr.flatten(), n_agents, n_states)}!')
        
    return X_trj, U_trj, obj_trj, np.mean(solve_times), obj_hist



def solve_mpc_centralized(n_agents, x0, xr, T, radius, Q, R, Qf, convex = False, n_trial = None):
    solver_type = 'CentralizedMPC'
    n_states = 6
    n_inputs = 3
    nx = n_agents * 6
    nu = n_agents * 3
    N = n_agents
    opti = Opti()
    Y_state = opti.variable((T+1)*nx + T*nu)
    
    x_dims = [n_states]*n_agents
    u_ref = np.array([0, 0, 0] * N).reshape(-1,1)
    n_dims = [3]*n_agents
    
    cost = 0
    for t in range(T-1):
        for idx in range(nx):
            cost += (Y_state[:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) *  \
            Q[idx,idx]* (Y_state[:(T+1)*nx][t*nx:(t+1)*nx][idx]-xr[idx]) 
             #Pair-wise Euclidean distance between each pair of agents
            distances = util.compute_pairwise_distance_nd_Sym(Y_state[:(T+1)*nx][t*nx:(t+1)*nx], x_dims, n_dims)
            #Collision avoidance cost
            for dist in distances:
                cost += fmin(0,(dist - 2*radius))**2 * 200
    for t in range(T):
        for idu in range(nu):
            cost += (Y_state[(T+1)*nx:][t*nu:(t+1)*nu][idu] - u_ref[idu]) *  \
            R[idu,idu] * (Y_state[(T+1)*nx:][t*nu:(t+1)*nu][idu] - u_ref[idu])
            
    for idf in range(nx):
        cost += (Y_state[:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf]) * \
        Qf[idf,idf] * (Y_state[:(T+1)*nx][T*nx:(T+1)*nx][idf] - xr[idf])

    obj_hist = [np.inf]
    x_curr = x0

    X_trj = np.zeros((0, nx))
    U_trj = np.zeros((0, nu))
    X_trj = np.r_[X_trj, x0.T]
    iters = 0

    solve_times = []
    t = 0
    dt = 0.1
    
    x_dims = [6]*N
    n_dims = [3]*N
    f = generate_f(x_dims)
    
    while not np.all(util.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, 6) <= 0.1):

        for k in range(T):
            
            k1 = f(Y_state[:(T+1)*nx][k*nx:(k+1)*nx],         Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            k2 = f(Y_state[:(T+1)*nx][k*nx:(k+1)*nx]+dt/2*k1, Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            k3 = f(Y_state[:(T+1)*nx][k*nx:(k+1)*nx]+dt/2*k2, Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            k4 = f(Y_state[:(T+1)*nx][k*nx:(k+1)*nx]+dt*k3,   Y_state[(T+1)*nx:][k*nu:(k+1)*nu])
            x_next = Y_state[:(T+1)*nx][k*nx:(k+1)*nx] + dt/6*(k1+2*k2+2*k3+k4) 

            opti.subject_to(Y_state[:(T+1)*nx][(k+1)*nx:(k+2)*nx]==x_next) # close the gaps
            
            opti.subject_to(Y_state[(T+1)*nx:][k*nu:(k+1)*nu] <= np.tile(np.array([np.pi/6, np.pi/6, 20]),(N,)).reshape(-1,1))
            opti.subject_to(np.tile(np.array([-np.pi/6, -np.pi/6, 0]),(N,)).reshape(-1,1) <= Y_state[(T+1)*nx:][k*nu:(k+1)*nu])


        X0 = opti.parameter(x0.shape[0],1)     
        opti.subject_to(Y_state[0:nx] == X0)
        
        cost_tot = cost 
        
        opti.minimize(cost_tot)

        opti.solver("ipopt")
        opti.set_value(X0,x_curr)
        
        if iters > 0:
            opti.set_initial(sol_prev.value_variables())
            
        t0 = perf_counter()
        try:
            sol = opti.solve()
            
        except RuntimeError:
            converged=False
            break
        
        times = perf_counter() - t0
        sol_prev = sol
        solve_times.append(times)
        
        cost_curr = sol.value(cost_tot)
        obj_hist.append(cost_curr)

        ctrl = sol.value(Y_state)[(T+1)*nx:].reshape((T, nu))[0]
        x_curr = sol.value(Y_state)[:(T+1)*nx].reshape((T+1,nx))[1] #Same as above
        X_trj = np.r_[X_trj, x_curr.reshape(1,-1)]
        U_trj = np.r_[U_trj, ctrl.reshape(1,-1)]
        
        opti.subject_to()
        
        iters += 1
        t += dt
        
        if iters > 60:
            converged = False
            print(f'Max MPC iters reached; exiting MPC loops.....')
            break
    
    if np.all(util.distance_to_goal(x_curr.flatten(), xr.flatten(), n_agents, n_states) <= 0.1):
        converged = True
        
    cost_trj =  util.evaluate_cost_trajectory(X_trj,U_trj,radius,x_dims,n_dims,xr,u_ref,Q,R,Qf)
    logging.info(
                f'{solver_type},{n_trial},{np.mean(solve_times)},{np.std(solve_times)},'
                f'{n_agents},{iters},{cost_trj },'
                )	
    
    return X_trj, U_trj, obj_hist


"""Nonlinear MPC, centralized"""
def solve_mpc_single_nonlinear(n_agents, x0, xr, T, Q, R, Qf):
    """This function computes the optimal control input over a single finite horizon T (no receding horizon implemented!)"""
    n_states = 12
    n_inputs = 4

    N = n_agents
    assert N == 1
 
    opti = Opti()
 
    X_state = opti.variable((T+1)*n_states) #Discrete state variable, concatenated along a horizon T+1
    U_state = opti.variable(T*n_inputs) #Discrete state variable, concatenated along a horizon T
    u_ref = np.array([0, 0, 0, 9.8/2]) #Reference input vector
    
    cost = 0
    #Running cost : x^T@Q@x + u^T@R@u
    for t in range(T):
        for i in range(Q.shape[0]):
            cost += (X_state[t*n_states:(t+1)*n_states][i] - xr[i])*  \
                    Q[i,i]*(X_state[t*n_states:(t+1)*n_states][i]-xr[i]) 
     
        for j in range(R.shape[0]):
            cost += (U_state[t*n_inputs:(t+1)*n_inputs][j] - u_ref[j]) *  \
                    R[j,j] * (U_state[t*n_inputs:(t+1)*n_inputs][j] - u_ref[j])

    #Terminal cost:
    for i in range(Qf.shape[0]):
        cost += (X_state[T*n_states:(T+1)*n_states][i] - xr[i]) * \
                Qf[i,i] * (X_state[T*n_states:(T+1)*n_states][i]- xr[i])
    # print(cost.is_scalar())
    # obj_hist = [np.inf]
    x_curr = x0
    u_curr = np.zeros((n_inputs,1))
 
    # X_trj = np.zeros((0, n_states))
    # U_trj = np.zeros((0, n_inputs))
    # X_trj = np.r_[X_trj, x0.T]
    iters = 0

    solve_times = []
    t = 0
    dt = 0.1
    
    for k in range(T):
        X_next = X_state[(k+1)*n_states:(k+2)*n_states]
        X_curr = X_state[k*n_states:(k+1)*n_states]
        U_curr = U_state[k*n_inputs:(k+1)*n_inputs]
        # f = generate_f_12DOF(X_curr, U_curr)

        k1 = generate_f_12DOF(X_curr,U_curr)
        k2 = generate_f_12DOF(X_curr+dt/2*k1, U_curr)
        k3 = generate_f_12DOF(X_curr+dt/2*k2, U_curr)
        k4 = generate_f_12DOF(X_curr+dt*k3,   U_curr)

        x_update = X_curr + dt/6*(k1+2*k2+2*k3+k4) 

        opti.subject_to(X_next==x_update) # close the gap
        opti.subject_to(U_curr <= np.array([2, 2, 2, 2.5*9.8])) #mass of the drone is 0.5 kg
        opti.subject_to(np.array([-2, -2, -2, -2.5*9.8]) <= U_curr) 

        
    X0 = opti.parameter(x0.shape[0],1)     
    opti.subject_to(X_state[0:n_states] == X0)
    
    opti.minimize(cost)

    opti.solver("ipopt")
    opti.set_value(X0,x_curr)
    
    t0 = perf_counter()
    try:
        sol = opti.solve()
        iter_time = perf_counter() - t0
  
    except RuntimeError:
        print("Runtime error encountered in single drone MPC !!! ")
        iter_time = None
        obj_val = None
        
        return x_curr, u_curr, iter_time, obj_val

    solve_times.append(perf_counter() - t0)
    obj_val = sol.value(cost)

    ctrl = sol.value(U_state).reshape((T, n_inputs))[0]
    u_curr = ctrl
    x_curr = sol.value(X_state).reshape((T+1,n_states))[1]
        
    return x_curr, u_curr, iter_time, obj_val