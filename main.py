import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import casadi as cs
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
from admm_mpc import *
from mpc import *
from logger import *
from util import *

if __name__ == "__main__":
    convex = True
    radius = 0.3
    # x0,xr = util.paper_setup_5_quads()
    x0, xr = util.paper_setup_3_quads()
    
    T = 10 # MPC horizon length
    n_states = 6
    n_inputs = 3
    n_agents = 3 #Change this accordingly
    ids = [100+n for n in range(n_agents)]
    Q = np.eye(n_states*n_agents)*1
    for i in range(n_agents):
        Q[i*n_states:(i+1)*n_states][0:3] = 5
    
    R = np.eye(n_inputs*n_agents)*0.1
    Qf = Q*100
    ADMM_ITER = 100
    # ADMM_ITER = 5
    MPC_ITER = 100
    setup_logger_admm()
        
    X_full, U_full, obj_trj, mean_time, obj_history = solve_admm_mpc(ids,
                                                                    n_states, 
                                                                    n_inputs, 
                                                                    n_agents, 
                                                                    x0, 
                                                                    xr, 
                                                                    T, 
                                                                    radius, 
                                                                    Q, 
                                                                    R, 
                                                                    Qf, 
                                                                    MPC_ITER,
                                                                    ADMM_ITER, convex = True, potential_ADMM = False, n_trial=None)
    
    np.savez("admm_consensus_BVC_convex_{}.npz".format(n_agents), X_full=X_full, obj_trj=obj_trj, obj_hist = obj_history, xr=xr, x0 = x0)
    
    # X_full, U_full, obj_trj, mean_time, obj_history = solve_admm_mpc(ids,
    #                                                                 n_states, 
    #                                                                 n_inputs, 
    #                                                                 n_agents, 
    #                                                                 x0, 
    #                                                                 xr, 
    #                                                                 T, 
    #                                                                 radius, 
    #                                                                 Q, 
    #                                                                 R, 
    #                                                                 Qf, 
    #                                                                 MPC_ITER,
    #                                                                 ADMM_ITER, convex = False, n_trial=None)
    # np.savez("admm_consensus_BVC_nonconvex_{}.npz".format(n_agents), X_full=X_full, obj_trj=obj_trj, obj_hist = obj_history, xr=xr, x0 = x0)
    
    
    
    # X_trj, U_trj, obj_trj, mean_times, obj_hist = solve_mpc_centralized(n_agents,
    #                                                                     x0, xr, 
    #                                                                     T, radius, 
    #                                                                     Q, R, Qf, 
    #                                                                     30, 
    #                                                                     )
    # np.savez("centralized_BVC_convex_{}.npz".format(n_agents), X_full=X_trj, obj_trj=obj_trj, obj_hist = obj_hist, xr=xr)
    
#     """Run non-convex version"""
#     X_full_nonConvex, U_full_nonConvex, \
#     obj_trj_nonConvex, mean_time_nonConvex, \
#     obj_history_nonConvex = solve_distributed_rhc(ids, n_states, \
#                                                 n_inputs, n_agents, \
#                                                 x0, xr, T, radius, \
#                                                 Q, R, Qf, ADMM_ITER, \
#                                                 convex_problem=False,
#                                                 n_trial=None)
    
#     np.savez("ADMM_BVC_nonconvex_{}.npz".format(n_agents), X_full=X_full, obj_trj=obj_trj, xr=xr)


