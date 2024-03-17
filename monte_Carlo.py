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
    radius = 0.3    
    n_states = 6
    n_inputs = 3
    
    n_agents =  [5, 10, 15]
    # n_agents = [3]
    n_trials_iters = range(30)
    
    setup_logger_admm()
    for n_quads in n_agents:
        for n_trial in n_trials_iters:   
            ids = [n for n in range(n_quads)]
            # x0,xr = util.setup_n_quads_V2(n_quads, 2*radius)
            x0,xr = util.setup_n_quads(n_quads, radius)
            # MPC horizon length:
            T = 10 
            # Weight matrices in the cost function:
            Q = np.eye(n_states*n_quads)
            for i in range(n_quads):
                Q[i*n_states:(i+1)*n_states][0:3] = 5
            R = np.eye(n_inputs*n_quads)*0.1
            Qf = Q*100
            
            #Parameters
            ADMM_ITER = 30
            MPC_ITER = 50
            try:
                
                # X_full, U_full, obj_val = solve_admm_mpc(ids, 
                #    n_states, 
                #    n_inputs, 
                #    n_quads, 
                #    x0, 
                #    xr, 
                #    T, 
                #    radius, 
                #    Q, 
                #    R, 
                #    Qf, 
                #    MPC_ITER, 
                #    ADMM_ITER, 
                #    convex = False, 
                #    potential_ADMM = True, 
                #    n_trial=n_trial)
                # if n_trial == 0:
                #     np.savez('logs/admmMPC_{}_quads.npz'.format(n_quads),X_traj = X_full, U_traj = U_full, ObjVals = obj_val)
                
                X_trj, U_trj, obj_hist = solve_mpc_centralized(n_quads, 
                                                               x0, 
                                                               xr, 
                                                               T, 
                                                               radius,
                                                               Q, 
                                                               R, 
                                                               Qf, 
                                                               n_trial = n_trial)
                if n_trial == 0:   
                    np.savez('logs/centralizedMPC_{}_quads.npz'.format(n_quads),X_traj = X_trj, U_traj = U_trj, ObjVals = obj_hist)
                
            except (EOFError, RuntimeError) as e:
                print("Error encountered, skipping to next iteration....")
                pass


