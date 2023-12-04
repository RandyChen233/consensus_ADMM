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
    setup_logger_admm()
    # n_agents = [3]
    n_agents =  [7]
    
    n_trials_iters = range(0,30)
    for n_quads in n_agents:
        for n_trial in n_trials_iters:     
            x0,xr = util.setup_n_quads_V2(n_quads,2*radius)
            T = 10 # MPC horizon length
            n_agents = 3 #Change this accordingly
            ids = [100+n for n in range(n_quads)]
            Q = np.eye(n_states*n_quads)*1
            for i in range(n_quads):
                Q[i*n_states:(i+1)*n_states][0:3] = 5
            
            R = np.eye(n_inputs*n_quads)*0.1
            Qf = Q*100
            ADMM_ITER = 100
            MPC_ITER = 300
            try:
                X_full, U_full, obj_trj, mean_time, obj_history = solve_admm_mpc(n_states, 
                                                                                n_inputs, 
                                                                                n_quads, 
                                                                                x0, 
                                                                                xr, 
                                                                                T, 
                                                                                radius, 
                                                                                Q, 
                                                                                R, 
                                                                                Qf, 
                                                                                MPC_ITER,
                                                                                ADMM_ITER, convex = True, n_trial=n_trial)
                
                X_full, U_full, obj_trj, mean_time, obj_history = solve_admm_mpc(n_states, 
                                                                                n_inputs, 
                                                                                n_quads, 
                                                                                x0, 
                                                                                xr, 
                                                                                T, 
                                                                                radius, 
                                                                                Q, 
                                                                                R, 
                                                                                Qf, 
                                                                                MPC_ITER,
                                                                                ADMM_ITER, convex = False, n_trial=n_trial)
            except (EOFError, RuntimeError) as e:
                continue


