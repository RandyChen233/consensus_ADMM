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
    n_states = 12
    n_inputs = 4
    # setup_logger_admm()
    n_agents =  [3,6,12,24]
    
    n_trials_iters = range(0,30)
    for n_quads in n_agents:
        ids = [100+n for n in range(n_quads)]
        for n_trial in n_trials_iters:     
            x0,xr = util.setup_n_quads_V2(n_quads, 1.5*radius)
            # MPC horizon length:
            T = 10 
            
            # Weight matrices in the cost function:
            Q = np.eye(n_states*n_quads)*1
            for i in range(n_quads):
                Q[i*n_states:(i+1)*n_states][0:3] = 5
            R = np.eye(n_inputs*n_quads)*0.1
            Qf = Q*100
            
            #Parameters
            ADMM_ITER = 100
            MPC_ITER = 300
            try:
                X_full, U_full, mean_time, obj_history = solve_admm_mpc(ids, 
                   n_states, 
                   n_inputs, 
                   n_quads, 
                   x0, 
                   xr, 
                   T, 
                   radius, 
                   Q, 
                   R, Qf, MPC_ITER, ADMM_ITER, convex = False, potential_ADMM = True, n_trial=None)
                
            except (EOFError, RuntimeError) as e:
                continue


