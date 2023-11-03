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

from admm_mpc import *


if __name__ == "__main__":
    
    x0,xr = util.setup_3_quads()
    T = 15
    radius = 0.4
    n_states = 6
    n_inputs = 3
    n_agents = 3
    ids = [100+n for n in range(n_agents)]
    Q = np.eye(n_states*n_agents)*1
    for i in range(n_agents):
        Q[i*n_states:(i+1)*n_states][0:3] = 5
    print(Q)
    R = np.eye(n_inputs*n_agents)*0.1
    Qf = Q*100
    ADMM_ITER = 10
    
    X_full, U_full, obj_trj, mean_time, obj_history = solve_distributed_rhc(ids, n_states, \
                                                                            n_inputs, n_agents, \
                                                                            x0, xr, T, radius, \
                                                                            Q, R, Qf, ADMM_ITER, \
                                                                            n_trial=None)
    
    # X_full, U_full, obj_trj ,_,_ = solve_admm_mpc(n_states, n_inputs, n_agents, x0, xr, T, radius, Q, R, Qf, ADMM_ITER)
    
    

    np.savez("ADMM_BVC_convex_{}.npz".format(n_agents), X_full=X_full, obj_trj=obj_trj, xr=xr)