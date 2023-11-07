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


if __name__ == "__main__":
    convex_problem = True
    # x0,xr = util.setup_3_quads()
    # x0, xr = util.setup_5_quads()
    x0, xr = util.four_quad_exchange()
    T = 30
    radius = 0.4
    n_states = 6
    n_inputs = 3
    # n_agents = 3
    n_agents = 4
    ids = [100+n for n in range(n_agents)]
    Q = np.eye(n_states*n_agents)*1
    for i in range(n_agents):
        Q[i*n_states:(i+1)*n_states][0:3] = 5

    R = np.eye(n_inputs*n_agents)*0.1
    Qf = Q*100
    # ADMM_ITER = 10
    ADMM_ITER = 5
    
    
    """Run convex version first:"""
    X_full, U_full, obj_trj, mean_time, obj_history = solve_distributed_rhc(ids, n_states, \
                                                                            n_inputs, n_agents, \
                                                                            x0, xr, T, radius, \
                                                                            Q, R, Qf, ADMM_ITER, \
                                                                            convex_problem=True,
                                                                            n_trial=None)
    
    np.savez("ADMM_BVC_convex_{}.npz".format(n_agents), X_full=X_full, obj_trj=obj_trj, obj_hist = obj_history, xr=xr)
    
    
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


