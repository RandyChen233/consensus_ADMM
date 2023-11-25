import logging
from pathlib import Path
import multiprocessing as mp
from os import getpid
import os
from time import strftime


def setup_logger_mpc():
    
    LOG_PATH = Path(__file__).parent/ "logs"
    LOG_FILE = LOG_PATH / strftime(
        "ADMM-mpc-_%m-%d-%y_%H.%M.%S_{getpid()}.csv"
    )
    if not LOG_PATH.is_dir():
        LOG_PATH.mkdir()
    print(f"Logging results to {LOG_FILE}")
    logging.basicConfig(filename=LOG_FILE, format="%(message)s", level=logging.INFO)
    logging.info(
        "i_trial, n_agents, t, converged, obj_trj,T,dt,radius,\
         SOVA_admm ,t_solve_avg, t_solve_std, MAX_ITER, dist_to_goal"
    )
    
    
def setup_logger_admm():
    LOG_PATH = Path(__file__).parent/ "logs"
    LOG_FILE = LOG_PATH / strftime(
        "ADMM-_%m-%d-%y_%H.%M.%S_{getpid()}.csv"
    )
    if not LOG_PATH.is_dir():
        LOG_PATH.mkdir()
    print(f"Logging results to {LOG_FILE}")
    logging.basicConfig(filename=LOG_FILE, format="%(message)s", level=logging.INFO)
    logging.info(
        "convex,n_trial,n_agents,iters,mpc_iter,obj_value,dual_res,primal_res"
    )
    
    