import os
import logging
import random
import numpy as np
import torch

def set_N_threads_(N_threads=1):
    logging.info(f'N_threads: {N_threads}')
    os.environ["OMP_NUM_THREADS"] = str(N_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(N_threads)
    os.environ["MKL_NUM_THREADS"] = str(N_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(N_threads)
    return N_threads

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.cuda.manual_seed_all(seed)  # If multiple GPUs