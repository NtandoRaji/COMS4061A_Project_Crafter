import random
import numpy as np
import torch as T


def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
