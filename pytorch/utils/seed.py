import random
import numpy as np
import torch

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
