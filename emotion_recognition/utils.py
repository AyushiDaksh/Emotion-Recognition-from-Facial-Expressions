import torch
import random
import numpy as np
from logging import warn

from constants import MODEL_NAME_MAP


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device="cpu"):
    # Select device on the machine
    if device == "cuda":
        if torch.cuda.is_available():
            device = device
        else:
            warn("Cuda not available, running on CPU")
            device = "cpu"
    else:
        device = device


def get_model(model_name):
    return MODEL_NAME_MAP[model_name]()
