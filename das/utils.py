"""Shared utilities for DAS."""

import os
import random

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed as _sb3_seed


def set_seed(seed: int) -> None:
    """Seed all RNGs used across the project.

    Controls:
      - PYTHONHASHSEED  (Python dict/set ordering)
      - Python random module
      - NumPy legacy API  (np.random.*)
      - PyTorch CPU and GPU
      - SB3 internal seeding (wraps the above + cudnn flags)

    Note: pypop7 optimizers use np.random.default_rng(), which is NOT
    controlled by np.random.seed().  Pass seed_rng in the optimizer options
    (see DASEnv._run_optimizer) to make them reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _sb3_seed(seed)
