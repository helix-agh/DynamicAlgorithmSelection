"""DAS agent — thin re-export of the existing das package.

The DAS agent uses Stable Baselines 3 PPO to select among a portfolio of
sub-optimisers at each checkpoint of a BBOB optimisation run.  All
implementation lives in the top-level ``das/`` package; this namespace exists
so both agents live under a common ``agents/`` hierarchy.

Typical usage
-------------
>>> from agents.das import DASEnv, PORTFOLIO, get_portfolio
>>> from stable_baselines3 import PPO
"""

from das.env.das_env import DASEnv
from das.env.observation import compute_observation, observation_dim
from das.env.reward import compute_reward
from das.optimizers.portfolio import PORTFOLIO, get_portfolio
from das.training.callbacks import DASEvalCallback

__all__ = [
    "DASEnv",
    "compute_observation",
    "observation_dim",
    "compute_reward",
    "PORTFOLIO",
    "get_portfolio",
    "DASEvalCallback",
]
