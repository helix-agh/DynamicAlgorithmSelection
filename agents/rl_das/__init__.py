"""RL-DAS agent.

Adaptation of "Deep Reinforcement Learning for Dynamic Algorithm Selection:
A Proof-of-Principle Study on Differential Evolution" (Guo et al., 2024).

Key differences from the original DAS agent:
- Observation: population state features + per-optimizer movement history
  (instead of ELA landscape features).
- Architecture: movement embedder networks collapse D-dim displacement vectors
  to scalars before the actor/critic backbone processes them.
- Training: hand-rolled PPO loop instead of Stable Baselines 3.
- Dimension-specific: a separate agent is trained for each problem dimension
  because the embedders have fixed input size.

Typical usage
-------------
>>> from agents.rl_das import RLDASEnv, PPOAgent, train, evaluate
"""

from agents.rl_das.env import RLDASEnv
from agents.rl_das.agent import PPOAgent
from agents.rl_das.trainer import train, evaluate

__all__ = ["RLDASEnv", "PPOAgent", "train", "evaluate"]
