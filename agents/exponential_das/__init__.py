"""Exponential-DAS agent.

Inspired by the PolicyGradientAgent from DynamicAlgorithmSelection.

Two "exponential" mechanisms define this agent:

1. **Exponential checkpoint spacing** (``cdb > 1.0`` in DASEnv) — the agent
   makes more algorithm-selection decisions early in the run, when the
   landscape is richest and the choice matters most.

2. **Exponential entropy annealing** — the PPO entropy bonus decays
   multiplicatively (×0.99) after each gradient update, gradually shifting
   the agent from exploration toward exploitation.

Other key differences from the plain DAS agent:
- Separate Adam optimisers for actor and critic (different sensible LR scales).
- KL-divergence–adaptive learning rate (TRPO-inspired; keeps updates stable).
- Random-policy warmup: uniform action probabilities until the rollout buffer
  fills for the first time, bootstrapping a useful replay dataset before
  any gradient step.
- Running Welford normaliser for both observations and rewards.
- LayerNorm + orthogonal-init network (from ppo_utils.py).
- GAE (γ=0.8, λ=0.5) instead of Monte-Carlo returns.

Typical usage
-------------
>>> from agents.exponential_das import ExpDASAgent, train, evaluate
"""

from agents.exponential_das.agent import ExpDASAgent
from agents.exponential_das.trainer import train, evaluate

__all__ = ["ExpDASAgent", "train", "evaluate"]
