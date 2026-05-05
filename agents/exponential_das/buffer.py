"""Rollout buffer and GAE computation for Exponential-DAS.

Design
------
The buffer accumulates (obs, action, log_prob, value, reward, done) tuples
until it reaches ``capacity``.  At that point:

  1. GAE returns and advantages are computed over the whole buffer.
  2. The caller runs the PPO minibatch update.
  3. The buffer is cleared; the next rollout begins.

This gives clean on-policy rollouts (no stale gradients from very old data)
while still amortising the PPO overhead over many steps.

The ``done`` flag is used inside GAE to reset the advantage bootstrap at
episode boundaries, so multiple episodes can coexist in one buffer.

GAE hyperparameters (γ=0.8, λ=0.5) are taken directly from ppo_utils.py in
the source project.
"""

from __future__ import annotations

import numpy as np
import torch

GAMMA = 0.8
LAMBDA = 0.5


class RolloutBuffer:
    """Fixed-capacity on-policy rollout buffer."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device
        self.clear()
        self._warmed_up = False  # override clear()'s True — False until first fill

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self.rewards)

    def is_full(self) -> bool:
        return len(self.rewards) >= self.capacity

    @property
    def warmed_up(self) -> bool:
        """True after the first full buffer (first PPO update has run)."""
        return self._warmed_up

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self._warmed_up = True  # stays True once set by first fill

    # ------------------------------------------------------------------
    # GAE + tensors
    # ------------------------------------------------------------------

    def compute_gae(
        self, bootstrap_value: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE returns and advantages (γ=0.8, λ=0.5)."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        prev_val = bootstrap_value
        for t in reversed(range(T)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + GAMMA * prev_val * mask - self.values[t]
            gae = delta + GAMMA * LAMBDA * mask * gae
            advantages[t] = gae
            prev_val = self.values[t]
        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages

    def as_tensors(
        self, bootstrap_value: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (obs, actions, old_log_probs, returns, advantages) as tensors."""
        returns, advantages = self.compute_gae(bootstrap_value)

        obs_t = torch.tensor(
            np.array(self.obs), dtype=torch.float32, device=self.device
        )
        act_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        lp_t = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, device=self.device)
        adv_t = torch.tensor(advantages, device=self.device)

        return obs_t, act_t, lp_t, ret_t, adv_t
