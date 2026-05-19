"""PPO agent for RL-DAS.

Hand-rolled PPO following Guo et al. 2024 (no Stable Baselines 3 dependency).
One episode = one problem instance; the agent updates at the end of each episode
for ``k_epoch`` gradient steps.

Hyper-parameters (defaults match the paper):
  gamma         0.99
  eps_clip      0.1
  max_grad_norm 0.1
  lr            1e-5  (actor + critic share one Adam optimiser)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.rl_das.network import Actor, Critic


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


class _Rollout:
    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PPOAgent:
    """PPO agent for the RL-DAS environment.

    Parameters
    ----------
    dim:
        Problem dimension (determines embedder input size).
    n_opt:
        Number of sub-optimizers (action space size).
    lr:
        Learning rate for the shared Adam optimizer.
    gamma:
        Discount factor.
    eps_clip:
        PPO clipping range ε.
    max_grad_norm:
        Gradient norm clipping threshold.
    device:
        ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        dim: int,
        n_opt: int,
        lr: float = 1e-5,
        gamma: float = 0.99,
        eps_clip: float = 0.1,
        max_grad_norm: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.actor = Actor(dim, n_opt).to(self.device)
        self.critic = Critic(dim, n_opt).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        self.rollout = _Rollout()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Sample an action from the current policy.

        Returns
        -------
        action    : int
        log_prob  : float
        value     : float (critic estimate)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(obs_t)
            value = self.critic(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.item()),
        )

    def predict(self, obs: np.ndarray) -> int:
        """Deterministic greedy action (for evaluation)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(obs_t)
        return int(torch.argmax(probs, dim=-1).item())

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def learn(self, k_epoch: int, bootstrap_value: float = 0.0) -> dict[str, float]:
        """Perform k_epoch PPO gradient steps on the current rollout.

        Call at the end of each episode before ``rollout.clear()``.
        Returns a dict with diagnostic scalars.
        """
        rollout = self.rollout
        if len(rollout) == 0:
            return {}

        obs_t = torch.tensor(
            np.array(rollout.obs), dtype=torch.float32, device=self.device
        )
        actions_t = torch.tensor(rollout.actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(
            rollout.log_probs, dtype=torch.float32, device=self.device
        )
        old_values_t = torch.tensor(
            rollout.values, dtype=torch.float32, device=self.device
        )

        # Discounted returns with terminal bootstrap
        returns = []
        R = bootstrap_value
        for r, done in zip(reversed(rollout.rewards), reversed(rollout.dones)):
            R = r + self.gamma * R * (1.0 - float(done))
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages_t = returns_t - old_values_t
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        total_actor_loss = total_critic_loss = total_entropy = 0.0

        for epoch_idx in range(k_epoch):
            probs = self.actor(obs_t)
            values = self.critic(obs_t)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs_t.detach())
            surr1 = ratio * advantages_t.detach()
            surr2 = (
                torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                * advantages_t.detach()
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value clipping applied from the first inner epoch.  Skipping it
            # on epoch 0 allowed an unconstrained large update on the first step,
            # breaking the PPO v2 guarantee that value changes stay within eps_clip.
            values_clipped = old_values_t + torch.clamp(
                values - old_values_t, -self.eps_clip, self.eps_clip
            )
            critic_loss = torch.max(
                (values - returns_t.detach()) ** 2,
                (values_clipped - returns_t.detach()) ** 2,
            ).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()

        return {
            "actor_loss": total_actor_loss / k_epoch,
            "critic_loss": total_critic_loss / k_epoch,
            "entropy": total_entropy / k_epoch,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, dim: int, n_opt: int, **kwargs) -> "PPOAgent":
        agent = cls(dim, n_opt, **kwargs)
        ckpt = torch.load(path, map_location=agent.device, weights_only=False)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            agent.optimizer.load_state_dict(ckpt["optimizer"])
        return agent
