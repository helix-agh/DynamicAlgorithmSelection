"""Exponential-DAS PPO agent.

Core behaviour (adapted from PolicyGradientAgent):

  Warmup phase (buffer not yet full):
    - Uniform random policy — all actions equally likely.
    - Normaliser statistics are updated from each observation and reward.

  Training phase (after first buffer fill):
    - Trained policy.
    - Normaliser statistics are frozen (stable input distribution).
    - After every PPO update:
        entropy_coef ←  max(entropy_coef × 0.99, 0.001)   [exponential decay]
        lr is adjusted based on mean KL vs. target_kl=0.03  [adaptive LR]

  PPO update (minibatch, shuffled):
    - Clipped surrogate objective (ε=0.2).
    - Separate actor/critic Adam optimisers (actor lr=3e-5, critic lr=1e-5).
    - Gradient norm clipping: 0.5 for each network.
    - Approximate KL (Schulman 2017) tracked for adaptive LR.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from agents.exponential_das.buffer import RolloutBuffer
from agents.exponential_das.network import Actor, Critic
from agents.exponential_das.normalizer import ObservationNormalizer, RewardNormalizer


class ExpDASAgent:
    """Exponential-DAS PPO agent.

    Parameters
    ----------
    obs_dim:
        Flat observation size (from DASEnv.observation_space.shape[0]).
    n_actions:
        Number of sub-optimizers (DASEnv action space size).
    buffer_capacity:
        Steps to collect before each PPO update.
    actor_lr / critic_lr:
        Initial learning rates.
    clip_eps:
        PPO clipping range ε.
    value_coef:
        Weight of the critic loss term.
    entropy_coef:
        Initial entropy bonus coefficient (decays exponentially).
    entropy_decay:
        Multiplicative decay applied to entropy_coef after each PPO update.
    entropy_min:
        Floor for the entropy coefficient.
    target_kl:
        Target approximate KL for learning-rate adaptation.
    ppo_epochs:
        Gradient epochs per PPO update.
    minibatch_size:
        Minibatch size for shuffled PPO updates.
    n_checkpoints:
        Episode length (used by the reward normaliser).
    device:
        ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        buffer_capacity: int = 2048,
        actor_lr: float = 3e-5,
        critic_lr: float = 1e-5,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_decay: float = 0.99,
        entropy_min: float = 0.001,
        target_kl: float = 0.03,
        ppo_epochs: int = 6,
        minibatch_size: int = 256,
        n_checkpoints: int = 10,
        device: str = "cpu",
    ) -> None:
        self.n_actions = n_actions
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.entropy_min = entropy_min
        self.target_kl = target_kl
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        self.actor = Actor(obs_dim, n_actions).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = RolloutBuffer(buffer_capacity, obs_dim, self.device)
        self.obs_norm = ObservationNormalizer(obs_dim)
        self.rew_norm = RewardNormalizer(n_checkpoints)

        # Diagnostics exposed for logging
        self.last_actor_loss: float = 0.0
        self.last_critic_loss: float = 0.0
        self.last_kl: float = 0.0
        self.current_lr: float = actor_lr

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, raw_obs: np.ndarray, step_idx: int = 0
    ) -> tuple[int, float, float]:
        """Return (action, log_prob, value).

        During warmup the policy is uniform; normaliser statistics update.
        After warmup the normalised obs feeds the trained networks; stats frozen.
        """
        updating_norm = not self.buffer.warmed_up
        obs = self.obs_norm.normalize(raw_obs, update=updating_norm)

        if not self.buffer.warmed_up:
            action = int(np.random.randint(self.n_actions))
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                value = float(self.critic(obs_t).item())
            log_prob = -np.log(self.n_actions)  # uniform distribution log-prob
            return action, log_prob, value

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            probs = self.actor(obs_t).squeeze(0).cpu().numpy()
            value = float(self.critic(obs_t).item())
        self.actor.train()
        self.critic.train()

        probs = np.nan_to_num(probs, nan=1.0 / self.n_actions)
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()

        action = int(np.random.choice(self.n_actions, p=probs))
        log_prob = float(np.log(probs[action] + 1e-12))
        return action, log_prob, value

    def predict(self, raw_obs: np.ndarray) -> int:
        """Deterministic greedy action (for evaluation)."""
        obs = self.obs_norm.normalize(raw_obs, update=False)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(obs_t)
        return int(torch.argmax(probs, dim=-1).item())

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def ppo_update(self, bootstrap_value: float = 0.0) -> None:
        """Compute GAE, run minibatch PPO, then decay entropy and adapt LR."""
        obs_t, act_t, old_lp_t, ret_t, adv_t = self.buffer.as_tensors(bootstrap_value)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T = obs_t.shape[0]
        total_kl = 0.0
        n_batches = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, self.minibatch_size):
                mb = indices[start : start + self.minibatch_size]
                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_old_lp = old_lp_t[mb]
                mb_ret = ret_t[mb]
                mb_adv = adv_t[mb]

                probs = self.actor(mb_obs)
                lp = torch.log(probs.gather(1, mb_act.unsqueeze(1)).squeeze(1) + 1e-12)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

                ratio = torch.exp(lp - mb_old_lp)

                with torch.no_grad():
                    log_ratio = lp - mb_old_lp
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(mb_obs).squeeze(1)
                critic_loss = nn.functional.mse_loss(values_pred, mb_ret)

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_opt.step()
                self.critic_opt.step()

                self.last_actor_loss = actor_loss.item()
                self.last_critic_loss = critic_loss.item()
                total_kl += approx_kl
                n_batches += 1

        mean_kl = total_kl / max(n_batches, 1)
        self.last_kl = mean_kl
        self._adapt_lr(mean_kl)

        # Exponential entropy decay
        self.entropy_coef = max(
            self.entropy_coef * self.entropy_decay, self.entropy_min
        )

    def _adapt_lr(self, mean_kl: float) -> None:
        """Scale learning rate up/down based on KL vs target (from PolicyGradientAgent)."""
        lr = self.actor_opt.param_groups[0]["lr"]
        if mean_kl > self.target_kl * 1.5:
            lr /= 1.5
        elif mean_kl < self.target_kl / 1.5:
            lr *= 1.5
        lr = float(np.clip(lr, 3e-6, 3e-4))
        for pg in self.actor_opt.param_groups:
            pg["lr"] = lr
        for pg in self.critic_opt.param_groups:
            pg["lr"] = lr
        self.current_lr = lr

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "entropy_coef": self.entropy_coef,
                "obs_norm_mean": self.obs_norm._mean,
                "obs_norm_M2": self.obs_norm._M2,
                "obs_norm_n": self.obs_norm._n,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, obs_dim: int, n_actions: int, **kwargs) -> "ExpDASAgent":
        agent = cls(obs_dim, n_actions, **kwargs)
        ckpt = torch.load(path, map_location=agent.device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        if "actor_opt" in ckpt:
            agent.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            agent.critic_opt.load_state_dict(ckpt["critic_opt"])
        agent.entropy_coef = float(ckpt.get("entropy_coef", agent.entropy_coef))
        if "obs_norm_mean" in ckpt:
            agent.obs_norm._mean = ckpt["obs_norm_mean"]
            agent.obs_norm._M2 = ckpt["obs_norm_M2"]
            agent.obs_norm._n = int(ckpt["obs_norm_n"])
        return agent
