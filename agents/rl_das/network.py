"""Neural network architecture for RL-DAS.

Actor and Critic share a backbone that embeds per-optimizer movement history
vectors (each of dimension ``dim``) down to scalars, then feeds the
concatenated representation through a small MLP.

Architecture (following Guo et al. 2024):

  Input: flat obs = [features_6d | best_move_0_dim | worst_move_0_dim | ...]

  For each of the 2*n_opt movement blocks:
      embedder_k : Linear(dim, 64) -> ReLU -> Linear(64, 1) -> ReLU

  backbone_input = cat(features_6d, *[emb_k(move_k) for k])   shape: (6+2*n_opt,)
  backbone       : Linear(6+2*n_opt, 64) -> Tanh -> Linear(64, 16) -> Tanh

  Actor head  : Linear(16, n_opt) -> Softmax
  Critic head : Linear(16, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MovementEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, 1)


class _RLDASBackbone(nn.Module):
    """Shared feature extractor used by both Actor and Critic."""

    N_FEATURES = 6  # must match env.RLDASEnv.N_FEATURES

    def __init__(self, dim: int, n_opt: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_opt = n_opt
        self.n_moves = 2 * n_opt  # best + worst per optimizer

        self.embedders = nn.ModuleList(
            [_MovementEmbedder(dim) for _ in range(self.n_moves)]
        )
        self.backbone = nn.Sequential(
            nn.Linear(self.N_FEATURES + self.n_moves, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, N_FEATURES + n_moves * dim)
        features = obs[:, : self.N_FEATURES]  # (batch, 6)
        moves = obs[:, self.N_FEATURES :].reshape(-1, self.n_moves, self.dim)

        embedded = torch.cat(
            [self.embedders[k](moves[:, k]) for k in range(self.n_moves)],
            dim=1,
        )  # (batch, n_moves)

        x = torch.cat([features, embedded], dim=1)  # (batch, 6 + n_moves)
        return self.backbone(x)  # (batch, 16)


class Actor(nn.Module):
    def __init__(self, dim: int, n_opt: int) -> None:
        super().__init__()
        self.backbone = _RLDASBackbone(dim, n_opt)
        self.head = nn.Linear(16, n_opt)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        logits = self.head(self.backbone(obs))
        logits = torch.nan_to_num(logits, nan=0.0)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, dim: int, n_opt: int) -> None:
        super().__init__()
        self.backbone = _RLDASBackbone(dim, n_opt)
        self.head = nn.Linear(16, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs)).squeeze(-1)  # (batch,)
