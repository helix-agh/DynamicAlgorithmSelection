"""Actor and Critic networks for Exponential-DAS.

Architecture (ported from ppo_utils.py in DynamicAlgorithmSelection):
  - LayerNorm after every hidden layer (stabilises training across problems
    with very different observation scales).
  - Orthogonal weight initialisation; output layer of Actor is zero-init so
    the initial policy is uniform over actions.
  - Hidden size: 96 (same as the source project).

Both networks are dimension-independent: they consume whatever flat vector
DASEnv provides, so the same agent works across all problem dimensions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


HIDDEN_SIZE = 96


def _layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0
) -> nn.Linear:
    if std == 0.0:
        init.constant_(layer.weight, 0.0)
    else:
        init.orthogonal_(layer.weight, std)
    init.constant_(layer.bias, bias)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, HIDDEN_SIZE)),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            _layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            _layer_init(nn.Linear(HIDDEN_SIZE, n_actions), std=0.0),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.head(self.feature_extractor(x))


class Critic(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, HIDDEN_SIZE)),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            _layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            _layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.head(self.feature_extractor(x))
