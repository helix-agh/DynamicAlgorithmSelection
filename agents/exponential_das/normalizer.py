"""Running normalizers for observations and rewards.

Both use Welford's online algorithm for numerically stable mean/variance.
Normalisation is only updated during the warmup phase (while the buffer is
filling for the first time); afterwards the statistics are frozen.  This
mirrors the StateNormalizer behaviour in the source project.
"""

from __future__ import annotations

import numpy as np


class ObservationNormalizer:
    """Per-dimension running z-score normaliser with clipping."""

    def __init__(self, obs_dim: int, clip: float = 5.0) -> None:
        self.clip = clip
        self._mean = np.zeros(obs_dim, dtype=np.float64)
        self._M2 = np.ones(obs_dim, dtype=np.float64)  # sum of squared deviations
        self._n = 0

    def update(self, x: np.ndarray) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (x - self._mean)

    def normalize(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.update(x)
        std = np.sqrt(self._M2 / max(self._n, 1) + 1e-8)
        normed = (x - self._mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


class RewardNormalizer:
    """Per-step running z-score normaliser for scalar rewards."""

    def __init__(self, n_checkpoints: int, clip: float = 10.0) -> None:
        self.clip = clip
        self._n_checkpoints = n_checkpoints
        # One running mean/var per checkpoint index
        self._mean = np.zeros(n_checkpoints, dtype=np.float64)
        self._M2 = np.ones(n_checkpoints, dtype=np.float64)
        self._counts = np.zeros(n_checkpoints, dtype=np.int64)

    def normalize(self, reward: float, step_idx: int, update: bool = True) -> float:
        idx = min(step_idx, self._n_checkpoints - 1)
        if update:
            self._counts[idx] += 1
            n = self._counts[idx]
            delta = reward - self._mean[idx]
            self._mean[idx] += delta / n
            self._M2[idx] += delta * (reward - self._mean[idx])
        std = float(np.sqrt(self._M2[idx] / max(self._counts[idx], 1) + 1e-8))
        return float(np.clip((reward - self._mean[idx]) / std, -self.clip, self.clip))
