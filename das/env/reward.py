"""Reward functions for the DAS environment.

All functions take (new_best_y, old_best_y, initial_value_range, is_final)
and return a scalar reward. Improvement is scaled by the initial fitness range
so rewards are comparable across different problem instances.
"""

import numpy as np


def _improvement_ratio(
    new_best_y: float, old_best_y: float, initial_range: tuple[float, float]
) -> float:
    scale = initial_range[1] - initial_range[0]
    return (old_best_y - new_best_y) / (scale + 1e-10)


def reward_log_scaled(new_best_y, old_best_y, initial_range, is_final=False):
    """Log-scaled incremental improvement (original r1)."""
    if old_best_y == float("inf"):
        return float(np.log(initial_range[1] - initial_range[0] + 1e-10))
    ratio = _improvement_ratio(new_best_y, old_best_y, initial_range)
    return float(np.log(np.clip(ratio, 0.0, 1.0) + 1e-5))


def reward_linear(new_best_y, old_best_y, initial_range, is_final=False):
    """Linear improvement clipped to [0, 1] (original r2)."""
    if old_best_y == float("inf"):
        return float(np.log(initial_range[1] - initial_range[0] + 1e-10))
    return float(
        np.clip(_improvement_ratio(new_best_y, old_best_y, initial_range), 0.0, 1.0)
    )


def reward_sparse(new_best_y, old_best_y, initial_range, is_final=False):
    """Sparse: only reward at the final checkpoint (original r3)."""
    if old_best_y == float("inf") or not is_final:
        return float(np.log(initial_range[1] - initial_range[0] + 1e-10))
    total_improvement = initial_range[0] - new_best_y
    scale = initial_range[1] - initial_range[0]
    return float(np.log(total_improvement / (scale + 1e-10) + 1e-5))


def reward_binary(new_best_y, old_best_y, initial_range, is_final=False):
    """Binary: 1 if improvement >= 0.1%, else 0 (original r4)."""
    if old_best_y == float("inf"):
        return 0.0
    ratio = _improvement_ratio(new_best_y, old_best_y, initial_range)
    return 1.0 if ratio >= 1e-3 else 0.0


REWARD_FNS = {
    1: reward_log_scaled,
    2: reward_linear,
    3: reward_sparse,
    4: reward_binary,
}


def compute_reward(
    new_best_y: float,
    old_best_y: float,
    initial_range: tuple[float, float],
    option: int = 1,
    is_final: bool = False,
) -> float:
    fn = REWARD_FNS.get(option)
    if fn is None:
        raise ValueError(
            f"Unknown reward option {option}. Choose from {list(REWARD_FNS)}"
        )
    return fn(new_best_y, old_best_y, initial_range, is_final)
