"""Reward functions for the DAS environment.

All functions take (new_best_y, old_best_y, initial_value_range, is_final,
optimum) and return a scalar reward. Improvement is scaled by the initial
fitness range so rewards are comparable across different problem instances.

``optimum`` (the known global minimum) is optional. When it is available,
optimum-aware functions measure progress in *orders of magnitude of the gap to
the optimum* (the natural BBOB metric) instead of a probe-relative ratio that
saturates near the optimum. It is training-only — the learned policy never sees
it — and every function falls back to its probe-relative behaviour when the
optimum is ``None`` (e.g. a non-BBOB suite), so the options stay portable.
"""

import numpy as np

_GAP_FLOOR = 1e-8  # BBOB precision target: gaps below this count as "solved".


def _improvement_ratio(
    new_best_y: float, old_best_y: float, initial_range: tuple[float, float]
) -> float:
    scale = initial_range[1] - initial_range[0]
    return (old_best_y - new_best_y) / (scale + 1e-10)


def _log_gap_orders(y_from: float, y_to: float, optimum: float) -> float:
    """Orders of magnitude the gap to the optimum shrinks going y_from -> y_to.

    Positive when ``y_to`` is closer to the optimum. Telescopes over a run to
    log10(initial_gap / final_gap), i.e. the total accuracy (in decades) gained.
    """
    old_gap = max(y_from - optimum, _GAP_FLOOR)
    new_gap = max(y_to - optimum, _GAP_FLOOR)
    return float(np.log10(old_gap) - np.log10(new_gap))


def _terminal_reward(final_y, initial_range, optimum) -> float:
    """Full-magnitude terminal reward, clipped to [-10, 10].

    With a known optimum: orders of magnitude of accuracy gained relative to the
    random-probe baseline — this does *not* saturate, so reaching gap 1e-8 is
    rewarded far more than gap 1e-2 (the probe-scaled version cannot tell them
    apart). Otherwise: probe-scaled total improvement (legacy behaviour).
    """
    if optimum is not None:
        return float(np.clip(_log_gap_orders(initial_range[0], final_y, optimum), -10.0, 10.0))
    raw = _improvement_ratio(final_y, initial_range[0], initial_range)
    return float(np.clip(raw, -10.0, 10.0))


def reward_log_scaled(new_best_y, old_best_y, initial_range, is_final=False, optimum=None):
    """Log-scaled incremental improvement (original r1)."""
    ratio = _improvement_ratio(new_best_y, old_best_y, initial_range)
    return float(np.log(np.clip(ratio, 0.0, 1.0) + 1e-5))


def reward_linear(new_best_y, old_best_y, initial_range, is_final=False, optimum=None):
    """Linear improvement clipped to [0, 1] (original r2)."""
    return float(
        np.clip(_improvement_ratio(new_best_y, old_best_y, initial_range), 0.0, 1.0)
    )


def reward_log_improvement(new_best_y, old_best_y, initial_range, is_final=False, optimum=None):
    """Constant reward per order-of-magnitude reduction toward the optimum."""
    if optimum is None:
        return reward_linear(new_best_y, old_best_y, initial_range, is_final)
    return float(max(_log_gap_orders(old_best_y, new_best_y, optimum), 0.0))


def reward_sparse(new_best_y, old_best_y, initial_range, is_final=False, optimum=None):
    """Sparse: reward only at the final checkpoint (original r3)."""
    if not is_final:
        return 0.0
    if optimum is not None:
        return float(np.clip(_log_gap_orders(initial_range[0], new_best_y, optimum), 0.0, 10.0))
    total_improvement = initial_range[0] - new_best_y
    scale = initial_range[1] - initial_range[0]
    return float(np.log(total_improvement / (scale + 1e-10) + 1e-5))


def reward_binary(new_best_y, old_best_y, initial_range, is_final=False, optimum=None):
    """Binary: 1 if improvement >= 0.1%, else 0 (original r4)."""
    ratio = _improvement_ratio(new_best_y, old_best_y, initial_range)
    return 1.0 if ratio >= 1e-3 else 0.0


def reward_hybrid_binary(new_best_y, old_best_y, initial_range, is_final=False, optimum=None):
    """Hybrid A: dense +0.1 progress bonus + full-magnitude terminal reward."""
    if is_final:
        return _terminal_reward(new_best_y, initial_range, optimum)
    ratio = _improvement_ratio(new_best_y, old_best_y, initial_range)
    return 0.1 if ratio > 1e-8 else 0


# Probably the best
def reward_hybrid_sign(new_best_y, old_best_y, initial_range, is_final=False, optimum=None):
    """Hybrid B: dense progress signal + full-magnitude terminal reward.

    Aggressive variant: higher base rewards, tighter threshold, stronger penalty.
    - Threshold 0.5% (step_ratio > scale * 5e-3) filters micro-steps
    - Reward range [0.1, 1.1]: large steps worth significantly more
    - Penalty -0.15 decayed: stronger motivation to keep searching
    """
    if is_final:
        return _terminal_reward(new_best_y, initial_range, optimum)

    scale = initial_range[1] - initial_range[0]
    step_ratio = _improvement_ratio(new_best_y, old_best_y, initial_range)
    if step_ratio > scale * 5e-3:
        return float(0.1 + 1.0 * np.clip(step_ratio, 0.0, 1.0))

    if optimum is not None:
        progress = max(_log_gap_orders(initial_range[0], new_best_y, optimum), 0.0)
    else:
        progress = max(_improvement_ratio(new_best_y, initial_range[0], initial_range), 0.0)
    return float(-0.15 / (1.0 + progress))


REWARD_FNS = {
    1: reward_log_scaled,
    2: reward_linear,
    3: reward_log_improvement,
    4: reward_sparse,
    5: reward_binary,
    6: reward_hybrid_binary,
    7: reward_hybrid_sign,
}


def compute_reward(
    new_best_y: float,
    old_best_y: float,
    initial_range: tuple[float, float],
    option: int = 1,
    is_final: bool = False,
    optimum: float | None = None,
) -> float:
    fn = REWARD_FNS.get(option)
    if fn is None:
        raise ValueError(
            f"Unknown reward option {option}. Choose from {list(REWARD_FNS)}"
        )
    return fn(new_best_y, old_best_y, initial_range, is_final, optimum)
