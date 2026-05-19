"""Observation (state) computation for the DAS environment.

The observation is a concatenation of:
  - 22 ELA (Exploratory Landscape Analysis) features from pflacco
  - 2*n_actions + 3 optimization-history features (action choices, entropy, etc.)
  - 2 progress features (FE ratio, stagnation ratio)

All features are returned unnormalized; normalization is applied via SB3's VecNormalize.
"""

import warnings

import numpy as np
import pandas as pd
from pflacco.classical_ela_features import (
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_dispersion,
    calculate_information_content,
    calculate_nbc,
)

MAX_DIM = 40
ELA_DIM = 22
MAX_HISTORY_SAMPLE = 2500

ELA_FEATURE_KEYS = [
    "ela_meta.lin_simple.coef.min",
    "ela_meta.lin_simple.coef.max",
    "ela_meta.lin_simple.coef.max_by_min",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_simple.cond",
    "ela_meta.quad_w_interact.adj_r2",
    "nbc.nn_nb.mean_ratio",
    "nbc.nn_nb.cor",
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor",
    "disp.ratio_mean_02",
    "disp.ratio_median_25",
    "disp.diff_mean_25",
    "disp.diff_median_02",
    "ic.h_max",
    "ic.eps_s",
    "ic.eps_max",
    "ic.m0",
    "ela_distr.skewness",
    "ela_distr.kurtosis",
    "ela_distr.number_of_peaks",
]


def compute_ela_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute 22 ELA features from a population sample.

    Falls back to zeros when the population is too small or degenerate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Slice to the most-recent samples first; deduplication is done below
        # in normalised space where it is actually meaningful — raw-space
        # np.unique missed points that become identical after normalisation and
        # was therefore doing redundant work without full correctness guarantees.
        x = x[-MAX_HISTORY_SAMPLE:]
        y = y[-MAX_HISTORY_SAMPLE:]

        x_norm_arr = (x - x.mean()) / (x.std() + 1e-8)
        y_norm_arr = (y - y.mean()) / (y.std() + 1e-8)

        x_df = pd.DataFrame(
            x_norm_arr, columns=[f"x_{i}" for i in range(x_norm_arr.shape[1])]
        )
        y_series = pd.Series(y_norm_arr)

        is_unique = ~x_df.duplicated()
        if not is_unique.all():
            x_df = x_df[is_unique].reset_index(drop=True)
            y_series = y_series[is_unique].reset_index(drop=True)

        if len(x_df) < 50 or np.var(y_series) < 1e-8:
            return np.zeros(ELA_DIM, dtype=np.float32)

        meta = calculate_ela_meta(x_df, y_series)
        nbc = calculate_nbc(x_df, y_series)
        disp = calculate_dispersion(x_df, y_series)
        ic = calculate_information_content(x_df, y_series)

        if (y**2).sum() > 0 and np.var(y_series) > 1e-8:
            ela_distr = calculate_ela_distribution(x_df, y_series)
        else:
            ela_distr = {
                k: 0.0
                for k in (
                    "ela_distr.skewness",
                    "ela_distr.kurtosis",
                    "ela_distr.number_of_peaks",
                )
            }

        # pflacco may return an incomplete dict for degenerate or edge-case
        # inputs that slipped past the variance guard above.  Fall back to
        # zeros rather than crashing training with a KeyError mid-run.
        try:
            all_feats = {**meta, **nbc, **disp, **ic, **ela_distr}
            return np.array([all_feats[k] for k in ELA_FEATURE_KEYS], dtype=np.float32)
        except (KeyError, ValueError):
            return np.zeros(ELA_DIM, dtype=np.float32)


def compute_action_history_features(
    choices_history: list[int],
    n_actions: int,
    n_checkpoints: int,
    ndim_problem: int,
) -> np.ndarray:
    """Encode action selection history as a fixed-size feature vector.

    Returns a vector of size 2*n_actions + 3:
      - one-hot of last chosen action (n_actions)
      - counter of consecutive same-action selections, normalized (1)
      - selection frequency per action (n_actions)
      - normalized choice entropy (1)
      - normalized problem dimensionality (1)
    """
    last_action = np.zeros(n_actions, dtype=np.float32)
    frequencies = np.zeros(n_actions, dtype=np.float32)
    same_action_count = 0.0
    entropy = 0.0

    if choices_history:
        last_idx = choices_history[-1]
        last_action[last_idx] = 1.0

        # O(n) instead of O(n_actions * n_steps) from calling list.count in a loop.
        counts = np.bincount(choices_history, minlength=n_actions).astype(np.float32)
        frequencies = counts / len(choices_history)

        run = 0
        for c in reversed(choices_history):
            if c == last_idx:
                run += 1
            else:
                break
        same_action_count = run / max(n_checkpoints, 1)

        log_n = np.log(n_actions) if n_actions > 1 else 1.0
        entropy = float(
            -(frequencies * np.nan_to_num(np.log(frequencies + 1e-12))).sum() / log_n
        )

    dim_norm = ndim_problem / MAX_DIM
    return np.concatenate(
        [last_action, [same_action_count], frequencies, [entropy, dim_norm]]
    )


def compute_progress_features(
    n_fe: int, max_fe: int, stagnation_count: int
) -> np.ndarray:
    """Encode optimization progress as a 2-element vector."""
    return np.array([n_fe / max_fe, stagnation_count / max_fe], dtype=np.float32)


def compute_observation(
    x_history: np.ndarray | None,
    y_history: np.ndarray | None,
    choices_history: list[int],
    n_actions: int,
    n_checkpoints: int,
    n_fe: int,
    max_fe: int,
    stagnation_count: int,
    ndim_problem: int,
    ela: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the full observation vector from its components."""
    # Accept a pre-computed ELA vector so the caller can cache it across steps
    # and avoid running pflacco on every observation build (pflacco is expensive).
    if ela is None:
        if x_history is not None and y_history is not None and len(x_history) >= 50:
            ela = compute_ela_features(x_history, y_history)
        else:
            ela = np.zeros(ELA_DIM, dtype=np.float32)

    action_hist = compute_action_history_features(
        choices_history, n_actions, n_checkpoints, ndim_problem
    )
    progress = compute_progress_features(n_fe, max_fe, stagnation_count)

    obs = np.concatenate([ela, action_hist, progress])
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def observation_dim(n_actions: int) -> int:
    """Total observation dimension for a given action space size."""
    return ELA_DIM + (2 * n_actions + 3) + 2
