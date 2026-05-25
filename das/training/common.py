"""Shared training utilities."""

import json
import os

import numpy as np

from das.env.das_env import DASEnv


def load_global_optima(path: str = "bbob_optima.jsonl") -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v for line in f for k, v in json.loads(line).items()}


# Standard BBOB precision targets (excess above the global optimum).
# Keys are used as JSON dict keys: "1e+02", "1e+01", ..., "1e-08".
ERT_TARGETS: tuple[float, ...] = tuple(10.0**k for k in range(2, -9, -1))


def _ert_key(t: float) -> str:
    return f"{t:.0e}"


def compute_run_stats(
    fitness_history: list[tuple[int, float]],
    max_fe: int,
    global_minimum: float,
) -> dict:
    """Compute AUOC, AOCC, hitting times and ERT budget from a run history.

    Parameters
    ----------
    fitness_history:
        List of (n_fe, best_y) pairs recorded at every improvement point.
        FE counts must be absolute (1-indexed, relative to episode start).
    max_fe:
        Total function-evaluation budget for the episode.
    global_minimum:
        Known optimum value; used to shift fitness so the excess is >= 0.
        Pass 0.0 when the optimum is unknown.

    Returns
    -------
    dict with keys:
        area_under_optimization_curve, aocc, final_fitness,
        hitting_times  – {target_key: first_fe | null} for each ERT_TARGET,
        max_fe         – the budget (needed for ERT aggregation).
    """
    empty_hits = {_ert_key(t): None for t in ERT_TARGETS}
    if not fitness_history or max_fe <= 0:
        return {
            "area_under_optimization_curve": 0.0,
            "aocc": 0.0,
            "final_fitness": 0.0,
            "hitting_times": empty_hits,
            "max_fe": max_fe,
        }

    lb, ub = 1e-8, 1e8
    log_lb, log_ub = -8.0, 8.0

    auoc_area = 0.0
    aocc_area = 0.0
    prev_fe = 0

    for fe, best_y in fitness_history:
        shifted = best_y - global_minimum
        width = fe - prev_fe

        auoc_area += shifted * width

        clipped = np.clip(shifted, lb, ub)
        normalized = (np.log10(clipped) - log_lb) / (log_ub - log_lb)
        aocc_area += (1.0 - normalized) * width

        prev_fe = fe

    # Final plateau from last improvement to end of budget
    final_shifted = fitness_history[-1][1] - global_minimum
    final_width = max_fe - fitness_history[-1][0]
    auoc_area += final_shifted * final_width
    final_clipped = np.clip(final_shifted, lb, ub)
    final_normalized = (np.log10(final_clipped) - log_lb) / (log_ub - log_lb)
    aocc_area += (1.0 - final_normalized) * final_width

    # Hitting times: first FE where shifted best_y ≤ target
    hitting_times: dict[str, int | None] = {}
    for target in ERT_TARGETS:
        hit_fe = None
        for fe, best_y in fitness_history:
            if best_y - global_minimum <= target:
                hit_fe = fe
                break
        hitting_times[_ert_key(target)] = hit_fe

    return {
        "area_under_optimization_curve": auoc_area / max_fe,
        "aocc": aocc_area / max_fe,
        "final_fitness": final_shifted,
        "hitting_times": hitting_times,
        "max_fe": max_fe,
    }


def make_das_env(problem_ids: list[str], optimizers: list, cfg: dict):
    """Return a zero-argument factory for use with SB3's make_vec_env."""

    def _init():
        from das.env.ioh_suite import IOHSuite

        _suite = IOHSuite()
        return DASEnv(
            problem_ids=problem_ids,
            suite=_suite,
            optimizers=optimizers,
            fe_multiplier=cfg["fe_multiplier"],
            n_checkpoints=cfg["n_checkpoints"],
            checkpoint_division_base=cfg["cdb"],
            reward_option=cfg["reward_option"],
            n_individuals=cfg["n_individuals"],
            seed=cfg.get("seed"),
        )

    return _init


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
