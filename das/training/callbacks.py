"""Stable Baselines 3 callbacks for DAS training.

DASEvalCallback  : Runs evaluation episodes on a held-out problem set and
                   logs mean best_y and AOCC metrics.
WandbCallback    : Forwards SB3 logger output to Weights & Biases.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ------------------------------------------------------------------ #
# AOCC metric                                                          #
# ------------------------------------------------------------------ #


def _aocc(
    fitness_history: list[tuple[int, float]], max_fe: int, optimum: float
) -> float:
    """Area Over the Convergence Curve, normalised to [0, 1]."""
    lb, ub = -8.0, 8.0
    area = 0.0
    prev_fe = 0
    for fe, f in fitness_history:
        shifted = np.clip(f - optimum, 1e-8, 1e8)
        normalised = (np.log10(shifted) - lb) / (ub - lb)
        area += (1.0 - normalised) * (fe - prev_fe)
        prev_fe = fe
    if fitness_history:
        last_f = np.clip(fitness_history[-1][1] - optimum, 1e-8, 1e8)
        normalised = (np.log10(last_f) - lb) / (ub - lb)
        area += (1.0 - normalised) * (max_fe - fitness_history[-1][0])
    return area / max_fe


# ------------------------------------------------------------------ #
# Result persistence                                                   #
# ------------------------------------------------------------------ #


def dump_result(name: str, problem_id: str, best_y: float, aocc: float):
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{name}.jsonl"), "a") as f:
        f.write(json.dumps({problem_id: {"best_y": best_y, "aocc": aocc}}) + "\n")


# ------------------------------------------------------------------ #
# Weights & Biases callback                                           #
# ------------------------------------------------------------------ #


class WandbCallback(BaseCallback):
    """Logs SB3 training metrics to an active wandb run."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        if not HAS_WANDB:
            raise ImportError("wandb is not installed.")

    def _on_step(self) -> bool:
        if self.logger is not None and HAS_WANDB and wandb.run is not None:
            for key, value in self.logger.name_to_value.items():
                wandb.log({key: value}, step=self.num_timesteps)
        return True


# ------------------------------------------------------------------ #
# Evaluation callback                                                  #
# ------------------------------------------------------------------ #


class DASEvalCallback(BaseCallback):
    """Periodically evaluates the current policy on held-out BBOB problems.

    Parameters
    ----------
    eval_env:
        A VecEnv wrapping a DASEnv configured with the test problem set.
    eval_freq:
        Evaluate every `eval_freq` training steps.
    n_eval_episodes:
        Number of problems to evaluate (taken from the start of the problem list).
    name:
        Prefix for result files written to results/.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        name: str = "eval",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.name = name
        self._best_mean_aocc = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        best_ys = []
        obs = self.eval_env.reset()

        for _ in range(self.n_eval_episodes):
            done = [False]
            ep_info: dict[str, Any] = {}
            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, infos = self.eval_env.step(action)
                if done[0]:
                    ep_info = infos[0]

            best_y = ep_info.get("best_y", float("inf"))
            best_ys.append(best_y)

        mean_best = float(np.mean(best_ys))
        self.logger.record("eval/mean_best_y", mean_best)

        if self.verbose:
            print(
                f"[EvalCallback] step={self.num_timesteps}  mean_best_y={mean_best:.4e}"
            )
