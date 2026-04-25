"""Base sub-optimizer for the DAS portfolio.

Every sub-optimizer in the portfolio must extend SubOptimizer. It adds:
  - x/y population history tracking (needed for ELA features)
  - target_fe early stopping (so we can run for exactly one checkpoint interval)
  - set_data / get_data interface for warm-starting between optimizer switches

The subclass interface follows pypop7 conventions so all existing pypop7-based
algorithms can be adapted with minimal changes.
"""

import time
from typing import Any

import numpy as np
from pypop7.optimizers.core import Optimizer as _Pypop7Base, Terminations


class SubOptimizer(_Pypop7Base):
    """Base class for all portfolio optimizers in DAS.

    Warm-starting contract
    ----------------------
    After running, call `get_data()` to get the population state dict.
    Before running, call `set_data(**state)` with that dict to warm-start.
    Subclasses override `set_data` / `get_data` to add algorithm-specific keys.
    Unknown keys in `set_data` are silently ignored so different optimizer types
    can hand off to each other without type checks.
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.best_so_far_y: float = options.get("best_so_far_y", float("inf"))
        self.best_so_far_x: np.ndarray | None = None
        self.worst_so_far_y: float = -np.inf
        self.worst_so_far_x: np.ndarray | None = None

        self.x_history: list[np.ndarray] = []
        self.y_history: list[float] = []
        self.fitness_history: list[tuple[int, float]] = []  # (n_fe, best_y) pairs

        self.target_fe: int = options.get("target_fe", int(1e9))
        self._warm_start: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # pypop7 hooks                                                         #
    # ------------------------------------------------------------------ #

    def _evaluate_fitness(self, x: np.ndarray, args=None) -> float:
        t0 = time.time()
        y = self.fitness_function(x) if args is None else self.fitness_function(x, args=args)
        self.time_function_evaluations += time.time() - t0
        self.n_function_evaluations += 1
        y_val = float(y)

        if y_val < self.best_so_far_y:
            self.best_so_far_x = np.copy(x)
            self.best_so_far_y = y_val
            self.fitness_history.append((self.n_function_evaluations, y_val))
        if y_val > self.worst_so_far_y:
            self.worst_so_far_x = np.copy(x)
            self.worst_so_far_y = y_val

        self.x_history.append(np.copy(x))
        self.y_history.append(y_val)
        return y_val

    def _check_terminations(self) -> bool:
        terminated = super()._check_terminations()
        if not terminated and self.n_function_evaluations >= self.target_fe:
            self.termination_signal = Terminations.MAX_FUNCTION_EVALUATIONS
            terminated = True
        return terminated

    def _collect(self, fitness: list) -> dict:
        result = super()._collect(fitness)
        result.update(
            {
                "x_history": np.array(self.x_history, dtype=np.float32),
                "y_history": np.array(self.y_history, dtype=np.float32),
                "fitness_history": self.fitness_history,
                "best_so_far_x": self.best_so_far_x,
                "best_so_far_y": self.best_so_far_y,
                "worst_so_far_x": self.worst_so_far_x,
                "worst_so_far_y": self.worst_so_far_y,
            }
        )
        return result

    # ------------------------------------------------------------------ #
    # Warm-start interface                                                 #
    # ------------------------------------------------------------------ #

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        """Load population from a previous optimizer run."""
        self._warm_start = {"x": x, "y": y}
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)

    def get_data(self) -> dict:
        """Return current population state for the next optimizer."""
        return dict(self._warm_start)


def get_checkpoints(n_checkpoints: int, max_fe: int, n_individuals: int, cdb: float) -> np.ndarray:
    """Compute exponentially-spaced checkpoint FE targets.

    cdb == 1.0  → uniform spacing
    cdb > 1.0   → early checkpoints are shorter (exponential growth)
    """
    ratios = np.cumprod(np.full(n_checkpoints, cdb))
    ratios = np.cumsum(ratios / ratios.sum())
    checkpoints = (ratios * max_fe).astype(int)
    checkpoints[-1] = max_fe
    checkpoints[0] = max(checkpoints[0], n_individuals)
    for i in range(1, n_checkpoints):
        checkpoints[i] = max(checkpoints[i - 1] + n_individuals, checkpoints[i])
    return checkpoints
