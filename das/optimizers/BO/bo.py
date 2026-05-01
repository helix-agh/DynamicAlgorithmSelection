"""Gaussian Process Bayesian Optimization base class.

Structural pattern mirrors pypop7/optimizers/bo/bo.py: a thin base that
defines the initialize/iterate interface and provides shared GP machinery.
Subclasses implement `_acquisition()` only.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from das.optimizers.base import SubOptimizer


class BO(SubOptimizer):
    """GP Bayesian Optimization base.

    Maintains a growing dataset of (obs_x, obs_y) observations, fits a GP
    surrogate at each iteration, and selects the next candidate by maximising
    the acquisition function via multi-start L-BFGS-B in normalised space.

    Warm-start contract
    -------------------
    ``get_data()`` returns ``obs_x``/``obs_y`` (all observations) plus
    ``x``/``y`` (observations sorted by fitness) so that population-based
    optimizers can warm-start from BO's data after a hand-off.
    ``set_data()`` accepts either ``obs_x``/``obs_y`` (BO→BO) or ``x``/``y``
    (population-based → BO).
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        if self.n_individuals is None:
            self.n_individuals = max(5, 5 * self.ndim_problem)

        self.n_restarts_optimizer: int = options.get("n_restarts_optimizer", 5)
        self.n_restarts_acq: int = options.get("n_restarts_acq", 20)
        # Cap GP training set size to avoid O(n³) slowdown
        self.max_gp_points: int = options.get("max_gp_points", 200)

        self._gp: GaussianProcessRegressor | None = None
        self._lb = self.lower_boundary.astype(float)
        self._ub = self.upper_boundary.astype(float)
        self._range = np.where(self._ub > self._lb, self._ub - self._lb, 1.0)
        self._n_generations = 0

    # ------------------------------------------------------------------ #
    # GP helpers                                                           #
    # ------------------------------------------------------------------ #

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._lb) / self._range

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return self._lb + x_norm * self._range

    def _make_gp(self) -> GaussianProcessRegressor:
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.ndim_problem),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=int(self.rng_optimization.integers(2**31)),
        )

    def _fit_gp(self, obs_x: np.ndarray, obs_y: np.ndarray) -> None:
        if len(obs_x) > self.max_gp_points:
            best_idx = int(np.argmin(obs_y))
            other = [i for i in range(len(obs_x)) if i != best_idx]
            chosen = self.rng_optimization.choice(
                other, self.max_gp_points - 1, replace=False
            )
            idx = np.concatenate([[best_idx], chosen])
            obs_x, obs_y = obs_x[idx], obs_y[idx]
        self._gp = self._make_gp()
        self._gp.fit(self._normalize(obs_x), obs_y)

    # ------------------------------------------------------------------ #
    # Acquisition interface (subclasses override)                          #
    # ------------------------------------------------------------------ #

    def _acquisition(self, x_norm: np.ndarray, obs_y: np.ndarray) -> float:
        """Return scalar to *minimise* — i.e. negative utility."""
        raise NotImplementedError

    def _next_candidate(self, obs_x: np.ndarray, obs_y: np.ndarray) -> np.ndarray:
        bounds = [(0.0, 1.0)] * self.ndim_problem
        best_x_norm, best_val = None, float("inf")
        starts = self.rng_optimization.uniform(
            0.0, 1.0, (self.n_restarts_acq, self.ndim_problem)
        )
        for x0 in starts:
            try:
                res = minimize(
                    self._acquisition,
                    x0,
                    args=(obs_y,),
                    bounds=bounds,
                    method="L-BFGS-B",
                )
                if res.fun < best_val:
                    best_val, best_x_norm = res.fun, res.x
            except Exception:
                continue
        if best_x_norm is None:
            best_x_norm = self.rng_optimization.uniform(0.0, 1.0, self.ndim_problem)
        return np.clip(self._denormalize(best_x_norm), self._lb, self._ub)

    # ------------------------------------------------------------------ #
    # pypop7-style initialize / iterate / optimize                         #
    # ------------------------------------------------------------------ #

    def initialize(self, obs_x=None, obs_y=None):
        if obs_x is not None and obs_y is not None and len(obs_x) >= 2:
            return np.asarray(obs_x, dtype=float), np.asarray(obs_y, dtype=float)
        n_init = self.n_individuals
        obs_x = self.rng_initialization.uniform(
            self.initial_lower_boundary,
            self.initial_upper_boundary,
            (n_init, self.ndim_problem),
        )
        obs_y = np.empty(n_init)
        for i in range(n_init):
            if self._check_terminations():
                return obs_x[:i], obs_y[:i]
            obs_y[i] = self._evaluate_fitness(obs_x[i])
        return obs_x, obs_y

    def iterate(self, obs_x: np.ndarray, obs_y: np.ndarray):
        if self._check_terminations():
            return obs_x, obs_y
        if len(obs_x) < 2:
            x_new = self.rng_optimization.uniform(self._lb, self._ub)
            y_new = self._evaluate_fitness(x_new)
            obs_x = x_new[np.newaxis] if len(obs_x) == 0 else np.vstack([obs_x, x_new])
            obs_y = np.append(obs_y, y_new)
            self._n_generations += 1
            self._warm_start = {"obs_x": obs_x, "obs_y": obs_y}
            return obs_x, obs_y

        self._fit_gp(obs_x, obs_y)
        candidate = self._next_candidate(obs_x, obs_y)
        y_new = self._evaluate_fitness(candidate)
        obs_x = np.vstack([obs_x, candidate])
        obs_y = np.append(obs_y, y_new)
        self._n_generations += 1
        self._warm_start = {"obs_x": obs_x, "obs_y": obs_y}
        return obs_x, obs_y

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        ws = self._warm_start
        obs_x, obs_y = self.initialize(ws.get("obs_x"), ws.get("obs_y"))
        while not self.termination_signal:
            obs_x, obs_y = self.iterate(obs_x, obs_y)
        return self._collect(fitness)

    def _collect(self, fitness):
        result = super()._collect(fitness)
        result["_n_generations"] = self._n_generations
        return result

    # ------------------------------------------------------------------ #
    # Warm-start interface                                                 #
    # ------------------------------------------------------------------ #

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        # Accept obs_x/obs_y (BO→BO) or fall back to x/y (population→BO)
        obs_x = kwargs.get("obs_x") if kwargs.get("obs_x") is not None else x
        obs_y = kwargs.get("obs_y") if kwargs.get("obs_y") is not None else y
        if obs_x is not None and obs_y is not None and len(obs_x) >= 2:
            self._warm_start = {
                "obs_x": np.asarray(obs_x, dtype=float),
                "obs_y": np.asarray(obs_y, dtype=float),
            }
        else:
            self._warm_start = {}
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)

    def get_data(self) -> dict:
        obs_x = self._warm_start.get("obs_x")
        obs_y = self._warm_start.get("obs_y")
        if obs_x is None or len(obs_x) == 0:
            return {}
        idx = np.argsort(obs_y)
        return {
            "obs_x": obs_x,
            "obs_y": obs_y,
            "x": obs_x[idx],  # sorted best-first for population-based hand-off
            "y": obs_y[idx],
        }
