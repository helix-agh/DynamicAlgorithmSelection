"""NM: Nelder-Mead downhill simplex method."""

import numpy as np

from das.optimizers.base import SubOptimizer


class NM(SubOptimizer):
    """Nelder-Mead simplex direct-search method.

    Maintains a simplex of n+1 vertices and iteratively replaces the worst
    vertex via reflection, expansion, contraction, or shrinkage.
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.sigma: float = options.get("sigma", 1.0)
        self.alpha: float = options.get("alpha", 1.0)
        self.beta: float = options.get("beta", 0.5)
        self.gamma: float = options.get("gamma", 2.0)
        self.shrinkage: float = options.get("shrinkage", 0.5)
        self.n_individuals: int = self.ndim_problem + 1
        self._n_generations: int = 0

    def initialize(self, x=None, y=None):
        n, d = self.n_individuals, self.ndim_problem

        if x is not None and len(x) == n:
            sx = np.array(x, dtype=float)
            sy = np.array(y, dtype=float) if y is not None else None
            if sy is None:
                sy = np.empty(n)
                for i in range(n):
                    if self._check_terminations():
                        return sx, sy
                    sy[i] = self._evaluate_fitness(sx[i])
            self._warm_start = {"x": sx, "y": sy}
            return sx, sy

        # Seed from best available point
        if x is not None and len(x) > 0:
            best_idx = int(np.argmin(y)) if y is not None else 0
            x0 = np.clip(np.copy(x[best_idx]), self.lower_boundary, self.upper_boundary)
        elif self.best_so_far_x is not None:
            x0 = np.clip(
                np.copy(self.best_so_far_x), self.lower_boundary, self.upper_boundary
            )
        else:
            x0 = self.rng_initialization.uniform(
                self.initial_lower_boundary, self.initial_upper_boundary, d
            )

        sx = np.empty((n, d))
        sy = np.empty(n)
        sx[0] = x0
        sy[0] = self._evaluate_fitness(sx[0])

        for i in range(1, n):
            if self._check_terminations():
                return sx, sy
            sx[i] = np.copy(x0)
            offset = self.sigma * self.rng_initialization.uniform(-1.0, 1.0)
            sx[i, i - 1] = np.clip(
                x0[i - 1] + offset,
                self.lower_boundary[i - 1],
                self.upper_boundary[i - 1],
            )
            sy[i] = self._evaluate_fitness(sx[i])

        self._warm_start = {"x": sx, "y": sy}
        return sx, sy

    def iterate(self, x, y):
        order = np.argsort(y)
        l, h = order[0], order[-1]
        p_mean = np.mean(x[order[:-1]], axis=0)

        # Reflection
        p_r = np.clip(
            (1 + self.alpha) * p_mean - self.alpha * x[h],
            self.lower_boundary,
            self.upper_boundary,
        )
        y_r = self._evaluate_fitness(p_r)
        if self._check_terminations():
            return x, y

        if y_r < y[l]:
            # Expansion
            p_e = np.clip(
                self.gamma * p_r + (1 - self.gamma) * p_mean,
                self.lower_boundary,
                self.upper_boundary,
            )
            y_e = self._evaluate_fitness(p_e)
            if self._check_terminations():
                return x, y
            x[h], y[h] = (p_e, y_e) if y_e < y_r else (p_r, y_r)
        else:
            if np.all(y_r > y[order[:-1]]):
                if y_r <= y[h]:
                    x[h], y[h] = p_r, y_r
                # Contraction
                p_c = np.clip(
                    self.beta * x[h] + (1 - self.beta) * p_mean,
                    self.lower_boundary,
                    self.upper_boundary,
                )
                y_c = self._evaluate_fitness(p_c)
                if self._check_terminations():
                    return x, y
                if y_c > y[h]:
                    # Shrinkage
                    for i in range(1, self.n_individuals):
                        x[order[i]] = np.clip(
                            x[l] + self.shrinkage * (x[order[i]] - x[l]),
                            self.lower_boundary,
                            self.upper_boundary,
                        )
                        y[order[i]] = self._evaluate_fitness(x[order[i]])
                        if self._check_terminations():
                            return x, y
                else:
                    x[h], y[h] = p_c, y_c
            else:
                x[h], y[h] = p_r, y_r

        self._n_generations += 1
        self._warm_start = {"x": x, "y": y}
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        x, y = self.initialize(self._warm_start.get("x"), self._warm_start.get("y"))
        while not self.termination_signal:
            x, y = self.iterate(x, y)
        return self._collect(fitness)

    def _collect(self, fitness):
        result = super()._collect(fitness)
        result["_n_generations"] = self._n_generations
        return result

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        if x is not None and y is not None:
            self._warm_start = {"x": np.asarray(x), "y": np.asarray(y)}
        else:
            self._warm_start = {}
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)
