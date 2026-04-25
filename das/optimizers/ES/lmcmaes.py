"""LMCMAES: Limited Memory CMA-ES."""

import numpy as np

from .base import ES


class LMCMAES(ES):
    """Limited Memory CMA-ES (Loshchilov, 2014).

    Approximates the covariance matrix with m direction vectors, making
    each generation O(n·m) instead of O(n²). Uses the Population Success
    Rule (PSR) for step-size adaptation.

    Warm-start state keys
    ---------------------
    mean, x, p_c, s, vm, pm, b, d, y
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options | {"sigma": options.get("sigma", 1.5)})
        n, mu = self.ndim_problem, self.n_parents

        w_base = np.log((self.n_individuals + 1.0) / 2.0)
        w_raw  = np.log(np.arange(mu) + 1.0)
        self._w = (w_base - w_raw) / (mu * w_base - np.sum(w_raw))
        self._mu_eff = 1.0 / np.sum(np.square(self._w))

        self.m       = options.get("m",      4 + int(3 * np.log(n)))
        self.n_steps = options.get("n_steps", self.m)
        self.c_c     = options.get("c_c",    1.0 / self.m)
        self.c_1     = options.get("c_1",    1.0 / (10.0 * np.log(n + 1.0)))
        self.c_s     = options.get("c_s",    0.3)
        self.d_s     = options.get("d_s",    1.0)
        self.z_star  = options.get("z_star", 0.25)

        self._a    = np.sqrt(1.0 - self.c_1)
        self._c    = 1.0 / np.sqrt(1.0 - self.c_1)
        self._bd_1 = np.sqrt(1.0 - self.c_1)
        self._bd_2 = self.c_1 / (1.0 - self.c_1)
        self._p_c_1 = 1.0 - self.c_c
        self._p_c_2 = np.sqrt(self.c_c * (2.0 - self.c_c) * self._mu_eff)
        self._rr    = np.arange(self.n_individuals * 2, 0, -1) - 1

        self._j: list = []
        self._l: list = []
        self._it: int = 0

    def initialize(self, mean=None, x=None, p_c=None, s=None, vm=None, pm=None, b=None, d=None, y=None):
        lam, n = self.n_individuals, self.ndim_problem
        if mean is None:
            mean = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        x  = x  if x  is not None else np.zeros((lam, n))
        p_c = p_c if p_c is not None else np.zeros(n)
        s  = s  if s  is not None else 0.0
        vm = vm if vm is not None else np.zeros((self.m, n))
        pm = pm if pm is not None else np.zeros((self.m, n))
        b  = b  if b  is not None else np.zeros(self.m)
        d  = d  if d  is not None else np.zeros(self.m)

        self._j  = [None] * self.m
        self._l  = [None] * self.m
        self._it = 0

        if y is None:
            y = np.empty(lam)
            for i in range(lam):
                if self._check_terminations():
                    return mean, x, p_c, s, vm, pm, b, d, y
                x[i] = np.clip(
                    mean + self.sigma * self.rng_optimization.standard_normal(n),
                    self.lower_boundary, self.upper_boundary,
                )
                y[i] = self._evaluate_fitness(x[i])
        return mean, x, p_c, s, vm, pm, b, d, y

    def _a_z(self, z, pm, vm, b) -> np.ndarray:
        """Algorithm 3 Az(): apply approximate covariance transformation."""
        x = np.copy(z)
        for t in range(self._it):
            x = self._a * x + b[self._j[t]] * np.dot(vm[self._j[t]], z) * pm[self._j[t]]
        return x

    def _a_inv_z(self, v, vm, d, i) -> np.ndarray:
        """Algorithm 4 Ainvz(): apply inverse transformation up to index i."""
        x = np.copy(v)
        for t in range(i):
            dot = np.dot(vm[self._j[t]], x)
            x = self._c * x - d[self._j[t]] * dot * vm[self._j[t]]
        return x

    def iterate(self, mean, x, pm, vm, y, b):
        sign = 1
        a_z  = np.empty(self.ndim_problem)
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            if sign == 1:
                z   = self.rng_optimization.standard_normal(self.ndim_problem)
                a_z = self._a_z(z, pm, vm, b)
            step = sign * self.sigma * a_z
            if not np.all(np.isfinite(step)):
                step = np.zeros_like(step)
            x[k] = mean + step
            y[k] = self._evaluate_fitness(x[k])
            sign *= -1
        return x, y

    def _update_distribution(self, mean, x, p_c, s, vm, pm, b, d, y, y_bak):
        order    = np.argsort(y)[: self.n_parents]
        mean_new = np.dot(self._w, x[order])

        safe_sigma = max(float(self.sigma), 1e-20)
        p_c = self._p_c_1 * p_c + self._p_c_2 * (mean_new - mean) / safe_sigma

        i_min = 1
        if self._n_generations < self.m:
            self._j[self._n_generations] = self._n_generations
        else:
            d_min = self._l[self._j[i_min]] - self._l[self._j[i_min - 1]]
            for j in range(2, self.m):
                d_cur = self._l[self._j[j]] - self._l[self._j[j - 1]]
                if d_cur < d_min:
                    d_min, i_min = d_cur, j
            i_min = 0 if d_min >= self.n_steps else i_min
            updated = self._j[i_min]
            for j in range(i_min, self.m - 1):
                self._j[j] = self._j[j + 1]
            self._j[self.m - 1] = updated

        self._it = min(self._n_generations + 1, self.m)
        self._l[self._j[self._it - 1]] = self._n_generations
        pm[self._j[self._it - 1]] = p_c

        start = 0 if i_min == 1 else i_min
        for i in range(start, self._it):
            vm[self._j[i]] = self._a_inv_z(pm[self._j[i]], vm, d, i)
            v_n = max(float(np.dot(vm[self._j[i]], vm[self._j[i]])), 1e-20)
            bd_3 = np.sqrt(1.0 + self._bd_2 * v_n)
            b[self._j[i]] = self._bd_1 / v_n * (bd_3 - 1.0)
            d[self._j[i]] = 1.0 / (self._bd_1 * v_n) * (1.0 - 1.0 / bd_3)

        if self._n_generations > 0:
            r     = np.argsort(np.hstack((y, y_bak)))
            z_psr = np.sum(self._rr[r < self.n_individuals] - self._rr[r >= self.n_individuals])
            z_psr = z_psr / self.n_individuals**2 - self.z_star
            s     = (1.0 - self.c_s) * s + self.c_s * z_psr
            s     = np.clip(s, -50.0, 50.0)
            self.sigma *= np.exp(s / self.d_s)

        return mean_new, p_c, s, vm, pm, b, d

    def optimize(self, fitness_function=None, args=None):
        fitness = super(ES, self).optimize(fitness_function)
        ws = self._warm_start
        mean, x, p_c, s, vm, pm, b, d, y = self.initialize(
            ws.get("mean"), ws.get("x"), ws.get("p_c"), ws.get("s"),
            ws.get("vm"),   ws.get("pm"), ws.get("b"),  ws.get("d"), ws.get("y"),
        )
        while not self.termination_signal:
            y_bak = np.copy(y)
            x, y = self.iterate(mean, x, pm, vm, y, b)
            if self._check_terminations():
                break
            mean, p_c, s, vm, pm, b, d = self._update_distribution(
                mean, x, p_c, s, vm, pm, b, d, y, y_bak
            )
            self._n_generations += 1
            self._warm_start = {
                "mean": mean, "x": x, "p_c": p_c, "s": s,
                "vm": vm, "pm": pm, "b": b, "d": d, "y": y,
            }
        return self._collect(fitness)

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        n, m = self.ndim_problem, self.m
        _shapes = {"p_c": (n,), "s": (), "vm": (m, n), "pm": (m, n), "b": (m,), "d": (m,)}

        def _valid(key, val):
            if val is None:
                return False
            if key == "s":
                return np.isscalar(val)
            return np.asarray(val).shape == _shapes[key]

        ws = {k: kwargs.get(k) if _valid(k, kwargs.get(k)) else None for k in _shapes}
        if x is not None and y is not None and len(x) >= 1:
            idx = np.argsort(y)[: self.n_individuals]
            x_sub, y_sub = x[idx], y[idx]
            ws["mean"] = x_sub.mean(axis=0)
            self.sigma = max(float(np.max(np.std(x_sub, axis=0))), 1e-8)
            ws["x"] = x_sub
            ws["y"] = y_sub
        else:
            ws["mean"] = None
            ws["x"]    = None
            ws["y"]    = None
        self._warm_start = ws
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)

    def get_data(self) -> dict:
        return dict(self._warm_start)
