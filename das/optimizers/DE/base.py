"""DE base classes shared across all Differential Evolution variants."""

import numpy as np

from das.optimizers.base import SubOptimizer


class DE(SubOptimizer):
    """DE base: current-to-best/1 mutation with binomial crossover."""

    Nmin: int = 30
    Nmax: int = 170

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100
        self.F:  float = options.get("F",  0.5)
        self.CR: float = options.get("CR", 0.9)
        self._n_generations = 0

    def initialize(self, x=None, y=None):
        needs_eval = y is None
        shape = (self.n_individuals, self.ndim_problem)
        x = x if x is not None else self.rng_initialization.uniform(
            self.initial_lower_boundary, self.initial_upper_boundary, shape
        )
        y = y if y is not None else np.empty(self.n_individuals)
        if needs_eval:
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return x, y
                y[i] = self._evaluate_fitness(x[i])
        return x, y

    def _mutate(self, x, y, i):
        others = [j for j in range(self.n_individuals) if j != i]
        r1, r2, r3 = self.rng_optimization.choice(others, 3, replace=False)
        best = int(np.argmin(y))
        return x[best] + self.F * (x[r1] - x[r2]) + self.F * (x[r3] - x[i])

    def _crossover(self, target, donor):
        mask = self.rng_optimization.random(self.ndim_problem) < self.CR
        mask[self.rng_optimization.integers(self.ndim_problem)] = True
        return np.clip(np.where(mask, donor, target), self.lower_boundary, self.upper_boundary)

    def iterate(self, x, y):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            trial_y = self._evaluate_fitness(self._crossover(x[i], self._mutate(x, y, i)))
            if trial_y <= y[i]:
                x[i], y[i] = self._crossover(x[i], self._mutate(x, y, i)), trial_y
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
        if x is None or y is None or len(x) < self.n_individuals:
            self._warm_start = {}
        else:
            self._warm_start = {"x": x[: self.n_individuals], "y": y[: self.n_individuals]}
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)


class _SHADEBase(DE):
    """Common memory / archive machinery shared by MADDE and NL_SHADE_RSP."""

    def __init__(self, problem: dict, options: dict):
        options = dict(options)
        options.setdefault("n_individuals", self.Nmax)
        super().__init__(problem, options)
        D = self.ndim_problem
        self.memory_size: int = 20 * D
        self.MF  = np.ones(self.memory_size) * 0.2
        self.MCr = np.ones(self.memory_size) * 0.2
        self.k_idx: int = 0
        self.archive = np.empty((0, D))

    def _choose_F_Cr(self, NP: int):
        idx = self.rng_optimization.integers(0, self.memory_size, size=NP)
        Cr  = np.clip(self.rng_optimization.normal(loc=self.MCr[idx], scale=0.1), 0.0, 1.0)
        locs = self.MF[idx]
        F   = locs + 0.1 * self.rng_optimization.standard_cauchy(size=NP)
        neg = F < 0
        F[neg] = 2 * locs[neg] - F[neg]
        return Cr, np.minimum(1.0, F)

    def _update_memory(self, SF, SCr, df):
        if len(SF) > 0:
            w = df / np.sum(df)
            self.MF[self.k_idx]  = np.sum(w * SF**2) / np.sum(w * SF)   if np.sum(w * SF)  > 1e-6 else 0.5
            self.MCr[self.k_idx] = np.sum(w * SCr**2) / np.sum(w * SCr) if np.sum(w * SCr) > 1e-6 else 0.5
        else:
            self.MF[self.k_idx] = 0.5; self.MCr[self.k_idx] = 0.5
        self.k_idx = (self.k_idx + 1) % self.memory_size

    def _nlpsr(self, x, y, A_rate: float = 2.1):
        ratio  = min(1.0, self.n_function_evaluations / self.max_function_evaluations)
        new_NP = max(self.Nmin, int(np.round(self.Nmax + (self.Nmin - self.Nmax) * ratio ** (1.0 - ratio))))
        if new_NP < x.shape[0]:
            keep = np.argsort(y)[:new_NP]
            x, y = x[keep], y[keep]
            self.n_individuals = new_NP
            self.NA = max(self.Nmin, int(np.round(A_rate * new_NP)))
            if len(self.archive) > self.NA:
                self.archive = self.archive[: self.NA]
        return x, y

    def _archive_add(self, loser: np.ndarray):
        if len(self.archive) < self.NA:
            self.archive = np.vstack([self.archive, loser[np.newaxis]])
        else:
            ri = self.rng_optimization.integers(len(self.archive))
            self.archive[ri] = loser

    def _binomial(self, x, v, Cr):
        NP, dim = x.shape
        mask = self.rng_optimization.random((NP, dim)) < Cr[:, np.newaxis]
        mask[np.arange(NP), self.rng_optimization.integers(dim, size=NP)] = True
        return np.where(mask, v, x)

    def _unique_indices(self, exclude: list[int], pool_size: int, n: int, max_tries: int = 25) -> np.ndarray:
        idx = self.rng_optimization.integers(0, pool_size, n)
        excl = np.array(exclude)
        for _ in range(max_tries):
            bad = np.isin(idx, excl)
            if not bad.any():
                break
            idx[bad] = self.rng_optimization.integers(0, pool_size, bad.sum())
            excl = np.append(excl, idx[~bad])
        return idx

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        if x is not None and y is not None and isinstance(y, np.ndarray):
            idx = np.argsort(y)[: self.n_individuals]
            self._warm_start = {"x": x[idx], "y": y[idx]}
            for key in ("archive", "MF", "MCr", "k_idx"):
                if key in kwargs and kwargs[key] is not None:
                    setattr(self, key, kwargs[key])
        else:
            self._warm_start = {}
        if best_x is not None: self.best_so_far_x = np.copy(best_x)
        if best_y is not None: self.best_so_far_y = float(best_y)
