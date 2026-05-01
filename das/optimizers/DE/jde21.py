"""JDE21: Self-adaptive DE with big + small sub-populations."""

import numpy as np

from .base import DE


class JDE21(DE):
    """Self-adaptive DE with big (NP=160) + small (NP=10) sub-populations."""

    def __init__(self, problem: dict, options: dict):
        options = dict(options)
        self.bNP = 160
        self.sNP = 10
        options["n_individuals"] = self.bNP + self.sNP
        super().__init__(problem, options)

        self.tau1 = 0.1
        self.tau2 = 0.1
        self.Fl_b = 0.1
        self.Fu = 1.1
        self.CRl_b = 0.0
        self.CRu_b = 1.1
        self.Fl_s = 0.17
        self.CRl_s = 0.1
        self.CRu_s = 0.8
        self.Finit = 0.5
        self.CRinit = 0.9
        self.age_limit = int(0.25 * self.ndim_problem * self.bNP)
        self._F = np.full(self.n_individuals, self.Finit)
        self._CR = np.full(self.n_individuals, self.CRinit)
        self._age = 0

    def _adapt(self, i, is_big):
        Fl = self.Fl_b if is_big else self.Fl_s
        F = (
            Fl + self.rng_optimization.random() * (self.Fu - Fl)
            if self.rng_optimization.random() < self.tau1
            else self._F[i]
        )
        CRl = self.CRl_b if is_big else self.CRl_s
        CRu = self.CRu_b if is_big else self.CRu_s
        CR = (
            CRl + self.rng_optimization.random() * (CRu - CRl)
            if self.rng_optimization.random() < self.tau2
            else self._CR[i]
        )
        return F, CR

    def iterate(self, x, y):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            F, CR = self._adapt(i, i < self.bNP)
            others = [j for j in range(self.n_individuals) if j != i]
            r1, r2, r3 = self.rng_optimization.choice(others, 3, replace=False)
            donor = np.clip(
                x[int(np.argmin(y))] + F * (x[r1] - x[r2]) + F * (x[r3] - x[i]),
                self.lower_boundary,
                self.upper_boundary,
            )
            mask = self.rng_optimization.random(self.ndim_problem) < CR
            mask[self.rng_optimization.integers(self.ndim_problem)] = True
            trial_y = self._evaluate_fitness(np.where(mask, donor, x[i]))
            if trial_y <= y[i]:
                x[i], y[i] = np.where(mask, donor, x[i]), trial_y
                self._F[i], self._CR[i] = F, CR

        self._age += 1
        if self._age >= self.age_limit:
            self._age = 0
            for slot in (
                int(np.argmax(y[: self.bNP])),
                self.bNP + int(np.argmin(y[self.bNP :])),
            ):
                x[slot] = self.rng_initialization.uniform(
                    self.initial_lower_boundary,
                    self.initial_upper_boundary,
                    self.ndim_problem,
                )
                y[slot] = self._evaluate_fitness(x[slot])

        self._n_generations += 1
        self._warm_start = {"x": x, "y": y, "F": self._F, "CR": self._CR}
        return x, y

    def initialize(self, x=None, y=None):
        F = self._warm_start.get("F")
        CR = self._warm_start.get("CR")
        if F is not None:
            self._F = np.copy(F)
        if CR is not None:
            self._CR = np.copy(CR)
        return super().initialize(x, y)

    def optimize(self, fitness_function=None, args=None):
        fitness = super(DE, self).optimize(fitness_function)
        x, y = self.initialize(self._warm_start.get("x"), self._warm_start.get("y"))
        while not self.termination_signal:
            x, y = self.iterate(x, y)
        return self._collect(fitness)

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        if x is None or y is None or len(x) < self.n_individuals:
            self._warm_start = {}
        else:
            F_kw = kwargs.get("F")
            CR_kw = kwargs.get("CR")
            if F_kw is not None:
                self._F = np.copy(F_kw)
            if CR_kw is not None:
                self._CR = np.copy(CR_kw)
            self._warm_start = {
                "x": x[: self.n_individuals],
                "y": y[: self.n_individuals],
                "F": self._F,
                "CR": self._CR,
            }
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)
