"""NL_SHADE_RSP: Non-linear SHADE with Rank-based Selection Pressure."""

import numpy as np

from .base import _SHADEBase, DE


class NL_SHADE_RSP(_SHADEBase):
    """Non-linear SHADE with Rank-based Selection Pressure."""

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.pa: float = 0.5
        self.NA: int   = int(self.Nmax * 2.1)

    def _sample_cauchy(self, loc, size):
        return loc + 0.1 * np.tan(np.pi * (self.rng_optimization.random(size) - 0.5))

    def _choose_F_Cr(self, NP):
        idx  = self.rng_optimization.integers(0, self.memory_size, NP)
        Cr   = np.clip(self.rng_optimization.normal(self.MCr[idx], 0.1, NP), 0.0, 1.0)
        locs = self.MF[idx]
        F    = self._sample_cauchy(locs, NP)
        neg  = F < 0; F[neg] = 2 * locs[neg] - F[neg]
        return Cr, np.minimum(1.0, F)

    def iterate(self, x, y):
        order = np.argsort(y); x, y = x[order], y[order]
        NP    = x.shape[0]
        ratio = self.n_function_evaluations / self.max_function_evaluations

        Cr, F = self._choose_F_Cr(NP)
        Cr    = np.sort(Cr)

        pb       = 0.4 - 0.2 * ratio
        pb_upper = max(2, int(np.round(NP * pb)))
        Cr_b     = 2.0 * (ratio - 0.5) if ratio < 0.5 else 0.0

        ranks = np.exp(-(np.arange(NP) + 1) / NP)
        pr    = ranks / ranks.sum()

        use_arc = self.rng_optimization.random(NP) < self.pa
        if len(self.archive) < 25:
            use_arc[:] = False

        pbest_idx = np.zeros(NP, dtype=int)
        r1        = np.zeros(NP, dtype=int)
        x2        = np.zeros_like(x)

        for i in range(NP):
            pb_i = self.rng_optimization.integers(0, pb_upper)
            if pb_i == i:
                pb_i = self.rng_optimization.integers(0, NP)
            pbest_idx[i] = pb_i

            r1_i = self.rng_optimization.integers(0, NP)
            for _ in range(25):
                if r1_i != i and r1_i != pb_i: break
                r1_i = self.rng_optimization.integers(0, NP)
            r1[i] = r1_i

            if use_arc[i] and len(self.archive) > 0:
                x2[i] = self.archive[self.rng_optimization.integers(0, min(len(self.archive), self.NA))]
            else:
                r2_i = int(self.rng_optimization.choice(NP, p=pr))
                for _ in range(25):
                    if r2_i not in (i, pb_i, r1_i): break
                    r2_i = int(self.rng_optimization.choice(NP, p=pr))
                x2[i] = x[r2_i]

        v = x + F[:, None] * (x[pbest_idx] - x) + F[:, None] * (x[r1] - x2)

        u = np.copy(x)
        if self.rng_optimization.random() < 0.5:
            for i in range(NP):
                jrand = self.rng_optimization.integers(self.ndim_problem)
                for j in range(self.ndim_problem):
                    if self.rng_optimization.random() < Cr_b or j == jrand:
                        u[i, j] = v[i, j]
        else:
            dim  = self.ndim_problem
            L    = self.rng_optimization.integers(dim, size=NP).repeat(dim).reshape(NP, dim) <= np.arange(dim)
            mask = np.where(self.rng_optimization.random((NP, dim)) > Cr[:, None], L, False)
            u    = np.where(mask, v, u)

        out = (u < -100) | (u > 100)
        if out.any():
            u[out] = self.rng_optimization.uniform(-100, 100, out.sum())

        new_y  = np.array([self._evaluate_fitness(ui) for ui in u])
        better = new_y < y
        df     = np.array([])

        if better.any():
            df    = (y[better] - new_y[better]) / (y[better] + 1e-9)
            arc_b = use_arc[better]
            fp    = df[ arc_b].sum(); fa = df[~arc_b].sum()
            na    = arc_b.sum()
            if na > 0 and fa > 0:
                self.pa = np.clip(fa / (na + 1e-15) / (fa / (na + 1e-15) + fp / (NP - na + 1e-15)), 0.1, 0.9)
            for i in np.where(better)[0]:
                self._archive_add(x[i])
            x[better] = u[better]; y[better] = new_y[better]

        self._update_memory(F[better], Cr[better], df)
        x, y = self._nlpsr(x, y, A_rate=2.1)
        self._n_generations += 1
        self._warm_start = {"x": x, "y": y, "archive": self.archive,
                            "MF": self.MF, "MCr": self.MCr, "k_idx": self.k_idx}
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = super(DE, self).optimize(fitness_function)
        self.pa  = 0.5
        x, y = self.initialize(self._warm_start.get("x"), self._warm_start.get("y"))
        while not self.termination_signal:
            prev_fe = self.n_function_evaluations
            x, y = self.iterate(x, y)
            if self._check_terminations() or self.n_function_evaluations == prev_fe:
                break
        return self._collect(fitness)
