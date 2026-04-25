"""MADDE: Multiple-strategy Adaptive DE with archive and NLPSR."""

import numpy as np

from .base import _SHADEBase, DE


class MADDE(_SHADEBase):
    """Multiple-strategy Adaptive DE with archive and NLPSR (MadDE, 2021)."""

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.p    = 0.18
        self.PqBX = 0.01
        self.pm   = np.ones(3) / 3       # strategy selection probabilities
        self.NA   = int(np.round(2.10 * self.Nmax))

    def _ctb_w_arc(self, x, p_best, F):
        """Current-to-pbest/1 with archive."""
        NP = x.shape[0]; combined = np.vstack([x, self.archive]) if len(self.archive) else x
        rb = self._unique_indices([], len(p_best),  NP)
        r1 = self._unique_indices([], NP,            NP)
        r2 = self._unique_indices([], len(combined), NP)
        return x + F[:, None] * (p_best[rb] - x) + F[:, None] * (x[r1] - combined[r2])

    def _ctr_w_arc(self, x, F):
        """Current-to-rand/1 with archive."""
        NP = x.shape[0]; combined = np.vstack([x, self.archive]) if len(self.archive) else x
        r1 = self._unique_indices([], NP,            NP)
        r2 = self._unique_indices([], len(combined), NP)
        return x + F[:, None] * (x[r1] - combined[r2])

    def _weighted_rtb(self, x, q_best, F, Fa):
        """Weighted rand-to-best/1."""
        NP = x.shape[0]
        rb = self._unique_indices([], len(q_best), NP)
        r1 = self._unique_indices([], NP, NP)
        r2 = self._unique_indices([], NP, NP)
        return F[:, None] * x[r1] + (F * Fa)[:, None] * (q_best[rb] - x[r2])

    def iterate(self, x, y):
        order = np.argsort(y); x, y = x[order], y[order]
        NP    = x.shape[0]
        ratio = self.n_function_evaluations / self.max_function_evaluations
        q     = 2 * self.p - self.p * ratio
        Fa    = 0.5 + 0.5 * ratio

        Cr, F = self._choose_F_Cr(NP)
        mu    = self.rng_optimization.choice(3, NP, p=self.pm)

        p_best = x[: max(int(self.p * NP), 2)]
        q_best = x[: max(int(q    * NP), 2)]

        v = np.zeros_like(x)
        m0, m1, m2 = mu == 0, mu == 1, mu == 2
        if m0.any(): v[m0] = self._ctb_w_arc(x[m0], p_best, F[m0])
        if m1.any(): v[m1] = self._ctr_w_arc(x[m1], F[m1])
        if m2.any(): v[m2] = self._weighted_rtb(x[m2], q_best, F[m2], Fa)

        lo, hi = self.lower_boundary, self.upper_boundary
        v = np.where(v < lo, (x + lo) / 2, np.where(v > hi, (x + hi) / 2, v))

        rvs = self.rng_optimization.random(NP)
        u   = np.copy(x)
        bu  = rvs >  self.PqBX
        qu  = ~bu
        if bu.any(): u[bu] = self._binomial(x[bu], v[bu], Cr[bu])
        if qu.any():
            combined = np.vstack([x, self.archive]) if len(self.archive) else x
            q_lim    = max(int(q * len(combined)), 2)
            qbest    = combined[: q_lim]
            cross    = qbest[self.rng_optimization.integers(0, len(qbest), qu.sum())]
            u[qu]    = self._binomial(cross, v[qu], Cr[qu])

        new_y  = np.array([self._evaluate_fitness(ui) for ui in u])
        better = new_y < y

        df_all = np.maximum(0, y - new_y)
        self._update_memory(F[better], Cr[better], df_all[better])

        count_S = np.array([
            np.mean(df_all[mu == i] / (y[mu == i] + 1e-10)) if (mu == i).any() else 0.0
            for i in range(3)
        ])
        if count_S.sum() > 0:
            self.pm = np.clip(count_S / count_S.sum(), 0.1, 0.9)
            self.pm /= self.pm.sum()

        for i in np.where(better)[0]:
            self._archive_add(x[i])
        x[better] = u[better]; y[better] = new_y[better]

        x, y = self._nlpsr(x, y, A_rate=2.10)
        self._n_generations += 1
        self._warm_start = {"x": x, "y": y, "archive": self.archive,
                            "MF": self.MF, "MCr": self.MCr, "k_idx": self.k_idx, "pm": self.pm}
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = super(DE, self).optimize(fitness_function)
        x, y = self.initialize(self._warm_start.get("x"), self._warm_start.get("y"))
        while not self.termination_signal:
            prev_fe = self.n_function_evaluations
            x, y = self.iterate(x, y)
            if self._check_terminations() or self.n_function_evaluations == prev_fe:
                break
        return self._collect(fitness)

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        super().set_data(x, y, best_x, best_y, **kwargs)
        if "pm" in kwargs and kwargs["pm"] is not None:
            self.pm = kwargs["pm"]
