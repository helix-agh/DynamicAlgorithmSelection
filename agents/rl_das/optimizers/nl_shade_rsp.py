"""NL_SHADE_RSP differential-evolution optimizer for RL-DAS."""

from __future__ import annotations

import numpy as np


def _eval(problem, xs: np.ndarray) -> np.ndarray:
    return np.array([float(problem(xs[i])) for i in range(len(xs))])


class NL_SHADE_RSP:
    def __init__(self) -> None:
        self.pb = 0.4
        self.pa = 0.5

    def _binomial(self, x: np.ndarray, v: np.ndarray, cr: np.ndarray) -> np.ndarray:
        NP, dim = x.shape
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < cr.repeat(dim).reshape(NP, dim), v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def _exponential(self, x: np.ndarray, v: np.ndarray, cr: np.ndarray) -> np.ndarray:
        NP, dim = x.shape
        u = x.copy()
        L = np.random.randint(dim, size=NP).repeat(dim).reshape(NP, dim)
        L = L <= np.arange(dim)
        rvs = np.random.rand(NP, dim)
        L = np.where(rvs > cr.repeat(dim).reshape(NP, dim), L, 0)
        return u * (1 - L) + v * L

    def _update_pa(self, fa: float, fp: float, na: int, NP: int) -> None:
        if na == 0 or fa == 0:
            self.pa = 0.5
            return
        self.pa = (fa / (na + 1e-15)) / ((fa / (na + 1e-15)) + (fp / (NP - na + 1e-15)))
        self.pa = float(np.clip(self.pa, 0.1, 0.9))

    def step(
        self,
        population,
        problem,
        FEs: int,
        FEs_end: int,
        MaxFEs: int,
        record_period: int = -1,
    ):
        if record_period <= 0:
            record_period = FEs_end - FEs
        NP, dim = population.NP, population.dim
        NA = int(NP * 2.1)
        if NA < population.archive.shape[0]:
            population.archive = population.archive[:NA]
        self.pa = 0.5
        k = 1
        Fevs = []
        while FEs >= k * record_period:
            k += 1
        population.sort(population.NP)

        while FEs < FEs_end and FEs < MaxFEs:
            Cr, F = population.choose_F_Cr()
            Cr = np.sort(Cr)
            pr = np.exp(-(np.arange(NP) + 1) / NP)
            pr /= np.sum(pr)
            cross_exponential = np.random.random() < 0.5
            pb_upper = int(max(2, NP * self.pb))
            pbs = np.random.randint(pb_upper, size=NP)
            count = 0
            duplicate = np.where(pbs == np.arange(NP))[0]
            while duplicate.shape[0] > 0 and count < 1:
                pbs[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
                duplicate = np.where(pbs == np.arange(NP))[0]
                count += 1
            xpb = population.group[pbs]
            r1 = np.random.randint(NP, size=NP)
            count = 0
            duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
                duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
                count += 1
            x1 = population.group[r1]
            rvs = np.random.rand(NP)
            r2_pop = np.where(rvs >= self.pa)[0]
            r2_arc = np.where(rvs < self.pa)[0]
            use_arc = np.zeros(NP, dtype=bool)
            use_arc[r2_arc] = True
            if population.archive.shape[0] < 25:
                r2_pop = np.arange(NP)
                r2_arc = np.array([], dtype=np.int32)
            r2 = np.random.choice(np.arange(NP), size=r2_pop.shape[0], p=pr)
            count = 0
            duplicate = np.where(
                (r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop])
            )[0]
            while duplicate.shape[0] > 0 and count < 25:
                r2[duplicate] = np.random.choice(
                    np.arange(NP), size=duplicate.shape[0], p=pr
                )
                duplicate = np.where(
                    (r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop])
                )[0]
                count += 1
            x2 = np.zeros((NP, dim))
            if r2_pop.shape[0] > 0:
                x2[r2_pop] = population.group[r2]
            if r2_arc.shape[0] > 0:
                x2[r2_arc] = population.archive[
                    np.random.randint(
                        min(population.archive.shape[0], NA),
                        size=r2_arc.shape[0],
                    )
                ]
            Fs = F.repeat(dim).reshape(NP, dim)
            vs = population.group + Fs * (xpb - population.group) + Fs * (x1 - x2)
            Crb = np.zeros(NP)
            tmp_id = np.where(np.arange(NP) + FEs < 0.5 * MaxFEs)[0]
            Crb[tmp_id] = 2 * ((FEs + tmp_id) / MaxFEs - 0.5)
            if cross_exponential:
                us = self._binomial(population.group, vs, Crb)
            else:
                us = self._exponential(population.group, vs, Cr)

            # Reinitialise out-of-bounds elements with uniform random in [lb, ub]
            oob = (us < population.Xmin) | (us > population.Xmax)
            if np.any(oob):
                rand_pts = (
                    np.random.rand(NP, dim) * (population.Xmax - population.Xmin)
                    + population.Xmin
                )
                us = np.where(oob, rand_pts, us)

            cost = _eval(problem, us)
            optim = np.where(cost < population.cost)[0]
            for i in optim:
                population.update_archive(i)
            population.F[optim] = F[optim]
            population.Cr[optim] = Cr[optim]
            SF = F[optim]
            SCr = Cr[optim]
            df = (population.cost[optim] - cost[optim]) / (
                population.cost[optim] + 1e-9
            )
            arc_usage = use_arc[optim]
            fp = float(np.sum(df[arc_usage]))
            fa = float(np.sum(df[~arc_usage]))
            na = int(np.sum(arc_usage))
            population.group[optim] = us[optim]
            population.cost[optim] = cost[optim]
            if float(np.min(cost)) < population.gbest:
                population.gbest = float(np.min(cost))
                population.gbest_solution = us[np.argmin(cost)].copy()
            FEs += NP
            self.pb = 0.4 - 0.2 * (FEs / MaxFEs)
            population.NLPSR(FEs, MaxFEs)
            population.update_M_F_Cr(SF, SCr, df)
            self._update_pa(fa, fp, na, NP)
            NP = population.NP
            NA = population.NA
            if FEs >= k * record_period:
                Fevs.append(population.gbest)
                k += 1

        return population, Fevs, min(FEs, MaxFEs)
