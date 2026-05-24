"""MadDE differential-evolution optimizer for RL-DAS."""

from __future__ import annotations

import numpy as np


def _eval(problem, xs: np.ndarray) -> np.ndarray:
    return np.array([float(problem(xs[i])) for i in range(len(xs))])


class MadDE:
    def __init__(self) -> None:
        self.p = 0.18
        self.PqBX = 0.01
        self.pm = np.ones(3) / 3

    def _ctb_w_arc(
        self,
        group: np.ndarray,
        best: np.ndarray,
        archive: np.ndarray,
        Fs: np.ndarray,
    ) -> np.ndarray:
        NP, dim = group.shape
        NB = best.shape[0]
        NA = archive.shape[0]
        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1
        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1
        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1
        xb = best[rb]
        x1 = group[r1]
        x2 = np.concatenate((group, archive), 0)[r2] if NA > 0 else group[r2]
        return group + Fs * (xb - group) + Fs * (x1 - x2)

    def _ctr_w_arc(
        self,
        group: np.ndarray,
        archive: np.ndarray,
        Fs: np.ndarray,
    ) -> np.ndarray:
        NP, dim = group.shape
        NA = archive.shape[0]
        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1
        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1
        x1 = group[r1]
        x2 = np.concatenate((group, archive), 0)[r2] if NA > 0 else group[r2]
        return group + Fs * (x1 - x2)

    def _weighted_rtb(
        self,
        group: np.ndarray,
        best: np.ndarray,
        Fs: np.ndarray,
        Fas: float,
    ) -> np.ndarray:
        NP, dim = group.shape
        NB = best.shape[0]
        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1
        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1
        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1
        return Fs * group[r1] + Fs * Fas * (best[rb] - group[r2])

    @staticmethod
    def _binomial(x: np.ndarray, v: np.ndarray, Crs: np.ndarray) -> np.ndarray:
        NP, dim = x.shape
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

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
        Fevs = []
        k = 1
        while FEs >= k * record_period:
            k += 1
        population.sort(population.NP)
        while FEs < FEs_end and FEs < MaxFEs:
            NP, dim = population.NP, population.dim
            q = 2 * self.p - self.p * FEs / MaxFEs
            Fa = 0.5 + 0.5 * FEs / MaxFEs
            Cr, F = population.choose_F_Cr()
            mu = np.random.choice(3, size=NP, p=self.pm)
            p1 = population.group[mu == 0]
            p2 = population.group[mu == 1]
            p3 = population.group[mu == 2]
            pbest = population.group[: max(int(self.p * NP), 2)]
            qbest = population.group[: max(int(q * NP), 2)]
            Fs = F.repeat(dim).reshape(NP, dim)
            v1 = self._ctb_w_arc(p1, pbest, population.archive, Fs[mu == 0])
            v2 = self._ctr_w_arc(p2, population.archive, Fs[mu == 1])
            v3 = self._weighted_rtb(p3, qbest, Fs[mu == 2], Fa)
            v = np.zeros((NP, dim))
            v[mu == 0] = v1
            v[mu == 1] = v2
            v[mu == 2] = v3

            # Bounds handling: midpoint reflection with lb/ub
            Xmin = population.Xmin
            Xmax = population.Xmax
            v = np.where(v < Xmin, (population.group + Xmin) / 2, v)
            v = np.where(v > Xmax, (population.group + Xmax) / 2, v)

            rvs = np.random.rand(NP)
            Crs = Cr.repeat(dim).reshape(NP, dim)
            u = np.zeros((NP, dim))
            if np.sum(rvs <= self.PqBX) > 0:
                qu = v[rvs <= self.PqBX]
                if population.archive.shape[0] > 0:
                    qbest = np.concatenate((population.group, population.archive), 0)[
                        : max(int(q * (NP + population.archive.shape[0])), 2)
                    ]
                cross_qbest = qbest[np.random.randint(qbest.shape[0], size=qu.shape[0])]
                qu = self._binomial(cross_qbest, qu, Crs[rvs <= self.PqBX])
                u[rvs <= self.PqBX] = qu
            bu = v[rvs > self.PqBX]
            bu = self._binomial(
                population.group[rvs > self.PqBX], bu, Crs[rvs > self.PqBX]
            )
            u[rvs > self.PqBX] = bu

            ncost = _eval(problem, u)
            FEs += NP
            optim = np.where(ncost < population.cost)[0]
            for i in optim:
                population.update_archive(i)
            SF = F[optim]
            SCr = Cr[optim]
            df = np.maximum(0, population.cost - ncost)
            population.update_M_F_Cr(SF, SCr, df[optim])
            count_S = np.zeros(3)
            for i in range(3):
                count_S[i] = np.mean(df[mu == i] / (population.cost[mu == i] + 1e-9))
            if np.sum(count_S) > 0:
                self.pm = np.clip(count_S / np.sum(count_S), 0.1, 0.9)
                self.pm /= np.sum(self.pm)
            else:
                self.pm = np.ones(3) / 3
            population.group[optim] = u[optim]
            population.cost = np.minimum(population.cost, ncost)
            population.NLPSR(FEs, MaxFEs)
            if float(np.min(population.cost)) < population.gbest:
                population.gbest = float(np.min(population.cost))
                population.gbest_solution = population.group[
                    np.argmin(population.cost)
                ].copy()
            if FEs >= k * record_period:
                Fevs.append(population.gbest)
                k += 1

        return population, Fevs, min(FEs, MaxFEs)
