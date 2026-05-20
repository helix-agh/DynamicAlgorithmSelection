"""DE optimizers for RL-DAS: NL_SHADE_RSP, JDE21, MadDE.

Ported from the original RL-DAS (Guo et al., 2024) with the following
adaptations for DAS2 / BBOB:

- Batch evaluation: ``np.array([float(problem(xs[i])) for i in range(len(xs))])``
  instead of ``problem.func(xs) - problem.optimum``.
- Bounds from ``population.Xmin`` / ``population.Xmax`` (not hardcoded ±100).
- No early-termination at ``cost < 1e-8`` (FE budget controls episode end).
- ``__init__`` takes no ``dim`` argument (dimension inferred from population).
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats


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


class JDE21:
    def __init__(self) -> None:
        self.sNP = 10
        self.tao1 = 0.1
        self.tao2 = 0.1
        self.Finit = 0.5
        self.CRinit = 0.9
        self.Fl_b = 0.1
        self.Fl_s = 0.17
        self.Fu = 1.1
        self.CRl_b = 0.0
        self.CRl_s = 0.1
        self.CRu_b = 1.1
        self.CRu_s = 0.8
        self.eps = 1e-12
        self.MyEps = 0.25
        self.nReset = 0
        self.sReset = 0
        self.cCopy = 0

    def _prevec_enakih(self, cost: np.ndarray, best: float) -> bool:
        eqs = len(cost[np.fabs(cost - best) < self.eps])
        return eqs > 2 and eqs > len(cost) * self.MyEps

    def _crowding(self, group: np.ndarray, vs: np.ndarray) -> np.ndarray:
        NP, dim = vs.shape
        dist = np.sum(
            ((group * np.ones((NP, NP, dim))).transpose(1, 0, 2) - vs) ** 2, -1
        ).transpose()
        return np.argmin(dist, -1)

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
        NP = population.NP
        dim = population.dim
        sNP = self.sNP
        bNP = max(NP - sNP, 2)
        age = 0
        k = 1
        Fevs = []

        def mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big):
            nonlocal bNP
            if big:
                xNP = bNP
                randF = np.random.rand(xNP) * self.Fu + self.Fl_b
                randCr = np.random.rand(xNP) * self.CRu_b + self.CRl_b
                pF = population.F[:xNP]
                pCr = population.Cr[:xNP]
            else:
                # Use actual slice size (may be < sNP when population shrank)
                xNP = population.F[bNP:].shape[0]
                if xNP == 0:
                    return SF, SCr, df, age
                randF = np.random.rand(xNP) * self.Fu + self.Fl_s
                randCr = np.random.rand(xNP) * self.CRu_b + self.CRl_s
                pF = population.F[bNP:]
                pCr = population.Cr[bNP:]

            rvs = np.random.rand(xNP)
            F = np.where(rvs < self.tao1, randF, pF)
            rvs = np.random.rand(xNP)
            Cr = np.where(rvs < self.tao2, randCr, pCr)
            Fs = F.repeat(dim).reshape(xNP, dim)
            Crs = Cr.repeat(dim).reshape(xNP, dim)
            v = population.group[r1] + Fs * (
                population.group[r2] - population.group[r3]
            )
            v = np.clip(v, population.Xmin, population.Xmax)
            jrand = np.random.randint(dim, size=xNP)
            u = np.where(
                np.random.rand(xNP, dim) < Crs,
                v,
                (population.group[:bNP] if big else population.group[bNP:]),
            )
            u[np.arange(xNP), jrand] = v[np.arange(xNP), jrand]
            cost = _eval(problem, u)
            crowding_ids = (
                self._crowding(population.group[:xNP], u)
                if big
                else np.arange(xNP) + bNP
            )
            age += xNP
            for i in range(xNP):
                idx = crowding_ids[i]
                if cost[i] < population.cost[idx]:
                    population.update_archive(idx)
                    population.group[idx] = u[i]
                    population.cost[idx] = cost[i]
                    population.F[idx] = F[i]
                    population.Cr[idx] = Cr[i]
                    SF = np.append(SF, F[i])
                    SCr = np.append(SCr, Cr[i])
                    d = (population.cost[i] - cost[i]) / (population.cost[i] + 1e-9)
                    df = np.append(df, d)
                    if cost[i] < population.cbest:
                        age = 0
                        population.cbest_id = idx
                        population.cbest = cost[i]
                        if cost[i] < population.gbest:
                            population.gbest = cost[i]
                            population.gbest_solution = u[i].copy()

            return SF, SCr, df, age

        population.sort(NP, True)
        while FEs >= k * record_period:
            k += 1
        while FEs < FEs_end:
            df = np.array([])
            SF = np.array([])
            SCr = np.array([])
            if (
                self._prevec_enakih(population.cost[:bNP], population.gbest)
                or age > MaxFEs / 10
            ):
                self.nReset += 1
                population.group[:bNP] = population.initialize_group(bNP)
                population.F[:bNP] = self.Finit
                population.Cr[:bNP] = self.CRinit
                population.cost[:bNP] = 1e15
                age = 0
                population.cbest = float(np.min(population.cost))
                population.cbest_id = int(np.argmin(population.cost))

            if FEs < MaxFEs / 3:
                mig = 1
            elif FEs < 2 * MaxFEs / 3:
                mig = 2
            else:
                mig = 3

            r1 = np.random.randint(bNP, size=bNP)
            count = 0
            duplicate = np.where((r1 == np.arange(bNP)) * (r1 == population.cbest_id))[
                0
            ]
            while duplicate.shape[0] > 0 and count < 25:
                r1[duplicate] = np.random.randint(bNP, size=duplicate.shape[0])
                duplicate = np.where(
                    (r1 == np.arange(bNP)) * (r1 == population.cbest_id)
                )[0]
                count += 1
            r2 = np.random.randint(bNP + mig, size=bNP)
            count = 0
            duplicate = np.where((r2 == np.arange(bNP)) + (r2 == r1))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r2[duplicate] = np.random.randint(bNP + mig, size=duplicate.shape[0])
                duplicate = np.where((r2 == np.arange(bNP)) + (r2 == r1))[0]
                count += 1
            r3 = np.random.randint(bNP + mig, size=bNP)
            count = 0
            duplicate = np.where((r3 == np.arange(bNP)) + (r3 == r1) + (r3 == r2))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r3[duplicate] = np.random.randint(bNP + mig, size=duplicate.shape[0])
                duplicate = np.where((r3 == np.arange(bNP)) + (r3 == r1) + (r3 == r2))[
                    0
                ]
                count += 1
            SF, SCr, df, age = mutate_cross_select(
                r1, r2, r3, SF, SCr, df, age, big=True
            )
            FEs += bNP
            if FEs >= k * record_period:
                Fevs.append(population.gbest)
                k += 1
            if FEs >= FEs_end or FEs >= MaxFEs:
                break

            curr_sNP = NP - bNP
            if (
                curr_sNP > 0
                and population.cbest_id >= bNP
                and self._prevec_enakih(population.cost[bNP:], population.cbest)
            ):
                self.sReset += 1
                cbest = population.cbest
                cbest_id = population.cbest_id
                tmp = population.group[cbest_id].copy()
                population.group[bNP:] = population.initialize_group(curr_sNP)
                population.F[bNP:] = self.Finit
                population.Cr[bNP:] = self.CRinit
                population.cost[bNP:] = 1e15
                population.cbest = cbest
                population.cbest_id = cbest_id
                population.group[cbest_id] = tmp
                population.cost[cbest_id] = cbest

            if population.cbest_id < bNP:
                self.cCopy += 1
                population.cost[bNP] = population.cbest
                population.group[bNP] = population.group[population.cbest_id].copy()
                population.cbest_id = bNP

            curr_sNP = NP - bNP  # actual small-pop size (= sNP unless shrunken)
            for _ in range(bNP // curr_sNP if curr_sNP > 0 else 0):
                r1 = np.random.randint(curr_sNP, size=curr_sNP) + bNP
                count = 0
                duplicate = np.where(r1 == (np.arange(curr_sNP) + bNP))[0]
                while duplicate.shape[0] > 0 and count < 25:
                    r1[duplicate] = (
                        np.random.randint(curr_sNP, size=duplicate.shape[0]) + bNP
                    )
                    duplicate = np.where(r1 == (np.arange(curr_sNP) + bNP))[0]
                    count += 1
                r2 = np.random.randint(curr_sNP, size=curr_sNP) + bNP
                count = 0
                duplicate = np.where((r2 == (np.arange(curr_sNP) + bNP)) + (r2 == r1))[
                    0
                ]
                while duplicate.shape[0] > 0 and count < 25:
                    r2[duplicate] = (
                        np.random.randint(curr_sNP, size=duplicate.shape[0]) + bNP
                    )
                    duplicate = np.where(
                        (r2 == (np.arange(curr_sNP) + bNP)) + (r2 == r1)
                    )[0]
                    count += 1
                r3 = np.random.randint(curr_sNP, size=curr_sNP) + bNP
                count = 0
                duplicate = np.where(
                    (r3 == (np.arange(curr_sNP) + bNP)) + (r3 == r1) + (r3 == r2)
                )[0]
                while duplicate.shape[0] > 0 and count < 25:
                    r3[duplicate] = (
                        np.random.randint(curr_sNP, size=duplicate.shape[0]) + bNP
                    )
                    duplicate = np.where(
                        (r3 == (np.arange(curr_sNP) + bNP)) + (r3 == r1) + (r3 == r2)
                    )[0]
                    count += 1
                SF, SCr, df, age = mutate_cross_select(
                    r1, r2, r3, SF, SCr, df, age, big=False
                )
                FEs += curr_sNP
                if FEs >= k * record_period:
                    Fevs.append(population.gbest)
                    k += 1
                if FEs >= FEs_end or FEs >= MaxFEs:
                    break

            population.update_M_F_Cr(SF, SCr, df)
            NP = min(
                int(population.cal_NP_next_gen(FEs, MaxFEs)), len(population.group)
            )
            population.NP = NP
            population.group = population.group[-NP:]
            population.cost = population.cost[-NP:]
            population.F = population.F[-NP:]
            population.Cr = population.Cr[-NP:]
            population.cbest_id = int(np.argmin(population.cost))
            population.cbest = float(np.min(population.cost))
            bNP = max(NP - sNP, 2)

        return population, Fevs, min(FEs, MaxFEs)


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


_PORTFOLIO: dict[str, type] = {
    "NL_SHADE_RSP": NL_SHADE_RSP,
    "MADDE": MadDE,
    "JDE21": JDE21,
}


def get_rldas_portfolio(names: list[str] | None = None) -> list:
    """Return instantiated RL-DAS optimizer objects.

    Parameters
    ----------
    names:
        Optimizer names. Defaults to ``["NL_SHADE_RSP", "MADDE", "JDE21"]``.
        Valid names: ``"NL_SHADE_RSP"``, ``"MADDE"``, ``"JDE21"``.
    """
    if names is None:
        names = ["NL_SHADE_RSP", "MADDE", "JDE21"]
    unknown = [n for n in names if n not in _PORTFOLIO]
    if unknown:
        raise ValueError(
            f"Unknown RL-DAS optimizer(s): {unknown}. Valid choices: {list(_PORTFOLIO)}"
        )
    return [_PORTFOLIO[n]() for n in names]
