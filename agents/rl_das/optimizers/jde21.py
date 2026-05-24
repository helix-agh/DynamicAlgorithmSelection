"""JDE21 differential-evolution optimizer for RL-DAS."""

from __future__ import annotations

import numpy as np


def _eval(problem, xs: np.ndarray) -> np.ndarray:
    return np.array([float(problem(xs[i])) for i in range(len(xs))])


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
