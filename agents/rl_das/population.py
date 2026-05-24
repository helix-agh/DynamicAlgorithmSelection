"""Population class for RL-DAS DE optimizers.

Direct port of the original RL-DAS Population with configurable bounds
so it works with BBOB problems ([-5, 5]) instead of CEC ([-100, 100]).
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats


class Population:
    def __init__(
        self,
        dim: int,
        lb: np.ndarray | float,
        ub: np.ndarray | float,
        Nmax: int = 170,
        Nmin: int = 30,
    ):
        self.dim = dim
        self.Nmax = Nmax
        self.Nmin = Nmin
        self.NP = Nmax
        self.NA = int(self.NP * 2.1)
        self.cost = np.zeros(self.NP)
        self.cbest: float = 1e15
        self.cbest_id: int = -1
        self.gbest: float = 1e15
        self.gbest_solution = np.zeros(dim)
        self.Xmin = (
            np.full(dim, lb, dtype=float)
            if np.isscalar(lb)
            else np.asarray(lb, dtype=float)
        )
        self.Xmax = (
            np.full(dim, ub, dtype=float)
            if np.isscalar(ub)
            else np.asarray(ub, dtype=float)
        )
        self.group = self._init_group()
        self.archive: np.ndarray = np.array([])
        self.MF = np.ones(dim * 20) * 0.2
        self.MCr = np.ones(dim * 20) * 0.2
        self.k: int = 0
        self.F = np.ones(self.NP) * 0.5
        self.Cr = np.ones(self.NP) * 0.9

    def _init_group(self, size: int = -1) -> np.ndarray:
        if size < 0:
            size = self.NP
        return np.random.rand(size, self.dim) * (self.Xmax - self.Xmin) + self.Xmin

    # Alias used by JDE21 reset logic
    def initialize_group(self, size: int = -1) -> np.ndarray:
        return self._init_group(size)

    def initialize_costs(self, problem) -> None:
        self.cost = np.array([float(problem(self.group[i])) for i in range(self.NP)])
        self.gbest = self.cbest = float(np.min(self.cost))
        self.cbest_id = int(np.argmin(self.cost))
        self.gbest_solution = self.group[self.cbest_id].copy()

    def sort(self, size: int, reverse: bool = False) -> None:
        r = -1 if reverse else 1
        ind = np.concatenate(
            (np.argsort(r * self.cost[:size]), np.arange(self.NP)[size:])
        )
        self.cost = self.cost[ind]
        self.cbest = float(np.min(self.cost))
        self.cbest_id = int(np.argmin(self.cost))
        self.group = self.group[ind]
        self.F = self.F[ind]
        self.Cr = self.Cr[ind]

    def cal_NP_next_gen(self, FEs: int, MaxFEs: int) -> float:
        return np.round(
            self.Nmax
            + (self.Nmin - self.Nmax) * np.power(FEs / MaxFEs, 1 - FEs / MaxFEs)
        )

    def slice(self, size: int) -> None:
        self.NP = size
        self.group = self.group[:size]
        self.cost = self.cost[:size]
        self.F = self.F[:size]
        self.Cr = self.Cr[:size]
        if self.cbest_id >= size:
            self.cbest_id = int(np.argmin(self.cost))
            self.cbest = float(np.min(self.cost))

    def _mean_wL(self, df: np.ndarray, s: np.ndarray) -> float:
        w = df / np.sum(df)
        if np.sum(w * s) > 1e-6:
            return float(np.sum(w * (s**2)) / np.sum(w * s))
        return 0.5

    def choose_F_Cr(self) -> tuple[np.ndarray, np.ndarray]:
        gs = self.NP
        ind_r = np.random.randint(0, self.MF.shape[0], size=gs)
        C_r = np.minimum(
            1, np.maximum(0, np.random.normal(loc=self.MCr[ind_r], scale=0.1, size=gs))
        )
        cauchy_locs = self.MF[ind_r]
        F = stats.cauchy.rvs(loc=cauchy_locs, scale=0.1, size=gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        return C_r, np.minimum(1, F)

    def update_M_F_Cr(self, SF: np.ndarray, SCr: np.ndarray, df: np.ndarray) -> None:
        if SF.shape[0] > 0:
            self.MF[self.k] = self._mean_wL(df, SF)
            self.MCr[self.k] = self._mean_wL(df, SCr)
            self.k = (self.k + 1) % self.MF.shape[0]
        else:
            self.MF[self.k] = 0.5
            self.MCr[self.k] = 0.5

    def NLPSR(self, FEs: int, MaxFEs: int) -> None:
        self.sort(self.NP)
        N = self.cal_NP_next_gen(FEs, MaxFEs)
        A = int(max(N * 2.1, self.Nmin))
        N = int(N)
        if N < self.NP:
            self.slice(N)
        if A < self.archive.shape[0]:
            self.NA = A
            self.archive = self.archive[:A]

    def update_archive(self, old_id: int) -> None:
        if self.archive.shape[0] < self.NA:
            self.archive = np.append(self.archive, self.group[old_id]).reshape(
                -1, self.dim
            )
        else:
            self.archive[np.random.randint(self.archive.shape[0])] = self.group[old_id]
