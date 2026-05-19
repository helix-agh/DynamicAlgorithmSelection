"""RL-DAS gymnasium environment (Population-based, 9-dim features).

Strictly follows the original RL-DAS design (Guo et al., 2024):

- A single mutable ``Population`` object is shared between DE sub-optimizers
  as warm-started state (no get_data/set_data contract).
- Portfolio is restricted to the three DE algorithms: NL_SHADE_RSP, JDE21, MadDE.
- Observation: 9-dim population-state features (computed with local sampling)
  concatenated with per-optimizer movement history embeddings of shape (dim,).

Observation layout: flat Box of shape (9 + 2 * n_opt * dim,)
  [0:9]    nine population-state features (see _pop_features)
  [9:]     movement history, 2*n_opt blocks of dim scalars each,
           interleaved as [best_0, worst_0, best_1, worst_1, ...]

Action space: Discrete(n_opt)

Reward: max(0, (prev_best - new_best) / cost_scale)  non-negative improvement
        normalised by the initial global-best cost.
"""

from __future__ import annotations

import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.rl_das.population import Population
from das.optimizers.base import get_checkpoints


# ---------------------------------------------------------------------------
# Feature helper functions (adapted from original RL-DAS utils.py)
# ---------------------------------------------------------------------------


def _cal_fdc(group_norm: np.ndarray, costs: np.ndarray) -> float:
    """Fitness-distance correlation. group_norm must be in [0, 1]^dim."""
    opt_x = group_norm[np.argmin(costs)]
    ds = np.sum((group_norm - opt_x) ** 2, axis=1)
    fs = 1.0 / (costs + 1e-8)
    C_fd = ((fs - fs.mean()) * (ds - ds.mean())).mean()
    delta_f = ((fs - fs.mean()) ** 2).mean()
    delta_d = ((ds - ds.mean()) ** 2).mean()
    return float(C_fd / (delta_d * delta_f + 1e-8))


def _dispersion(group_norm: np.ndarray, costs: np.ndarray) -> tuple[float, float]:
    """Dispersion and dispersion-ratio metrics. group_norm in [0, 1]^dim."""
    gs, dim = group_norm.shape
    group_sorted = group_norm[np.argsort(costs)]
    diam = float(np.sqrt(dim))
    disp = 0.0
    max_dis = 0.0
    for i in range(1, gs):
        shift = np.concatenate((group_sorted[i:], group_sorted[:i]), 0)
        distances = np.sqrt(np.sum((group_sorted - shift) ** 2, -1))
        disp += float(np.sum(distances))
        cur_max = float(np.max(distances))
        if cur_max > max_dis:
            max_dis = cur_max
    disp /= gs**2
    gs10 = max(gs * 10 // 100, 1)
    top10 = group_sorted[:gs10]
    disp10 = 0.0
    for i in range(1, gs10):
        shift = np.concatenate((top10[i:], top10[:i]), 0)
        disp10 += float(np.sum(np.sqrt(np.sum((top10 - shift) ** 2, -1))))
    if gs10 > 1:
        disp10 /= gs10**2
    return float(disp10 - disp), float(max_dis / (diam + 1e-10))


def _negative_slope_coefficient(
    group_cost: np.ndarray, sample_cost: np.ndarray
) -> float:
    """Negative slope coefficient (NSC)."""
    gs = sample_cost.shape[0]
    m = 10
    gs -= gs % m
    if gs < m:
        return 0.0
    pairs = sorted(zip(group_cost[:gs].tolist(), sample_cost[:gs].tolist()))
    arr = np.array(pairs)
    sorted_group = arr[:, 0].reshape(m, -1)
    sorted_sample = arr[:, 1].reshape(m, -1)
    Ms = np.mean(sorted_group, -1)
    Ns = np.mean(sorted_sample, -1)
    nsc = np.minimum((Ns[1:] - Ns[:-1]) / (Ms[1:] - Ms[:-1] + 1e-8), 0)
    return float(np.sum(nsc))


def _average_neutral_ratio(
    group_cost: np.ndarray, sample_costs: np.ndarray, eps: float = 1.0
) -> float:
    """Average neutral ratio (ANR)."""
    gs = sample_costs.shape[1]
    dcost = np.fabs(sample_costs - group_cost[:gs])
    return float(np.mean(np.sum(dcost < eps, axis=0) / sample_costs.shape[0]))


def _non_improvable_worsenable(
    group_cost: np.ndarray, sample_costs: np.ndarray
) -> tuple[float, float]:
    """Non-improvable (NI) and non-worsenable (NW) ratios."""
    gs = sample_costs.shape[1]
    NI = (
        1.0
        - np.count_nonzero(np.sum(group_cost[:gs] > sample_costs, axis=-1))
        / sample_costs.shape[0]
    )
    NW = (
        1.0
        - np.count_nonzero(np.sum(group_cost[:gs] < sample_costs, axis=-1))
        / sample_costs.shape[0]
    )
    return float(NI), float(NW)


def _pop_features(
    population: Population,
    sample_costs: np.ndarray,
    cost_scale: float,
    n_fe: int,
    max_fe: int,
) -> np.ndarray:
    """Return 9-dim feature vector matching original RL-DAS Population.get_feature.

    Features (in order):
      gbc        normalised global-best cost
      fdc        fitness-distance correlation
      disp       dispersion (disp10 - disp_all)
      disp_ratio max_pairwise_dist / sqrt(dim)
      nsc        negative slope coefficient
      anr        average neutral ratio
      ni         non-improvable ratio
      nw         non-worsenable ratio
      progress   n_fe / max_fe
    """
    group = population.group
    cost = population.cost
    lb = population.Xmin
    ub = population.Xmax

    group_norm = (group - lb) / (ub - lb + 1e-10)

    gbc = float(population.gbest / (cost_scale + 1e-10))
    fdc = _cal_fdc(group_norm, cost / (cost_scale + 1e-10))
    disp, disp_ratio = _dispersion(group_norm, cost)
    nsc = _negative_slope_coefficient(cost, sample_costs[0])
    anr = _average_neutral_ratio(cost, sample_costs)
    ni, nw = _non_improvable_worsenable(cost, sample_costs)
    progress = float(n_fe / max(max_fe, 1))

    out = np.array(
        [gbc, fdc, disp, disp_ratio, nsc, anr, ni, nw, progress], dtype=np.float32
    )
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RLDASEnv(gym.Env):
    """RL-DAS environment wrapping optimization problems via an IOHSuite.

    Uses a Population object as shared warm-started state across all DE
    sub-optimizers (matching the original RL-DAS design).

    Parameters
    ----------
    problem_ids:
        BBOB problem IDs to cycle through (one per episode).
    suite:
        IOHSuite object.
    optimizers:
        List of instantiated DE optimizer objects (NL_SHADE_RSP, JDE21, MadDE).
    dim:
        Problem dimension.
    fe_multiplier:
        Budget = fe_multiplier * dim.
    n_checkpoints:
        Number of optimizer-selection steps per episode.
    checkpoint_division_base:
        1.0 → uniform checkpoints; >1.0 → exponentially growing.
    n_individuals:
        Initial population size (Nmax).
    seed:
        Unused (kept for API compatibility with other DAS envs).
    """

    metadata = {"render_modes": []}
    N_FEATURES = 9
    SAMPLE_TIMES = 2

    def __init__(
        self,
        problem_ids: list[str],
        suite,
        optimizers: list,
        dim: int,
        fe_multiplier: int = 10_000,
        n_checkpoints: int = 20,
        checkpoint_division_base: float = 1.0,
        n_individuals: int = 170,
        seed: int | None = None,
    ):
        super().__init__()

        self._problem_ids = [pid for pid in problem_ids if pid.endswith(f"_d{dim:02d}")]
        if not self._problem_ids:
            raise ValueError(f"No problem_ids match dimension {dim}.")

        self.suite = suite
        self._optimizers = optimizers
        self.dim = dim
        self.fe_multiplier = fe_multiplier
        self.n_checkpoints = n_checkpoints
        self.cdb = checkpoint_division_base
        self.n_individuals = n_individuals

        self.n_opt = len(optimizers)
        obs_dim = self.N_FEATURES + 2 * self.n_opt * dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_opt)

        self._problem = None
        self._problem_idx = 0
        self._max_fe = 0
        self._n_fe = 0
        self._checkpoints: np.ndarray | None = None
        self._checkpoint_idx = 0
        self._population: Population | None = None
        self._cost_scale: float = 1.0
        self._best_history: list[list[np.ndarray]] = [[] for _ in range(self.n_opt)]
        self._worst_history: list[list[np.ndarray]] = [[] for _ in range(self.n_opt)]

    @property
    def problem_ids(self) -> list[str]:
        # Public accessor — callers should not reach into _problem_ids directly
        # because it is filtered (dimension-matched) and may differ from the
        # original list passed to the constructor.
        return self._problem_ids

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        problem_id = self._problem_ids[self._problem_idx % len(self._problem_ids)]
        self._problem_idx += 1
        self._problem = self.suite.get_problem(problem_id)

        lb = float(self._problem.lower_bounds[0])
        ub = float(self._problem.upper_bounds[0])
        self._max_fe = self.fe_multiplier * self.dim
        self._checkpoints = get_checkpoints(
            self.n_checkpoints, self._max_fe, self.n_individuals, self.cdb
        )
        self._checkpoint_idx = 0

        self._population = Population(self.dim, lb, ub, Nmax=self.n_individuals)
        self._population.initialize_costs(self._problem)
        self._cost_scale = max(float(self._population.gbest), 1e-10)
        self._n_fe = self._population.NP

        self._best_history = [[] for _ in range(self.n_opt)]
        self._worst_history = [[] for _ in range(self.n_opt)]

        obs = self._build_observation()
        return obs, {"problem_id": problem_id, "dimension": self.dim}

    def step(self, action: int):
        assert self._problem is not None, "Call reset() before step()"

        target_fe = int(self._checkpoints[self._checkpoint_idx])
        prev_gbest = self._population.gbest

        prev_best_x = self._population.gbest_solution.copy()
        worst_idx = int(np.argmax(self._population.cost))
        prev_worst_x = self._population.group[worst_idx].copy()

        self._population, _, end_fes = self._optimizers[action].step(
            self._population, self._problem, self._n_fe, target_fe, self._max_fe
        )
        self._n_fe = end_fes

        new_best_x = self._population.gbest_solution.copy()
        new_worst_x = self._population.group[
            int(np.argmax(self._population.cost))
        ].copy()
        scale = float(self.dim**0.5) + 1e-10
        self._best_history[action].append((new_best_x - prev_best_x) / scale)
        self._worst_history[action].append((new_worst_x - prev_worst_x) / scale)

        self._checkpoint_idx += 1
        terminated = (
            self._checkpoint_idx >= self.n_checkpoints or self._n_fe >= self._max_fe
        )

        reward = float(
            max(0.0, (prev_gbest - self._population.gbest) / (self._cost_scale + 1e-10))
        )

        obs = self._build_observation()
        return (
            obs,
            reward,
            terminated,
            False,
            {"best_y": self._population.gbest, "n_fe": self._n_fe},
        )

    # ------------------------------------------------------------------
    # Local sampling (counts FEs toward budget)
    # ------------------------------------------------------------------

    def _local_sample(self) -> np.ndarray:
        """Run SAMPLE_TIMES independent trials on a deepcopy of the population.

        Returns
        -------
        sample_costs : ndarray of shape (SAMPLE_TIMES, min_NP)
        """
        sample_size = self._population.NP
        costs = []
        min_len = sample_size
        for _ in range(self.SAMPLE_TIMES):
            pop_copy = copy.deepcopy(self._population)
            opt = self._optimizers[np.random.randint(self.n_opt)]
            popped, _, _ = opt.step(
                pop_copy,
                self._problem,
                self._n_fe,
                self._n_fe + sample_size,
                self._max_fe,
            )
            costs.append(popped.cost.copy())
            if popped.cost.shape[0] < min_len:
                min_len = popped.cost.shape[0]
        self._n_fe = min(self._n_fe + sample_size * self.SAMPLE_TIMES, self._max_fe)
        return np.array([c[:min_len] for c in costs])

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        sample_costs = self._local_sample()
        features = _pop_features(
            self._population, sample_costs, self._cost_scale, self._n_fe, self._max_fe
        )

        movements: list[np.ndarray] = []
        for k in range(self.n_opt):
            if self._best_history[k]:
                best_emb = np.mean(self._best_history[k], axis=0).astype(np.float32)
            else:
                best_emb = np.zeros(self.dim, dtype=np.float32)
            if self._worst_history[k]:
                worst_emb = np.mean(self._worst_history[k], axis=0).astype(np.float32)
            else:
                worst_emb = np.zeros(self.dim, dtype=np.float32)
            movements.append(best_emb)
            movements.append(worst_emb)

        obs = np.concatenate([features, *movements])
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
