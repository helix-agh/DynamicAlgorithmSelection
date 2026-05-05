"""RL-DAS gymnasium environment.

Observation space: flat Box of shape (6 + 2 * n_opt * dim,)
  - [0:6]           six population-state features
  - [6:]            movement history, 2*n_opt blocks of ``dim`` scalars each,
                    interleaved as [best_0, worst_0, best_1, worst_1, ...]

Action space: Discrete(n_opt)

Reward: max(0, (prev_best - new_best) / cost_scale)  — non-negative improvement
        normalised by the initial global-best cost so rewards are comparable
        across problems with different fitness scales.

The environment operates on BBOB problems accessed through a cocoex Suite.
Only problems whose dimension matches ``dim`` are used; others are skipped.
"""

from __future__ import annotations

import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from das.optimizers.base import get_checkpoints


# ---------------------------------------------------------------------------
# Population-state feature extraction (6-D, dimension-independent)
# ---------------------------------------------------------------------------


def _pop_features(
    x: np.ndarray,
    y: np.ndarray,
    gbest_y: float,
    cost_scale: float,
    n_fes: int,
    max_fes: int,
) -> np.ndarray:
    """Return a 6-D population-state feature vector.

    Features
    --------
    gbc       : normalised global-best cost
    fdc       : fitness–distance correlation
    disp      : mean distance to centroid (dispersion)
    disp_ratio: dispersion / search-space range
    nsc       : negative slope coefficient (distance vs fitness)
    progress  : n_fes / max_fes
    """
    NP = len(y)

    # 1. Normalised global best
    gbc = float(gbest_y / cost_scale) if cost_scale > 1e-10 else 0.0

    # 2. Fitness–distance correlation
    gbest_x = x[int(np.argmin(y))]
    dists = np.linalg.norm(x - gbest_x, axis=1)
    if dists.std() > 1e-10 and y.std() > 1e-10 and NP > 2:
        fdc = float(np.corrcoef(dists, y)[0, 1])
    else:
        fdc = 0.0

    # 3. Dispersion metrics
    centroid = x.mean(axis=0)
    disp_dists = np.linalg.norm(x - centroid, axis=1)
    disp = float(disp_dists.mean())
    spread = float(np.ptp(x))
    disp_ratio = disp / (spread + 1e-10)

    # 4. Negative slope coefficient
    if dists.std() > 1e-10 and NP > 2:
        try:
            slope = float(np.polyfit(dists, y, 1)[0])
            nsc = -slope if np.isfinite(slope) else 0.0
        except (np.linalg.LinAlgError, ValueError):
            nsc = 0.0
    else:
        nsc = 0.0

    # 5. Progress
    progress = float(n_fes / max(max_fes, 1))

    out = np.array([gbc, fdc, disp, disp_ratio, nsc, progress], dtype=np.float32)
    # Guard against any NaN/inf that could corrupt the network weights
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RLDASEnv(gym.Env):
    """RL-DAS environment wrapping BBOB problems via a cocoex Suite.

    Parameters
    ----------
    problem_ids:
        BBOB problem IDs to cycle through (one per episode).  Only IDs whose
        dimension matches ``dim`` are used.
    suite:
        cocoex Suite object.
    optimizers:
        Ordered list of sub-optimizer classes (same pypop7-compatible classes
        as used in DASEnv — defines the action space).
    dim:
        Problem dimension.  The movement embeddings have shape (dim,), so the
        agent is dimension-specific.
    fe_multiplier:
        Budget = fe_multiplier * dim.
    n_checkpoints:
        Number of optimizer-selection steps per episode.
    checkpoint_division_base:
        cdb=1.0 → uniform checkpoints; cdb>1.0 → exponentially growing.
    n_individuals:
        Population size.
    seed:
        Seed for deterministic optimizer RNGs (passed to pypop7 as seed_rng).
    """

    metadata = {"render_modes": []}
    N_FEATURES = 6

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

        # Filter to matching dimension (BBOB IDs end with _d{dim:02d})
        self._problem_ids = [pid for pid in problem_ids if pid.endswith(f"_d{dim:02d}")]
        if not self._problem_ids:
            raise ValueError(f"No problem_ids match dimension {dim}.")

        self.suite = suite
        self.optimizers = optimizers
        self.dim = dim
        self.fe_multiplier = fe_multiplier
        self.n_checkpoints = n_checkpoints
        self.cdb = checkpoint_division_base
        self.n_individuals = n_individuals
        self._seed = seed

        self.n_opt = len(optimizers)
        obs_dim = self.N_FEATURES + 2 * self.n_opt * dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_opt)

        # Episode state
        self._problem = None
        self._problem_idx = 0
        self._max_fe = 0
        self._n_fe = 0
        self._checkpoints: np.ndarray | None = None
        self._checkpoint_idx = 0

        # Current population arrays (shared warm-start state)
        self._pop_x: np.ndarray | None = None
        self._pop_y: np.ndarray | None = None
        self._optimizer_state: dict = {}

        # Global best tracking
        self._gbest_y: float = np.inf
        self._gbest_x: np.ndarray | None = None
        self._cost_scale: float = 1.0
        self._worst_y: float = -np.inf
        self._initial_range: tuple[float, float] = (np.inf, -np.inf)

        # Movement history: list of accumulated movement vectors per optimizer
        # shape: (n_opt,) lists of (dim,) arrays; averaged in observation
        self._best_history: list[list[np.ndarray]] = [[] for _ in range(self.n_opt)]
        self._worst_history: list[list[np.ndarray]] = [[] for _ in range(self.n_opt)]

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        problem_id = self._problem_ids[self._problem_idx % len(self._problem_ids)]
        self._problem_idx += 1

        self._problem = self.suite.get_problem(problem_id)
        self._max_fe = self.fe_multiplier * self.dim
        self._checkpoints = get_checkpoints(
            self.n_checkpoints, self._max_fe, self.n_individuals, self.cdb
        )

        self._n_fe = 0
        self._checkpoint_idx = 0
        self._optimizer_state = {}
        self._pop_x = None
        self._pop_y = None

        self._gbest_y = np.inf
        self._gbest_x = None
        self._cost_scale = 1.0
        self._worst_y = -np.inf
        self._initial_range = (np.inf, -np.inf)

        self._best_history = [[] for _ in range(self.n_opt)]
        self._worst_history = [[] for _ in range(self.n_opt)]

        obs = self._build_observation()
        return obs, {"problem_id": problem_id, "dimension": self.dim}

    def step(self, action: int):
        assert self._problem is not None, "Call reset() before step()"

        target_fe = int(self._checkpoints[self._checkpoint_idx])
        prev_gbest = self._gbest_y

        # Record pre-step best/worst positions for movement computation
        prev_best_x, prev_worst_x = self._get_best_worst_positions()

        result = self._run_optimizer(action, target_fe)
        self._update_state(result)

        # Compute movement for the chosen optimizer
        new_best_x, new_worst_x = self._get_best_worst_positions()
        best_move = (new_best_x - prev_best_x) / (self.dim**0.5 + 1e-10)
        worst_move = (new_worst_x - prev_worst_x) / (self.dim**0.5 + 1e-10)
        self._best_history[action].append(best_move)
        self._worst_history[action].append(worst_move)

        self._checkpoint_idx += 1
        terminated = (
            self._checkpoint_idx >= self.n_checkpoints or self._n_fe >= self._max_fe
        )

        improvement = (
            max(0.0, prev_gbest - self._gbest_y)
            if np.isfinite(prev_gbest) and np.isfinite(self._gbest_y)
            else 0.0
        )
        reward = float(improvement / (self._cost_scale + 1e-10))
        if not np.isfinite(reward):
            reward = 0.0

        obs = self._build_observation()
        info = {"best_y": self._gbest_y, "n_fe": self._n_fe}
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Optimizer execution (reuses pypop7 warm-start machinery)
    # ------------------------------------------------------------------

    def _run_optimizer(self, action: int, target_fe: int) -> dict:
        optimizer_class = self.optimizers[action]
        problem_config = {
            "fitness_function": self._problem,
            "ndim_problem": self.dim,
            "lower_boundary": self._problem.lower_bounds,
            "upper_boundary": self._problem.upper_bounds,
        }
        options = {
            "max_function_evaluations": self._max_fe,
            "target_fe": target_fe,
            "n_individuals": self.n_individuals,
            "best_so_far_y": self._gbest_y,
            "verbose": False,
        }
        if self._seed is not None:
            options["seed_rng"] = (
                self._seed * 1_000_000
                + self._problem_idx * 1_000
                + self._checkpoint_idx
            ) % (2**31)

        optimizer = optimizer_class(problem_config, options)
        optimizer.n_function_evaluations = self._n_fe
        optimizer.set_data(
            best_x=self._gbest_x,
            best_y=self._gbest_y if self._gbest_y < np.inf else None,
            **self._optimizer_state,
        )
        result = optimizer.optimize()
        if isinstance(result, tuple):
            result = result[0]

        new_state = optimizer.get_data()
        if new_state:
            self._optimizer_state = new_state
        elif len(optimizer.x_history) > 0:
            self._optimizer_state = {
                "x": np.array(optimizer.x_history[-self.n_individuals :]),
                "y": np.array(optimizer.y_history[-self.n_individuals :]),
            }

        # Update cached population arrays
        if "x" in self._optimizer_state:
            self._pop_x = self._optimizer_state["x"]
            self._pop_y = self._optimizer_state["y"]

        return result

    def _update_state(self, result: dict) -> None:
        new_best_y: float = result.get("best_so_far_y", np.inf)
        new_best_x: np.ndarray | None = result.get("best_so_far_x")
        worst_y: float = result.get("worst_so_far_y", -np.inf)

        if new_best_y < self._gbest_y:
            self._gbest_y = new_best_y
            self._gbest_x = new_best_x

        if worst_y > self._worst_y:
            self._worst_y = worst_y

        if self._initial_range[0] == np.inf:
            self._initial_range = (new_best_y, max(worst_y, new_best_y + 1e-5))
            self._cost_scale = max(abs(new_best_y), 1e-10)

        y_hist = result.get("y_history")
        n_fe_step = len(y_hist) if y_hist is not None else 0
        self._n_fe = result.get("n_function_evaluations", self._n_fe + n_fe_step)

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_best_worst_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current best and worst particle positions."""
        if self._pop_x is None or len(self._pop_x) == 0:
            zeros = np.zeros(self.dim, dtype=np.float32)
            return zeros, zeros
        best_idx = int(np.argmin(self._pop_y))
        worst_idx = int(np.argmax(self._pop_y))
        return self._pop_x[best_idx].astype(np.float32), self._pop_x[worst_idx].astype(
            np.float32
        )

    def _build_observation(self) -> np.ndarray:
        # Population-state features
        if self._pop_x is not None and len(self._pop_x) > 0:
            features = _pop_features(
                self._pop_x,
                self._pop_y,
                self._gbest_y,
                self._cost_scale,
                self._n_fe,
                self._max_fe,
            )
        else:
            features = np.zeros(self.N_FEATURES, dtype=np.float32)

        # Per-optimizer movement history (averaged, or zero if unused)
        movements = []
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
