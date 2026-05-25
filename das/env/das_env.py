"""Gymnasium environment for Dynamic Algorithm Selection on COCO-BBOB.

Each episode corresponds to one optimization run on a single BBOB problem.
At every timestep the agent picks which sub-optimizer to run next; the
optimizer then runs until the next exponentially-spaced checkpoint.

Observation space : Box(-inf, +inf, shape=(state_dim,)) – normalized externally
                    via stable-baselines3's VecNormalize.
Action space      : Discrete(n_optimizers)
Reward            : Fitness improvement, scaled and shaped by reward_option.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from das.env.observation import (compute_observation, observation_dim, MAX_HISTORY_SAMPLE)
from das.env.reward import compute_reward
from das.optimizers.base import get_checkpoints


class DASEnv(gym.Env):
    """DAS environment.

    Parameters
    ----------
    problem_ids:
        BBOB problem IDs to cycle through (one per episode).
    suite:
        IOHSuite (or compatible) object to fetch problems from.
    optimizers:
        Ordered list of sub-optimizer classes (defines the action space).
    fe_multiplier:
        Budget = fe_multiplier * problem_dimension.
    n_checkpoints:
        Number of optimizer-selection steps per episode.
    checkpoint_division_base (cdb):
        cdb=1.0 → uniform checkpoints; cdb>1.0 → exponentially growing intervals.
    reward_option:
        1=log-scaled, 2=linear, 3=sparse, 4=binary (see das/env/reward.py).
    n_individuals:
        Population size per sub-optimizer.  ``None`` (default) lets each
        algorithm use its own built-in default.  Pass a single ``int`` to
        override all algorithms with the same value, or a list of ``int |
        None`` with one entry per optimizer for a mixed setup.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        problem_ids: list[str],
        suite,
        optimizers: list,
        fe_multiplier: int = 10_000,
        n_checkpoints: int = 10,
        checkpoint_division_base: float = 1.0,
        reward_option: int = 1,
        n_individuals: int | list[int | None] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.problem_ids = problem_ids
        self.suite = suite
        self.optimizers = optimizers
        self.fe_multiplier = fe_multiplier
        self.n_checkpoints = n_checkpoints
        self.cdb = checkpoint_division_base
        self.reward_option = reward_option
        n_opt = len(optimizers)
        if n_individuals is None or isinstance(n_individuals, int):
            self.n_individuals: list[int | None] = [n_individuals] * n_opt
        else:
            pop = list(n_individuals)
            if len(pop) == 1:
                pop = pop * n_opt
            if len(pop) != n_opt:
                raise ValueError(
                    f"n_individuals has {len(pop)} entries but portfolio has {n_opt}"
                )
            self.n_individuals = pop
        self._seed = seed

        n_actions = len(optimizers)
        obs_dim = observation_dim(n_actions)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

        # Episode state – reset() initialises these
        self._problem = None
        self._problem_idx = 0
        self._max_fe = 0
        self._n_fe = 0
        self._checkpoints: np.ndarray | None = None
        self._checkpoint_idx = 0

        self._optimizer_state: dict = {}  # passed between sub-optimizers for warm-starting
        self._x_history: np.ndarray | None = None
        self._y_history: np.ndarray | None = None

        self._best_y = float("inf")
        self._best_x: np.ndarray | None = None
        self._worst_y = -np.inf
        self._initial_range: tuple[float, float] = (float("inf"), -np.inf)
        self._stagnation_count = 0
        self._choices_history: list[int] = []

    # ------------------------------------------------------------------ #
    # Gymnasium interface                                                  #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        problem_id = self.problem_ids[self._problem_idx % len(self.problem_ids)]
        self._problem_idx += 1

        self._problem = self.suite.get_problem(problem_id)
        dim = self._problem.dimension
        self._max_fe = self.fe_multiplier * dim
        known = [n for n in self.n_individuals if n is not None]
        self._checkpoints = get_checkpoints(
            self.n_checkpoints, self._max_fe, max(known) if known else 1, self.cdb
        )

        # Reset episode bookkeeping
        self._n_fe = 0
        self._checkpoint_idx = 0
        self._optimizer_state = {}
        self._x_history = None
        self._y_history = None
        self._best_y = float("inf")
        self._best_x = None
        self._worst_y = -np.inf
        self._initial_range = (float("inf"), -np.inf)
        self._stagnation_count = 0
        self._choices_history = []

        obs = self._build_observation()
        info = {"problem_id": problem_id, "dimension": dim}
        return obs, info

    def step(self, action: int):
        assert self._problem is not None, "Call reset() before step()"

        target_fe = int(self._checkpoints[self._checkpoint_idx])
        prev_best_y = self._best_y

        result = self._run_optimizer(action, target_fe)

        self._update_episode_state(result, prev_best_y)
        self._choices_history.append(action)
        self._checkpoint_idx += 1

        terminated = (
            self._checkpoint_idx >= self.n_checkpoints or self._n_fe >= self._max_fe
        )
        reward = compute_reward(
            self._best_y,
            prev_best_y,
            self._initial_range,
            option=self.reward_option,
            is_final=terminated,
        )

        obs = self._build_observation()
        info = {
            "best_y": self._best_y,
            "n_fe": self._n_fe,
            "checkpoint": self._checkpoint_idx,
            "fitness_history_step": result.get("fitness_history", []),
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_optimizer(self, action: int, target_fe: int) -> dict:
        """Instantiate the selected sub-optimizer and run it to target_fe."""
        optimizer_class = self.optimizers[action]
        problem_config = {
            "fitness_function": self._problem,
            "ndim_problem": self._problem.dimension,
            "lower_boundary": self._problem.lower_bounds,
            "upper_boundary": self._problem.upper_bounds,
        }
        n_ind = self.n_individuals[action]
        options = {
            "max_function_evaluations": self._max_fe,
            "target_fe": target_fe,
            "best_so_far_y": self._best_y,
            "verbose": False,
        }
        if n_ind is not None:
            options["n_individuals"] = n_ind
        # Derive a deterministic seed for pypop7's default_rng, which is
        # independent of np.random and must be seeded explicitly.
        if self._seed is not None:
            options["seed_rng"] = (
                self._seed * 1_000_000
                + self._problem_idx * 1_000
                + self._checkpoint_idx
            ) % (2**31)
        optimizer = optimizer_class(problem_config, options)
        optimizer.n_function_evaluations = self._n_fe

        optimizer.set_data(
            best_x=self._best_x,
            best_y=self._best_y if self._best_y < float("inf") else None,
            **self._optimizer_state,
        )
        result = optimizer.optimize()
        # result may be (result_dict, agent_state) tuple in subclasses; normalise
        if isinstance(result, tuple):
            result = result[0]

        # Update warm-start state for next step
        new_state = optimizer.get_data()
        if new_state:
            self._optimizer_state = new_state
        else:
            # Fallback: carry x/y from the population history
            if len(optimizer.x_history) > 0:
                x_hist = optimizer.x_history
                y_hist = optimizer.y_history
                if n_ind is not None:
                    x_hist = x_hist[-n_ind:]
                    y_hist = y_hist[-n_ind:]
                self._optimizer_state = {
                    "x": np.array(x_hist),
                    "y": np.array(y_hist),
                }

        return result

    def _update_episode_state(self, result: dict, prev_best_y: float):
        new_best_y: float = result.get("best_so_far_y", float("inf"))
        new_best_x: np.ndarray | None = result.get("best_so_far_x")
        worst_y: float = result.get("worst_so_far_y", -np.inf)

        if new_best_y < self._best_y:
            self._best_y = new_best_y
            self._best_x = new_best_x

        if worst_y > self._worst_y:
            self._worst_y = worst_y

        # Set initial range on first step.
        # When worst_so_far_y is absent the default is -inf, which collapses
        # scale to 1e-5 and inflates every subsequent reward by 1e5.  Instead,
        # derive scale from the magnitude of the initial best fitness.
        if self._initial_range[0] == float("inf"):
            safe_worst = (
                worst_y if np.isfinite(worst_y) else new_best_y + max(abs(new_best_y), 1.0)
            )
            self._initial_range = (new_best_y, max(safe_worst, new_best_y + 1e-5))

        # Stagnation counter — prefer the FE delta from the result dict so that
        # stagnation accumulates correctly even when y_history is not returned.
        x_hist: np.ndarray | None = result.get("x_history")
        y_hist: np.ndarray | None = result.get("y_history")
        n_fe_reported = result.get("n_function_evaluations")
        if n_fe_reported is not None:
            n_fe_step = max(0, n_fe_reported - self._n_fe)
        else:
            n_fe_step = len(y_hist) if y_hist is not None else 0

        if new_best_y >= prev_best_y:
            self._stagnation_count += n_fe_step
        else:
            self._stagnation_count = 0

        self._n_fe = result.get("n_function_evaluations", self._n_fe + n_fe_step)

        # Accumulate population history for ELA, capped at MAX_HISTORY_SAMPLE rows.
        # Without the cap, large budgets (e.g. 40-dim × 10 000 FE) accumulate
        # hundreds of thousands of rows — GBs of RAM for a single episode.
        if x_hist is not None and len(x_hist) > 0:
            self._x_history = (
                x_hist[-MAX_HISTORY_SAMPLE:]
                if self._x_history is None
                else np.concatenate([self._x_history, x_hist])[-MAX_HISTORY_SAMPLE:]
            )
            self._y_history = (
                y_hist[-MAX_HISTORY_SAMPLE:]
                if self._y_history is None
                else np.concatenate([self._y_history, y_hist])[-MAX_HISTORY_SAMPLE:]
            )

    def _build_observation(self) -> np.ndarray:

        return compute_observation(
            x_history=self._x_history,
            y_history=self._y_history,
            choices_history=self._choices_history,
            n_actions=len(self.optimizers),
            n_checkpoints=self.n_checkpoints,
            n_fe=self._n_fe,
            max_fe=max(self._max_fe, 1),
            stagnation_count=self._stagnation_count,
            ndim_problem=self._problem.dimension if self._problem is not None else 1,
        )
