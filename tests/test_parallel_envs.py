"""Tests for parallel DASEnv behaviour.

How parallelism works here
--------------------------
Each parallel environment is an independent DASEnv instance.  In training,
SubprocVecEnv runs each instance in its own subprocess (forked on macOS/Linux);
DummyVecEnv runs them sequentially in the same process (used in tests).

Independence contract:
  - Each env has its own _problem_idx, episode state, and optimizer warm-start.
  - There is no shared mutable state between env instances.

Shared (read-only):
  - The problem_ids list (each env uses its own counter into it).
  - The cocoex Suite object (fetches problems by ID, no mutation).

VecNormalize:
  - One RunningMeanStd for the whole vectorised env.
  - Every step() call feeds all n_envs observations into it at once.
  - In training=False (eval) mode the stats are frozen.

Auto-reset:
  - When terminated=True the vec wrapper immediately calls env.reset()
    inside the same step() call.  The returned obs for that slot is the
    new episode's first observation; the terminal observation is in
    infos["terminal_observation"].
"""

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from das.env.das_env import DASEnv
from das.env.observation import observation_dim
from das.optimizers.portfolio import get_portfolio


# ------------------------------------------------------------------ #
# Lightweight mock BBOB suite (no cocoex dependency in tests)         #
# ------------------------------------------------------------------ #


class MockProblem:
    """Sphere function — fast and deterministic."""

    def __init__(self, problem_id: str, dim: int = 2):
        self.id = problem_id
        self.dimension = dim
        self.lower_bounds = np.full(dim, -5.0)
        self.upper_bounds = np.full(dim, 5.0)
        self._shift = (hash(problem_id) % 100) / 20.0

    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum((np.asarray(x) - self._shift) ** 2))


class MockSuite:
    def __init__(self, dim: int = 2):
        self._dim = dim
        self._cache: dict[str, MockProblem] = {}

    def get_problem(self, problem_id: str) -> MockProblem:
        if problem_id not in self._cache:
            self._cache[problem_id] = MockProblem(problem_id, self._dim)
        return self._cache[problem_id]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

PROBLEM_IDS = [f"mock_f{i:02d}" for i in range(6)]
N_CHECKPOINTS = 3
FE_MULTIPLIER = 50  # 50×dim=2 → 100 FEs per episode
N_INDIVIDUALS = 10
PORTFOLIO = ["SPSO"]


def make_env(problem_ids=PROBLEM_IDS, suite=None):
    return DASEnv(
        problem_ids=problem_ids,
        suite=suite or MockSuite(),
        optimizers=get_portfolio(PORTFOLIO),
        fe_multiplier=FE_MULTIPLIER,
        n_checkpoints=N_CHECKPOINTS,
        checkpoint_division_base=1.0,
        reward_option=1,
        n_individuals=N_INDIVIDUALS,
    )


def factory(problem_ids=PROBLEM_IDS, suite=None):
    """Zero-argument factory for DummyVecEnv."""
    _suite = suite or MockSuite()

    def _init():
        return make_env(problem_ids, _suite)

    return _init


def drain_episode(env):
    """Step a single DASEnv to completion; return final info."""
    done = False
    info = {}
    while not done:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    return info


def drain_vec_episode(vec, env_idx: int):
    """Step a DummyVecEnv until env_idx finishes its episode."""
    n = vec.num_envs
    done = False
    last_infos = [{}] * n
    while not done:
        _, _, dones, infos = vec.step(np.zeros(n, dtype=int))
        last_infos = infos
        done = dones[env_idx]
    return last_infos


# ------------------------------------------------------------------ #
# 1. Single-env basics                                                 #
# ------------------------------------------------------------------ #


class TestSingleEnv:
    def test_observation_shape(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.shape == (observation_dim(n_actions=1),)

    def test_episode_ends_after_n_checkpoints(self):
        env = make_env()
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            steps += 1
        assert steps == N_CHECKPOINTS

    def test_problem_cycling_order(self):
        """Each reset() advances to the next problem_id in sequence."""
        ids = PROBLEM_IDS[:3]
        env = make_env(problem_ids=ids)
        for expected in ids * 2:  # wrap around once
            _, info = env.reset()
            assert info["problem_id"] == expected
            drain_episode(env)

    def test_best_y_nondecreasing_within_episode(self):
        env = make_env()
        env.reset()
        prev = float("inf")
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            assert info["best_y"] <= prev + 1e-9
            prev = info["best_y"]

    def test_reset_clears_all_episode_state(self):
        env = make_env()
        env.reset()
        env.step(0)
        env.step(0)  # mid-episode
        env.reset()  # full reset
        # reset() runs a random probe, so _n_fe > 0 and _best_y is finite
        assert env._n_fe > 0
        assert np.isfinite(env._best_y)
        assert env._checkpoint_idx == 0
        assert env._choices_history == []
        assert env._optimizer_state == {}


# ------------------------------------------------------------------ #
# 2. Isolation between two independent env instances                  #
# ------------------------------------------------------------------ #


class TestEnvIsolation:
    def test_problem_idx_is_independent(self):
        """Advancing env_a's problem counter does not move env_b's."""
        suite = MockSuite()
        env_a = make_env(suite=suite)
        env_b = make_env(suite=suite)

        env_a.reset()
        env_a.reset()  # env_a: idx → 2
        env_b.reset()  # env_b: idx → 1

        assert env_a._problem_idx == 2
        assert env_b._problem_idx == 1

    def test_best_y_is_independent(self):
        """Stepping env_a does not change env_b's best_y."""
        suite = MockSuite()
        env_a = make_env(suite=suite)
        env_b = make_env(suite=suite)
        env_a.reset()
        env_b.reset()
        best_y_before = env_b._best_y  # probe value set during reset

        env_a.step(0)

        assert env_b._best_y == best_y_before

    def test_optimizer_state_does_not_leak(self):
        """Warm-start population in env_a must not appear in env_b."""
        suite = MockSuite()
        env_a = make_env(suite=suite)
        env_b = make_env(suite=suite)
        env_a.reset()
        env_b.reset()

        env_a.step(0)

        assert env_b._optimizer_state == {}

    def test_stagnation_count_is_independent(self):
        """Stagnation accumulation in env_a does not affect env_b."""
        suite = MockSuite()
        env_a = make_env(suite=suite)
        env_b = make_env(suite=suite)
        env_a.reset()
        env_b.reset()

        for _ in range(N_CHECKPOINTS):
            env_a.step(0)

        assert env_b._stagnation_count == 0

    def test_episode_completion_does_not_reset_sibling(self):
        """Finishing env_a's episode does not touch env_b's state."""
        suite = MockSuite()
        env_a = make_env(suite=suite)
        env_b = make_env(suite=suite)
        env_a.reset()
        env_b.reset()

        drain_episode(env_a)

        # env_b is still on its first step (untouched)
        assert env_b._checkpoint_idx == 0


# ------------------------------------------------------------------ #
# 3. Vectorised environment (DummyVecEnv)                             #
# ------------------------------------------------------------------ #


class TestDummyVecEnv:
    def _make(self, n_envs: int, suite=None):
        _suite = suite or MockSuite()
        return DummyVecEnv([factory(suite=_suite) for _ in range(n_envs)])

    def test_reset_obs_shape_single(self):
        vec = self._make(1)
        obs = vec.reset()
        assert obs.shape == (1, observation_dim(n_actions=1))

    def test_reset_obs_shape_multi(self):
        vec = self._make(3)
        obs = vec.reset()
        assert obs.shape == (3, observation_dim(n_actions=1))

    def test_step_returns_correct_batch_size(self):
        n = 2
        vec = self._make(n)
        vec.reset()
        obs, rewards, dones, infos = vec.step(np.zeros(n, dtype=int))
        assert obs.shape[0] == n
        assert rewards.shape == (n,)
        assert dones.shape == (n,)
        assert len(infos) == n

    def test_each_env_reports_its_own_info(self):
        """After an episode, each info dict belongs to its own env."""
        vec = self._make(2)
        vec.reset()
        infos = drain_vec_episode(vec, env_idx=0)
        # The final info for each env contains best_y
        assert "best_y" in infos[0]

    def test_auto_reset_on_termination(self):
        """When env terminates, the vec wrapper resets it automatically.
        The obs slot holds the new episode's first observation, and the
        terminal observation is preserved in infos['terminal_observation'].
        """
        vec = self._make(1)
        vec.reset()
        done = False
        last_obs = last_infos = None
        while not done:
            last_obs, _, dones, last_infos = vec.step(np.zeros(1, dtype=int))
            done = dones[0]

        # The obs returned when done=True is the NEW episode's first obs
        # (env was already reset inside step())
        assert last_obs.shape == (1, observation_dim(n_actions=1))
        # Terminal observation is stashed separately
        assert "terminal_observation" in last_infos[0]

    def test_envs_have_independent_problem_idx_in_vec(self):
        """Each env inside DummyVecEnv has its own problem_idx counter.

        After vec.reset() both envs are at idx=1.  Calling reset() on only
        the first underlying env advances it to idx=2 while env1 stays at 1.
        """
        vec = self._make(2)
        vec.reset()  # both envs reset once → idx=1 each

        env0 = vec.envs[0]
        env1 = vec.envs[1]
        assert env0._problem_idx == 1
        assert env1._problem_idx == 1

        env0.reset()  # advance only env0
        assert env0._problem_idx == 2
        assert env1._problem_idx == 1  # untouched

    def test_n_checkpoints_steps_per_episode_in_vec(self):
        """Each env terminates after exactly N_CHECKPOINTS steps."""
        vec = self._make(1)
        vec.reset()
        steps = 0
        done = False
        while not done:
            _, _, dones, _ = vec.step(np.zeros(1, dtype=int))
            done = dones[0]
            steps += 1
        assert steps == N_CHECKPOINTS


# ------------------------------------------------------------------ #
# 4. VecNormalize — shared running stats                              #
# ------------------------------------------------------------------ #


class TestVecNormalize:
    def _make_normalized(self, n_envs: int, training: bool = True):
        suite = MockSuite()
        vec = DummyVecEnv([factory(suite=suite) for _ in range(n_envs)])
        return VecNormalize(
            vec, norm_obs=True, norm_reward=True, clip_obs=5.0, training=training
        )

    def test_obs_stats_update_after_step(self):
        """Running obs mean must shift after at least one step."""
        vn = self._make_normalized(n_envs=1)
        vn.reset()
        mean_before = vn.obs_rms.mean.copy()
        vn.step(np.zeros(1, dtype=int))
        assert not np.allclose(vn.obs_rms.mean, mean_before)

    def test_two_envs_accumulate_stats_faster(self):
        """With 2 envs, one step feeds 2 observations into obs_rms.
        The sample count should be double that of a single-env setup.
        """
        vn1 = self._make_normalized(n_envs=1)
        vn2 = self._make_normalized(n_envs=2)

        vn1.reset()
        vn2.reset()
        vn1.step(np.zeros(1, dtype=int))
        vn2.step(np.zeros(2, dtype=int))

        # vn2 processed 2 observations per step; vn1 processed 1.
        # RunningMeanStd initialises count at 1e-4 (not 0), so
        # vn2.count = eps + 2*N  and  2*vn1.count = 2*eps + 2*N — differ by eps.
        assert vn2.obs_rms.count == pytest.approx(2 * vn1.obs_rms.count, abs=1e-3)

    def test_frozen_stats_do_not_update(self):
        """With training=False, obs_rms must not change after stepping."""
        vn = self._make_normalized(n_envs=1, training=False)
        vn.reset()
        mean_before = vn.obs_rms.mean.copy()
        vn.step(np.zeros(1, dtype=int))
        assert np.allclose(vn.obs_rms.mean, mean_before)

    def test_normalized_obs_bounded_by_clip(self):
        """Normalised observations must not exceed clip_obs=5.0 in abs value."""
        vn = self._make_normalized(n_envs=2)
        vn.reset()
        for _ in range(N_CHECKPOINTS):
            obs, _, _, _ = vn.step(np.zeros(2, dtype=int))
            assert np.all(np.abs(obs) <= 5.0 + 1e-6)

    def test_each_env_contributes_to_shared_stats(self):
        """Stats after 1 step with 2 envs must differ from stats after
        1 step with 1 env — both envs must be contributing."""
        vn1 = self._make_normalized(n_envs=1)
        vn2 = self._make_normalized(n_envs=2)

        vn1.reset()
        vn2.reset()
        vn1.step(np.zeros(1, dtype=int))
        vn2.step(np.zeros(2, dtype=int))

        # More observations → closer to true mean; the two means will differ
        # unless obs are identical, which they aren't (different problems/seeds)
        assert not np.allclose(vn1.obs_rms.mean, vn2.obs_rms.mean)
