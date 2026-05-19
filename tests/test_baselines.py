"""Tests for baselines.py — random policy, fixed policies, single algorithm, oracle.

Each test uses the MockSuite/MockProblem stubs from test_parallel_envs so no
real BBOB/cocoex is needed and episodes complete in milliseconds.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from das.env.das_env import DASEnv
from das.optimizers.portfolio import get_portfolio
from das.utils import set_seed
from baselines import (
    collect_env_results,
    collect_single_results,
    compute_oracle,
    fixed_policy,
    random_policy,
    run_episode,
    run_single_algorithm,
    summarise,
)


# ------------------------------------------------------------------ #
# Shared stubs (same as test_parallel_envs)                           #
# ------------------------------------------------------------------ #


class MockProblem:
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


PROBLEM_IDS = [f"mock_f{i:02d}" for i in range(4)]
N_CHECKPOINTS = 3
FE_MULTIPLIER = (
    5  # keep total budget (FE_MULTIPLIER × dim) < 50 to avoid ELA computation
)
N_INDIVIDUALS = 10
PORTFOLIO = ["SPSO", "IPSO"]


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


def make_cfg():
    return dict(
        fe_multiplier=FE_MULTIPLIER,
        n_checkpoints=N_CHECKPOINTS,
        cdb=1.0,
        reward_option=1,
        n_individuals=N_INDIVIDUALS,
    )


# ------------------------------------------------------------------ #
# 1. Policy functions                                                  #
# ------------------------------------------------------------------ #


class TestPolicyFunctions:
    def test_random_policy_returns_valid_action(self):
        n = 3
        obs = np.zeros(10)
        actions = {random_policy(obs, n) for _ in range(50)}
        # With 50 samples and 3 actions, all should appear
        assert actions == {0, 1, 2}

    def test_random_policy_is_within_bounds(self):
        obs = np.zeros(10)
        for _ in range(100):
            a = random_policy(obs, 5)
            assert 0 <= a < 5

    def test_fixed_policy_always_returns_same_action(self):
        obs = np.zeros(10)
        for target in range(3):
            policy = fixed_policy(target)
            for _ in range(10):
                assert policy(obs, 3) == target

    def test_fixed_policy_ignores_obs(self):
        policy = fixed_policy(1)
        assert policy(np.zeros(5), 3) == 1
        assert policy(np.ones(5) * 999, 3) == 1


# ------------------------------------------------------------------ #
# 2. run_episode                                                       #
# ------------------------------------------------------------------ #


class TestRunEpisode:
    def test_returns_info_dict_with_best_y(self):
        env = make_env()
        info = run_episode(env, random_policy)
        assert "best_y" in info
        assert isinstance(info["best_y"], float)

    def test_fixed_policy_runs_full_episode(self):
        env = make_env()
        info = run_episode(env, fixed_policy(0))
        assert "best_y" in info

    def test_best_y_is_finite(self):
        env = make_env()
        info = run_episode(env, random_policy)
        assert np.isfinite(info["best_y"])

    def test_episode_advances_problem_idx(self):
        env = make_env()
        assert env._problem_idx == 0
        run_episode(env, random_policy)
        assert env._problem_idx == 1

    def test_multiple_episodes_cycle_problems(self):
        env = make_env(problem_ids=PROBLEM_IDS[:3])
        for expected_idx in range(6):  # wrap around twice
            run_episode(env, random_policy)
        assert env._problem_idx == 6


# ------------------------------------------------------------------ #
# 3. collect_env_results                                               #
# ------------------------------------------------------------------ #


class TestCollectEnvResults:
    def test_returns_one_record_per_problem(self):
        suite = MockSuite()
        ids = PROBLEM_IDS[:2]
        records = collect_env_results(
            "random",
            random_policy,
            ids,
            suite,
            get_portfolio(PORTFOLIO),
            make_cfg(),
            {},
        )
        assert len(records) == 2

    def test_record_structure(self):
        suite = MockSuite()
        records = collect_env_results(
            "random",
            random_policy,
            PROBLEM_IDS[:1],
            suite,
            get_portfolio(PORTFOLIO),
            make_cfg(),
            {},
        )
        r = records[0]
        assert set(r.keys()) >= {"problem_id", "agent", "best_y", "gap"}

    def test_agent_tag_in_records(self):
        suite = MockSuite()
        records = collect_env_results(
            "fixed:SPSO",
            fixed_policy(0),
            PROBLEM_IDS[:1],
            suite,
            get_portfolio(PORTFOLIO),
            make_cfg(),
            {},
        )
        assert records[0]["agent"] == "fixed:SPSO"

    def test_gap_equals_best_y_minus_optimum(self):
        suite = MockSuite()
        optima = {PROBLEM_IDS[0]: 1.0}
        records = collect_env_results(
            "random",
            random_policy,
            PROBLEM_IDS[:1],
            suite,
            get_portfolio(PORTFOLIO),
            make_cfg(),
            optima,
        )
        r = records[0]
        assert abs(r["gap"] - (r["best_y"] - 1.0)) < 1e-9

    def test_problem_ids_match_order(self):
        suite = MockSuite()
        ids = PROBLEM_IDS[:3]
        records = collect_env_results(
            "random",
            random_policy,
            ids,
            suite,
            get_portfolio(PORTFOLIO),
            make_cfg(),
            {},
        )
        assert [r["problem_id"] for r in records] == ids

    def test_fixed_policy_only_uses_one_action(self):
        """A fixed-action policy must always select the same optimizer.

        We verify this by checking that env._choices_history contains only
        the chosen action index after one complete episode.
        """
        suite = MockSuite()
        for action_idx in range(len(PORTFOLIO)):
            env = make_env(suite=suite)
            run_episode(env, fixed_policy(action_idx))
            assert all(c == action_idx for c in env._choices_history), (
                f"Expected only action {action_idx}, got {env._choices_history}"
            )


# ------------------------------------------------------------------ #
# 4. run_single_algorithm / collect_single_results                    #
# ------------------------------------------------------------------ #


class TestSingleAlgorithm:
    def test_run_single_returns_float(self):
        opt_class = get_portfolio(["SPSO"])[0]
        problem = MockProblem("test_p", dim=2)
        result = run_single_algorithm(opt_class, problem, FE_MULTIPLIER, N_INDIVIDUALS)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_run_single_uses_full_budget(self):
        """The optimizer must receive the full budget (fe_multiplier × dim)."""
        opt_class = get_portfolio(["SPSO"])[0]
        problem = MockProblem("test_p", dim=2)
        result_small = run_single_algorithm(opt_class, problem, 2, N_INDIVIDUALS)
        result_large = run_single_algorithm(opt_class, problem, 50, N_INDIVIDUALS)
        # Larger budget should not produce a worse result
        assert result_large <= result_small + 1e-6

    def test_collect_single_returns_one_record_per_problem(self):
        suite = MockSuite()
        opt_class = get_portfolio(["SPSO"])[0]
        records = collect_single_results(
            "single:SPSO",
            opt_class,
            PROBLEM_IDS[:2],
            suite,
            FE_MULTIPLIER,
            N_INDIVIDUALS,
            {},
        )
        assert len(records) == 2

    def test_collect_single_record_structure(self):
        suite = MockSuite()
        opt_class = get_portfolio(["SPSO"])[0]
        records = collect_single_results(
            "single:SPSO",
            opt_class,
            PROBLEM_IDS[:1],
            suite,
            FE_MULTIPLIER,
            N_INDIVIDUALS,
            {},
        )
        r = records[0]
        assert set(r.keys()) >= {"problem_id", "agent", "best_y", "gap"}
        assert r["agent"] == "single:SPSO"

    def test_single_vs_fixed_both_finite(self):
        """Both single and fixed-policy agents should produce finite gaps."""
        suite = MockSuite()
        opt_class = get_portfolio(["SPSO"])[0]

        single_records = collect_single_results(
            "single:SPSO",
            opt_class,
            PROBLEM_IDS[:2],
            suite,
            FE_MULTIPLIER,
            N_INDIVIDUALS,
            {},
        )
        fixed_records = collect_env_results(
            "fixed:SPSO",
            fixed_policy(0),
            PROBLEM_IDS[:2],
            suite,
            get_portfolio(PORTFOLIO),
            make_cfg(),
            {},
        )
        for r in single_records + fixed_records:
            assert np.isfinite(r["gap"])


# ------------------------------------------------------------------ #
# 5. Oracle computation                                                #
# ------------------------------------------------------------------ #


class TestComputeOracle:
    def _make_results(self):
        """Two agents with known per-problem gaps for testing."""
        return {
            "agent_A": [
                {"problem_id": "p1", "agent": "agent_A", "best_y": 5.0, "gap": 5.0},
                {"problem_id": "p2", "agent": "agent_A", "best_y": 3.0, "gap": 3.0},
            ],
            "agent_B": [
                {"problem_id": "p1", "agent": "agent_B", "best_y": 2.0, "gap": 2.0},
                {"problem_id": "p2", "agent": "agent_B", "best_y": 8.0, "gap": 8.0},
            ],
        }

    def test_oracle_best_picks_minimum_gap(self):
        best, _ = compute_oracle(self._make_results())
        by_pid = {r["problem_id"]: r for r in best}
        assert by_pid["p1"]["gap"] == 2.0  # agent_B wins
        assert by_pid["p2"]["gap"] == 3.0  # agent_A wins

    def test_oracle_worst_picks_maximum_gap(self):
        _, worst = compute_oracle(self._make_results())
        by_pid = {r["problem_id"]: r for r in worst}
        assert by_pid["p1"]["gap"] == 5.0  # agent_A is worst
        assert by_pid["p2"]["gap"] == 8.0  # agent_B is worst

    def test_oracle_best_records_winning_agent(self):
        best, _ = compute_oracle(self._make_results())
        by_pid = {r["problem_id"]: r for r in best}
        assert by_pid["p1"]["best_agent"] == "agent_B"
        assert by_pid["p2"]["best_agent"] == "agent_A"

    def test_oracle_worst_records_worst_agent(self):
        _, worst = compute_oracle(self._make_results())
        by_pid = {r["problem_id"]: r for r in worst}
        assert by_pid["p1"]["worst_agent"] == "agent_A"
        assert by_pid["p2"]["worst_agent"] == "agent_B"

    def test_oracle_covers_all_problems(self):
        best, worst = compute_oracle(self._make_results())
        pids_best = {r["problem_id"] for r in best}
        pids_worst = {r["problem_id"] for r in worst}
        assert pids_best == {"p1", "p2"}
        assert pids_worst == {"p1", "p2"}

    def test_oracle_best_le_oracle_worst(self):
        """Per problem, oracle-best gap ≤ oracle-worst gap."""
        best, worst = compute_oracle(self._make_results())
        best_by_pid = {r["problem_id"]: r["gap"] for r in best}
        worst_by_pid = {r["problem_id"]: r["gap"] for r in worst}
        for pid in best_by_pid:
            assert best_by_pid[pid] <= worst_by_pid[pid]

    def test_oracle_with_real_runs(self):
        """Oracle computed from actual SPSO and IPSO fixed-policy runs."""
        suite = MockSuite()
        ids = PROBLEM_IDS[:2]
        cfg = make_cfg()
        optimizers = get_portfolio(PORTFOLIO)
        global_optima = {}

        records_spso = collect_env_results(
            "fixed:SPSO", fixed_policy(0), ids, suite, optimizers, cfg, global_optima
        )
        records_ipso = collect_env_results(
            "fixed:IPSO", fixed_policy(1), ids, suite, optimizers, cfg, global_optima
        )

        best, worst = compute_oracle(
            {
                "fixed:SPSO": records_spso,
                "fixed:IPSO": records_ipso,
            }
        )
        assert len(best) == len(ids)
        assert len(worst) == len(ids)
        for b, w in zip(
            sorted(best, key=lambda r: r["problem_id"]),
            sorted(worst, key=lambda r: r["problem_id"]),
        ):
            assert b["gap"] <= w["gap"] + 1e-9


# ------------------------------------------------------------------ #
# 6. summarise                                                         #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# 7. Reproducibility via seed                                          #
# ------------------------------------------------------------------ #


class TestReproducibility:
    def _seeded_env(self, seed):
        return DASEnv(
            problem_ids=PROBLEM_IDS[:2],
            suite=MockSuite(),
            optimizers=get_portfolio(PORTFOLIO),
            fe_multiplier=FE_MULTIPLIER,
            n_checkpoints=N_CHECKPOINTS,
            checkpoint_division_base=1.0,
            reward_option=1,
            n_individuals=N_INDIVIDUALS,
            seed=seed,
        )

    def test_same_seed_same_best_y(self):
        """Two envs with the same seed and same fixed policy produce the same best_y."""
        for seed in [0, 42, 999]:
            set_seed(seed)
            env1 = self._seeded_env(seed)
            info1 = run_episode(env1, fixed_policy(0))

            set_seed(seed)
            env2 = self._seeded_env(seed)
            info2 = run_episode(env2, fixed_policy(0))

            assert info1["best_y"] == pytest.approx(info2["best_y"]), (
                f"seed={seed}: {info1['best_y']} != {info2['best_y']}"
            )

    def test_different_seeds_different_best_y(self):
        """Different seeds produce different optimizer trajectories (with high probability)."""
        results = set()
        for seed in [0, 1, 2, 3, 4]:
            set_seed(seed)
            env = self._seeded_env(seed)
            info = run_episode(env, fixed_policy(0))
            results.add(round(info["best_y"], 8))
        # With 5 different seeds, at least 2 distinct outcomes are expected
        assert len(results) > 1

    def test_seed_rng_set_in_optimizer_options(self):
        """DASEnv must pass seed_rng to pypop7 when env._seed is set."""
        from unittest.mock import patch, MagicMock

        env = self._seeded_env(seed=42)
        env.reset()

        captured_options = {}

        original_class = get_portfolio(PORTFOLIO)[0]

        class CapturingOptimizer(original_class):
            def __init__(self, problem, options):
                captured_options.update(options)
                super().__init__(problem, options)

        env.optimizers = [CapturingOptimizer] + env.optimizers[1:]
        env.step(0)

        assert "seed_rng" in captured_options, (
            "seed_rng was not passed to the optimizer options"
        )
        assert isinstance(captured_options["seed_rng"], int)

    def test_no_seed_does_not_set_seed_rng(self):
        """Without a seed, seed_rng must NOT be injected (pypop7 uses random state)."""
        from unittest.mock import patch

        env = DASEnv(
            problem_ids=PROBLEM_IDS[:1],
            suite=MockSuite(),
            optimizers=get_portfolio(PORTFOLIO),
            fe_multiplier=FE_MULTIPLIER,
            n_checkpoints=N_CHECKPOINTS,
            checkpoint_division_base=1.0,
            reward_option=1,
            n_individuals=N_INDIVIDUALS,
            seed=None,
        )
        env.reset()

        captured_options = {}
        original_class = get_portfolio(PORTFOLIO)[0]

        class CapturingOptimizer(original_class):
            def __init__(self, problem, options):
                captured_options.update(options)
                super().__init__(problem, options)

        env.optimizers = [CapturingOptimizer] + env.optimizers[1:]
        env.step(0)

        assert "seed_rng" not in captured_options

    def test_seed_rng_differs_per_checkpoint(self):
        """Each checkpoint must get a unique seed_rng to avoid correlated RNG states."""
        env = self._seeded_env(seed=42)
        env.reset()

        seeds_seen = []
        original_class = get_portfolio(PORTFOLIO)[0]

        class CapturingOptimizer(original_class):
            def __init__(self, problem, options):
                seeds_seen.append(options.get("seed_rng"))
                super().__init__(problem, options)

        env.optimizers = [CapturingOptimizer] + env.optimizers[1:]
        for _ in range(N_CHECKPOINTS):
            done = env._checkpoint_idx >= env.n_checkpoints
            if not done:
                env.step(0)

        assert len(set(seeds_seen)) == N_CHECKPOINTS, (
            f"Expected {N_CHECKPOINTS} unique seed_rng values, got {seeds_seen}"
        )


class TestSummarise:
    def _records(self, gaps):
        return [
            {"problem_id": f"p{i}", "agent": "x", "best_y": g, "gap": g}
            for i, g in enumerate(gaps)
        ]

    def test_mean_gap(self):
        s = summarise("x", self._records([1.0, 3.0]))
        assert s["mean_gap"] == pytest.approx(2.0)

    def test_median_gap(self):
        s = summarise("x", self._records([1.0, 2.0, 9.0]))
        assert s["median_gap"] == pytest.approx(2.0)

    def test_best_gap(self):
        s = summarise("x", self._records([5.0, 1.0, 3.0]))
        assert s["best_gap"] == pytest.approx(1.0)

    def test_worst_gap(self):
        s = summarise("x", self._records([5.0, 1.0, 3.0]))
        assert s["worst_gap"] == pytest.approx(5.0)

    def test_n_problems(self):
        s = summarise("x", self._records([1.0, 2.0, 3.0]))
        assert s["n_problems"] == 3
