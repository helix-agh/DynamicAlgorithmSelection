"""Tests for baselines.py — random policy, fixed policies, single algorithm, oracle.

Each test uses the MockSuite/MockProblem stubs from test_parallel_envs so no
real BBOB/cocoex is needed and episodes complete in milliseconds.
"""

import numpy as np
import pytest

from das.env.das_env import DASEnv
from das.optimizers.portfolio import get_portfolio
from das.training.common import compute_run_stats
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

METRICS_KEYS = {"area_under_optimization_curve", "aocc", "final_fitness"}


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


def _metrics(record: dict) -> dict:
    """Extract the metrics dict from a {problem_id: metrics} record."""
    return next(iter(record.values()))


def _pid(record: dict) -> str:
    """Extract the problem_id key from a {problem_id: metrics} record."""
    return next(iter(record.keys()))


# ------------------------------------------------------------------ #
# 0. compute_run_stats (unit tests for the core metric function)      #
# ------------------------------------------------------------------ #


class TestComputeRunStats:
    def test_aocc_in_unit_interval(self):
        history = [(100, 5.0), (300, 2.0), (700, 0.5)]
        stats = compute_run_stats(history, 1000, global_minimum=0.0)
        assert 0.0 <= stats["aocc"] <= 1.0

    def test_auoc_known_value(self):
        # AUOC = integral of shifted fitness / max_fe (piecewise constant)
        # history = [(100, 5), (300, 2), (700, 0.5)], max_fe=1000, g=0
        # area = 5*100 + 2*200 + 0.5*400 + 0.5*300  (final plateau 700→1000)
        #      = 500 + 400 + 200 + 150 = 1250
        # AUOC = 1250 / 1000 = 1.25
        history = [(100, 5.0), (300, 2.0), (700, 0.5)]
        stats = compute_run_stats(history, 1000, global_minimum=0.0)
        assert stats["area_under_optimization_curve"] == pytest.approx(1.25)

    def test_final_fitness_is_shifted(self):
        history = [(500, 3.0)]
        stats = compute_run_stats(history, 1000, global_minimum=1.0)
        assert stats["final_fitness"] == pytest.approx(2.0)  # 3.0 - 1.0

    def test_final_fitness_zero_global_minimum(self):
        history = [(500, -4.5)]
        stats = compute_run_stats(history, 1000, global_minimum=0.0)
        assert stats["final_fitness"] == pytest.approx(-4.5)

    def test_empty_history_returns_zeros(self):
        stats = compute_run_stats([], 1000, global_minimum=0.0)
        assert stats["aocc"] == 0.0
        assert stats["area_under_optimization_curve"] == 0.0
        assert stats["final_fitness"] == 0.0

    def test_single_improvement_at_start(self):
        # Improvement at first FE, constant for the rest → AUOC = shifted_y
        history = [(1, 2.0)]
        stats = compute_run_stats(history, 1000, global_minimum=0.0)
        # area = 2.0*1 (loop) + 2.0*999 (plateau) = 2000 → / 1000 = 2.0
        assert stats["area_under_optimization_curve"] == pytest.approx(2.0)

    def test_monotone_improvement_raises_aocc(self):
        # More improvement points should yield higher AOCC
        fast = [(10, 1.0), (20, 0.1), (30, 0.01)]
        slow = [(10, 1.0), (500, 0.1), (900, 0.01)]
        stats_fast = compute_run_stats(fast, 1000, global_minimum=0.0)
        stats_slow = compute_run_stats(slow, 1000, global_minimum=0.0)
        assert stats_fast["aocc"] > stats_slow["aocc"]

    def test_all_keys_present(self):
        history = [(100, 1.0)]
        stats = compute_run_stats(history, 1000, global_minimum=0.0)
        assert METRICS_KEYS == set(stats.keys())

    def test_global_minimum_shifts_auoc(self):
        history = [(500, 5.0)]
        stats_g0 = compute_run_stats(history, 1000, global_minimum=0.0)
        stats_g2 = compute_run_stats(history, 1000, global_minimum=2.0)
        # With g=2, shifted=3 < shifted=5 → lower AUOC
        assert (
            stats_g2["area_under_optimization_curve"]
            < stats_g0["area_under_optimization_curve"]
        )

    def test_aocc_perfect_convergence(self):
        # If the optimizer reaches very close to the optimum immediately, AOCC ≈ 1
        history = [(1, 1e-7)]  # shifted ≈ 1e-7 ≈ lb → normalized ≈ 0 → (1-0)*width
        stats = compute_run_stats(history, 1000, global_minimum=0.0)
        assert stats["aocc"] > 0.9

    def test_aocc_no_convergence(self):
        # If the optimizer stays near ub (1e8), AOCC ≈ 0
        history = [(1, 1e8)]
        stats = compute_run_stats(history, 1000, global_minimum=0.0)
        assert stats["aocc"] < 0.1


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
    def test_returns_two_element_tuple(self):
        env = make_env()
        result = run_episode(env, random_policy)
        assert isinstance(result, tuple) and len(result) == 2

    def test_step_info_has_best_y(self):
        env = make_env()
        step_info, _ = run_episode(env, random_policy)
        assert "best_y" in step_info
        assert isinstance(step_info["best_y"], float)

    def test_best_y_is_finite(self):
        env = make_env()
        step_info, _ = run_episode(env, random_policy)
        assert np.isfinite(step_info["best_y"])

    def test_fitness_history_is_list_of_tuples(self):
        env = make_env()
        _, fitness_history = run_episode(env, random_policy)
        assert isinstance(fitness_history, list)
        for fe, y in fitness_history:
            assert isinstance(fe, int)
            assert isinstance(y, float)

    def test_fitness_history_fe_is_increasing(self):
        env = make_env()
        _, fitness_history = run_episode(env, random_policy)
        fes = [fe for fe, _ in fitness_history]
        assert fes == sorted(fes)

    def test_fitness_history_y_is_nonincreasing(self):
        """Improvement history must be strictly improving (best-so-far never worsens)."""
        env = make_env()
        _, fitness_history = run_episode(env, random_policy)
        ys = [y for _, y in fitness_history]
        for i in range(1, len(ys)):
            assert ys[i] < ys[i - 1]

    def test_fitness_history_nonempty_after_episode(self):
        """At least one improvement must occur (first evaluation beats inf)."""
        env = make_env()
        _, fitness_history = run_episode(env, random_policy)
        assert len(fitness_history) >= 1

    def test_fixed_policy_runs_full_episode(self):
        env = make_env()
        step_info, fitness_history = run_episode(env, fixed_policy(0))
        assert np.isfinite(step_info["best_y"])
        assert len(fitness_history) >= 1

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

    def test_record_is_nested_dict(self):
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
        assert len(r) == 1
        pid, metrics = next(iter(r.items()))
        assert pid == PROBLEM_IDS[0]
        assert METRICS_KEYS <= set(metrics.keys())

    def test_all_metric_keys_present(self):
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
        metrics = _metrics(records[0])
        assert METRICS_KEYS <= set(metrics.keys())

    def test_aocc_in_unit_interval(self):
        suite = MockSuite()
        records = collect_env_results(
            "random",
            random_policy,
            PROBLEM_IDS[:2],
            suite,
            get_portfolio(PORTFOLIO),
            make_cfg(),
            {},
        )
        for r in records:
            assert 0.0 <= _metrics(r)["aocc"] <= 1.0

    def test_final_fitness_is_finite(self):
        suite = MockSuite()
        records = collect_env_results(
            "random",
            random_policy,
            PROBLEM_IDS[:2],
            suite,
            get_portfolio(PORTFOLIO),
            make_cfg(),
            {},
        )
        for r in records:
            assert np.isfinite(_metrics(r)["final_fitness"])

    def test_final_fitness_shifted_by_global_minimum(self):
        # Use compute_run_stats directly with a fixed history to verify shift logic.
        history = [(100, 5.0)]
        stats_g0 = compute_run_stats(history, 1000, global_minimum=0.0)
        stats_g1 = compute_run_stats(history, 1000, global_minimum=1.0)
        assert stats_g0["final_fitness"] - stats_g1["final_fitness"] == pytest.approx(
            1.0
        )

    def test_agent_tag_in_metrics(self):
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
        assert _metrics(records[0])["agent"] == "fixed:SPSO"

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
        assert [_pid(r) for r in records] == ids

    def test_fixed_policy_only_uses_one_action(self):
        """A fixed-action policy must always select the same optimizer."""
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
    def test_run_single_returns_metrics_dict(self):
        opt_class = get_portfolio(["SPSO"])[0]
        problem = MockProblem("test_p", dim=2)
        stats = run_single_algorithm(opt_class, problem, FE_MULTIPLIER, N_INDIVIDUALS)
        assert isinstance(stats, dict)
        assert METRICS_KEYS <= set(stats.keys())

    def test_run_single_all_metrics_finite(self):
        opt_class = get_portfolio(["SPSO"])[0]
        problem = MockProblem("test_p", dim=2)
        stats = run_single_algorithm(opt_class, problem, FE_MULTIPLIER, N_INDIVIDUALS)
        assert np.isfinite(stats["final_fitness"])
        assert np.isfinite(stats["area_under_optimization_curve"])
        assert np.isfinite(stats["aocc"])

    def test_run_single_aocc_in_unit_interval(self):
        opt_class = get_portfolio(["SPSO"])[0]
        problem = MockProblem("test_p", dim=2)
        stats = run_single_algorithm(opt_class, problem, FE_MULTIPLIER, N_INDIVIDUALS)
        assert 0.0 <= stats["aocc"] <= 1.0

    def test_run_single_larger_budget_not_worse(self):
        opt_class = get_portfolio(["SPSO"])[0]
        problem = MockProblem("test_p", dim=2)
        small = run_single_algorithm(opt_class, problem, 2, N_INDIVIDUALS)
        large = run_single_algorithm(opt_class, problem, 50, N_INDIVIDUALS)
        assert large["final_fitness"] <= small["final_fitness"] + 1e-6

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
        metrics = _metrics(r)
        assert METRICS_KEYS <= set(metrics.keys())
        assert metrics["agent"] == "single:SPSO"

    def test_collect_single_aocc_in_unit_interval(self):
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
        for r in records:
            assert 0.0 <= _metrics(r)["aocc"] <= 1.0

    def test_single_vs_fixed_both_have_finite_final_fitness(self):
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
            assert np.isfinite(_metrics(r)["final_fitness"])


# ------------------------------------------------------------------ #
# 5. Oracle computation                                                #
# ------------------------------------------------------------------ #


class TestComputeOracle:
    def _make_results(self):
        """Two agents with known per-problem metrics for testing."""
        return {
            "agent_A": [
                {
                    "p1": {
                        "final_fitness": 5.0,
                        "aocc": 0.3,
                        "area_under_optimization_curve": 5.0,
                        "agent": "agent_A",
                    }
                },
                {
                    "p2": {
                        "final_fitness": 3.0,
                        "aocc": 0.5,
                        "area_under_optimization_curve": 3.0,
                        "agent": "agent_A",
                    }
                },
            ],
            "agent_B": [
                {
                    "p1": {
                        "final_fitness": 2.0,
                        "aocc": 0.7,
                        "area_under_optimization_curve": 2.0,
                        "agent": "agent_B",
                    }
                },
                {
                    "p2": {
                        "final_fitness": 8.0,
                        "aocc": 0.1,
                        "area_under_optimization_curve": 8.0,
                        "agent": "agent_B",
                    }
                },
            ],
        }

    def test_oracle_best_picks_minimum_final_fitness(self):
        best, _ = compute_oracle(self._make_results())
        by_pid = {_pid(r): _metrics(r) for r in best}
        assert by_pid["p1"]["final_fitness"] == 2.0  # agent_B wins
        assert by_pid["p2"]["final_fitness"] == 3.0  # agent_A wins

    def test_oracle_worst_picks_maximum_final_fitness(self):
        _, worst = compute_oracle(self._make_results())
        by_pid = {_pid(r): _metrics(r) for r in worst}
        assert by_pid["p1"]["final_fitness"] == 5.0  # agent_A is worst
        assert by_pid["p2"]["final_fitness"] == 8.0  # agent_B is worst

    def test_oracle_best_records_winning_agent(self):
        best, _ = compute_oracle(self._make_results())
        by_pid = {_pid(r): _metrics(r) for r in best}
        assert by_pid["p1"]["best_agent"] == "agent_B"
        assert by_pid["p2"]["best_agent"] == "agent_A"

    def test_oracle_worst_records_worst_agent(self):
        _, worst = compute_oracle(self._make_results())
        by_pid = {_pid(r): _metrics(r) for r in worst}
        assert by_pid["p1"]["worst_agent"] == "agent_A"
        assert by_pid["p2"]["worst_agent"] == "agent_B"

    def test_oracle_covers_all_problems(self):
        best, worst = compute_oracle(self._make_results())
        assert {_pid(r) for r in best} == {"p1", "p2"}
        assert {_pid(r) for r in worst} == {"p1", "p2"}

    def test_oracle_best_le_oracle_worst(self):
        """Per problem, oracle-best final_fitness ≤ oracle-worst final_fitness."""
        best, worst = compute_oracle(self._make_results())
        best_by_pid = {_pid(r): _metrics(r)["final_fitness"] for r in best}
        worst_by_pid = {_pid(r): _metrics(r)["final_fitness"] for r in worst}
        for pid in best_by_pid:
            assert best_by_pid[pid] <= worst_by_pid[pid]

    def test_oracle_with_real_runs(self):
        """Oracle computed from actual SPSO and IPSO fixed-policy runs."""
        suite = MockSuite()
        ids = PROBLEM_IDS[:2]
        cfg = make_cfg()
        optimizers = get_portfolio(PORTFOLIO)

        records_spso = collect_env_results(
            "fixed:SPSO", fixed_policy(0), ids, suite, optimizers, cfg, {}
        )
        records_ipso = collect_env_results(
            "fixed:IPSO", fixed_policy(1), ids, suite, optimizers, cfg, {}
        )

        best, worst = compute_oracle(
            {"fixed:SPSO": records_spso, "fixed:IPSO": records_ipso}
        )
        assert len(best) == len(ids)
        assert len(worst) == len(ids)
        for b, w in zip(
            sorted(best, key=_pid),
            sorted(worst, key=_pid),
        ):
            assert _metrics(b)["final_fitness"] <= _metrics(w)["final_fitness"] + 1e-9


# ------------------------------------------------------------------ #
# 6. summarise                                                         #
# ------------------------------------------------------------------ #


class TestSummarise:
    def _records(self, final_fitnesses, aocc_values=None):
        if aocc_values is None:
            aocc_values = [0.5] * len(final_fitnesses)
        return [
            {
                f"p{i}": {
                    "final_fitness": ff,
                    "aocc": a,
                    "area_under_optimization_curve": ff,
                    "agent": "x",
                }
            }
            for i, (ff, a) in enumerate(zip(final_fitnesses, aocc_values))
        ]

    def test_mean_final_fitness(self):
        s = summarise("x", self._records([1.0, 3.0]))
        assert s["mean_final_fitness"] == pytest.approx(2.0)

    def test_median_final_fitness(self):
        s = summarise("x", self._records([1.0, 2.0, 9.0]))
        assert s["median_final_fitness"] == pytest.approx(2.0)

    def test_best_final_fitness(self):
        s = summarise("x", self._records([5.0, 1.0, 3.0]))
        assert s["best_final_fitness"] == pytest.approx(1.0)

    def test_worst_final_fitness(self):
        s = summarise("x", self._records([5.0, 1.0, 3.0]))
        assert s["worst_final_fitness"] == pytest.approx(5.0)

    def test_mean_aocc(self):
        s = summarise("x", self._records([1.0, 2.0], aocc_values=[0.4, 0.6]))
        assert s["mean_aocc"] == pytest.approx(0.5)

    def test_n_problems(self):
        s = summarise("x", self._records([1.0, 2.0, 3.0]))
        assert s["n_problems"] == 3

    def test_agent_field(self):
        s = summarise("my_agent", self._records([1.0]))
        assert s["agent"] == "my_agent"


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
            step_info1, _ = run_episode(env1, fixed_policy(0))

            set_seed(seed)
            env2 = self._seeded_env(seed)
            step_info2, _ = run_episode(env2, fixed_policy(0))

            assert step_info1["best_y"] == pytest.approx(step_info2["best_y"]), (
                f"seed={seed}: {step_info1['best_y']} != {step_info2['best_y']}"
            )

    def test_same_seed_same_aocc(self):
        """Two envs with the same seed produce identical AOCC via identical fitness history."""
        for seed in [0, 42]:
            set_seed(seed)
            env1 = self._seeded_env(seed)
            _, hist1 = run_episode(env1, fixed_policy(0))

            set_seed(seed)
            env2 = self._seeded_env(seed)
            _, hist2 = run_episode(env2, fixed_policy(0))

            assert hist1 == hist2

    def test_different_seeds_different_best_y(self):
        """Different seeds produce different optimizer trajectories (with high probability)."""
        results = set()
        for seed in [0, 1, 2, 3, 4]:
            set_seed(seed)
            env = self._seeded_env(seed)
            step_info, _ = run_episode(env, fixed_policy(0))
            results.add(round(step_info["best_y"], 8))
        # With 5 different seeds, at least 2 distinct outcomes are expected
        assert len(results) > 1

    def test_seed_rng_set_in_optimizer_options(self):
        """DASEnv must pass seed_rng to pypop7 when env._seed is set."""
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


# ------------------------------------------------------------------ #
# 8. DASEnv step info — fitness_history_step                          #
# ------------------------------------------------------------------ #


class TestFitnessHistoryStep:
    def test_step_info_contains_fitness_history_step(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "fitness_history_step" in info

    def test_fitness_history_step_is_list(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(0)
        assert isinstance(info["fitness_history_step"], list)

    def test_fitness_history_step_entries_are_int_float_tuples(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(0)
        for fe, y in info["fitness_history_step"]:
            assert isinstance(fe, int)
            assert isinstance(y, float)

    def test_fitness_history_step_fe_within_budget(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(0)
        max_fe = FE_MULTIPLIER * 2  # dim=2
        for fe, _ in info["fitness_history_step"]:
            assert 1 <= fe <= max_fe

    def test_fitness_history_step_accumulated_across_checkpoints(self):
        """Full episode fitness history must contain at least as many points as one step."""
        env = make_env()
        env.reset()
        all_history = []
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            all_history.extend(info["fitness_history_step"])

        # At minimum one improvement in the first checkpoint (from inf)
        assert len(all_history) >= 1

    def test_fitness_history_step_fe_monotone_across_episode(self):
        """FE values accumulated across all checkpoints must be strictly increasing."""
        env = make_env()
        env.reset()
        all_history = []
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            all_history.extend(info["fitness_history_step"])

        fes = [fe for fe, _ in all_history]
        assert fes == sorted(set(fes)), "Duplicate or out-of-order FE values in history"
