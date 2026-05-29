"""Integration test: Exponential-DAS on all 10D BBOB problems, random split.

All 360 10D problem IDs are split randomly (2/3 train, 1/3 test).
A small sample from each partition is used for the actual train/eval run
so the test finishes in reasonable time while the split logic covers the
full benchmark.
"""

import json

import numpy as np
import pytest

from agents.exponential_das.agent import ExpDASAgent
from agents.exponential_das.trainer import evaluate, train
from das.env.bbob_splits import ALL_FUNCTIONS, build_problem_ids
from das.env.das_env import DASEnv
from das.env.ioh_suite import IOHSuite
from das.env.observation import observation_dim
from das.optimizers.portfolio import get_portfolio
from das.training.common import write_jsonl

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DIM = 10
PORTFOLIO = ["SPSOL", "TDE", "NM"]
FE_MULTIPLIER = 20  # 20 × 10 = 200 FEs per episode
N_CHECKPOINTS = 3
CDB = 2.0
N_INDIVIDUALS = 20
SEED = 42

N_TRAIN_SAMPLE = 6  # problems sampled from the train partition
N_TEST_SAMPLE = 4  # problems sampled from the test partition

# ------------------------------------------------------------------
# Full random split of all 10D BBOB problems (360 IDs)
# 2/3 train / 1/3 test, fixed seed for reproducibility
# ------------------------------------------------------------------

_all_10d_ids = build_problem_ids(ALL_FUNCTIONS, [DIM])
_perm = list(_all_10d_ids)
np.random.default_rng(SEED).shuffle(_perm)
_split_at = 2 * len(_perm) // 3

ALL_TRAIN_IDS: list[str] = _perm[:_split_at]
ALL_TEST_IDS: list[str] = _perm[_split_at:]

# Subsample for the actual training / eval run
TRAIN_IDS: list[str] = ALL_TRAIN_IDS[:N_TRAIN_SAMPLE]
TEST_IDS: list[str] = ALL_TEST_IDS[:N_TEST_SAMPLE]

EXPECTED_METRIC_KEYS = {
    "area_under_optimization_curve",
    "aocc",
    "final_fitness",
    "hitting_times",
    "max_fe",
    "reward",
}


# ------------------------------------------------------------------
# Module-scoped fixture — pipeline runs once, shared across all tests
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp_das_integration")
    suite = IOHSuite()
    optimizers = get_portfolio(PORTFOLIO)
    n_opt = len(optimizers)
    obs_dim = observation_dim(n_opt)

    env_cfg = dict(
        suite=suite,
        optimizers=optimizers,
        fe_multiplier=FE_MULTIPLIER,
        n_checkpoints=N_CHECKPOINTS,
        checkpoint_division_base=CDB,
        reward_option=1,
        n_individuals=N_INDIVIDUALS,
        seed=SEED,
    )

    train_env = DASEnv(problem_ids=TRAIN_IDS, **env_cfg)
    test_env = DASEnv(problem_ids=TEST_IDS, **env_cfg)

    agent = ExpDASAgent(
        obs_dim=obs_dim,
        n_actions=n_opt,
        buffer_capacity=N_CHECKPOINTS,
        actor_lr=3e-5,
        critic_lr=1e-5,
        ppo_epochs=1,
        n_checkpoints=N_CHECKPOINTS,
        device="cpu",
    )

    save_dir = str(tmp / "models")
    result_path = tmp / "results" / "test_exp_das_eval.jsonl"
    result_path.parent.mkdir()

    total_episodes = 2 * len(TRAIN_IDS)
    train(
        train_env=train_env,
        test_env=None,
        agent=agent,
        total_episodes=total_episodes,
        eval_interval=total_episodes + 1,
        save_interval=total_episodes + 1,
        save_dir=save_dir,
        name="test_exp_das",
    )

    results = evaluate(test_env, agent, n_episodes=len(TEST_IDS))
    write_jsonl(str(result_path), results)

    train_env.close()
    test_env.close()

    yield result_path, results


# ------------------------------------------------------------------
# 1. Split-level checks (no IOH / optimizer needed)
# ------------------------------------------------------------------


class TestRandomSplit:
    def test_split_covers_all_10d_problems(self):
        assert set(ALL_TRAIN_IDS) | set(ALL_TEST_IDS) == set(_all_10d_ids)

    def test_split_has_no_overlap(self):
        assert not (set(ALL_TRAIN_IDS) & set(ALL_TEST_IDS))

    def test_train_is_twice_the_size_of_test(self):
        assert len(ALL_TRAIN_IDS) == 2 * len(ALL_TEST_IDS)

    def test_all_ids_are_10d(self):
        for pid in ALL_TRAIN_IDS + ALL_TEST_IDS:
            assert pid.endswith("_d10"), f"{pid} is not a 10D problem"

    def test_sample_ids_come_from_correct_partition(self):
        assert all(pid in ALL_TRAIN_IDS for pid in TRAIN_IDS)
        assert all(pid in ALL_TEST_IDS for pid in TEST_IDS)


# ------------------------------------------------------------------
# 2. JSONL file checks
# ------------------------------------------------------------------


@pytest.mark.integration
class TestExpDasIntegration:
    def test_result_file_is_created(self, pipeline):
        result_path, _ = pipeline
        assert result_path.exists(), "eval JSONL file was not created"

    def test_result_file_has_one_record_per_test_problem(self, pipeline):
        result_path, _ = pipeline
        records = [json.loads(line) for line in result_path.read_text().splitlines()]
        assert len(records) == len(TEST_IDS)

    def test_result_file_records_are_valid_json(self, pipeline):
        result_path, _ = pipeline
        for line in result_path.read_text().splitlines():
            assert isinstance(json.loads(line), dict)

    def test_result_file_problem_ids_match_test_set(self, pipeline):
        result_path, _ = pipeline
        records = [json.loads(line) for line in result_path.read_text().splitlines()]
        assert {next(iter(r)) for r in records} == set(TEST_IDS)

    def test_all_metric_keys_present(self, pipeline):
        _, results = pipeline
        for record in results:
            metrics = next(iter(record.values()))
            missing = EXPECTED_METRIC_KEYS - set(metrics.keys())
            assert not missing, f"missing keys: {missing}"

    def test_aocc_in_unit_interval(self, pipeline):
        _, results = pipeline
        for record in results:
            aocc = next(iter(record.values()))["aocc"]
            assert 0.0 <= aocc <= 1.0, f"aocc={aocc} outside [0, 1]"

    def test_final_fitness_is_finite(self, pipeline):
        _, results = pipeline
        for record in results:
            ff = next(iter(record.values()))["final_fitness"]
            assert np.isfinite(ff), f"final_fitness={ff} is not finite"

    def test_max_fe_equals_budget(self, pipeline):
        _, results = pipeline
        expected = FE_MULTIPLIER * DIM
        for record in results:
            max_fe = next(iter(record.values()))["max_fe"]
            assert max_fe == expected, f"max_fe={max_fe}, expected {expected}"

    def test_hitting_times_keys_are_strings(self, pipeline):
        _, results = pipeline
        for record in results:
            ht = next(iter(record.values()))["hitting_times"]
            assert all(isinstance(k, str) for k in ht)
