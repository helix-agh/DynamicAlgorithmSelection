"""Integration tests for heterogeneous multi-family DAS portfolios.

Each test exercises a *combination* of algorithm families (BO + PSO + ES,
BO + PSO + DE, etc.) running inside a real DASEnv episode.  The focus is
on correct warm-start hand-offs when the policy switches between algorithm
families mid-episode.

Problem variety
---------------
Five distinct landscapes at low dimensionality (d ∈ {1, 2, 3, 5}) ensure
the suites cover unimodal, non-smooth, multimodal, and asymmetric basins.

Speed contract
--------------
FE_MULTIPLIER=100 gives 200 FEs at d=2 — enough for BO (+ Fast* subclasses
to cut GP restarts) and for PSO/ES/MADDE/NL_SHADE_RSP with env-provided NP.
JDE21 hard-codes NP=170 so DE-containing portfolios use FE_MULTIPLIER=250.
"""

import warnings

import numpy as np
import pytest

from das.env.das_env import DASEnv
from das.optimizers.BO import GPBO_EI, GPBO_UCB
from das.optimizers.portfolio import get_portfolio

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
# Fast BO variants (fewer restarts — test-only)                      #
# ------------------------------------------------------------------ #


class FastGPBO_EI(GPBO_EI):
    def __init__(self, problem, options):
        super().__init__(
            problem, dict(options, n_restarts_optimizer=1, n_restarts_acq=5)
        )


class FastGPBO_UCB(GPBO_UCB):
    def __init__(self, problem, options):
        super().__init__(
            problem, dict(options, n_restarts_optimizer=1, n_restarts_acq=5)
        )


_BO_FAST = {"GPBO_EI": FastGPBO_EI, "GPBO_UCB": FastGPBO_UCB}


def resolve(spec: list[str]) -> list:
    """Resolve a string portfolio spec, substituting Fast* for BO names."""
    classes = []
    for name in spec:
        classes.append(_BO_FAST[name] if name in _BO_FAST else get_portfolio([name])[0])
    return classes


# ------------------------------------------------------------------ #
# Problem landscapes                                                  #
# ------------------------------------------------------------------ #

FUNCTIONS = {
    "sphere": lambda x: float(np.sum(x**2)),
    "abs": lambda x: float(np.sum(np.abs(x))),
    "multimodal": lambda x: float(np.sum(x**2 - np.cos(2 * np.pi * x))),
    "asymmetric": lambda x: float(np.sum(np.where(x > 0, x**2, 10 * x**2))),
    "step": lambda x: float(np.sum(np.floor(np.abs(x) + 0.5) ** 2)),
}


class MockProblem:
    def __init__(self, pid: str, dim: int, fn=None):
        self.id = pid
        self.dimension = dim
        self.lower_bounds = np.full(dim, -5.0)
        self.upper_bounds = np.full(dim, 5.0)
        shift = (hash(pid) % 100) / 20.0
        self._fn = fn or (lambda x: float(np.sum((np.asarray(x) - shift) ** 2)))

    def __call__(self, x):
        return self._fn(np.asarray(x, dtype=float))


class MockSuite:
    def __init__(self, dim: int = 2, fn=None):
        self._dim, self._fn = dim, fn
        self._cache: dict = {}

    def get_problem(self, pid: str) -> MockProblem:
        if pid not in self._cache:
            self._cache[pid] = MockProblem(pid, self._dim, self._fn)
        return self._cache[pid]


# ------------------------------------------------------------------ #
# Env factory                                                         #
# ------------------------------------------------------------------ #

PROBLEM_IDS = [f"mock_f{i:02d}" for i in range(4)]
N_CHECKPOINTS = 3
N_INDIVIDUALS = 10


def make_env(optimizer_classes, *, dim=2, fn=None, fe_multiplier=100):
    return DASEnv(
        problem_ids=PROBLEM_IDS,
        suite=MockSuite(dim=dim, fn=fn),
        optimizers=optimizer_classes,
        fe_multiplier=fe_multiplier,
        n_checkpoints=N_CHECKPOINTS,
        checkpoint_division_base=1.0,
        reward_option=1,
        n_individuals=N_INDIVIDUALS,
    )


def drain(env, policy=None):
    """Run one episode to completion; return final step info."""
    done, info = False, {}
    while not done:
        action = policy(env) if policy else env.action_space.sample()
        _, _, t, tr, info = env.step(action)
        done = t or tr
    return info


def fixed(idx):
    return lambda env: idx


def round_robin(n):
    """Cycles through actions 0, 1, 2, …, n-1 across successive steps."""
    state = {"i": 0}

    def _policy(env):
        a = state["i"] % n
        state["i"] += 1
        return a

    return _policy


# ------------------------------------------------------------------ #
# 1. BO + PSO + ES — three-family portfolios                         #
# ------------------------------------------------------------------ #

_BO_PSO_ES = [
    (["GPBO_EI", "SPSO", "LMCMAES"], 2, "sphere"),
    (["GPBO_UCB", "IPSO", "CMAES"], 2, "multimodal"),
    (["GPBO_EI", "SPSOL", "LMCMAES"], 3, "asymmetric"),
    (["GPBO_EI", "SPSO", "LMCMAES"], 5, "sphere"),
]


class TestBOPSOES:
    """Three-family episodes: BO + one PSO variant + one ES variant."""

    @pytest.mark.parametrize("spec,dim,fn_name", _BO_PSO_ES)
    def test_random_policy(self, spec, dim, fn_name):
        env = make_env(resolve(spec), dim=dim, fn=FUNCTIONS[fn_name])
        env.reset()
        info = drain(env)
        assert np.isfinite(info["best_y"])

    @pytest.mark.parametrize("spec,dim,fn_name", _BO_PSO_ES)
    def test_round_robin_exercises_all_handoffs(self, spec, dim, fn_name):
        """Round-robin forces every consecutive pair to hand off."""
        classes = resolve(spec)
        env = make_env(classes, dim=dim, fn=FUNCTIONS[fn_name])
        env.reset()
        info = drain(env, policy=round_robin(len(classes)))
        assert np.isfinite(info["best_y"])
        assert set(env._choices_history) == set(range(len(classes)))


# ------------------------------------------------------------------ #
# 2. BO + PSO + DE — three-family portfolios                         #
# ------------------------------------------------------------------ #

_BO_PSO_DE_LIGHT = [
    (["GPBO_EI", "SPSO", "MADDE"], 2, "sphere", 100),
    (["GPBO_UCB", "CPSO", "NL_SHADE_RSP"], 2, "step", 100),
]
_BO_PSO_DE_HEAVY = [
    (["GPBO_EI", "SPSO", "JDE21"], 2, "sphere", 250),
]


class TestBOPSODE:
    """Three-family episodes: BO + one PSO variant + one DE variant."""

    @pytest.mark.parametrize(
        "spec,dim,fn_name,fe_mult", _BO_PSO_DE_LIGHT + _BO_PSO_DE_HEAVY
    )
    def test_random_policy(self, spec, dim, fn_name, fe_mult):
        env = make_env(
            resolve(spec), dim=dim, fn=FUNCTIONS[fn_name], fe_multiplier=fe_mult
        )
        env.reset()
        info = drain(env)
        assert np.isfinite(info["best_y"])

    @pytest.mark.parametrize("spec,dim,fn_name,fe_mult", _BO_PSO_DE_LIGHT)
    def test_round_robin_all_families(self, spec, dim, fn_name, fe_mult):
        classes = resolve(spec)
        env = make_env(classes, dim=dim, fn=FUNCTIONS[fn_name], fe_multiplier=fe_mult)
        env.reset()
        info = drain(env, policy=round_robin(len(classes)))
        assert np.isfinite(info["best_y"])
        assert set(env._choices_history) == set(range(len(classes)))


# ------------------------------------------------------------------ #
# 3. BO + PSO + ES + DE — all four families                          #
# ------------------------------------------------------------------ #

_ALL_FOUR = [
    (["GPBO_EI", "SPSO", "CMAES", "MADDE"], 2, "sphere", 100),
    (["GPBO_UCB", "IPSO", "LMCMAES", "NL_SHADE_RSP"], 2, "abs", 100),
]


class TestAllFourFamilies:
    """Four-family portfolios: every algorithm family represented."""

    @pytest.mark.parametrize("spec,dim,fn_name,fe_mult", _ALL_FOUR)
    def test_random_policy(self, spec, dim, fn_name, fe_mult):
        env = make_env(
            resolve(spec), dim=dim, fn=FUNCTIONS[fn_name], fe_multiplier=fe_mult
        )
        env.reset()
        info = drain(env)
        assert np.isfinite(info["best_y"])

    def test_round_robin_visits_all_four(self):
        spec, dim, fn_name, fe_mult = _ALL_FOUR[0]
        classes = resolve(spec)
        env = make_env(classes, dim=dim, fn=FUNCTIONS[fn_name], fe_multiplier=fe_mult)
        env.reset()
        info = drain(env, policy=round_robin(len(classes)))
        assert np.isfinite(info["best_y"])

    def test_best_y_nondecreasing_all_families(self):
        classes = resolve(["GPBO_EI", "SPSO", "CMAES", "MADDE"])
        env = make_env(classes, fn=FUNCTIONS["sphere"])
        env.reset()
        prev, done = float("inf"), False
        while not done:
            _, _, t, tr, info = env.step(env.action_space.sample())
            done = t or tr
            assert info["best_y"] <= prev + 1e-9
            prev = info["best_y"]


# ------------------------------------------------------------------ #
# 4. Warm-start hand-off chains                                       #
# ------------------------------------------------------------------ #


class TestWarmStartChains:
    """Explicit hand-off sequences through DASEnv.

    With N_CHECKPOINTS=3 we can force exactly three transitions.
    Each test name describes the chain: A→B→C means checkpoint-0 uses A,
    checkpoint-1 uses B, checkpoint-2 uses C.
    """

    def _run_chain(self, spec, actions, dim=2, fn=None, fe_multiplier=100):
        assert len(actions) == N_CHECKPOINTS
        classes = resolve(spec)
        env = make_env(
            classes, dim=dim, fn=fn or FUNCTIONS["sphere"], fe_multiplier=fe_multiplier
        )
        env.reset()
        for a in actions:
            _, _, t, tr, info = env.step(a)
        assert env._choices_history == list(actions)
        assert np.isfinite(info["best_y"])
        return info

    # -- BO ↔ PSO --------------------------------------------------- #

    def test_bo_pso_bo(self):
        self._run_chain(["GPBO_EI", "SPSO", "LMCMAES"], actions=[0, 1, 0])

    def test_pso_bo_pso(self):
        self._run_chain(["GPBO_EI", "SPSO", "LMCMAES"], actions=[1, 0, 1])

    def test_pso_bo_es(self):
        self._run_chain(["GPBO_UCB", "SPSO", "LMCMAES"], actions=[1, 0, 2])

    # -- BO ↔ ES ---------------------------------------------------- #

    def test_bo_es_bo(self):
        self._run_chain(["GPBO_EI", "SPSO", "CMAES"], actions=[0, 2, 0])

    # -- BO ↔ DE ---------------------------------------------------- #

    def test_de_bo_pso(self):
        self._run_chain(["GPBO_EI", "SPSO", "MADDE"], actions=[2, 0, 1])

    def test_pso_bo_de(self):
        self._run_chain(["GPBO_UCB", "SPSO", "NL_SHADE_RSP"], actions=[1, 0, 2])

    def test_bo_de_bo(self):
        self._run_chain(["GPBO_EI", "SPSO", "MADDE"], actions=[0, 2, 0])

    # -- BO ↔ BO ---------------------------------------------------- #

    def test_ei_pso_ucb(self):
        self._run_chain(["GPBO_EI", "GPBO_UCB", "SPSO"], actions=[0, 2, 1])


# ------------------------------------------------------------------ #
# 5. Env contract invariants across all portfolios                   #
# ------------------------------------------------------------------ #

_CONTRACT_PORTFOLIOS = [
    (["GPBO_EI", "SPSO", "LMCMAES"], 100),
    (["GPBO_EI", "SPSO", "CMAES", "MADDE"], 100),
]


class TestEnvContractHeterogeneous:
    """Core DASEnv invariants hold for every heterogeneous portfolio."""

    @pytest.mark.parametrize("spec,fe_mult", _CONTRACT_PORTFOLIOS)
    def test_episode_terminates_in_n_checkpoints(self, spec, fe_mult):
        env = make_env(resolve(spec), fe_multiplier=fe_mult)
        env.reset()
        steps, done = 0, False
        while not done:
            _, _, t, tr, _ = env.step(env.action_space.sample())
            done = t or tr
            steps += 1
        assert steps == N_CHECKPOINTS

    @pytest.mark.parametrize("spec,fe_mult", _CONTRACT_PORTFOLIOS)
    def test_best_y_nondecreasing(self, spec, fe_mult):
        env = make_env(resolve(spec), fe_multiplier=fe_mult)
        env.reset()
        prev, done = float("inf"), False
        while not done:
            _, _, t, tr, info = env.step(env.action_space.sample())
            done = t or tr
            assert info["best_y"] <= prev + 1e-9
            prev = info["best_y"]

    @pytest.mark.parametrize("spec,fe_mult", _CONTRACT_PORTFOLIOS)
    def test_optimizer_state_nonempty_after_first_step(self, spec, fe_mult):
        env = make_env(resolve(spec), fe_multiplier=fe_mult)
        env.reset()
        env.step(0)
        assert env._optimizer_state

    @pytest.mark.parametrize("spec,fe_mult", _CONTRACT_PORTFOLIOS)
    def test_problem_advances_after_episode(self, spec, fe_mult):
        env = make_env(resolve(spec), fe_multiplier=fe_mult)
        env.reset()
        drain(env)
        assert env._problem_idx == 1

    @pytest.mark.parametrize("spec,fe_mult", _CONTRACT_PORTFOLIOS)
    def test_reset_clears_state_between_episodes(self, spec, fe_mult):
        env = make_env(resolve(spec), fe_multiplier=fe_mult)
        env.reset()
        drain(env)
        env.reset()
        # reset() runs a random probe, so _n_fe > 0 and _best_y is finite
        assert env._n_fe > 0
        assert np.isfinite(env._best_y)
        assert env._optimizer_state == {}
