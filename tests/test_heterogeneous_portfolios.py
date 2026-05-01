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


def reverse_round_robin(n):
    """Cycles through actions n-1, n-2, …, 0 across successive steps."""
    state = {"i": 0}

    def _policy(env):
        a = (n - 1) - (state["i"] % n)
        state["i"] += 1
        return a

    return _policy


def _make_rr(n, direction):
    return round_robin(n) if direction == "forward" else reverse_round_robin(n)


# ------------------------------------------------------------------ #
# 1. BO + PSO + ES — three-family portfolios                         #
# ------------------------------------------------------------------ #

# Each row: (portfolio spec, dim, fn_name)
_BO_PSO_ES = [
    (["GPBO_EI", "SPSO", "LMCMAES"], 2, "sphere"),
    (["GPBO_UCB", "SPSO", "LMCMAES"], 2, "abs"),
    (["GPBO_EI", "IPSO", "CMAES"], 2, "multimodal"),
    (["GPBO_UCB", "IPSO", "CMAES"], 3, "sphere"),
    (["GPBO_EI", "SPSOL", "LMCMAES"], 3, "asymmetric"),
    (["GPBO_UCB", "CPSO", "CMAES"], 2, "step"),
    (["GPBO_EI", "SPSO", "CMAES"], 1, "sphere"),
    (["GPBO_UCB", "IPSO", "LMCMAES"], 1, "abs"),
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

    @pytest.mark.parametrize("direction", ["forward", "reverse"])
    @pytest.mark.parametrize("spec,dim,fn_name", _BO_PSO_ES)
    def test_round_robin_exercises_all_handoffs(self, spec, dim, fn_name, direction):
        """Round-robin (forward and reverse) forces every consecutive pair to hand off."""
        classes = resolve(spec)
        env = make_env(classes, dim=dim, fn=FUNCTIONS[fn_name])
        env.reset()
        info = drain(env, policy=_make_rr(len(classes), direction))
        assert np.isfinite(info["best_y"])
        assert set(env._choices_history) == set(range(len(classes)))

    @pytest.mark.parametrize("spec,dim,fn_name", _BO_PSO_ES[:4])
    def test_fixed_bo_policy_only_calls_bo(self, spec, dim, fn_name):
        env = make_env(resolve(spec), dim=dim, fn=FUNCTIONS[fn_name])
        env.reset()
        drain(env, policy=fixed(0))
        assert all(c == 0 for c in env._choices_history)
        assert np.isfinite(env._best_y)

    @pytest.mark.parametrize("spec,dim,fn_name", _BO_PSO_ES[:4])
    def test_fixed_pso_policy_never_calls_bo(self, spec, dim, fn_name):
        env = make_env(resolve(spec), dim=dim, fn=FUNCTIONS[fn_name])
        env.reset()
        drain(env, policy=fixed(1))  # PSO is always at index 1
        assert all(c == 1 for c in env._choices_history)
        assert np.isfinite(env._best_y)

    @pytest.mark.parametrize("spec,dim,fn_name", _BO_PSO_ES[:4])
    def test_fixed_es_policy_never_calls_bo(self, spec, dim, fn_name):
        env = make_env(resolve(spec), dim=dim, fn=FUNCTIONS[fn_name])
        env.reset()
        drain(env, policy=fixed(2))  # ES is always at index 2
        assert all(c == 2 for c in env._choices_history)
        assert np.isfinite(env._best_y)


# ------------------------------------------------------------------ #
# 2. BO + PSO + DE — three-family portfolios                         #
# ------------------------------------------------------------------ #

# MADDE/NL_SHADE_RSP respect env n_individuals (=10 here).
# JDE21 forces NP=170, so it gets a larger budget via fe_multiplier=250.
_BO_PSO_DE_LIGHT = [
    (["GPBO_EI", "SPSO", "MADDE"], 2, "sphere", 100),
    (["GPBO_UCB", "SPSO", "NL_SHADE_RSP"], 2, "abs", 100),
    (["GPBO_EI", "IPSO", "MADDE"], 3, "multimodal", 100),
    (["GPBO_UCB", "CPSO", "NL_SHADE_RSP"], 2, "step", 100),
]
_BO_PSO_DE_HEAVY = [
    (["GPBO_EI", "SPSO", "JDE21"], 2, "sphere", 250),
    (["GPBO_UCB", "IPSO", "JDE21"], 3, "abs", 250),
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

    @pytest.mark.parametrize("direction", ["forward", "reverse"])
    @pytest.mark.parametrize("spec,dim,fn_name,fe_mult", _BO_PSO_DE_LIGHT)
    def test_round_robin_all_families(self, spec, dim, fn_name, fe_mult, direction):
        classes = resolve(spec)
        env = make_env(classes, dim=dim, fn=FUNCTIONS[fn_name], fe_multiplier=fe_mult)
        env.reset()
        info = drain(env, policy=_make_rr(len(classes), direction))
        assert np.isfinite(info["best_y"])
        assert set(env._choices_history) == set(range(len(classes)))


# ------------------------------------------------------------------ #
# 3. BO + PSO + ES + DE — all four families                          #
# ------------------------------------------------------------------ #

_ALL_FOUR = [
    (["GPBO_EI", "SPSO", "CMAES", "MADDE"], 2, "sphere", 100),
    (["GPBO_UCB", "IPSO", "LMCMAES", "NL_SHADE_RSP"], 2, "abs", 100),
    (["GPBO_EI", "SPSO", "LMCMAES", "MADDE"], 3, "multimodal", 100),
    (["GPBO_UCB", "CPSO", "CMAES", "NL_SHADE_RSP"], 3, "asymmetric", 100),
    (["GPBO_EI", "SPSO", "CMAES", "JDE21"], 2, "sphere", 250),
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

    @pytest.mark.parametrize("direction", ["forward", "reverse"])
    @pytest.mark.parametrize("spec,dim,fn_name,fe_mult", _ALL_FOUR[:3])
    def test_round_robin_visits_all_four(self, spec, dim, fn_name, fe_mult, direction):
        # With N_CHECKPOINTS=3 and 4 optimizers, forward visits [0,1,2], reverse visits [3,2,1]
        classes = resolve(spec)
        env = make_env(classes, dim=dim, fn=FUNCTIONS[fn_name], fe_multiplier=fe_mult)
        env.reset()
        info = drain(env, policy=_make_rr(len(classes), direction))
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
        # GPBO_EI → SPSO → GPBO_EI: BO hands population to PSO, gets it back
        self._run_chain(["GPBO_EI", "SPSO", "LMCMAES"], actions=[0, 1, 0])

    def test_pso_bo_pso(self):
        # SPSO → GPBO_EI → SPSO: population seeded into GP, then back to PSO
        self._run_chain(["GPBO_EI", "SPSO", "LMCMAES"], actions=[1, 0, 1])

    def test_pso_bo_es(self):
        # SPSO → GPBO_UCB → LMCMAES
        self._run_chain(["GPBO_UCB", "SPSO", "LMCMAES"], actions=[1, 0, 2])

    # -- BO ↔ ES ---------------------------------------------------- #

    def test_bo_es_bo(self):
        # GPBO_EI → CMAES → GPBO_EI
        self._run_chain(["GPBO_EI", "SPSO", "CMAES"], actions=[0, 2, 0])

    def test_es_bo_pso(self):
        # CMAES → GPBO_UCB → SPSO
        self._run_chain(["GPBO_UCB", "SPSO", "CMAES"], actions=[2, 0, 1])

    def test_es_bo_es(self):
        # LMCMAES → GPBO_EI → CMAES: cross-ES transition via BO bridge
        self._run_chain(
            ["GPBO_EI", "SPSO", "LMCMAES", "CMAES"],
            actions=[2, 0, 3],
            fe_multiplier=100,
        )

    # -- BO ↔ DE ---------------------------------------------------- #

    def test_de_bo_pso(self):
        # MADDE → GPBO_EI → SPSO
        self._run_chain(["GPBO_EI", "SPSO", "MADDE"], actions=[2, 0, 1])

    def test_pso_bo_de(self):
        # SPSO → GPBO_UCB → NL_SHADE_RSP
        self._run_chain(["GPBO_UCB", "SPSO", "NL_SHADE_RSP"], actions=[1, 0, 2])

    def test_bo_de_bo(self):
        # GPBO_EI → MADDE → GPBO_EI: BO hands off to DE and reclaims state
        self._run_chain(["GPBO_EI", "SPSO", "MADDE"], actions=[0, 2, 0])

    # -- BO ↔ BO ---------------------------------------------------- #

    def test_ei_pso_ucb(self):
        # GPBO_EI → SPSO → GPBO_UCB: EI observations flow through PSO to UCB
        self._run_chain(["GPBO_EI", "GPBO_UCB", "SPSO"], actions=[0, 2, 1])

    def test_ucb_pso_ei(self):
        # GPBO_UCB → SPSO → GPBO_EI
        self._run_chain(["GPBO_EI", "GPBO_UCB", "SPSO"], actions=[1, 2, 0])

    def test_ei_ucb_pso(self):
        # GPBO_EI → GPBO_UCB → SPSO: BO→BO obs hand-off then PSO
        self._run_chain(["GPBO_EI", "GPBO_UCB", "SPSO"], actions=[0, 1, 2])

    # -- Higher-dimension chains ------------------------------------ #

    @pytest.mark.parametrize("dim", [3, 5])
    def test_bo_pso_es_dim(self, dim):
        self._run_chain(["GPBO_EI", "SPSO", "CMAES"], actions=[0, 1, 2], dim=dim)

    @pytest.mark.parametrize("fn_name", ["abs", "multimodal", "asymmetric", "step"])
    def test_bo_pso_es_landscape(self, fn_name):
        self._run_chain(
            ["GPBO_EI", "SPSO", "LMCMAES"], actions=[0, 1, 2], fn=FUNCTIONS[fn_name]
        )


# ------------------------------------------------------------------ #
# 5. Env contract invariants across all portfolios                   #
# ------------------------------------------------------------------ #

_CONTRACT_PORTFOLIOS = [
    (["GPBO_EI", "SPSO", "LMCMAES"], 100),
    (["GPBO_UCB", "IPSO", "CMAES"], 100),
    (["GPBO_EI", "GPBO_UCB", "SPSO"], 100),
    (["GPBO_EI", "SPSO", "MADDE"], 100),
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
        assert env._n_fe == 0
        assert env._best_y == float("inf")
        assert env._optimizer_state == {}
