"""Tests for das/env/reward.py — the reward functions used by DASEnv.

Three layers of coverage:

1. **Per-reward unit tests** — each of the seven reward options is checked
   against the contract stated in its docstring (signs, bounds, optimum-aware
   vs probe-relative behaviour, the terminal/dense split, …).

2. **Cross-cutting property tests** — invariants that must hold for *every*
   reward: finite float output, purity/determinism, ``compute_reward``
   dispatch, monotonicity in the amount of improvement.

3. **Sparsity / signal-density tests** — over a bank of simulated optimization
   trajectories we assert that the reward actually *carries signal*: its
   variance is non-zero and (for the dense options) it is non-zero in at least
   ~30% of transitions. The sparse reward (option 4) is intentionally sparse
   and is checked separately.

The functions are pure NumPy, so no IOH/pypop7/cocoex is needed and the whole
module runs in milliseconds.
"""

import numpy as np
import pytest

from das.env.reward import (
    REWARD_FNS,
    compute_reward,
    reward_binary,
    reward_hybrid_binary,
    reward_hybrid_sign,
    reward_linear,
    reward_log_improvement,
    reward_log_scaled,
    reward_sparse,
    _GAP_FLOOR,
    _improvement_ratio,
    _log_gap_orders,
    _terminal_reward,
)

# Reward options grouped by intended density.
DENSE_OPTIONS = [1, 2, 3, 5, 6, 7]  # emit signal on (most) intermediate steps
SPARSE_OPTIONS = [4]  # reward only at the final checkpoint — sparse by design
ALL_OPTIONS = sorted(REWARD_FNS)


# ------------------------------------------------------------------ #
# Trajectory simulator (shared by the property / sparsity tests)      #
# ------------------------------------------------------------------ #


def simulate_episode(rng, n_steps, optimum, p_stall=0.35):
    """One optimization run as a list of reward-function argument tuples.

    Each entry is ``(new_best_y, old_best_y, initial_range, is_final, optimum)``
    — exactly the signature every reward function takes. ``best_y`` is monotone
    non-increasing (best-so-far never worsens); with probability ``p_stall`` a
    step makes no progress (``new == old``), otherwise the gap to the reference
    shrinks geometrically (a realistic log-linear convergence model). The last
    step of each episode is flagged ``is_final``.
    """
    ref = optimum if optimum is not None else 0.0
    initial_gap = rng.uniform(1.0, 100.0)
    y0 = ref + initial_gap
    initial_range = (y0, y0 + rng.uniform(1.0, 50.0))
    best = y0
    transitions = []
    for t in range(n_steps):
        is_final = t == n_steps - 1
        old = best
        if rng.random() < p_stall:
            new = best  # stall: no improvement this step
        else:
            gap = max(best - ref, 1e-12)
            new = ref + gap * 10 ** (-rng.uniform(0.0, 1.5))  # shrink gap
            best = new
        transitions.append((new, old, initial_range, is_final, optimum))
    return transitions


def simulate_bank(seed=123, n_episodes=200, n_steps=10, optimum=0.0):
    """Flatten ``n_episodes`` simulated runs into one list of transitions."""
    rng = np.random.default_rng(seed)
    transitions = []
    for _ in range(n_episodes):
        transitions.extend(simulate_episode(rng, n_steps, optimum))
    return transitions


# Representative scenarios covering improvement/stall, final/non-final and the
# optimum / no-optimum branches — used by the cross-cutting property tests.
SCENARIOS = [
    (1.0, 10.0, (10.0, 50.0), False, 0.0),  # improvement, optimum-aware
    (10.0, 10.0, (10.0, 50.0), False, 0.0),  # stall, optimum-aware
    (1.0, 10.0, (10.0, 50.0), True, 0.0),  # final, optimum-aware
    (5.0, 50.0, (50.0, 100.0), False, None),  # improvement, probe-relative
    (50.0, 50.0, (50.0, 100.0), False, None),  # stall, probe-relative
    (5.0, 50.0, (50.0, 100.0), True, None),  # final, probe-relative
]


# ------------------------------------------------------------------ #
# 0. Private helpers                                                   #
# ------------------------------------------------------------------ #


class TestImprovementRatio:
    def test_positive_on_improvement(self):
        # old=10, new=5, scale=10 → (10-5)/10 = 0.5
        assert _improvement_ratio(5.0, 10.0, (0.0, 10.0)) == pytest.approx(0.5)

    def test_zero_on_stall(self):
        assert _improvement_ratio(7.0, 7.0, (0.0, 10.0)) == pytest.approx(0.0)

    def test_negative_on_worsening(self):
        assert _improvement_ratio(8.0, 5.0, (0.0, 10.0)) < 0.0

    def test_no_zero_division_on_degenerate_range(self):
        # scale == 0 must not raise (the +1e-10 guard); result is finite.
        assert np.isfinite(_improvement_ratio(5.0, 5.0, (5.0, 5.0)))


class TestLogGapOrders:
    def test_one_order_of_magnitude(self):
        # gap 10 → gap 1 is exactly one decade gained.
        assert _log_gap_orders(10.0, 1.0, 0.0) == pytest.approx(1.0)

    def test_sign_reverses_on_worsening(self):
        assert _log_gap_orders(1.0, 10.0, 0.0) == pytest.approx(-1.0)

    def test_floor_prevents_blowup_at_optimum(self):
        # Reaching the optimum exactly: gap is floored, result stays finite.
        r = _log_gap_orders(1.0, 0.0, 0.0)
        assert np.isfinite(r)
        # log10(1) - log10(1e-8) = 8
        assert r == pytest.approx(8.0)

    def test_telescopes_over_a_run(self):
        # Summing per-step gains over a monotone run == total decades gained.
        ys = [100.0, 30.0, 7.0, 0.5, 0.01]
        opt = 0.0
        step_sum = sum(_log_gap_orders(a, b, opt) for a, b in zip(ys, ys[1:]))
        total = _log_gap_orders(ys[0], ys[-1], opt)
        assert step_sum == pytest.approx(total)


class TestTerminalReward:
    def test_optimum_aware_does_not_saturate(self):
        # The headline design claim: with a known optimum, a far deeper gap is
        # rewarded much more — unlike the probe-relative version which saturates.
        rng = (1.0, 5.0)  # reference gap = 1
        coarse = _terminal_reward(1e-2, rng, optimum=0.0)
        fine = _terminal_reward(1e-8, rng, optimum=0.0)
        assert fine > coarse + 1.0  # 8.0 vs 2.0 — clearly separated

    def test_probe_relative_saturates(self):
        rng = (1.0, 5.0)
        coarse = _terminal_reward(1e-2, rng, optimum=None)
        fine = _terminal_reward(1e-8, rng, optimum=None)
        # Both ≈ reference/scale — the two gaps are nearly indistinguishable.
        assert abs(fine - coarse) < 0.01

    def test_clipped_to_ten(self):
        # A gigantic accuracy gain is clipped at +10.
        r = _terminal_reward(_GAP_FLOOR, (1e12, 1e12 + 1.0), optimum=0.0)
        assert r == pytest.approx(10.0)

    def test_clipped_at_minus_ten(self):
        # Massive regression (optimum-aware) clips at -10.
        r = _terminal_reward(1e12, (1.0, 2.0), optimum=0.0)
        assert r == pytest.approx(-10.0)


# ------------------------------------------------------------------ #
# 1. Per-reward unit tests                                             #
# ------------------------------------------------------------------ #


class TestRewardLogScaled:
    def test_no_improvement_is_log_floor(self):
        # log(clip(0)+1e-5) = log(1e-5)
        assert reward_log_scaled(5.0, 5.0, (0.0, 10.0)) == pytest.approx(np.log(1e-5))

    def test_full_improvement_near_zero(self):
        # ratio clipped to 1 → log(1+1e-5) ≈ 0
        r = reward_log_scaled(0.0, 10.0, (0.0, 10.0))
        assert r == pytest.approx(np.log(1 + 1e-5))

    def test_monotonic_in_improvement(self):
        small = reward_log_scaled(9.0, 10.0, (0.0, 10.0))
        large = reward_log_scaled(2.0, 10.0, (0.0, 10.0))
        assert large > small

    def test_ignores_is_final_and_optimum(self):
        base = reward_log_scaled(3.0, 10.0, (0.0, 10.0))
        assert reward_log_scaled(3.0, 10.0, (0.0, 10.0), True, 0.0) == base


class TestRewardLinear:
    def test_no_improvement_is_zero(self):
        assert reward_linear(5.0, 5.0, (0.0, 10.0)) == 0.0

    def test_half_scale_improvement(self):
        assert reward_linear(5.0, 10.0, (0.0, 10.0)) == pytest.approx(0.5)

    def test_clipped_above_one(self):
        # improvement larger than the scale clips at 1.
        assert reward_linear(-10.0, 10.0, (0.0, 10.0)) == pytest.approx(1.0)

    def test_clipped_below_zero_on_worsening(self):
        assert reward_linear(20.0, 10.0, (0.0, 10.0)) == 0.0


class TestRewardLogImprovement:
    def test_falls_back_to_linear_without_optimum(self):
        for new, old in [(5.0, 10.0), (10.0, 10.0), (12.0, 10.0)]:
            assert reward_log_improvement(
                new, old, (0.0, 10.0), optimum=None
            ) == reward_linear(new, old, (0.0, 10.0))

    def test_one_order_with_optimum(self):
        assert reward_log_improvement(1.0, 10.0, (10.0, 50.0), optimum=0.0) == (
            pytest.approx(1.0)
        )

    def test_non_negative_on_worsening_with_optimum(self):
        # A worsening step would give negative log-gap; it is floored at 0.
        assert reward_log_improvement(10.0, 1.0, (1.0, 50.0), optimum=0.0) == 0.0

    def test_dense_return_telescopes_to_terminal_orders(self):
        # Over a monotone run, the *sum* of dense per-step log-improvement
        # rewards equals the total decades of accuracy gained (= sparse final).
        ys = [110.0, 40.0, 9.0, 0.7, 0.02]
        opt, rng = 0.0, (110.0, 160.0)
        dense_sum = sum(
            reward_log_improvement(b, a, rng, optimum=opt) for a, b in zip(ys, ys[1:])
        )
        sparse_final = reward_sparse(ys[-1], ys[-2], rng, is_final=True, optimum=opt)
        assert dense_sum == pytest.approx(sparse_final)


class TestRewardSparse:
    @pytest.mark.parametrize("optimum", [0.0, None])
    def test_zero_when_not_final(self, optimum):
        # No matter how large the improvement, a non-final step pays nothing.
        assert reward_sparse(0.01, 100.0, (100.0, 150.0), False, optimum) == 0.0

    def test_final_with_optimum_counts_orders(self):
        # range[0]=10 (gap 10) → new=1 (gap 1): one decade.
        assert reward_sparse(1.0, 5.0, (10.0, 50.0), True, 0.0) == pytest.approx(1.0)

    def test_final_with_optimum_floored_at_zero(self):
        # No net progress vs the reference → clipped to 0 (not negative).
        assert reward_sparse(10.0, 10.0, (10.0, 50.0), True, 0.0) == 0.0

    def test_final_without_optimum_uses_log_ratio(self):
        # log((range[0]-new)/scale + 1e-5); range=(10,50) scale=40, new=0
        r = reward_sparse(0.0, 5.0, (10.0, 50.0), True, None)
        assert r == pytest.approx(np.log(10.0 / 40.0 + 1e-5))


class TestRewardBinary:
    def test_one_above_threshold(self):
        # improvement 2 over scale 1000 → ratio ≈ 2e-3 ≥ 1e-3
        assert reward_binary(998.0, 1000.0, (0.0, 1000.0)) == 1.0

    def test_zero_below_threshold(self):
        # improvement 0.5 over scale 1000 → ratio ≈ 5e-4 < 1e-3
        assert reward_binary(999.5, 1000.0, (0.0, 1000.0)) == 0.0

    def test_zero_on_stall(self):
        assert reward_binary(5.0, 5.0, (0.0, 10.0)) == 0.0

    def test_returns_only_zero_or_one(self):
        for tr in simulate_bank(n_episodes=20):
            assert reward_binary(*tr) in (0.0, 1.0)


class TestRewardHybridBinary:
    def test_progress_bonus_on_improvement(self):
        assert reward_hybrid_binary(9.0, 10.0, (0.0, 10.0), False) == pytest.approx(0.1)

    def test_zero_on_stall(self):
        assert reward_hybrid_binary(10.0, 10.0, (0.0, 10.0), False) == 0.0

    def test_final_uses_full_magnitude_terminal(self):
        r = reward_hybrid_binary(1e-8, 5.0, (10.0, 50.0), True, 0.0)
        assert r == pytest.approx(
            _terminal_reward(1e-8, (10.0, 50.0), 0.0)
        )

    def test_terminal_outweighs_step_bonus(self):
        # The whole point of the hybrid: the terminal payout dwarfs +0.1 steps.
        step = reward_hybrid_binary(9.0, 10.0, (10.0, 50.0), False, 0.0)
        final = reward_hybrid_binary(1e-8, 5.0, (10.0, 50.0), True, 0.0)
        assert final > step


class TestRewardHybridSign:
    def test_positive_bonus_on_good_step(self):
        # gap 10 → 1 (one decade > 0.05 threshold): base + slope*clip(1) = 1.1
        r = reward_hybrid_sign(1.0, 10.0, (10.0, 50.0), False, 0.0)
        assert r == pytest.approx(1.1)

    def test_penalty_on_stall(self):
        # Stall with no accumulated progress → full -penalty.
        r = reward_hybrid_sign(10.0, 10.0, (10.0, 50.0), False, 0.0)
        assert r == pytest.approx(-0.15)

    def test_non_final_bounded(self):
        # Non-final reward lives in [-penalty, base+slope].
        for tr in simulate_bank(n_episodes=50):
            new, old, rng, is_final, opt = tr
            if is_final:
                continue
            r = reward_hybrid_sign(new, old, rng, False, opt)
            assert -0.15 - 1e-9 <= r <= 1.1 + 1e-9

    def test_no_penalty_at_precision_target(self):
        # Stalling *at* the optimum is the goal state, not stagnation → 0.
        assert reward_hybrid_sign(0.0, 0.0, (10.0, 50.0), False, 0.0) == 0.0

    def test_penalty_shrinks_as_progress_grows(self):
        # The same stall is penalised less once a lot of progress is banked.
        rng, opt = (100.0, 150.0), 0.0
        early_stall = reward_hybrid_sign(99.0, 99.0, rng, False, opt)
        late_stall = reward_hybrid_sign(1.0, 1.0, rng, False, opt)
        assert late_stall > early_stall  # less negative
        assert late_stall < 0.0

    def test_final_uses_terminal_reward(self):
        r = reward_hybrid_sign(1e-8, 5.0, (10.0, 50.0), True, 0.0)
        assert r == pytest.approx(_terminal_reward(1e-8, (10.0, 50.0), 0.0))

    def test_probe_relative_branch_rewards_big_step(self):
        # Without optimum: a step improving by > step_threshold (5e-3) of the
        # scale gets the positive bonus.
        r = reward_hybrid_sign(40.0, 50.0, (50.0, 100.0), False, None)
        assert r > 0.1


# ------------------------------------------------------------------ #
# 2. Cross-cutting property tests (every reward option)               #
# ------------------------------------------------------------------ #


class TestRewardProperties:
    @pytest.mark.parametrize("option", ALL_OPTIONS)
    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_returns_finite_float(self, option, scenario):
        r = REWARD_FNS[option](*scenario)
        assert isinstance(r, float)
        assert np.isfinite(r)

    @pytest.mark.parametrize("option", ALL_OPTIONS)
    def test_pure_and_deterministic(self, option):
        fn = REWARD_FNS[option]
        for scenario in SCENARIOS:
            assert fn(*scenario) == fn(*scenario)

    @pytest.mark.parametrize("option", ALL_OPTIONS)
    def test_compute_reward_matches_direct_call(self, option):
        for new, old, rng, is_final, opt in SCENARIOS:
            direct = REWARD_FNS[option](new, old, rng, is_final, opt)
            dispatched = compute_reward(
                new, old, rng, option=option, is_final=is_final, optimum=opt
            )
            assert dispatched == direct

    def test_compute_reward_unknown_option_raises(self):
        with pytest.raises(ValueError):
            compute_reward(1.0, 2.0, (0.0, 10.0), option=999)

    def test_compute_reward_default_option_is_log_scaled(self):
        assert compute_reward(3.0, 10.0, (0.0, 10.0)) == reward_log_scaled(
            3.0, 10.0, (0.0, 10.0)
        )

    @pytest.mark.parametrize("option", ALL_OPTIONS)
    def test_improvement_not_worse_than_stall(self, option):
        """A genuine improvement never scores below an identical stall.

        This is the core sign property a reward must respect to be learnable:
        making progress is at least as good as making none. Checked on a
        non-final step for both the optimum-aware and probe-relative branches.
        """
        fn = REWARD_FNS[option]
        for rng, opt in [((10.0, 50.0), 0.0), ((50.0, 100.0), None)]:
            old = rng[0] + 5.0
            stall = fn(old, old, rng, False, opt)
            improve = fn(old - 3.0, old, rng, False, opt)
            assert improve >= stall - 1e-12


# ------------------------------------------------------------------ #
# 3. Sparsity / signal-density tests                                  #
# ------------------------------------------------------------------ #
#
# The reviewer asked specifically for a "reward sparsity" test: that the reward
# variance is not zero and that the reward is non-zero in at least ~30% of
# transitions. We check this over a bank of simulated optimization runs.

NONZERO_FRACTION_FLOOR = 0.30


class TestRewardSparsity:
    @pytest.mark.parametrize("optimum", [0.0, None])
    @pytest.mark.parametrize("option", ALL_OPTIONS)
    def test_variance_is_nonzero(self, option, optimum):
        """No reward may collapse to a constant — that carries no signal.

        (Even the sparse reward varies: its terminal payout differs per run.)
        """
        vals = np.array([REWARD_FNS[option](*tr) for tr in simulate_bank(optimum=optimum)])
        assert np.var(vals) > 0.0

    @pytest.mark.parametrize("optimum", [0.0, None])
    @pytest.mark.parametrize("option", DENSE_OPTIONS)
    def test_dense_rewards_nonzero_often(self, option, optimum):
        """Dense rewards must fire on a meaningful fraction (~≥30%) of steps."""
        vals = np.array([REWARD_FNS[option](*tr) for tr in simulate_bank(optimum=optimum)])
        frac_nonzero = float(np.mean(vals != 0.0))
        assert frac_nonzero >= NONZERO_FRACTION_FLOOR, (
            f"option {option}: only {frac_nonzero:.1%} non-zero rewards"
        )

    @pytest.mark.parametrize("optimum", [0.0, None])
    def test_sparse_reward_fires_only_at_final(self, optimum):
        """Counterpart to the dense test: option 4 is sparse *by design* — it
        must be non-zero only on the final step (≈ 1/n_steps of transitions)."""
        n_steps = 10
        transitions = simulate_bank(n_steps=n_steps, optimum=optimum)
        nonzero_idx = [
            i for i, tr in enumerate(transitions) if reward_sparse(*tr) != 0.0
        ]
        # Every non-zero reward must come from a final step …
        assert all(transitions[i][3] is True for i in nonzero_idx)
        # … and the non-zero fraction is about 1/n_steps, i.e. clearly sparse.
        frac_nonzero = len(nonzero_idx) / len(transitions)
        assert frac_nonzero == pytest.approx(1.0 / n_steps, abs=1e-9)
        assert frac_nonzero < NONZERO_FRACTION_FLOOR

    def test_episode_return_tracks_solution_quality(self):
        """A run that converges deeper must earn a higher episode return.

        This is the property that makes the reward usable for RL: summed over an
        episode, a better final solution yields a larger return. Checked on the
        dense hybrid-sign reward (the one in production use).
        """
        rng_range, opt = (100.0, 150.0), 0.0

        def episode_return(final_gap):
            # Identical 4-step descent, different depth on the last step.
            ys = [100.0, 10.0, 1.0, final_gap]
            total = 0.0
            for i in range(1, len(ys)):
                is_final = i == len(ys) - 1
                total += reward_hybrid_sign(
                    ys[i], ys[i - 1], rng_range, is_final, opt
                )
            return total

        assert episode_return(1e-6) > episode_return(1e-1)
