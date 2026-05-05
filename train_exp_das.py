"""Training script for the Exponential-DAS agent.

Usage
-----
    python train_exp_das.py <name> [options]

Outputs
-------
    models/<name>_best.pt        (checkpoint at best test performance)
    models/<name>_final.pt
    models/<name>_ep<N>.pt       (periodic checkpoints)
    models/<name>_train_log.jsonl
    results/<name>_eval.jsonl    (final evaluation)

Examples
--------
    # Default: 3 DE optimisers, cdb=2.0 (exponential checkpoints)
    python train_exp_das.py my_run

    # Larger portfolio, more checkpoints
    python train_exp_das.py my_run --portfolio NL_SHADE_RSP MADDE JDE21 CMAES \\
        --n-checkpoints 20 --cdb 3.0 --total-episodes 10000
"""

import argparse
import json
import os
import warnings
from itertools import product
from pathlib import Path

import cocoex as cx
import numpy as np

from agents.exponential_das import ExpDASAgent, train, evaluate
from das.env.das_env import DASEnv
from das.env.observation import observation_dim
from das.optimizers.portfolio import get_portfolio

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# BBOB problem sets (same constants as train.py)                               #
# --------------------------------------------------------------------------- #

ALL_DIMS = [2, 3, 5, 10, 20, 40]
ALL_FUNCTIONS = set(range(1, 25))
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
EASY_TRAIN_FUNCTIONS = {4, *range(6, 15), 18, 19, 20, 22, 23, 24}


def _build_problem_ids(
    functions: set[int],
    dims: list[int],
    instances: list[int] | None = None,
) -> list[str]:
    insts = instances if instances is not None else INSTANCE_IDS
    return [
        f"bbob_f{f:03d}_i{i:02d}_d{d:02d}"
        for i, f, d in product(insts, sorted(functions), dims)
    ]


def _get_train_test_split(mode: str, dims: list[int]) -> tuple[list[str], list[str]]:
    if mode == "easy":
        return (
            _build_problem_ids(EASY_TRAIN_FUNCTIONS, dims),
            _build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dims),
        )
    if mode == "hard":
        return (
            _build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dims),
            _build_problem_ids(EASY_TRAIN_FUNCTIONS, dims),
        )
    all_ids = _build_problem_ids(ALL_FUNCTIONS, dims)
    rng = np.random.default_rng()
    rng.shuffle(all_ids)
    split = 2 * len(all_ids) // 3
    return all_ids[:split], all_ids[split:]


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="Train an Exponential-DAS agent")
    p.add_argument("name", help="Experiment name (used for output file names)")
    p.add_argument(
        "-p",
        "--portfolio",
        nargs="+",
        default=["NL_SHADE_RSP", "MADDE", "JDE21"],
        help="Sub-optimizer names (default: NL_SHADE_RSP MADDE JDE21)",
    )
    p.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[2, 5, 10],
        help="Problem dimensions (default: 2 5 10)",
    )
    p.add_argument(
        "--mode",
        choices=["easy", "hard", "random"],
        default="easy",
        help="Train/test split strategy (default: easy)",
    )
    p.add_argument(
        "--fe-multiplier",
        type=int,
        default=10_000,
        help="Budget = fe_multiplier * dim (default: 10000)",
    )
    p.add_argument(
        "--n-checkpoints",
        type=int,
        default=10,
        help="RL steps per episode (default: 10)",
    )
    p.add_argument(
        "--cdb",
        type=float,
        default=2.0,
        help="Checkpoint division base >1 gives exponential spacing "
        "(default: 2.0; set to 1.0 for uniform)",
    )
    p.add_argument(
        "--reward-option",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Reward shaping option (default: 1)",
    )
    p.add_argument(
        "--n-individuals", type=int, default=100, help="Population size (default: 100)"
    )
    p.add_argument(
        "--buffer-capacity",
        type=int,
        default=None,
        help="PPO rollout buffer size in steps (default: 16 * n_checkpoints)",
    )
    p.add_argument(
        "--total-episodes",
        type=int,
        default=5000,
        help="Total training episodes (default: 5000)",
    )
    p.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate on test set every N episodes (default: 100)",
    )
    p.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Checkpoint every N episodes (default: 500)",
    )
    p.add_argument(
        "--actor-lr",
        type=float,
        default=3e-5,
        help="Actor learning rate (default: 3e-5)",
    )
    p.add_argument(
        "--critic-lr",
        type=float,
        default=1e-5,
        help="Critic learning rate (default: 1e-5)",
    )
    p.add_argument(
        "--ppo-epochs",
        type=int,
        default=6,
        help="PPO gradient epochs per update (default: 6)",
    )
    p.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    optimizers = get_portfolio(args.portfolio)
    n_opt = len(optimizers)
    obs_dim = observation_dim(n_opt)

    train_ids, test_ids = _get_train_test_split(args.mode, args.dims)
    print(f"Train: {len(train_ids)} problems  |  Test: {len(test_ids)} problems")
    print(f"obs_dim={obs_dim}  cdb={args.cdb}  n_checkpoints={args.n_checkpoints}")

    suite = cx.Suite("bbob", "", "")

    env_cfg = dict(
        suite=suite,
        optimizers=optimizers,
        fe_multiplier=args.fe_multiplier,
        n_checkpoints=args.n_checkpoints,
        checkpoint_division_base=args.cdb,
        reward_option=args.reward_option,
        n_individuals=args.n_individuals,
        seed=args.seed,
    )
    train_env = DASEnv(problem_ids=train_ids, **env_cfg)
    test_env = DASEnv(problem_ids=test_ids, **env_cfg)

    buffer_capacity = args.buffer_capacity or (16 * args.n_checkpoints)

    agent = ExpDASAgent(
        obs_dim=obs_dim,
        n_actions=n_opt,
        buffer_capacity=buffer_capacity,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        ppo_epochs=args.ppo_epochs,
        n_checkpoints=args.n_checkpoints,
        device=args.device,
    )

    print(
        f"Exponential-DAS  |  portfolio={args.portfolio}"
        f"  |  buffer_capacity={buffer_capacity}"
        f"  |  ppo_epochs={args.ppo_epochs}"
    )

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    log = train(
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        total_episodes=args.total_episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir="models",
        name=args.name,
    )

    print("\nFinal evaluation on test set …")
    test_results = evaluate(test_env, agent, n_episodes=min(len(test_ids), 50))
    mean_best_y = float(np.mean([r["best_y"] for r in test_results]))
    print(f"Test mean best_y = {mean_best_y:.6e}")

    eval_path = f"results/{args.name}_eval.jsonl"
    with open(eval_path, "w") as f:
        for r in test_results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {eval_path}")


if __name__ == "__main__":
    main()
