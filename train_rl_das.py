"""Training script for the RL-DAS agent.

Usage
-----
    python train_rl_das.py <name> [options]

Outputs
-------
    models/<name>_final.pt
    models/<name>_epoch<N>.pt    (checkpoints)
    models/<name>_train_log.jsonl
    results/<name>_eval.jsonl    (evaluation on test set)

Examples
--------
    # Default: 3 DE optimizers, dim=10, 500 epochs
    python train_rl_das.py my_run

    # Custom portfolio and dimension
    python train_rl_das.py my_run --portfolio NL_SHADE_RSP MADDE JDE21 --dim 5

    # Fewer epochs for a quick sanity check
    python train_rl_das.py test_run --n-epochs 10 --eval-interval 5
"""

import argparse
import json
import os
import warnings
from itertools import product
from pathlib import Path

import cocoex as cx
import numpy as np

from agents.rl_das import RLDASEnv, PPOAgent, train, evaluate
from das.optimizers.portfolio import get_portfolio

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# BBOB problem sets (same constants as train.py)                               #
# --------------------------------------------------------------------------- #

ALL_FUNCTIONS = set(range(1, 25))
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
EASY_TRAIN_FUNCTIONS = {4, *range(6, 15), 18, 19, 20, 22, 23, 24}


def _build_problem_ids(
    functions: set[int],
    dim: int,
    instances: list[int] | None = None,
) -> list[str]:
    insts = instances if instances is not None else INSTANCE_IDS
    return [
        f"bbob_f{f:03d}_i{i:02d}_d{dim:02d}"
        for i, f in product(insts, sorted(functions))
    ]


def _get_train_test_split(mode: str, dim: int) -> tuple[list[str], list[str]]:
    if mode == "easy":
        return (
            _build_problem_ids(EASY_TRAIN_FUNCTIONS, dim),
            _build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dim),
        )
    if mode == "hard":
        return (
            _build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dim),
            _build_problem_ids(EASY_TRAIN_FUNCTIONS, dim),
        )
    # Random split
    all_ids = _build_problem_ids(ALL_FUNCTIONS, dim)
    rng = np.random.default_rng()
    rng.shuffle(all_ids)
    split = 2 * len(all_ids) // 3
    return all_ids[:split], all_ids[split:]


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="Train an RL-DAS agent")
    p.add_argument("name", help="Experiment name (used for output file names)")
    p.add_argument(
        "-p",
        "--portfolio",
        nargs="+",
        default=["NL_SHADE_RSP", "MADDE", "JDE21"],
        help="Sub-optimizer names from the portfolio (default: NL_SHADE_RSP MADDE JDE21)",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=10,
        help="Problem dimension (agent is dim-specific, default: 10)",
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
        default=20,
        help="RL steps per episode (default: 20)",
    )
    p.add_argument(
        "--n-individuals", type=int, default=170, help="Population size (default: 170)"
    )
    p.add_argument(
        "--n-epochs", type=int, default=500, help="Training epochs (default: 500)"
    )
    p.add_argument(
        "--k-epoch",
        type=int,
        default=None,
        help="PPO gradient steps per episode (default: int(0.3 * max_fes / period))",
    )
    p.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)"
    )
    p.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Evaluate on test set every N epochs (default: 5)",
    )
    p.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Checkpoint every N epochs (default: 50)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, e.g. 'cpu' or 'cuda' (default: cpu)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Optimizer portfolio
    optimizers = get_portfolio(args.portfolio)
    if not optimizers:
        raise ValueError(f"Unknown optimizers: {args.portfolio}")

    # Problem splits (dim-specific)
    train_ids, test_ids = _get_train_test_split(args.mode, args.dim)
    print(f"Train: {len(train_ids)} problems  |  Test: {len(test_ids)} problems")

    # Shared cocoex suite
    suite = cx.Suite("bbob", "", "")

    # k_epoch default: 30% of the decisions made per episode
    # (mirrors the paper's int(0.3 * MaxFEs / period))
    if args.k_epoch is None:
        args.k_epoch = max(1, int(0.3 * args.n_checkpoints))

    # Environments
    train_env = RLDASEnv(
        problem_ids=train_ids,
        suite=suite,
        optimizers=optimizers,
        dim=args.dim,
        fe_multiplier=args.fe_multiplier,
        n_checkpoints=args.n_checkpoints,
        n_individuals=args.n_individuals,
        seed=args.seed,
    )
    test_env = RLDASEnv(
        problem_ids=test_ids,
        suite=suite,
        optimizers=optimizers,
        dim=args.dim,
        fe_multiplier=args.fe_multiplier,
        n_checkpoints=args.n_checkpoints,
        n_individuals=args.n_individuals,
        seed=args.seed,
    )

    print(
        f"RL-DAS  |  dim={args.dim}  |  optimizers={args.portfolio}"
        f"  |  obs_dim={train_env.observation_space.shape[0]}"
        f"  |  k_epoch={args.k_epoch}"
    )

    # Agent
    agent = PPOAgent(
        dim=args.dim,
        n_opt=len(optimizers),
        lr=args.lr,
        device=args.device,
    )

    # Train
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    log = train(
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        n_epochs=args.n_epochs,
        k_epoch=args.k_epoch,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir="models",
        name=args.name,
    )

    # Final evaluation on full test set
    print("\nRunning final evaluation on test set …")
    test_results = evaluate(test_env, agent, n_episodes=len(test_env._problem_ids))
    mean_best_y = float(np.mean([r["best_y"] for r in test_results]))
    print(f"Test mean best_y = {mean_best_y:.6e}")

    eval_path = f"results/{args.name}_eval.jsonl"
    with open(eval_path, "w") as f:
        for r in test_results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {eval_path}")


if __name__ == "__main__":
    main()
