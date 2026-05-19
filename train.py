"""Unified training entry point for all DAS agents.

Usage
-----
    python train.py ppo     <name> [options]
    python train.py rl-das  <name> [options]
    python train.py exp-das <name> [options]

ppo outputs
-----------
    models/<name>.zip
    models/<name>_vecnorm.pkl
    results/<name>_eval.jsonl          (with --eval)

rl-das outputs
--------------
    models/<name>_final.pt
    models/<name>_epoch<N>.pt
    models/<name>_train_log.jsonl
    results/<name>_eval.jsonl

exp-das outputs
---------------
    models/<name>_best.pt  /  _final.pt  /  _ep<N>.pt
    models/<name>_train_log.jsonl
    results/<name>_eval.jsonl
"""

import argparse
import warnings
from pathlib import Path

from das.env.bbob_splits import ALL_DIMS
from das.utils import set_seed

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
# Argument parsing                                                     #
# ------------------------------------------------------------------ #


def _add_shared_args(
    p: argparse.ArgumentParser, *, include_portfolio: bool = True
) -> None:
    p.add_argument("name", help="Experiment name (used for output file names)")
    if include_portfolio:
        p.add_argument(
            "-p",
            "--portfolio",
            nargs="+",
            default=["SPSO", "IPSO", "SPSOL"],
            help="Sub-optimizer names from the portfolio",
        )
    p.add_argument(
        "--mode",
        choices=["easy", "hard", "random"],
        default="easy",
        help="Train/test split strategy",
    )
    p.add_argument(
        "--fe-multiplier",
        type=int,
        default=10_000,
        help="Budget = fe_multiplier × dimension",
    )
    p.add_argument(
        "--n-checkpoints",
        type=int,
        default=10,
        help="Optimizer-selection steps per episode",
    )
    p.add_argument(
        "--n-individuals",
        type=int,
        default=None,
        help="Population size override (default: each algorithm uses its own built-in default)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ela-recompute-every",
        type=int,
        default=500,
        help=(
            "Recompute ELA features every N new population samples. "
            "Set to 1 to recompute on every step (slow but maximally fresh). "
            "Default: 500."
        ),
    )


def _parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description="Train a DAS agent.  Choose an agent with a sub-command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(
        dest="agent", required=True, metavar="{ppo,rl-das,exp-das}"
    )

    # ---- PPO --------------------------------------------------------
    ppo = sub.add_parser(
        "ppo",
        help="SB3 PPO with VecNormalize (multi-dim)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_args(ppo)
    ppo.add_argument(
        "-d",
        "--dims",
        nargs="+",
        type=int,
        default=ALL_DIMS,
        choices=ALL_DIMS,
        help="Problem dimensions",
    )
    ppo.add_argument(
        "-x", "--cdb", type=float, default=1.0, help="Checkpoint division base"
    )
    ppo.add_argument(
        "-O",
        "--reward-option",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Reward shaping option",
    )
    ppo.add_argument(
        "-E",
        "--n-epochs",
        type=int,
        default=20,
        help="Passes over the full training set. total_timesteps = n_epochs × |train_ids| × n_checkpoints",
    )
    ppo.add_argument(
        "-j", "--n-envs", type=int, default=1, help="Parallel training envs"
    )
    ppo.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    ppo.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate on the test set immediately after training",
    )

    # ---- RL-DAS -----------------------------------------------------
    rl = sub.add_parser(
        "rl-das",
        help="Custom RL-DAS: single-dimension, pure-PyTorch PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_args(rl, include_portfolio=False)
    rl.add_argument(
        "--dim", type=int, default=10, help="Problem dimension (agent is dim-specific)"
    )
    rl.add_argument("--n-epochs", type=int, default=20, help="Training epochs")
    rl.add_argument(
        "--k-epoch",
        type=int,
        default=None,
        help="PPO gradient steps per episode (default: int(0.3 × n_checkpoints))",
    )
    rl.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    rl.add_argument(
        "--eval-interval", type=int, default=5, help="Evaluate every N epochs"
    )
    rl.add_argument(
        "--save-interval", type=int, default=50, help="Checkpoint every N epochs"
    )
    rl.add_argument("--device", default="cpu", help="PyTorch device")
    rl.add_argument(
        "--no-eval", dest="eval", action="store_false", help="Skip final evaluation"
    )
    rl.set_defaults(
        eval=True,
        portfolio=["NL_SHADE_RSP", "MADDE", "JDE21"],
        n_individuals=None,
    )

    # ---- Exp-DAS ----------------------------------------------------
    exp = sub.add_parser(
        "exp-das",
        help="Exponential-DAS: custom PPO with exponential checkpoint spacing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_args(exp)
    exp.add_argument(
        "--dims", nargs="+", type=int, default=[2, 5, 10], help="Problem dimensions"
    )
    exp.add_argument(
        "--cdb",
        type=float,
        default=2.0,
        help="Checkpoint division base (>1 = exponential)",
    )
    exp.add_argument(
        "--reward-option",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Reward shaping option",
    )
    exp.add_argument(
        "--buffer-capacity",
        type=int,
        default=None,
        help="PPO rollout buffer size in steps (default: 16 × n_checkpoints)",
    )
    exp.add_argument(
        "-E",
        "--n-epochs",
        type=int,
        default=3,
        help="Passes over the training set. total_episodes = n_epochs × |train_ids|",
    )
    exp.add_argument(
        "--eval-interval", type=int, default=100, help="Evaluate every N episodes"
    )
    exp.add_argument(
        "--save-interval", type=int, default=500, help="Checkpoint every N episodes"
    )
    exp.add_argument("--actor-lr", type=float, default=3e-5, help="Actor learning rate")
    exp.add_argument(
        "--critic-lr", type=float, default=1e-5, help="Critic learning rate"
    )
    exp.add_argument(
        "--ppo-epochs", type=int, default=6, help="PPO gradient epochs per update"
    )
    exp.add_argument("--device", default="cpu", help="PyTorch device")
    exp.add_argument(
        "--no-eval", dest="eval", action="store_false", help="Skip final evaluation"
    )
    exp.set_defaults(eval=True)

    return root.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    if args.agent == "ppo":
        from das.training.ppo import run_ppo

        run_ppo(args)
    elif args.agent == "rl-das":
        from das.training.rldas import run_rl_das

        run_rl_das(args)
    elif args.agent == "exp-das":
        from das.training.expdas import run_exp_das

        run_exp_das(args)


if __name__ == "__main__":
    main()
