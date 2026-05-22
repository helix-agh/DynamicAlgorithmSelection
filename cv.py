"""Cross-validation entry point: train one model per fold, evaluate on held-out split.

Usage
-----
    python cv.py ppo     <name> [options]
    python cv.py rl-das  <name> [options]
    python cv.py exp-das <name> [options]

Outputs (per fold)
------------------
    models/<name>_cv_<fold>.zip / _final.pt   trained model
    results/<name>_cv_<fold>.jsonl            per-problem test results
    results/<name>_cv_summary.jsonl           aggregated stats across all folds
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
    p.add_argument("--n-individuals", type=int, default=100, help="Population size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cv-mode",
        default="LOIO",
        choices=["LOIO", "LOPO"],
        help="LOIO: hold out instances per fold; LOPO: hold out functions per fold",
    )
    p.add_argument("--n-folds", type=int, default=3, help="Number of CV folds")
    p.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Zero-based fold indices to run (default: all)",
    )


def _parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description="Cross-validation for DAS agents.  Choose an agent with a sub-command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(
        dest="agent", required=True, metavar="{ppo,rl-das,exp-das}"
    )

    # ---- PPO --------------------------------------------------------
    ppo = sub.add_parser(
        "ppo",
        help="SB3 PPO with VecNormalize",
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
        help="Training passes per fold. total_timesteps = n_epochs × |train_ids| × n_checkpoints",
    )
    ppo.add_argument(
        "-j", "--n-envs", type=int, default=1, help="Parallel training envs"
    )
    ppo.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")

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
    rl.add_argument("--n-epochs", type=int, default=20, help="Training epochs per fold")
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
    rl.set_defaults(
        portfolio=["NL_SHADE_RSP", "MADDE", "JDE21"],
        n_individuals=170,
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
        help="Passes over the training set per fold. total_episodes = n_epochs × |train_ids|",
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
        from das.training.ppo import run_cv_ppo

        run_cv_ppo(args)
    elif args.agent == "rl-das":
        from das.training.rldas import run_cv_rl_das

        run_cv_rl_das(args)
    elif args.agent == "exp-das":
        from das.training.expdas import run_cv_exp_das

        run_cv_exp_das(args)


if __name__ == "__main__":
    main()
