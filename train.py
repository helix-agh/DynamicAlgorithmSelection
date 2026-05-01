"""Training script for DAS using Stable Baselines 3 PPO.

Usage
-----
    # Regular training (single train/test split):
    python train.py <name> [options]

    # Cross-validation (LOIO = 15 folds, LOPO = 24 folds):
    python train.py <name> --cv-mode LOIO [options]
    python train.py <name> --cv-mode LOPO [options]
    python train.py <name> --cv-mode LOPO --folds 0 1 2  # specific folds only

Regular mode outputs
--------------------
    models/<name>.zip
    models/<name>_vecnorm.pkl

CV mode outputs (per fold + summary)
-------------------------------------
    models/<name>_cv_<fold_tag>.zip
    models/<name>_cv_<fold_tag>_vecnorm.pkl
    results/<name>_cv_<fold_tag>.jsonl
    results/<name>_cv_summary.jsonl
"""

import argparse
import json
import os
import warnings
from itertools import product

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from tqdm import tqdm

from das.env.das_env import DASEnv
from das.optimizers.portfolio import get_portfolio
from das.training.callbacks import WandbCallback, DASEvalCallback
from das.utils import set_seed

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# BBOB problem sets                                                    #
# ------------------------------------------------------------------ #

ALL_DIMS = [2, 3, 5, 10, 20, 40]
ALL_FUNCTIONS = set(range(1, 25))
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
EASY_TRAIN_FUNCTIONS = {4, *range(6, 15), 18, 19, 20, 22, 23, 24}


def build_problem_ids(
    functions: set[int],
    dims: list[int],
    instances: list[int] | None = None,
) -> list[str]:
    insts = instances if instances is not None else INSTANCE_IDS
    return [
        f"bbob_f{f:03d}_i{i:02d}_d{d:02d}"
        for i, f, d in product(insts, sorted(functions), dims)
    ]


def get_train_test_split(mode: str, dims: list[int]) -> tuple[list[str], list[str]]:
    if mode == "easy":
        train_fns = EASY_TRAIN_FUNCTIONS
        test_fns = ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS
    elif mode == "hard":
        train_fns = ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS
        test_fns = EASY_TRAIN_FUNCTIONS
    else:  # LOIO – random 2/3 split on problem IDs
        all_ids = build_problem_ids(ALL_FUNCTIONS, dims)
        rng = np.random.default_rng()
        rng.shuffle(all_ids)
        split = 2 * len(all_ids) // 3
        return all_ids[:split], all_ids[split:]

    return build_problem_ids(train_fns, dims), build_problem_ids(test_fns, dims)


# ------------------------------------------------------------------ #
# Cross-validation splits                                              #
# ------------------------------------------------------------------ #


def get_cv_folds(
    cv_mode: str, dims: list[int]
) -> list[tuple[list[str], list[str], str]]:
    """Return (train_ids, test_ids, fold_tag) for each CV fold.

    LOIO: 15 folds – hold out one BBOB instance ID at a time.
    LOPO: 24 folds – hold out one BBOB function at a time.
    """
    folds = []
    if cv_mode == "LOIO":
        for inst in INSTANCE_IDS:
            train_insts = [i for i in INSTANCE_IDS if i != inst]
            folds.append(
                (
                    build_problem_ids(ALL_FUNCTIONS, train_insts, dims),
                    build_problem_ids(ALL_FUNCTIONS, [inst], dims),
                    f"inst{inst:02d}",
                )
            )
    else:  # LOPO
        for fn in sorted(ALL_FUNCTIONS):
            train_fns = ALL_FUNCTIONS - {fn}
            folds.append(
                (
                    build_problem_ids(train_fns, dims),
                    build_problem_ids({fn}, dims),
                    f"f{fn:03d}",
                )
            )
    return folds


# ------------------------------------------------------------------ #
# Global optima loader                                                 #
# ------------------------------------------------------------------ #


def load_global_optima(path: str = "bbob_optima.jsonl") -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v for line in f for k, v in json.loads(line).items()}


# ------------------------------------------------------------------ #
# Environment factory                                                  #
# ------------------------------------------------------------------ #


def make_das_env(problem_ids: list[str], optimizers: list, cfg: dict):
    def _init():
        import cocoex as _cx

        _cx.utilities.MiniPrint()
        _suite = _cx.Suite("bbob", "", "")
        return DASEnv(
            problem_ids=problem_ids,
            suite=_suite,
            optimizers=optimizers,
            fe_multiplier=cfg["fe_multiplier"],
            n_checkpoints=cfg["n_checkpoints"],
            checkpoint_division_base=cfg["cdb"],
            reward_option=cfg["reward_option"],
            n_individuals=cfg["n_individuals"],
            seed=cfg.get("seed"),
        )

    return _init


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def parse_args():
    p = argparse.ArgumentParser(description="Train a DAS agent with PPO")
    p.add_argument("name", help="Experiment name (used for file names)")
    p.add_argument(
        "-p",
        "--portfolio",
        nargs="+",
        default=["SPSO", "IPSO", "SPSOL"],
        help="Sub-optimizer names from the portfolio",
    )
    p.add_argument(
        "-m",
        "--mode",
        default="easy",
        choices=["easy", "hard", "LOIO"],
        help="Train/test split strategy (ignored when --cv-mode is set)",
    )
    p.add_argument(
        "-d",
        "--dims",
        nargs="+",
        type=int,
        default=ALL_DIMS,
        choices=ALL_DIMS,
        help="Problem dimensions",
    )
    p.add_argument(
        "-f",
        "--fe-multiplier",
        type=int,
        default=10_000,
        help="Budget = fe_multiplier * dimension",
    )
    p.add_argument(
        "-s",
        "--n-checkpoints",
        type=int,
        default=10,
        help="Optimizer selection steps per episode",
    )
    p.add_argument(
        "-x",
        "--cdb",
        type=float,
        default=1.0,
        help="Checkpoint division base (1.0 = uniform)",
    )
    p.add_argument("-O", "--reward-option", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument(
        "-n",
        "--n-individuals",
        type=int,
        default=100,
        help="Shared population size across all sub-optimizers",
    )
    p.add_argument(
        "-E",
        "--n-epochs",
        type=int,
        default=1,
        help="Number of passes over the full training set (per fold in CV mode). "
        "total_timesteps = n_epochs × |train_ids| × n_checkpoints. "
        "Evaluation runs once per epoch.",
    )
    p.add_argument(
        "-j", "--n-envs", type=int, default=1, help="Number of parallel environments"
    )
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument("--seed", type=int, default=42)
    # Cross-validation options
    p.add_argument(
        "--cv-mode",
        default=None,
        choices=["LOIO", "LOPO"],
        help="Run cross-validation: LOIO (15 folds) or LOPO (24 folds). "
        "Omit for regular single-run training.",
    )
    p.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Zero-based fold indices to run (CV mode only, default: all). "
        "Useful for parallelism or resuming interrupted runs.",
    )
    return p.parse_args()


# ------------------------------------------------------------------ #
# Single-run training                                                  #
# ------------------------------------------------------------------ #


def train_model(
    name: str,
    train_ids: list[str],
    test_ids: list[str],
    optimizers: list,
    cfg: dict,
    args,
) -> None:
    """Train one PPO model and save it to models/<name>."""
    model_path = os.path.join("models", name)
    vecnorm_path = model_path + "_vecnorm.pkl"

    # One epoch = one full pass over the training set.
    # SB3 counts individual env steps; each episode has n_checkpoints steps,
    # and epoch size is independent of n_envs (more envs → fewer rollouts, same total).
    steps_per_epoch = len(train_ids) * cfg["n_checkpoints"]
    total_timesteps = args.n_epochs * steps_per_epoch
    eval_freq = steps_per_epoch  # evaluate once at the end of every epoch

    print(
        f"  Epochs    : {args.n_epochs}  "
        f"({steps_per_epoch:,} steps/epoch → {total_timesteps:,} total steps)"
    )

    train_env = make_vec_env(
        make_das_env(train_ids, optimizers, cfg),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
        seed=args.seed,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    eval_sample = test_ids[: min(20, len(test_ids))]
    eval_env = make_vec_env(
        make_das_env(eval_sample, optimizers, cfg),
        n_envs=1,
        seed=args.seed + 1,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    policy_kwargs = dict(net_arch=[96, 96])
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-5,
        n_steps=cfg["n_checkpoints"],
        batch_size=256,
        n_epochs=6,  # PPO gradient update epochs per rollout (different from dataset epochs)
        gamma=0.8,
        gae_lambda=0.5,
        clip_range=0.3,
        ent_coef=0.01,
        vf_coef=0.3,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
    )

    callbacks = [
        DASEvalCallback(
            eval_env, eval_freq=eval_freq, n_eval_episodes=len(eval_sample), name=name
        ),
    ]
    if args.wandb:
        callbacks.append(WandbCallback())

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    model.save(model_path)
    train_env.save(vecnorm_path)
    train_env.close()
    eval_env.close()
    print(f"Saved model → {model_path}.zip")


# ------------------------------------------------------------------ #
# Cross-validation                                                     #
# ------------------------------------------------------------------ #


def run_cv(args, optimizers, cfg) -> None:
    """Run k-fold cross-validation and write per-fold + summary results."""
    global_optima = load_global_optima()
    all_folds = get_cv_folds(args.cv_mode, args.dims)
    fold_indices = args.folds if args.folds is not None else list(range(len(all_folds)))

    print(
        f"CV mode   : {args.cv_mode}  "
        f"({len(fold_indices)}/{len(all_folds)} folds selected)"
    )
    print(f"Epochs/fold: {args.n_epochs}")

    fold_summaries = []

    for run_idx, fold_idx in enumerate(fold_indices):
        train_ids, test_ids, fold_tag = all_folds[fold_idx]
        fold_name = f"{args.name}_cv_{fold_tag}"
        model_path = os.path.join("models", fold_name)
        vecnorm_path = model_path + "_vecnorm.pkl"
        result_path = os.path.join("results", f"{fold_name}.jsonl")

        print(f"\n{'=' * 60}")
        print(
            f"Fold {run_idx + 1}/{len(fold_indices)}: {fold_tag}  "
            f"({len(train_ids)} train / {len(test_ids)} test)"
        )
        print(f"{'=' * 60}")

        # ---- Training -------------------------------------------
        if os.path.exists(model_path + ".zip"):
            print(f"  [skip training] {model_path}.zip already exists")
            model = PPO.load(model_path)
        else:
            train_model(fold_name, train_ids, test_ids, optimizers, cfg, args)
            model = PPO.load(model_path)

        # ---- Evaluation -----------------------------------------
        if os.path.exists(result_path):
            print(f"  [skip evaluation] {result_path} already exists")
            with open(result_path) as f:
                fold_results = [json.loads(line) for line in f]
        else:
            eval_env = make_vec_env(
                make_das_env(test_ids, optimizers, cfg),
                n_envs=1,
                seed=args.seed,
            )
            if os.path.exists(vecnorm_path):
                eval_env = VecNormalize.load(vecnorm_path, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False

            fold_results = []
            for problem_id in tqdm(test_ids, desc=f"  eval {fold_tag}", smoothing=0.0):
                obs = eval_env.reset()
                done = [False]
                info = {}
                while not done[0]:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, infos = eval_env.step(action)
                    if done[0]:
                        info = infos[0]

                best_y = info.get("best_y", float("inf"))
                optimum = global_optima.get(problem_id, 0.0)
                fold_results.append(
                    {
                        "problem_id": problem_id,
                        "fold": fold_tag,
                        "best_y": best_y,
                        "gap": best_y - optimum,
                    }
                )

            with open(result_path, "w") as f:
                for r in fold_results:
                    f.write(json.dumps(r) + "\n")
            eval_env.close()

        gaps = [r["gap"] for r in fold_results]
        summary = {
            "fold": fold_tag,
            "fold_idx": fold_idx,
            "n_test": len(fold_results),
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
        }
        fold_summaries.append(summary)
        print(
            f"  mean gap={summary['mean_gap']:.4e}  "
            f"median gap={summary['median_gap']:.4e}"
        )

    # ---- Aggregate summary --------------------------------------
    all_gaps = []
    for fold_idx in fold_indices:
        _, _, fold_tag = all_folds[fold_idx]
        rpath = os.path.join("results", f"{args.name}_cv_{fold_tag}.jsonl")
        if os.path.exists(rpath):
            with open(rpath) as fh:
                all_gaps.extend(json.loads(line)["gap"] for line in fh)

    overall = {
        "cv_mode": args.cv_mode,
        "name": args.name,
        "portfolio": args.portfolio,
        "dims": args.dims,
        "n_epochs_per_fold": args.n_epochs,
        "n_folds_run": len(fold_summaries),
        "overall_mean_gap": float(np.mean(all_gaps)) if all_gaps else None,
        "overall_median_gap": float(np.median(all_gaps)) if all_gaps else None,
        "folds": fold_summaries,
    }

    summary_path = os.path.join("results", f"{args.name}_cv_summary.jsonl")
    with open(summary_path, "w") as f:
        f.write(json.dumps(overall, indent=2) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Cross-validation complete  ({args.cv_mode})")
    print(f"  Folds run          : {len(fold_summaries)}")
    if all_gaps:
        print(f"  Overall mean gap   : {overall['overall_mean_gap']:.4e}")
        print(f"  Overall median gap : {overall['overall_median_gap']:.4e}")
    print(f"  Summary            : {summary_path}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    optimizers = get_portfolio(args.portfolio)

    cfg = {
        "fe_multiplier": args.fe_multiplier,
        "n_checkpoints": args.n_checkpoints,
        "cdb": args.cdb,
        "reward_option": args.reward_option,
        "n_individuals": args.n_individuals,
        "seed": args.seed,
    }

    print(f"Portfolio : {args.portfolio}")
    print(f"Budget    : {args.fe_multiplier}×dim  |  checkpoints={args.n_checkpoints}")

    if args.cv_mode:
        run_cv(args, optimizers, cfg)
    else:
        train_ids, test_ids = get_train_test_split(args.mode, args.dims)
        print(
            f"Mode      : {args.mode}  "
            f"({len(train_ids)} train / {len(test_ids)} test problems)"
        )
        train_model(args.name, train_ids, test_ids, optimizers, cfg, args)


if __name__ == "__main__":
    main()
