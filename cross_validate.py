"""Cross-validation for DAS on BBOB.

Two modes:
  LOIO (Leave-One-Instance-Out): 15 folds, one per BBOB instance ID.
      Fold k trains on all problems whose instance ID ≠ inst_k;
      tests on all problems with instance ID == inst_k.

  LOPO (Leave-One-Problem-Out): 24 folds, one per BBOB function.
      Fold k trains on all problems whose function ID ≠ f_k;
      tests on all problems with function ID == f_k.

Usage
-----
    python cross_validate.py <name> --cv-mode LOIO [options]
    python cross_validate.py <name> --cv-mode LOPO [options]
    python cross_validate.py <name> --cv-mode LOPO --folds 0 1 2  # subset

Outputs
-------
    models/<name>_cv_<fold_tag>.zip          – trained model per fold
    models/<name>_cv_<fold_tag>_vecnorm.pkl  – normalisation stats
    results/<name>_cv_<fold_tag>.jsonl       – per-problem results
    results/<name>_cv_summary.jsonl          – aggregated summary
"""

import argparse
import json
import os
import warnings
from itertools import product as iterproduct

import cocoex
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from tqdm import tqdm

from das.optimizers.portfolio import get_portfolio
from das.training.callbacks import DASEvalCallback
from train import make_das_env, load_global_optima, ALL_DIMS

warnings.filterwarnings("ignore")

ALL_FUNCTIONS = list(range(1, 25))
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]


# ------------------------------------------------------------------ #
# Problem ID helpers                                                   #
# ------------------------------------------------------------------ #


def build_problem_ids(
    functions: list[int], instances: list[int], dims: list[int]
) -> list[str]:
    return [
        f"bbob_f{f:03d}_i{i:02d}_d{d:02d}"
        for i, f, d in iterproduct(instances, sorted(functions), dims)
    ]


def get_cv_folds(
    cv_mode: str, dims: list[int]
) -> list[tuple[list[str], list[str], str]]:
    """Return a list of (train_ids, test_ids, fold_tag) for every fold."""
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
        for fn in ALL_FUNCTIONS:
            train_fns = [f for f in ALL_FUNCTIONS if f != fn]
            folds.append(
                (
                    build_problem_ids(train_fns, INSTANCE_IDS, dims),
                    build_problem_ids([fn], INSTANCE_IDS, dims),
                    f"f{fn:03d}",
                )
            )
    return folds


# ------------------------------------------------------------------ #
# AOCC metric (matches evaluate.py)                                   #
# ------------------------------------------------------------------ #


def aocc(
    fitness_history: list[tuple[int, float]], max_fe: int, optimum: float
) -> float:
    lb, ub = -8.0, 8.0
    area, prev_fe = 0.0, 0
    for fe, f in fitness_history:
        v = np.clip(f - optimum, 1e-8, 1e8)
        area += (1.0 - (np.log10(v) - lb) / (ub - lb)) * (fe - prev_fe)
        prev_fe = fe
    if fitness_history:
        last_v = np.clip(fitness_history[-1][1] - optimum, 1e-8, 1e8)
        area += (1.0 - (np.log10(last_v) - lb) / (ub - lb)) * (
            max_fe - fitness_history[-1][0]
        )
    return area / max_fe


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def parse_args():
    p = argparse.ArgumentParser(description="K-fold cross-validation for DAS on BBOB")
    p.add_argument(
        "name", help="Experiment name prefix (used in model/result filenames)"
    )
    p.add_argument(
        "--cv-mode",
        default="LOIO",
        choices=["LOIO", "LOPO"],
        help="LOIO=Leave-One-Instance-Out (15 folds) | LOPO=Leave-One-Problem-Out (24 folds)",
    )
    p.add_argument(
        "-p",
        "--portfolio",
        nargs="+",
        default=["SPSO", "IPSO", "SPSOL"],
        help="Sub-optimizer names from the portfolio",
    )
    p.add_argument(
        "-d",
        "--dims",
        nargs="+",
        type=int,
        default=ALL_DIMS,
        choices=ALL_DIMS,
        help="Problem dimensions to include",
    )
    p.add_argument(
        "-f",
        "--fe-multiplier",
        type=int,
        default=10_000,
        help="Budget = fe_multiplier × dimension",
    )
    p.add_argument(
        "-s",
        "--n-checkpoints",
        type=int,
        default=10,
        help="Optimizer-selection steps per episode",
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
        "-n", "--n-individuals", type=int, default=100, help="Shared population size"
    )
    p.add_argument(
        "-t",
        "--total-steps",
        type=int,
        default=500_000,
        help="PPO training timesteps per fold",
    )
    p.add_argument(
        "-j",
        "--n-envs",
        type=int,
        default=1,
        help="Parallel training environments per fold",
    )
    p.add_argument(
        "-e",
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluate every N training steps (during training)",
    )
    p.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Zero-based fold indices to run (default: all). "
        "Useful for parallelism or resuming.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    optimizers = get_portfolio(args.portfolio)
    global_optima = load_global_optima()

    cfg = {
        "fe_multiplier": args.fe_multiplier,
        "n_checkpoints": args.n_checkpoints,
        "cdb": args.cdb,
        "reward_option": args.reward_option,
        "n_individuals": args.n_individuals,
    }

    all_folds = get_cv_folds(args.cv_mode, args.dims)
    fold_indices = args.folds if args.folds is not None else list(range(len(all_folds)))

    print(
        f"CV mode   : {args.cv_mode}  "
        f"({len(fold_indices)}/{len(all_folds)} folds selected)"
    )
    print(f"Portfolio : {args.portfolio}")
    print(f"Budget    : {args.fe_multiplier}×dim  |  checkpoints={args.n_checkpoints}")
    print(f"Steps/fold: {args.total_steps:,}")

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
            f"({len(train_ids)} train / {len(test_ids)} test problems)"
        )
        print(f"{'=' * 60}")

        # ---- Training ------------------------------------------- #
        if os.path.exists(model_path + ".zip"):
            print(f"  [skip training] {model_path}.zip already exists")
            model = PPO.load(model_path)
        else:
            train_env = make_vec_env(
                make_das_env(train_ids, optimizers, cfg),
                n_envs=args.n_envs,
                vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
                seed=args.seed,
            )
            train_env = VecNormalize(
                train_env, norm_obs=True, norm_reward=True, clip_obs=5.0
            )

            eval_sample = test_ids[: min(20, len(test_ids))]
            eval_env = make_vec_env(
                make_das_env(eval_sample, optimizers, cfg),
                n_envs=1,
                seed=args.seed + 1,
            )
            eval_env = VecNormalize(
                eval_env, norm_obs=True, norm_reward=False, training=False
            )

            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=3e-5,
                n_steps=args.n_checkpoints,
                batch_size=256,
                n_epochs=6,
                gamma=0.8,
                gae_lambda=0.5,
                clip_range=0.3,
                ent_coef=0.01,
                vf_coef=0.3,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[96, 96]),
                verbose=0,
                seed=args.seed,
            )

            callback = DASEvalCallback(
                eval_env,
                eval_freq=args.eval_freq,
                n_eval_episodes=len(eval_sample),
                name=fold_name,
            )
            model.learn(
                total_timesteps=args.total_steps,
                callback=callback,
                progress_bar=True,
            )

            model.save(model_path)
            train_env.save(vecnorm_path)
            train_env.close()
            eval_env.close()
            print(f"  Saved model → {model_path}.zip")

        # ---- Evaluation ----------------------------------------- #
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
            print(f"  Saved results → {result_path}")

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
            f"  mean gap={summary['mean_gap']:.4e}  median gap={summary['median_gap']:.4e}"
        )

    # ---- Aggregate summary -------------------------------------- #
    all_gaps = []
    for idx, fold_idx in enumerate(fold_indices):
        _, _, fold_tag = all_folds[fold_idx]
        rpath = os.path.join("results", f"{args.name}_cv_{fold_tag}.jsonl")
        if os.path.exists(rpath):
            with open(rpath) as f:
                for line in f:
                    all_gaps.append(json.loads(line)["gap"])

    overall = {
        "cv_mode": args.cv_mode,
        "name": args.name,
        "portfolio": args.portfolio,
        "dims": args.dims,
        "total_steps_per_fold": args.total_steps,
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


if __name__ == "__main__":
    main()
