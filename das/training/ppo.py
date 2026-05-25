"""PPO training and evaluation logic (SB3-based)."""

import json
import os

import numpy as np
from tqdm import tqdm

from das.env.bbob_splits import get_cv_folds, get_train_test_split
from das.optimizers.portfolio import get_portfolio
from das.training.common import get_ioh_optimum, make_das_env, write_jsonl


def _eval_loop(
    model, eval_env, problem_ids: list[str], desc: str = "eval"
) -> list[dict]:
    results = []
    for problem_id in tqdm(problem_ids, desc=f"  {desc}", smoothing=0.0):
        obs = eval_env.reset()
        done = [False]
        info = {}
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = eval_env.step(action)
            if done[0]:
                info = infos[0]
        best_y = info.get("best_y", float("inf"))
        optimum = get_ioh_optimum(problem_id)
        results.append(
            {"problem_id": problem_id, "best_y": best_y, "gap": best_y - optimum}
        )
    return results


def _train_single(
    name: str,
    train_ids: list[str],
    test_ids: list[str],
    optimizers: list,
    cfg: dict,
    args,
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from das.training.callbacks import DASEvalCallback, WandbCallback

    model_path = os.path.join("models", name)
    vecnorm_path = model_path + "_vecnorm.pkl"

    steps_per_epoch = len(train_ids) * cfg["n_checkpoints"]
    total_timesteps = args.n_epochs * steps_per_epoch

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

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-5,
        n_steps=cfg["n_checkpoints"],
        batch_size=256,
        n_epochs=6,
        gamma=0.8,
        gae_lambda=0.5,
        clip_range=0.3,
        ent_coef=0.01,
        vf_coef=0.3,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[96, 96]),
        verbose=1,
        seed=args.seed,
    )

    callbacks = [
        DASEvalCallback(
            eval_env,
            eval_freq=steps_per_epoch,
            n_eval_episodes=len(eval_sample),
            name=name,
        )
    ]
    if getattr(args, "wandb", False):
        callbacks.append(WandbCallback())

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    model.save(model_path)
    train_env.save(vecnorm_path)
    train_env.close()
    eval_env.close()
    print(f"  Saved model → {model_path}.zip")


def _load_eval_env(
    model_path: str, test_ids: list[str], optimizers: list, cfg: dict, seed: int
):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    vecnorm_path = model_path + "_vecnorm.pkl"
    eval_env = make_vec_env(
        make_das_env(test_ids, optimizers, cfg), n_envs=1, seed=seed
    )
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    return eval_env


def run_ppo(args) -> None:
    from stable_baselines3 import PPO

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

    train_ids, test_ids = get_train_test_split(args.mode, args.dims)
    print(
        f"Mode      : {args.mode}  ({len(train_ids)} train / {len(test_ids)} test problems)"
    )

    _train_single(args.name, train_ids, test_ids, optimizers, cfg, args)

    if args.eval:
        print("\nRunning post-training evaluation …")
        model_path = os.path.join("models", args.name)
        model = PPO.load(model_path)
        eval_env = _load_eval_env(model_path, test_ids, optimizers, cfg, args.seed)
        results = _eval_loop(model, eval_env, test_ids)
        eval_env.close()
        out_path = os.path.join("results", f"{args.name}_eval.jsonl")
        write_jsonl(out_path, results)
        gaps = [r["gap"] for r in results]
        print(f"  Mean gap   : {np.mean(gaps):.4e}")
        print(f"  Median gap : {np.median(gaps):.4e}")
        print(f"  Results    : {out_path}")


def run_cv_ppo(args) -> None:
    from stable_baselines3 import PPO

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

    all_folds = get_cv_folds(
        args.cv_mode, args.dims, seed=args.seed, n_folds=args.n_folds
    )
    fold_indices = list(range(len(all_folds))) if args.folds is None else args.folds

    print(
        f"CV mode    : {args.cv_mode}  ({len(fold_indices)}/{len(all_folds)} folds selected)"
    )
    print(f"Epochs/fold: {args.n_epochs}")

    fold_summaries = []

    for run_idx, fold_idx in enumerate(fold_indices):
        train_ids, test_ids, fold_tag = all_folds[fold_idx]
        fold_name = f"{args.name}_cv_{fold_tag}"
        model_path = os.path.join("models", fold_name)
        result_path = os.path.join("results", f"{fold_name}.jsonl")

        print(f"\n{'=' * 60}")
        print(
            f"Fold {run_idx + 1}/{len(fold_indices)}: {fold_tag}  ({len(train_ids)} train / {len(test_ids)} test)"
        )
        print(f"{'=' * 60}")

        if os.path.exists(model_path + ".zip"):
            print(f"  [skip training] {model_path}.zip already exists")
            model = PPO.load(model_path)
        else:
            _train_single(fold_name, train_ids, test_ids, optimizers, cfg, args)
            model = PPO.load(model_path)

        if os.path.exists(result_path):
            print(f"  [skip evaluation] {result_path} already exists")
            with open(result_path) as fh:
                fold_results = [json.loads(line) for line in fh]
        else:
            eval_env = _load_eval_env(model_path, test_ids, optimizers, cfg, args.seed)
            fold_results = _eval_loop(
                model, eval_env, test_ids, desc=f"eval {fold_tag}"
            )
            for r in fold_results:
                r["fold"] = fold_tag
            eval_env.close()
            write_jsonl(result_path, fold_results)
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
    write_jsonl(summary_path, [overall])

    print(f"\n{'=' * 60}")
    print(f"Cross-validation complete  ({args.cv_mode})")
    print(f"  Folds run          : {len(fold_summaries)}")
    if all_gaps:
        print(f"  Overall mean gap   : {overall['overall_mean_gap']:.4e}")
        print(f"  Overall median gap : {overall['overall_median_gap']:.4e}")
    print(f"  Summary            : {summary_path}")
