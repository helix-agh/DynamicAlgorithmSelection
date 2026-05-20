"""Exponential-DAS training runner (custom PyTorch PPO with exp checkpoint spacing)."""

import json
import os

import cocoex as cx
import numpy as np

from das.env.bbob_splits import get_cv_folds, get_train_test_split
from das.env.das_env import DASEnv
from das.env.observation import observation_dim
from das.optimizers.portfolio import get_portfolio
from das.training.common import write_jsonl


def run_exp_das(args) -> None:
    from agents.exponential_das import ExpDASAgent
    from agents.exponential_das import train, evaluate

    optimizers = get_portfolio(args.portfolio)
    n_opt = len(optimizers)
    obs_dim = observation_dim(n_opt)

    train_ids, test_ids = get_train_test_split(args.mode, args.dims)
    total_episodes = args.n_epochs * len(train_ids)
    print(f"Train: {len(train_ids)} problems  |  Test: {len(test_ids)} problems")
    print(
        f"obs_dim={obs_dim}  cdb={args.cdb}  n_checkpoints={args.n_checkpoints}"
        f"  n_epochs={args.n_epochs}  total_episodes={total_episodes}"
    )

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

    train(
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        total_episodes=total_episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir="models",
        name=args.name,
    )

    if args.eval:
        print("\nFinal evaluation on test set …")
        test_results = evaluate(test_env, agent, n_episodes=min(len(test_ids), 50))
        mean_best_y = float(np.mean([r["best_y"] for r in test_results]))
        print(f"Test mean best_y = {mean_best_y:.6e}")
        eval_path = os.path.join("results", f"{args.name}_eval.jsonl")
        write_jsonl(eval_path, test_results)
        print(f"Results saved to {eval_path}")


def run_cv_exp_das(args) -> None:
    from agents.exponential_das import ExpDASAgent
    from agents.exponential_das import train, evaluate

    optimizers = get_portfolio(args.portfolio)
    n_opt = len(optimizers)
    obs_dim = observation_dim(n_opt)

    suite = cx.Suite("bbob", "", "")

    all_folds = get_cv_folds(
        args.cv_mode, args.dims, seed=args.seed, n_folds=args.n_folds
    )
    fold_indices = list(range(len(all_folds))) if args.folds is None else args.folds

    print(f"CV mode    : {args.cv_mode}  ({len(fold_indices)}/{len(all_folds)} folds)")
    print(f"n_epochs/fold: {args.n_epochs}  |  dims={args.dims}")

    buffer_capacity = args.buffer_capacity or (16 * args.n_checkpoints)

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

    fold_summaries = []

    for run_idx, fold_idx in enumerate(fold_indices):
        train_ids, test_ids, fold_tag = all_folds[fold_idx]
        fold_name = f"{args.name}_cv_{fold_tag}"
        model_path = os.path.join("models", f"{fold_name}_final.pt")
        result_path = os.path.join("results", f"{fold_name}.jsonl")

        print(f"\n{'=' * 60}")
        print(
            f"Fold {run_idx + 1}/{len(fold_indices)}: {fold_tag}"
            f"  ({len(train_ids)} train / {len(test_ids)} test)"
        )
        print(f"{'=' * 60}")

        total_episodes = args.n_epochs * len(train_ids)

        if os.path.exists(model_path):
            print(f"  [skip training] {model_path} already exists")
            agent = ExpDASAgent.load(model_path, obs_dim=obs_dim, n_actions=n_opt)
        else:
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
            train_env = DASEnv(problem_ids=train_ids, **env_cfg)
            train(
                train_env=train_env,
                test_env=None,
                agent=agent,
                total_episodes=total_episodes,
                eval_interval=total_episodes + 1,
                save_interval=args.save_interval,
                save_dir="models",
                name=fold_name,
            )
            train_env.close()

        if os.path.exists(result_path):
            print(f"  [skip evaluation] {result_path} already exists")
            with open(result_path) as fh:
                fold_results = [json.loads(line) for line in fh]
        else:
            eval_env = DASEnv(problem_ids=test_ids, **env_cfg)
            raw = evaluate(eval_env, agent, n_episodes=len(test_ids))
            fold_results = [{**r, "fold": fold_tag} for r in raw]
            eval_env.close()
            write_jsonl(result_path, fold_results)
            print(f"  Saved results → {result_path}")

        mean_best_y = float(np.mean([r["best_y"] for r in fold_results]))
        fold_summaries.append(
            {
                "fold": fold_tag,
                "fold_idx": fold_idx,
                "n_test": len(fold_results),
                "mean_best_y": mean_best_y,
            }
        )
        print(f"  mean best_y={mean_best_y:.4e}")

    all_best_y = []
    for fold_idx in fold_indices:
        _, _, fold_tag = all_folds[fold_idx]
        rpath = os.path.join("results", f"{args.name}_cv_{fold_tag}.jsonl")
        if os.path.exists(rpath):
            with open(rpath) as fh:
                all_best_y.extend(json.loads(line)["best_y"] for line in fh)

    overall = {
        "cv_mode": args.cv_mode,
        "name": args.name,
        "portfolio": args.portfolio,
        "dims": args.dims,
        "n_epochs_per_fold": args.n_epochs,
        "n_folds_run": len(fold_summaries),
        "overall_mean_best_y": float(np.mean(all_best_y)) if all_best_y else None,
        "folds": fold_summaries,
    }
    summary_path = os.path.join("results", f"{args.name}_cv_summary.jsonl")
    write_jsonl(summary_path, [overall])

    print(f"\n{'=' * 60}")
    print(f"Cross-validation complete  ({args.cv_mode}  dims={args.dims})")
    print(f"  Folds run          : {len(fold_summaries)}")
    if all_best_y:
        print(f"  Overall mean best_y: {overall['overall_mean_best_y']:.4e}")
    print(f"  Summary            : {summary_path}")
