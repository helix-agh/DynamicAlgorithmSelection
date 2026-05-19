"""RL-DAS training runner (custom single-dim PyTorch PPO)."""

import json
import os

import numpy as np

from das.env.bbob_splits import get_cv_folds, get_train_test_split
from das.training.common import write_jsonl


def run_rl_das(args) -> None:
    from das.env.ioh_suite import IOHSuite
    from agents.rl_das import RLDASEnv, PPOAgent
    from agents.rl_das import train, evaluate
    from agents.rl_das.optimizers import get_rldas_portfolio

    optimizers = get_rldas_portfolio(args.portfolio)

    train_ids, test_ids = get_train_test_split(args.mode, [args.dim])
    print(f"Train: {len(train_ids)} problems  |  Test: {len(test_ids)} problems")

    suite = IOHSuite()

    # Local variable — avoid mutating args so the caller's namespace stays predictable.
    k_epoch = (
        args.k_epoch if args.k_epoch is not None
        else max(1, int(0.3 * args.n_checkpoints))
    )

    env_kwargs = dict(
        suite=suite,
        optimizers=optimizers,
        dim=args.dim,
        fe_multiplier=args.fe_multiplier,
        n_checkpoints=args.n_checkpoints,
        seed=args.seed,
    )
    if args.n_individuals is not None:
        env_kwargs["n_individuals"] = args.n_individuals
    train_env = RLDASEnv(problem_ids=train_ids, **env_kwargs)
    test_env = RLDASEnv(problem_ids=test_ids, **env_kwargs)

    agent = PPOAgent(
        dim=args.dim, n_opt=len(optimizers), lr=args.lr, device=args.device
    )

    print(
        f"RL-DAS  |  dim={args.dim}  |  portfolio={args.portfolio}"
        f"  |  obs_dim={train_env.observation_space.shape[0]}"
        f"  |  k_epoch={k_epoch}"
    )

    train(
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        n_epochs=args.n_epochs,
        k_epoch=k_epoch,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir="models",
        name=args.name,
    )

    if args.eval:
        print("\nRunning final evaluation on test set …")

        # Fresh env so _problem_idx starts at 0.  test_env accumulated increments
        # from periodic evaluations inside train() and would start from a rotated
        # offset rather than problem 0, making results hard to reproduce.
        eval_env = RLDASEnv(problem_ids=test_ids, **env_kwargs)
        n_problems = len(test_ids)
        test_results = evaluate(eval_env, agent, n_episodes=n_problems)

        # Create the output directory before writing — write_jsonl does not
        # create parent directories and would raise FileNotFoundError otherwise.
        os.makedirs("results", exist_ok=True)
        n_problems = len(test_env._problem_ids)
        test_results = evaluate(test_env, agent, n_episodes=n_problems)
        mean_best_y = float(np.mean([r["best_y"] for r in test_results]))
        print(f"Test mean best_y = {mean_best_y:.6e}")
        eval_path = os.path.join("results", f"{args.name}_eval.jsonl")
        write_jsonl(eval_path, test_results)
        print(f"Results saved to {eval_path}")


def run_cv_rl_das(args) -> None:
    from das.env.ioh_suite import IOHSuite
    from agents.rl_das import RLDASEnv, PPOAgent
    from agents.rl_das import train, evaluate
    from agents.rl_das.optimizers import get_rldas_portfolio

    optimizers = get_rldas_portfolio(args.portfolio)

    suite = IOHSuite()

    if args.k_epoch is None:
        args.k_epoch = max(1, int(0.3 * args.n_checkpoints))

    all_folds = get_cv_folds(
        args.cv_mode, [args.dim], seed=args.seed, n_folds=args.n_folds
    )
    fold_indices = list(range(len(all_folds))) if args.folds is None else args.folds

    print(
        f"CV mode    : {args.cv_mode}  ({len(fold_indices)}/{len(all_folds)} folds selected)"
    )
    print(f"Epochs/fold: {args.n_epochs}  |  dim={args.dim}")

    env_kwargs = dict(
        suite=suite,
        optimizers=optimizers,
        dim=args.dim,
        fe_multiplier=args.fe_multiplier,
        n_checkpoints=args.n_checkpoints,
        seed=args.seed,
    )
    if args.n_individuals is not None:
        env_kwargs["n_individuals"] = args.n_individuals

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

        agent = PPOAgent(
            dim=args.dim, n_opt=len(optimizers), lr=args.lr, device=args.device
        )

        if os.path.exists(model_path):
            print(f"  [skip training] {model_path} already exists")
            agent = PPOAgent.load(model_path, dim=args.dim, n_opt=len(optimizers))
        else:
            train_env = RLDASEnv(problem_ids=train_ids, **env_kwargs)
            test_env = RLDASEnv(problem_ids=test_ids, **env_kwargs)
            print(
                f"RL-DAS  |  obs_dim={train_env.observation_space.shape[0]}"
                f"  |  k_epoch={args.k_epoch}"
            )
            train(
                train_env=train_env,
                test_env=test_env,
                agent=agent,
                n_epochs=args.n_epochs,
                k_epoch=args.k_epoch,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval,
                save_dir="models",
                name=fold_name,
            )
            train_env.close()
            test_env.close()

        if os.path.exists(result_path):
            print(f"  [skip evaluation] {result_path} already exists")
            with open(result_path) as fh:
                fold_results = [json.loads(line) for line in fh]
        else:
            eval_env = RLDASEnv(problem_ids=test_ids, **env_kwargs)
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
        "dim": args.dim,
        "n_epochs_per_fold": args.n_epochs,
        "n_folds_run": len(fold_summaries),
        "overall_mean_best_y": float(np.mean(all_best_y)) if all_best_y else None,
        "folds": fold_summaries,
    }
    summary_path = os.path.join("results", f"{args.name}_cv_summary.jsonl")
    write_jsonl(summary_path, [overall])

    print(f"\n{'=' * 60}")
    print(f"Cross-validation complete  ({args.cv_mode}  dim={args.dim})")
    print(f"  Folds run          : {len(fold_summaries)}")
    if all_best_y:
        print(f"  Overall mean best_y: {overall['overall_mean_best_y']:.4e}")
    print(f"  Summary            : {summary_path}")
