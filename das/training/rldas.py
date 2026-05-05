"""RL-DAS training runner (custom single-dim PyTorch PPO)."""

import os

import numpy as np

from das.env.bbob_splits import get_train_test_split
from das.optimizers.portfolio import get_portfolio
from das.training.common import write_jsonl


def run_rl_das(args) -> None:
    import cocoex as cx
    from agents.rl_das import RLDASEnv, PPOAgent
    from agents.rl_das import train, evaluate

    optimizers = get_portfolio(args.portfolio)
    if not optimizers:
        raise ValueError(f"Unknown optimizers: {args.portfolio}")

    train_ids, test_ids = get_train_test_split(args.mode, [args.dim])
    print(f"Train: {len(train_ids)} problems  |  Test: {len(test_ids)} problems")

    suite = cx.Suite("bbob", "", "")

    if args.k_epoch is None:
        args.k_epoch = max(1, int(0.3 * args.n_checkpoints))

    env_kwargs = dict(
        suite=suite,
        optimizers=optimizers,
        dim=args.dim,
        fe_multiplier=args.fe_multiplier,
        n_checkpoints=args.n_checkpoints,
        n_individuals=args.n_individuals,
        seed=args.seed,
    )
    train_env = RLDASEnv(problem_ids=train_ids, **env_kwargs)
    test_env = RLDASEnv(problem_ids=test_ids, **env_kwargs)

    agent = PPOAgent(
        dim=args.dim, n_opt=len(optimizers), lr=args.lr, device=args.device
    )

    print(
        f"RL-DAS  |  dim={args.dim}  |  portfolio={args.portfolio}"
        f"  |  obs_dim={train_env.observation_space.shape[0]}"
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
        name=args.name,
    )

    if args.eval:
        print("\nRunning final evaluation on test set …")
        n_problems = len(test_env.problem_ids)
        test_results = evaluate(test_env, agent, n_episodes=n_problems)
        mean_best_y = float(np.mean([r["best_y"] for r in test_results]))
        print(f"Test mean best_y = {mean_best_y:.6e}")
        eval_path = os.path.join("results", f"{args.name}_eval.jsonl")
        write_jsonl(eval_path, test_results)
        print(f"Results saved to {eval_path}")
