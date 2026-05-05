"""Exponential-DAS training runner (custom PyTorch PPO with exp checkpoint spacing)."""

import os

import cocoex as cx
import numpy as np

from das.env.bbob_splits import get_train_test_split
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

    train(
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        total_episodes=args.total_episodes,
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
