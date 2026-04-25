"""Training script for DAS using Stable Baselines 3 PPO.

Usage
-----
    python train.py <name> [options]

The trained model is saved to models/<name>.zip.
A VecNormalize statistics file is saved to models/<name>_vecnorm.pkl.
"""

import argparse
import json
import os
import warnings
from itertools import product

import cocoex
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from das.env.das_env import DASEnv
from das.optimizers.portfolio import get_portfolio
from das.training.callbacks import WandbCallback, DASEvalCallback

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# BBOB problem sets                                                    #
# ------------------------------------------------------------------ #

ALL_DIMS = [2, 3, 5, 10, 20, 40]
ALL_FUNCTIONS = set(range(1, 25))
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
EASY_TRAIN_FUNCTIONS = {4, *range(6, 15), 18, 19, 20, 22, 23, 24}


def build_problem_ids(functions: set[int], dims: list[int]) -> list[str]:
    return [
        f"bbob_f{f:03d}_i{i:02d}_d{d:02d}"
        for i, f, d in product(INSTANCE_IDS, sorted(functions), dims)
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

def make_das_env(problem_ids: list[str], suite, optimizers: list, cfg: dict):
    def _init():
        return DASEnv(
            problem_ids=problem_ids,
            suite=suite,
            optimizers=optimizers,
            fe_multiplier=cfg["fe_multiplier"],
            n_checkpoints=cfg["n_checkpoints"],
            checkpoint_division_base=cfg["cdb"],
            reward_option=cfg["reward_option"],
            n_individuals=cfg["n_individuals"],
        )
    return _init


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Train a DAS agent with PPO")
    p.add_argument("name", help="Experiment name (used for file names)")
    p.add_argument("-p", "--portfolio", nargs="+", default=["SPSO", "IPSO", "SPSOL"],
                   help="Sub-optimizer names from the portfolio")
    p.add_argument("-m", "--mode", default="easy", choices=["easy", "hard", "LOIO"],
                   help="Train/test split strategy")
    p.add_argument("-d", "--dims", nargs="+", type=int, default=ALL_DIMS,
                   choices=ALL_DIMS, help="Problem dimensions")
    p.add_argument("-f", "--fe-multiplier", type=int, default=10_000,
                   help="Budget = fe_multiplier * dimension")
    p.add_argument("-s", "--n-checkpoints", type=int, default=10,
                   help="Optimizer selection steps per episode")
    p.add_argument("-x", "--cdb", type=float, default=1.0,
                   help="Checkpoint division base (1.0 = uniform)")
    p.add_argument("-O", "--reward-option", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("-n", "--n-individuals", type=int, default=100,
                   help="Shared population size across all sub-optimizers")
    p.add_argument("-t", "--total-steps", type=int, default=500_000,
                   help="Total PPO training timesteps")
    p.add_argument("-j", "--n-envs", type=int, default=1,
                   help="Number of parallel environments")
    p.add_argument("-e", "--eval-freq", type=int, default=50_000,
                   help="Evaluate every N training steps")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
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
    cocoex.utilities.MiniPrint()
    suite = cocoex.Suite("bbob", "", "")
    train_ids, test_ids = get_train_test_split(args.mode, args.dims)

    cfg = {
        "fe_multiplier": args.fe_multiplier,
        "n_checkpoints": args.n_checkpoints,
        "cdb": args.cdb,
        "reward_option": args.reward_option,
        "n_individuals": args.n_individuals,
    }

    print(f"Portfolio : {args.portfolio}")
    print(f"Mode      : {args.mode}  ({len(train_ids)} train / {len(test_ids)} test problems)")
    print(f"Budget    : {args.fe_multiplier}×dim  |  checkpoints={args.n_checkpoints}")

    # Training env (vectorised + normalised)
    train_env = make_vec_env(
        make_das_env(train_ids, suite, optimizers, cfg),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
        seed=args.seed,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    # Evaluation env (single env, normalisation params frozen after training)
    eval_env = make_vec_env(
        make_das_env(test_ids[:20], suite, optimizers, cfg),
        n_envs=1,
        seed=args.seed + 1,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # PPO – policy net size mirrors the original (96-unit hidden layers)
    policy_kwargs = dict(net_arch=[96, 96])
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-5,
        n_steps=args.n_checkpoints,       # collect one full episode per update batch
        batch_size=256,
        n_epochs=6,
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
        DASEvalCallback(eval_env, eval_freq=args.eval_freq, n_eval_episodes=20, name=args.name),
    ]
    if args.wandb:
        callbacks.append(WandbCallback())

    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=True)

    model_path = os.path.join("models", args.name)
    model.save(model_path)
    train_env.save(model_path + "_vecnorm.pkl")
    print(f"Saved model to {model_path}.zip")


if __name__ == "__main__":
    main()
