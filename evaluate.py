"""Evaluation script: load a trained DAS model and run it on BBOB test problems.

Usage
-----
    python evaluate.py <model_name> [options]

Results are written per-problem to results/<model_name>_eval.jsonl.
An optional cocoex observer writes COCO-compatible data for cocopp post-processing.
"""

import argparse
import json
import os
import warnings
from itertools import product

import cocoex
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from das.env.bbob_splits import ALL_DIMS, get_train_test_split
from das.env.das_env import DASEnv
from das.optimizers.portfolio import get_portfolio
from das.utils import set_seed
from das.training.common import load_global_optima, make_das_env

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
# AOCC metric                                                          #
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
    p = argparse.ArgumentParser(description="Evaluate a trained DAS agent")
    p.add_argument("name", help="Model name (looks for models/<name>.zip)")
    p.add_argument("-p", "--portfolio", nargs="+", default=["SPSO", "IPSO", "SPSOL"])
    p.add_argument("-m", "--mode", default="easy", choices=["easy", "hard", "random"])
    p.add_argument(
        "-d", "--dims", nargs="+", type=int, default=ALL_DIMS, choices=ALL_DIMS
    )
    p.add_argument("-f", "--fe-multiplier", type=int, default=10_000)
    p.add_argument("-s", "--n-checkpoints", type=int, default=10)
    p.add_argument("-x", "--cdb", type=float, default=1.0)
    p.add_argument("-O", "--reward-option", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("-n", "--n-individuals", type=int, default=100)
    p.add_argument("--coco", action="store_true", help="Write COCO observer data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("results", exist_ok=True)

    optimizers = get_portfolio(args.portfolio)
    _, test_ids = get_train_test_split(args.mode, args.dims)

    cfg = {
        "fe_multiplier": args.fe_multiplier,
        "n_checkpoints": args.n_checkpoints,
        "cdb": args.cdb,
        "reward_option": args.reward_option,
        "n_individuals": args.n_individuals,
    }

    # Load model
    model_path = os.path.join("models", args.name)
    vecnorm_path = model_path + "_vecnorm.pkl"
    model = PPO.load(model_path)

    eval_env = make_vec_env(
        make_das_env(test_ids, optimizers, cfg), n_envs=1, seed=args.seed
    )
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    global_optima = load_global_optima()
    observer = (
        cocoex.Observer("bbob", f"result_folder: {args.name}") if args.coco else None
    )

    out_path = os.path.join("results", f"{args.name}_eval.jsonl")
    results_all = []

    for problem_id in tqdm(test_ids, smoothing=0.0):
        # Run one episode
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
        record = {
            "problem_id": problem_id,
            "best_y": best_y,
            "gap": best_y - optimum,
        }
        results_all.append(record)

    with open(out_path, "w") as f:
        for r in results_all:
            f.write(json.dumps(r) + "\n")

    gaps = [r["gap"] for r in results_all]
    print(f"\nEvaluation complete  ({len(results_all)} problems)")
    print(f"  Mean gap  : {np.mean(gaps):.4e}")
    print(f"  Median gap: {np.median(gaps):.4e}")
    print(f"  Results   : {out_path}")


if __name__ == "__main__":
    main()
