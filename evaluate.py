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

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from das.env.bbob_splits import ALL_DIMS, get_train_test_split
from das.optimizers.portfolio import get_portfolio
from das.utils import set_seed
from das.training.common import compute_run_stats, get_ioh_optimum, make_das_env

warnings.filterwarnings("ignore")


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
    p.add_argument("-n", "--n-individuals", type=int, default=None)
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

    out_path = os.path.join("results", f"{args.name}_eval.jsonl")
    results_all = []

    for problem_id in tqdm(test_ids, smoothing=0.0):
        obs = eval_env.reset()
        done = [False]
        fitness_history: list[tuple[int, float]] = []
        step_info: dict = {}
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = eval_env.step(action)
            step_info = infos[0]
            fitness_history.extend(step_info.get("fitness_history_step", []))

        max_fe = step_info.get("n_fe", 0)
        global_minimum = get_ioh_optimum(problem_id)
        stats = compute_run_stats(fitness_history, max_fe, global_minimum)
        results_all.append({problem_id: stats})

    with open(out_path, "w") as f:
        for r in results_all:
            f.write(json.dumps(r) + "\n")

    metrics = [next(iter(r.values())) for r in results_all]
    print(f"\nEvaluation complete  ({len(results_all)} problems)")
    print(
        f"  Mean final_fitness  : {np.mean([m['final_fitness'] for m in metrics]):.4e}"
    )
    print(f"  Mean AOCC           : {np.mean([m['aocc'] for m in metrics]):.4f}")
    print(f"  Results             : {out_path}")


if __name__ == "__main__":
    main()
