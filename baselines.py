"""Baseline agents for DAS on BBOB.

Three types of agent, two modes of execution:

  DASEnv-based (uses checkpoints + warm-starting, comparable to RL agent)
  -----------------------------------------------------------------------
    random          Uniform random action at every checkpoint.
    fixed:<name>    Always pick one optimizer, e.g. fixed:SPSO.

  Pure single-algorithm (full budget, no checkpoint splitting)
  ------------------------------------------------------------
    single:<name>   Run one optimizer for the full budget in one shot,
                    bypassing the checkpoint mechanism entirely.

  Aggregate
  ---------
    all             Run random + fixed:<each> + single:<each>, then
                    derive oracle-best and oracle-worst across all
                    fixed-policy results.

Usage
-----
    python baselines.py <name> --agent random [options]
    python baselines.py <name> --agent fixed:SPSO [options]
    python baselines.py <name> --agent single:SPSO [options]
    python baselines.py <name> --agent all [options]

Outputs
-------
    results/<name>_<agent_tag>.jsonl     per-problem {problem_id, agent, best_y, gap}
    results/<name>_baselines_summary.jsonl   aggregate comparison (when --agent all)
"""

import argparse
import json
import os
import warnings
from itertools import product

import cocoex
import numpy as np
from tqdm import tqdm

from das.env.das_env import DASEnv
from das.optimizers.portfolio import get_portfolio
from das.utils import set_seed
from das.env.bbob_splits import ALL_DIMS, get_train_test_split
from train import load_global_optima

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
# Policy functions                                                     #
# ------------------------------------------------------------------ #


def random_policy(obs: np.ndarray, n_actions: int) -> int:
    return int(np.random.randint(n_actions))


def fixed_policy(action: int):
    def _policy(obs: np.ndarray, n_actions: int) -> int:
        return action

    return _policy


# ------------------------------------------------------------------ #
# DASEnv-based episode runner                                          #
# ------------------------------------------------------------------ #


def run_episode(env: DASEnv, policy_fn) -> dict:
    """Run one complete episode; return the final info dict."""
    obs, info = env.reset()
    done = False
    while not done:
        action = policy_fn(obs, env.action_space.n)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def collect_env_results(
    agent_tag: str,
    policy_fn,
    test_ids: list[str],
    suite,
    optimizers: list,
    cfg: dict,
    global_optima: dict[str, float],
) -> list[dict]:
    """Run policy_fn on every problem in test_ids via DASEnv."""
    env = DASEnv(
        problem_ids=test_ids,
        suite=suite,
        optimizers=optimizers,
        fe_multiplier=cfg["fe_multiplier"],
        n_checkpoints=cfg["n_checkpoints"],
        checkpoint_division_base=cfg["cdb"],
        reward_option=cfg["reward_option"],
        n_individuals=cfg["n_individuals"],
    )
    results = []
    for problem_id in tqdm(test_ids, desc=f"  {agent_tag}", smoothing=0.0):
        info = run_episode(env, policy_fn)
        best_y = info.get("best_y", float("inf"))
        optimum = global_optima.get(problem_id, 0.0)
        results.append(
            {
                "problem_id": problem_id,
                "agent": agent_tag,
                "best_y": best_y,
                "gap": best_y - optimum,
            }
        )
    env.close()
    return results


# ------------------------------------------------------------------ #
# Pure single-algorithm runner (no DASEnv, no checkpoint splitting)   #
# ------------------------------------------------------------------ #


def run_single_algorithm(
    optimizer_class, problem, fe_multiplier: int, n_individuals: int
) -> float:
    """Run one optimizer for the full budget in one uninterrupted call."""
    max_fe = fe_multiplier * problem.dimension
    problem_config = {
        "fitness_function": problem,
        "ndim_problem": problem.dimension,
        "lower_boundary": problem.lower_bounds,
        "upper_boundary": problem.upper_bounds,
    }
    options = {
        "max_function_evaluations": max_fe,
        "target_fe": max_fe,
        "n_individuals": n_individuals,
        "verbose": False,
    }
    optimizer = optimizer_class(problem_config, options)
    result = optimizer.optimize()
    if isinstance(result, tuple):
        result = result[0]
    return float(result.get("best_so_far_y", float("inf")))


def collect_single_results(
    agent_tag: str,
    optimizer_class,
    test_ids: list[str],
    suite,
    fe_multiplier: int,
    n_individuals: int,
    global_optima: dict[str, float],
) -> list[dict]:
    """Run the optimizer independently on every problem in test_ids."""
    results = []
    for problem_id in tqdm(test_ids, desc=f"  {agent_tag}", smoothing=0.0):
        problem = suite.get_problem(problem_id)
        best_y = run_single_algorithm(
            optimizer_class, problem, fe_multiplier, n_individuals
        )
        optimum = global_optima.get(problem_id, 0.0)
        results.append(
            {
                "problem_id": problem_id,
                "agent": agent_tag,
                "best_y": best_y,
                "gap": best_y - optimum,
            }
        )
    return results


# ------------------------------------------------------------------ #
# Oracle: per-problem best / worst across a set of result lists        #
# ------------------------------------------------------------------ #


def compute_oracle(all_results: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    """Compute per-problem oracle-best and oracle-worst from multiple agent runs.

    Parameters
    ----------
    all_results: mapping agent_tag → list of per-problem dicts

    Returns
    -------
    oracle_best, oracle_worst: per-problem dicts annotated with winning agent
    """
    # Index by problem_id
    by_problem: dict[str, list[dict]] = {}
    for tag, records in all_results.items():
        for r in records:
            pid = r["problem_id"]
            by_problem.setdefault(pid, []).append(r)

    oracle_best, oracle_worst = [], []
    for pid, records in by_problem.items():
        best = min(records, key=lambda r: r["gap"])
        worst = max(records, key=lambda r: r["gap"])
        oracle_best.append(
            {
                "problem_id": pid,
                "agent": "oracle-best",
                "best_agent": best["agent"],
                "best_y": best["best_y"],
                "gap": best["gap"],
            }
        )
        oracle_worst.append(
            {
                "problem_id": pid,
                "agent": "oracle-worst",
                "worst_agent": worst["agent"],
                "best_y": worst["best_y"],
                "gap": worst["gap"],
            }
        )

    oracle_best.sort(key=lambda r: r["problem_id"])
    oracle_worst.sort(key=lambda r: r["problem_id"])
    return oracle_best, oracle_worst


# ------------------------------------------------------------------ #
# Summary helpers                                                      #
# ------------------------------------------------------------------ #


def summarise(tag: str, records: list[dict]) -> dict:
    gaps = [r["gap"] for r in records]
    return {
        "agent": tag,
        "n_problems": len(gaps),
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "best_gap": float(np.min(gaps)),
        "worst_gap": float(np.max(gaps)),
    }


def save_results(records: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def print_summary(summaries: list[dict]) -> None:
    width = max(len(s["agent"]) for s in summaries) + 2
    header = (
        f"  {'Agent':<{width}}  {'Mean gap':>12}  {'Median gap':>12}  {'Best gap':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in summaries:
        print(
            f"  {s['agent']:<{width}}  "
            f"{s['mean_gap']:>12.4e}  "
            f"{s['median_gap']:>12.4e}  "
            f"{s['best_gap']:>12.4e}"
        )


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def parse_args():
    p = argparse.ArgumentParser(description="Baseline agents for DAS on BBOB")
    p.add_argument("name", help="Experiment name prefix for result files")
    p.add_argument(
        "--agent",
        default="random",
        help=(
            "Agent to run. One of: "
            "random | fixed:<opt_name> | single:<opt_name> | all. "
            "Examples: --agent fixed:SPSO  --agent single:CMAES  --agent all"
        ),
    )
    p.add_argument(
        "-p",
        "--portfolio",
        nargs="+",
        default=["SPSO", "IPSO", "SPSOL"],
        help="Optimizer portfolio (defines action space for env-based agents)",
    )
    p.add_argument(
        "-m",
        "--mode",
        default="easy",
        choices=["easy", "hard", "LOIO"],
        help="Train/test split strategy (baselines run on the test split)",
    )
    p.add_argument(
        "-d", "--dims", nargs="+", type=int, default=ALL_DIMS, choices=ALL_DIMS
    )
    p.add_argument("-f", "--fe-multiplier", type=int, default=10_000)
    p.add_argument("-s", "--n-checkpoints", type=int, default=10)
    p.add_argument("-x", "--cdb", type=float, default=1.0)
    p.add_argument("-O", "--reward-option", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("-n", "--n-individuals", type=int, default=100)
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
    opt_names = args.portfolio
    cocoex.utilities.MiniPrint()
    suite = cocoex.Suite("bbob", "", "")
    _, test_ids = get_train_test_split(args.mode, args.dims)
    global_optima = load_global_optima()

    cfg = {
        "fe_multiplier": args.fe_multiplier,
        "n_checkpoints": args.n_checkpoints,
        "cdb": args.cdb,
        "reward_option": args.reward_option,
        "n_individuals": args.n_individuals,
    }

    print(f"Portfolio : {opt_names}")
    print(f"Mode      : {args.mode}  ({len(test_ids)} test problems)")
    print(f"Budget    : {args.fe_multiplier}×dim  |  checkpoints={args.n_checkpoints}")

    # Determine which agents to run
    agent_arg = args.agent
    if agent_arg == "all":
        run_tags = (
            ["random"]
            + [f"fixed:{n}" for n in opt_names]
            + [f"single:{n}" for n in opt_names]
        )
    else:
        run_tags = [agent_arg]

    all_results: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    for tag in run_tags:
        result_path = os.path.join(
            "results", f"{args.name}_{tag.replace(':', '_')}.jsonl"
        )

        if os.path.exists(result_path):
            print(f"\n[skip] {tag}  →  {result_path} already exists")
            with open(result_path) as f:
                records = [json.loads(line) for line in f]
            all_results[tag] = records
            summaries.append(summarise(tag, records))
            continue

        print(f"\n{'─' * 50}")
        print(f"Running: {tag}")
        print(f"{'─' * 50}")

        if tag == "random":
            records = collect_env_results(
                tag, random_policy, test_ids, suite, optimizers, cfg, global_optima
            )

        elif tag.startswith("fixed:"):
            opt_name = tag[len("fixed:") :]
            if opt_name not in opt_names:
                raise ValueError(
                    f"Optimizer '{opt_name}' not in portfolio {opt_names}. "
                    "Pass it via -p / --portfolio."
                )
            action = opt_names.index(opt_name)
            records = collect_env_results(
                tag,
                fixed_policy(action),
                test_ids,
                suite,
                optimizers,
                cfg,
                global_optima,
            )

        elif tag.startswith("single:"):
            opt_name = tag[len("single:") :]
            opt_class = get_portfolio([opt_name])[0]
            records = collect_single_results(
                tag,
                opt_class,
                test_ids,
                suite,
                args.fe_multiplier,
                args.n_individuals,
                global_optima,
            )

        else:
            raise ValueError(
                f"Unknown agent '{tag}'. "
                "Use: random | fixed:<name> | single:<name> | all"
            )

        save_results(records, result_path)
        print(f"  Saved → {result_path}")

        all_results[tag] = records
        summaries.append(summarise(tag, records))

    # Oracle (only when multiple fixed runs are available)
    fixed_results = {
        tag: recs
        for tag, recs in all_results.items()
        if tag.startswith("fixed:") or tag.startswith("single:")
    }
    if len(fixed_results) > 1:
        oracle_best, oracle_worst = compute_oracle(fixed_results)

        for oracle_tag, oracle_records in [
            ("oracle-best", oracle_best),
            ("oracle-worst", oracle_worst),
        ]:
            oracle_path = os.path.join("results", f"{args.name}_{oracle_tag}.jsonl")
            save_results(oracle_records, oracle_path)
            all_results[oracle_tag] = oracle_records
            summaries.append(summarise(oracle_tag, oracle_records))
            print(f"  Saved {oracle_tag} → {oracle_path}")

    # Print comparison table
    print(f"\n{'=' * 60}")
    print(f"Summary  ({len(test_ids)} test problems)")
    print(f"{'=' * 60}")
    print_summary(summaries)

    # Aggregate summary file
    if len(summaries) > 1:
        summary_path = os.path.join("results", f"{args.name}_baselines_summary.jsonl")
        with open(summary_path, "w") as f:
            f.write(json.dumps({"agents": summaries}, indent=2) + "\n")
        print(f"\n  Summary → {summary_path}")


if __name__ == "__main__":
    main()
