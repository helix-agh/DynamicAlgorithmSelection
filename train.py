"""Unified training entry point for all DAS agents.

Usage
-----
    python train.py ppo     <name> [options]
    python train.py rl-das  <name> [options]
    python train.py exp-das <name> [options]

ppo outputs
-----------
    models/<name>.zip
    models/<name>_vecnorm.pkl
    results/<name>_eval.jsonl          (with --eval)

    CV mode (--cv-mode LOIO|LOPO):
    models/<name>_cv_<fold>.zip  +  _vecnorm.pkl
    results/<name>_cv_<fold>.jsonl
    results/<name>_cv_summary.jsonl

rl-das outputs
--------------
    models/<name>_final.pt
    models/<name>_epoch<N>.pt
    models/<name>_train_log.jsonl
    results/<name>_eval.jsonl

exp-das outputs
---------------
    models/<name>_best.pt  /  _final.pt  /  _ep<N>.pt
    models/<name>_train_log.jsonl
    results/<name>_eval.jsonl
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from das.env.bbob_splits import ALL_DIMS, get_train_test_split, get_cv_folds
from das.env.das_env import DASEnv
from das.optimizers.portfolio import get_portfolio
from das.utils import set_seed

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
# Shared helpers                                                       #
# ------------------------------------------------------------------ #


def load_global_optima(path: str = "bbob_optima.jsonl") -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v for line in f for k, v in json.loads(line).items()}


def make_das_env(problem_ids: list[str], optimizers: list, cfg: dict):
    """Return a zero-argument factory for use with SB3's make_vec_env."""

    def _init():
        import cocoex as _cx

        _cx.utilities.MiniPrint()
        _suite = _cx.Suite("bbob", "", "")
        return DASEnv(
            problem_ids=problem_ids,
            suite=_suite,
            optimizers=optimizers,
            fe_multiplier=cfg["fe_multiplier"],
            n_checkpoints=cfg["n_checkpoints"],
            checkpoint_division_base=cfg["cdb"],
            reward_option=cfg["reward_option"],
            n_individuals=cfg["n_individuals"],
            seed=cfg.get("seed"),
        )

    return _init


# ------------------------------------------------------------------ #
# PPO agent                                                            #
# ------------------------------------------------------------------ #


def _ppo_eval_loop(
    model, eval_env, problem_ids: list[str], global_optima: dict, desc: str = "eval"
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
        optimum = global_optima.get(problem_id, 0.0)
        results.append(
            {"problem_id": problem_id, "best_y": best_y, "gap": best_y - optimum}
        )
    return results


def _ppo_train_single(
    name: str,
    train_ids: list[str],
    test_ids: list[str],
    optimizers: list,
    cfg: dict,
    args,
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
    from das.training.callbacks import WandbCallback, DASEvalCallback

    model_path = os.path.join("models", name)
    vecnorm_path = model_path + "_vecnorm.pkl"

    steps_per_epoch = len(train_ids) * cfg["n_checkpoints"]
    total_timesteps = args.n_epochs * steps_per_epoch
    eval_freq = steps_per_epoch

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
            eval_freq=eval_freq,
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


def run_ppo(args) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

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

    if args.cv_mode:
        _run_ppo_cv(args, optimizers, cfg)
        return

    train_ids, test_ids = get_train_test_split(args.mode, args.dims)
    print(
        f"Mode      : {args.mode}  ({len(train_ids)} train / {len(test_ids)} test problems)"
    )

    _ppo_train_single(args.name, train_ids, test_ids, optimizers, cfg, args)

    if args.eval:
        print("\nRunning post-training evaluation …")
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
        results = _ppo_eval_loop(model, eval_env, test_ids, global_optima)
        eval_env.close()
        out_path = os.path.join("results", f"{args.name}_eval.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        gaps = [r["gap"] for r in results]
        print(f"  Mean gap   : {np.mean(gaps):.4e}")
        print(f"  Median gap : {np.median(gaps):.4e}")
        print(f"  Results    : {out_path}")


def _run_ppo_cv(args, optimizers: list, cfg: dict) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    global_optima = load_global_optima()
    all_folds = get_cv_folds(args.cv_mode, args.dims, seed=args.seed)
    fold_indices = args.folds if args.folds is not None else list(range(len(all_folds)))

    print(
        f"CV mode    : {args.cv_mode}  ({len(fold_indices)}/{len(all_folds)} folds selected)"
    )
    print(f"Epochs/fold: {args.n_epochs}")

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
            f"({len(train_ids)} train / {len(test_ids)} test)"
        )
        print(f"{'=' * 60}")

        if os.path.exists(model_path + ".zip"):
            print(f"  [skip training] {model_path}.zip already exists")
            model = PPO.load(model_path)
        else:
            _ppo_train_single(fold_name, train_ids, test_ids, optimizers, cfg, args)
            model = PPO.load(model_path)

        if os.path.exists(result_path):
            print(f"  [skip evaluation] {result_path} already exists")
            with open(result_path) as fh:
                fold_results = [json.loads(line) for line in fh]
        else:
            eval_env = make_vec_env(
                make_das_env(test_ids, optimizers, cfg), n_envs=1, seed=args.seed
            )
            if os.path.exists(vecnorm_path):
                eval_env = VecNormalize.load(vecnorm_path, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
            fold_results = _ppo_eval_loop(
                model, eval_env, test_ids, global_optima, desc=f"eval {fold_tag}"
            )
            for r in fold_results:
                r["fold"] = fold_tag
            eval_env.close()
            with open(result_path, "w") as fh:
                for r in fold_results:
                    fh.write(json.dumps(r) + "\n")
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
    with open(summary_path, "w") as fh:
        fh.write(json.dumps(overall, indent=2) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Cross-validation complete  ({args.cv_mode})")
    print(f"  Folds run          : {len(fold_summaries)}")
    if all_gaps:
        print(f"  Overall mean gap   : {overall['overall_mean_gap']:.4e}")
        print(f"  Overall median gap : {overall['overall_median_gap']:.4e}")
    print(f"  Summary            : {summary_path}")


# ------------------------------------------------------------------ #
# RL-DAS agent                                                         #
# ------------------------------------------------------------------ #


def run_rl_das(args) -> None:
    import cocoex as cx
    from agents.rl_das import RLDASEnv, PPOAgent, train, evaluate

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

    print(
        f"RL-DAS  |  dim={args.dim}  |  portfolio={args.portfolio}"
        f"  |  obs_dim={train_env.observation_space.shape[0]}"
        f"  |  k_epoch={args.k_epoch}"
    )

    agent = PPOAgent(
        dim=args.dim, n_opt=len(optimizers), lr=args.lr, device=args.device
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
        test_results = evaluate(test_env, agent, n_episodes=len(test_env._problem_ids))
        mean_best_y = float(np.mean([r["best_y"] for r in test_results]))
        print(f"Test mean best_y = {mean_best_y:.6e}")
        eval_path = os.path.join("results", f"{args.name}_eval.jsonl")
        with open(eval_path, "w") as f:
            for r in test_results:
                f.write(json.dumps(r) + "\n")
        print(f"Results saved to {eval_path}")


# ------------------------------------------------------------------ #
# Exponential-DAS agent                                                #
# ------------------------------------------------------------------ #


def run_exp_das(args) -> None:
    import cocoex as cx
    from agents.exponential_das import ExpDASAgent, train, evaluate
    from das.env.observation import observation_dim

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
        with open(eval_path, "w") as f:
            for r in test_results:
                f.write(json.dumps(r) + "\n")
        print(f"Results saved to {eval_path}")


# ------------------------------------------------------------------ #
# Argument parsing                                                     #
# ------------------------------------------------------------------ #


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("name", help="Experiment name (used for output file names)")
    p.add_argument(
        "-p",
        "--portfolio",
        nargs="+",
        default=["SPSO", "IPSO", "SPSOL"],
        help="Sub-optimizer names from the portfolio",
    )
    p.add_argument(
        "--mode",
        choices=["easy", "hard", "random"],
        default="easy",
        help="Train/test split strategy",
    )
    p.add_argument(
        "--fe-multiplier",
        type=int,
        default=10_000,
        help="Budget = fe_multiplier × dimension",
    )
    p.add_argument(
        "--n-checkpoints",
        type=int,
        default=10,
        help="Optimizer-selection steps per episode",
    )
    p.add_argument("--n-individuals", type=int, default=100, help="Population size")
    p.add_argument("--seed", type=int, default=42)


def _parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description="Train a DAS agent.  Choose an agent with a sub-command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(
        dest="agent", required=True, metavar="{ppo,rl-das,exp-das}"
    )

    # ---- PPO --------------------------------------------------------
    ppo = sub.add_parser(
        "ppo",
        help="SB3 PPO with VecNormalize (multi-dim, CV support)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_args(ppo)
    ppo.add_argument(
        "-d",
        "--dims",
        nargs="+",
        type=int,
        default=ALL_DIMS,
        choices=ALL_DIMS,
        help="Problem dimensions",
    )
    ppo.add_argument(
        "-x", "--cdb", type=float, default=1.0, help="Checkpoint division base"
    )
    ppo.add_argument(
        "-O",
        "--reward-option",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Reward shaping option",
    )
    ppo.add_argument(
        "-E",
        "--n-epochs",
        type=int,
        default=1,
        help="Passes over the full training set. "
        "total_timesteps = n_epochs × |train_ids| × n_checkpoints",
    )
    ppo.add_argument(
        "-j", "--n-envs", type=int, default=1, help="Parallel training envs"
    )
    ppo.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    ppo.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation on the test set immediately after training",
    )
    ppo.add_argument(
        "--cv-mode",
        default=None,
        choices=["LOIO", "LOPO"],
        help="3-fold CV: LOIO holds out 5 of 15 instances per fold; LOPO holds out 8 of 24 functions per fold",
    )
    ppo.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Zero-based fold indices to run (CV mode only, default: all)",
    )

    # ---- RL-DAS -----------------------------------------------------
    rl = sub.add_parser(
        "rl-das",
        help="Custom RL-DAS: single-dimension, pure-PyTorch PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_args(rl)
    rl.add_argument(
        "--dim", type=int, default=10, help="Problem dimension (agent is dim-specific)"
    )
    rl.add_argument("--n-epochs", type=int, default=500, help="Training epochs")
    rl.add_argument(
        "--k-epoch",
        type=int,
        default=None,
        help="PPO gradient steps per episode (default: int(0.3 × n_checkpoints))",
    )
    rl.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    rl.add_argument(
        "--eval-interval", type=int, default=5, help="Evaluate every N epochs"
    )
    rl.add_argument(
        "--save-interval", type=int, default=50, help="Checkpoint every N epochs"
    )
    rl.add_argument("--device", default="cpu", help="PyTorch device")
    rl.add_argument(
        "--no-eval", dest="eval", action="store_false", help="Skip final evaluation"
    )
    rl.set_defaults(eval=True)

    # ---- Exp-DAS ----------------------------------------------------
    exp = sub.add_parser(
        "exp-das",
        help="Exponential-DAS: custom PPO with exponential checkpoint spacing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_args(exp)
    exp.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[2, 5, 10],
        help="Problem dimensions",
    )
    exp.add_argument(
        "--cdb",
        type=float,
        default=2.0,
        help="Checkpoint division base (>1 = exponential)",
    )
    exp.add_argument(
        "--reward-option",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Reward shaping option",
    )
    exp.add_argument(
        "--buffer-capacity",
        type=int,
        default=None,
        help="PPO rollout buffer size in steps (default: 16 × n_checkpoints)",
    )
    exp.add_argument(
        "--total-episodes", type=int, default=5000, help="Total training episodes"
    )
    exp.add_argument(
        "--eval-interval", type=int, default=100, help="Evaluate every N episodes"
    )
    exp.add_argument(
        "--save-interval", type=int, default=500, help="Checkpoint every N episodes"
    )
    exp.add_argument("--actor-lr", type=float, default=3e-5, help="Actor learning rate")
    exp.add_argument(
        "--critic-lr", type=float, default=1e-5, help="Critic learning rate"
    )
    exp.add_argument(
        "--ppo-epochs", type=int, default=6, help="PPO gradient epochs per update"
    )
    exp.add_argument("--device", default="cpu", help="PyTorch device")
    exp.add_argument(
        "--no-eval", dest="eval", action="store_false", help="Skip final evaluation"
    )
    exp.set_defaults(eval=True)

    return root.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    dispatch = {"ppo": run_ppo, "rl-das": run_rl_das, "exp-das": run_exp_das}
    dispatch[args.agent](args)


if __name__ == "__main__":
    main()
