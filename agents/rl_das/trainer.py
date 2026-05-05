"""Training and evaluation routines for the RL-DAS agent.

Training loop (following Guo et al. 2024):
  For each epoch:
    For each problem in the training set:
      Run one episode, collecting (obs, action, log_prob, reward, value, done)
      At episode end: call agent.learn(k_epoch)
    Every eval_interval epochs: evaluate on test set and log
    Every save_interval epochs: checkpoint model
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from agents.rl_das.agent import PPOAgent
from agents.rl_das.env import RLDASEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_episode(env: RLDASEnv, agent: PPOAgent, deterministic: bool = False) -> dict:
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if deterministic:
            action = agent.predict(obs)
            log_prob = 0.0
            value = 0.0
        else:
            action, log_prob, value = agent.select_action(obs)

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if not deterministic:
            agent.rollout.add(obs, action, log_prob, reward, value, done)

        obs = next_obs

    return {
        "total_reward": total_reward,
        "best_y": step_info.get("best_y", float("inf")),
        "n_fe": step_info.get("n_fe", 0),
        "problem_id": info.get("problem_id", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    train_env: RLDASEnv,
    test_env: RLDASEnv,
    agent: PPOAgent,
    n_epochs: int = 500,
    k_epoch: int = 24,
    eval_interval: int = 5,
    save_interval: int = 10,
    save_dir: str = "models",
    name: str = "rl_das",
) -> list[dict]:
    """Train the RL-DAS agent.

    Parameters
    ----------
    train_env:
        Environment cycled over training problems.
    test_env:
        Environment cycled over test problems (used for periodic evaluation).
    agent:
        The PPOAgent to train in-place.
    n_epochs:
        Total number of training epochs.
    k_epoch:
        PPO gradient steps per episode (paper default: int(0.3 * max_fes / period)).
    eval_interval:
        Evaluate on test set every this many epochs.
    save_interval:
        Checkpoint model every this many epochs.
    save_dir:
        Directory to write checkpoints and result logs.
    name:
        Base name for saved files.

    Returns
    -------
    Log of per-epoch statistics.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    n_train = len(train_env._problem_ids)

    for epoch in range(1, n_epochs + 1):
        epoch_rewards = []
        epoch_start = time.time()

        for _ in range(n_train):
            ep = _run_episode(train_env, agent, deterministic=False)
            epoch_rewards.append(ep["total_reward"])

            diagnostics = agent.learn(k_epoch)
            agent.rollout.clear()

        mean_train_reward = float(np.mean(epoch_rewards))
        entry: dict = {
            "epoch": epoch,
            "mean_train_reward": mean_train_reward,
            "elapsed_s": round(time.time() - epoch_start, 2),
        }

        if epoch % eval_interval == 0:
            test_results = evaluate(
                test_env, agent, n_episodes=len(test_env._problem_ids)
            )
            entry["mean_test_reward"] = float(
                np.mean([r["total_reward"] for r in test_results])
            )
            entry["mean_test_best_y"] = float(
                np.mean([r["best_y"] for r in test_results])
            )
            print(
                f"Epoch {epoch:4d}/{n_epochs}"
                f"  train_r={mean_train_reward:.4f}"
                f"  test_r={entry['mean_test_reward']:.4f}"
                f"  test_best_y={entry['mean_test_best_y']:.4e}"
                f"  ({entry['elapsed_s']:.1f}s)"
            )
        else:
            print(
                f"Epoch {epoch:4d}/{n_epochs}"
                f"  train_r={mean_train_reward:.4f}"
                f"  ({entry['elapsed_s']:.1f}s)"
            )

        log.append(entry)

        if epoch % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"{name}_epoch{epoch}.pt")
            agent.save(ckpt_path)
            print(f"  -> saved {ckpt_path}")

    # Final checkpoint
    agent.save(os.path.join(save_dir, f"{name}_final.pt"))

    # Write log
    log_path = os.path.join(save_dir, f"{name}_train_log.jsonl")
    with open(log_path, "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")

    return log


def evaluate(
    env: RLDASEnv,
    agent: PPOAgent,
    n_episodes: int | None = None,
) -> list[dict]:
    """Run the agent deterministically and return per-episode results.

    Parameters
    ----------
    env:
        The evaluation environment.
    agent:
        Trained agent (greedy policy).
    n_episodes:
        How many episodes to run.  Defaults to the full problem list.

    Returns
    -------
    List of dicts with keys: problem_id, total_reward, best_y, n_fe.
    """
    if n_episodes is None:
        n_episodes = len(env._problem_ids)

    results = []
    for _ in range(n_episodes):
        ep = _run_episode(env, agent, deterministic=True)
        results.append(ep)

    return results
