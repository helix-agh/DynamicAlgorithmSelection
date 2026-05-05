"""Training and evaluation routines for Exponential-DAS.

Training loop
-------------
  The buffer fills across episodes (not restarted between episodes).
  When the buffer reaches capacity the agent runs PPO and clears it.
  The ``step_idx`` counter tells the reward normaliser which checkpoint slot
  this transition belongs to (so rewards at early checkpoints are compared
  only to other early-checkpoint rewards, matching StepwiseRewardNormalizer).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from agents.exponential_das.agent import ExpDASAgent
from das.env.das_env import DASEnv


def train(
    train_env: DASEnv,
    test_env: DASEnv,
    agent: ExpDASAgent,
    total_episodes: int = 5000,
    eval_interval: int = 100,
    save_interval: int = 500,
    save_dir: str = "models",
    name: str = "exp_das",
) -> list[dict]:
    """Train the Exponential-DAS agent.

    Parameters
    ----------
    train_env:
        DASEnv (ideally with ``checkpoint_division_base > 1`` for exponential
        checkpoint spacing).
    test_env:
        Separate DASEnv for periodic evaluation.
    agent:
        The ExpDASAgent to train in-place.
    total_episodes:
        Number of training episodes.
    eval_interval:
        Evaluate on test set every this many episodes.
    save_interval:
        Save a checkpoint every this many episodes.
    save_dir:
        Directory for checkpoints and logs.
    name:
        Experiment name prefix for saved files.

    Returns
    -------
    List of per-episode log entries.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    episode_rewards: list[float] = []
    best_test_reward = -np.inf

    for ep in range(1, total_episodes + 1):
        obs, info = train_env.reset()
        done = False
        step_idx = 0
        ep_reward = 0.0

        while not done:
            action, log_prob, value = agent.select_action(obs, step_idx)
            next_obs, reward, terminated, truncated, step_info = train_env.step(action)
            done = terminated or truncated

            # Reward normalisation (update only during warmup)
            normed_reward = agent.rew_norm.normalize(
                reward, step_idx, update=not agent.buffer.warmed_up
            )
            ep_reward += reward

            agent.buffer.add(obs, action, log_prob, value, normed_reward, done)

            if agent.buffer.is_full():
                # Bootstrap value for the last state
                if not done:
                    n_obs = agent.obs_norm.normalize(next_obs, update=False)
                    n_obs_t = torch.tensor(
                        n_obs, dtype=torch.float32, device=agent.device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        bootstrap = float(agent.critic(n_obs_t).item())
                else:
                    bootstrap = 0.0

                agent.ppo_update(bootstrap_value=bootstrap)
                agent.buffer.clear()

            obs = next_obs
            step_idx += 1

        episode_rewards.append(ep_reward)

        entry: dict = {
            "episode": ep,
            "reward": ep_reward,
            "entropy_coef": round(agent.entropy_coef, 6),
            "lr": agent.current_lr,
        }

        if ep % eval_interval == 0:
            test_results = evaluate(test_env, agent, n_episodes=20)
            mean_test_r = float(np.mean([r["reward"] for r in test_results]))
            entry["mean_test_reward"] = mean_test_r
            mean_train_r = float(np.mean(episode_rewards[-eval_interval:]))
            print(
                f"Ep {ep:5d}/{total_episodes}"
                f"  train={mean_train_r:.4f}"
                f"  test={mean_test_r:.4f}"
                f"  entropy={agent.entropy_coef:.4f}"
                f"  lr={agent.current_lr:.2e}"
                f"  kl={agent.last_kl:.4f}"
            )
            if mean_test_r > best_test_reward:
                best_test_reward = mean_test_r
                agent.save(os.path.join(save_dir, f"{name}_best.pt"))

        if ep % save_interval == 0:
            ckpt = os.path.join(save_dir, f"{name}_ep{ep}.pt")
            agent.save(ckpt)

        log.append(entry)

    agent.save(os.path.join(save_dir, f"{name}_final.pt"))

    with open(os.path.join(save_dir, f"{name}_train_log.jsonl"), "w") as f:
        for e in log:
            f.write(json.dumps(e) + "\n")

    return log


def evaluate(
    env: DASEnv,
    agent: ExpDASAgent,
    n_episodes: int = 20,
) -> list[dict]:
    """Run the agent deterministically and return per-episode results."""
    results = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        results.append(
            {
                "problem_id": info.get("problem_id", ""),
                "reward": total_reward,
                "best_y": step_info.get("best_y", float("inf")),
                "n_fe": step_info.get("n_fe", 0),
            }
        )
    return results
