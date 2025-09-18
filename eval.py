from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from rldqn.agent import DQNAgent


def load_agent(model_dir: str, env: gym.Env) -> DQNAgent:
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DQNAgent(
        state_shape=obs_shape,
        num_actions=num_actions,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
    )
    agent.load(model_dir)
    return agent


def evaluate(env_id: str, model_dir: str, episodes: int = 20, seed: int | None = 42) -> Dict[str, Any]:
    env = gym.make(env_id)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
    agent = load_agent(model_dir, env)

    returns: List[float] = []
    lengths: List[int] = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total = 0.0
        steps = 0
        while not (done or truncated):
            action = agent.act(state, explore=False)
            state, reward, done, truncated, _ = env.step(action)
            total += reward
            steps += 1
        returns.append(total)
        lengths.append(steps)
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "returns": returns,
        "lengths": lengths,
    }


def compare_runs(log_files: List[str], labels: List[str], out_png: str | None = None) -> None:
    plt.figure(figsize=(8, 5))
    for path, label in zip(log_files, labels):
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        steps = data[:, 0]
        rewards = data[:, 1]
        plt.plot(steps, rewards, label=label, linewidth=1.5)
    plt.xlabel("steps")
    plt.ylabel("episode_reward")
    plt.title("Training curves comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent or compare runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--env", required=True, help="Gym env id")
    p_eval.add_argument("--model", required=True, help="Saved model directory")
    p_eval.add_argument("--episodes", type=int, default=20)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--out", type=str, default=None, help="Optional JSON output path")

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--logs", nargs="+", required=True, help="CSV files with step,reward")
    p_cmp.add_argument("--labels", nargs="+", required=True, help="labels for each curve")
    p_cmp.add_argument("--out_png", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "eval":
        res = evaluate(args.env, args.model, args.episodes, args.seed)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
    else:
        compare_runs(args.logs, args.labels, args.out_png)


if __name__ == "__main__":
    main()


