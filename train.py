from __future__ import annotations

import os
import time
import argparse
import yaml
import gym
import numpy as np
import tensorflow as tf

from rldqn.agent import DQNAgent
from rldqn.memory import ReplayBuffer, PrioritizedReplayBuffer, Transition


def make_env(env_id: str, seed: int | None = None) -> gym.Env:
    env = gym.make(env_id)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
    return env


def load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    env_id = config.get("env_id", "LunarLander-v2")
    seed = config.get("seed", 42)
    total_steps = int(config.get("total_steps", 200_000))
    start_learning = int(config.get("start_learning", 10_000))
    batch_size = int(config.get("batch_size", 64))
    buffer_size = int(config.get("buffer_size", 100_000))
    use_per = bool(config.get("use_per", True))
    alpha = float(config.get("per_alpha", 0.6))
    beta = float(config.get("per_beta", 0.4))
    gamma = float(config.get("gamma", 0.99))
    lr = float(config.get("learning_rate", 1e-3))
    double_dqn = bool(config.get("double_dqn", True))
    dueling = bool(config.get("dueling", True))
    target_update_period = int(config.get("target_update_period", 1000))
    target_update_tau = float(config.get("target_update_tau", 1.0))
    epsilon = config.get("epsilon", {"start": 1.0, "end": 0.05, "decay_steps": 100_000})
    hidden_layers = tuple(config.get("hidden_layers", [256, 256]))
    logdir = config.get("logdir", "runs")
    run_name = config.get("run_name", time.strftime("%Y%m%d-%H%M%S"))

    env = make_env(env_id, seed)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = DQNAgent(
        state_shape=obs_shape,
        num_actions=num_actions,
        hidden_layers=hidden_layers,
        learning_rate=lr,
        gamma=gamma,
        epsilon_start=float(epsilon.get("start", 1.0)),
        epsilon_end=float(epsilon.get("end", 0.05)),
        epsilon_decay_steps=int(epsilon.get("decay_steps", 100_000)),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        double_dqn=double_dqn,
        dueling=dueling,
        seed=seed,
    )

    if use_per:
        buffer = PrioritizedReplayBuffer(buffer_size, obs_shape, alpha=alpha, beta=beta)
    else:
        buffer = ReplayBuffer(buffer_size, obs_shape)

    summary_dir = os.path.join(logdir, f"{env_id}_{run_name}")
    writer = tf.summary.create_file_writer(summary_dir)
    csv_path = os.path.join(summary_dir, "episode_rewards.csv")
    os.makedirs(summary_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,episode_reward\n")

    state, _ = env.reset()
    episode_reward = 0.0
    episode_len = 0
    episode = 0

    for step in range(1, total_steps + 1):
        action = agent.act(state, explore=True)
        next_state, reward, done, truncated, _ = env.step(action)
        real_done = done or truncated
        buffer.push(Transition(state, action, reward, next_state, real_done))

        state = next_state
        episode_reward += reward
        episode_len += 1

        if real_done:
            with writer.as_default():
                tf.summary.scalar("episode_reward", episode_reward, step=step)
                tf.summary.scalar("episode_length", episode_len, step=step)
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{episode_reward}\n")
            state, _ = env.reset()
            episode_reward = 0.0
            episode_len = 0
            episode += 1

        if step >= start_learning and buffer.can_sample(batch_size):
            batch = buffer.sample(batch_size)
            weights = batch.get("weights")
            loss, td_abs = agent.train_on_batch(batch, weights)
            if isinstance(buffer, PrioritizedReplayBuffer):
                buffer.update_priorities(batch["indices"], td_abs)

            if step % 100 == 0:
                with writer.as_default():
                    tf.summary.scalar("train/loss", loss, step=step)
                    tf.summary.scalar("train/epsilon", agent.epsilon(), step=step)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN/Double+Dueling on Gym envs")
    parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()


