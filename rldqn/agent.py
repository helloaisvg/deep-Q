from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any

from .networks import build_q_network


class DQNAgent:
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        hidden_layers: Tuple[int, ...] = (256, 256),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        target_update_tau: float = 1.0,
        target_update_period: int = 1000,
        double_dqn: bool = True,
        dueling: bool = True,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.target_update_tau = target_update_tau
        self.target_update_period = target_update_period

        self.q_online = build_q_network(state_shape, num_actions, hidden_layers, dueling)
        self.q_target = build_q_network(state_shape, num_actions, hidden_layers, dueling)
        self.q_target.set_weights(self.q_online.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.Huber()

        # epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_step = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.train_step / max(1, self.epsilon_decay_steps))
        return float(self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start))

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.rand() < self.epsilon():
            return int(np.random.randint(self.num_actions))
        state = np.asarray(state, dtype=np.float32)[None, ...]
        q_values = self.q_online(state, training=False).numpy()[0]
        return int(np.argmax(q_values))

    @tf.function(jit_compile=False)
    def _train_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
        weights: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            q_values = self.q_online(states, training=True)
            action_q = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)

            # target computation
            next_q_online = self.q_online(next_states, training=False)
            next_actions = tf.argmax(next_q_online, axis=1)
            next_q_target = self.q_target(next_states, training=False)

            if self.double_dqn:
                next_q = tf.reduce_sum(
                    next_q_target * tf.one_hot(next_actions, self.num_actions), axis=1
                )
            else:
                next_q = tf.reduce_max(next_q_target, axis=1)

            target_q = rewards + (1.0 - dones) * self.gamma * next_q

            td_errors = target_q - action_q
            loss = tf.reduce_mean(weights * self.loss_fn(target_q, action_q))

        grads = tape.gradient(loss, self.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_online.trainable_variables))
        return loss, tf.abs(td_errors)

    def train_on_batch(
        self,
        batch: Dict[str, np.ndarray],
        weights: np.ndarray | None = None,
    ) -> Tuple[float, np.ndarray]:
        states = tf.convert_to_tensor(batch["states"], dtype=tf.float32)
        actions = tf.convert_to_tensor(batch["actions"], dtype=tf.int32)
        rewards = tf.convert_to_tensor(batch["rewards"], dtype=tf.float32)
        next_states = tf.convert_to_tensor(batch["next_states"], dtype=tf.float32)
        dones = tf.convert_to_tensor(batch["dones"], dtype=tf.float32)
        if weights is None:
            weights = np.ones((states.shape[0],), dtype=np.float32)
        weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)

        loss, td_abs = self._train_step(states, actions, rewards, next_states, dones, weights_tf)

        self.train_step += 1
        if self.train_step % self.target_update_period == 0:
            self._update_target()

        return float(loss.numpy()), td_abs.numpy()

    def _update_target(self) -> None:
        if self.target_update_tau >= 1.0:
            self.q_target.set_weights(self.q_online.get_weights())
            return
        online_weights = self.q_online.get_weights()
        target_weights = self.q_target.get_weights()
        new_weights = [
            self.target_update_tau * ow + (1.0 - self.target_update_tau) * tw
            for ow, tw in zip(online_weights, target_weights)
        ]
        self.q_target.set_weights(new_weights)

    def save(self, path: str) -> None:
        self.q_online.save(path)

    def load(self, path: str) -> None:
        self.q_online = tf.keras.models.load_model(path)
        self.q_target = tf.keras.models.clone_model(self.q_online)
        self.q_target.set_weights(self.q_online.get_weights())


