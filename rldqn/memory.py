from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: Tuple[int, ...]) -> None:
        self.capacity = capacity
        self.size = 0
        self.position = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, transition: Transition) -> None:
        idx = self.position
        self.states[idx] = transition.state
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.next_states[idx] = transition.next_state
        self.dones[idx] = float(transition.done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_states": self.next_states[idxs],
            "dones": self.dones[idxs],
            "indices": idxs,
        }
        return batch


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, state_shape: Tuple[int, ...], alpha: float = 0.6, beta: float = 0.4) -> None:
        super().__init__(capacity, state_shape)
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, transition: Transition) -> None:
        super().push(transition)
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[(self.position - 1) % self.capacity] = max_prio

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.size]
        probs = (prios + self.eps) ** self.alpha
        probs = probs / probs.sum()

        idxs = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[idxs]) ** (-self.beta)
        weights = weights / weights.max()

        batch = {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_states": self.next_states[idxs],
            "dones": self.dones[idxs],
            "indices": idxs,
            "weights": weights.astype(np.float32),
        }
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors_abs: np.ndarray) -> None:
        self.priorities[indices] = np.abs(td_errors_abs) + self.eps


