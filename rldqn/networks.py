from __future__ import annotations

import tensorflow as tf
from typing import Tuple


def build_q_network(
    state_shape: Tuple[int, ...],
    num_actions: int,
    hidden_layers: Tuple[int, ...] = (256, 256),
    dueling: bool = True,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=state_shape, dtype=tf.float32)
    x = inputs
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation="relu", kernel_initializer="he_uniform")(x)

    if dueling:
        value = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)
        value = tf.keras.layers.Dense(1, activation=None)(value)

        advantage = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)
        advantage = tf.keras.layers.Dense(num_actions, activation=None)(advantage)

        advantage_mean = tf.keras.layers.Lambda(lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        q_values = tf.keras.layers.Add()([value, advantage_mean])
    else:
        q_values = tf.keras.layers.Dense(num_actions, activation=None)(x)

    model = tf.keras.Model(inputs=inputs, outputs=q_values, name="q_network")
    return model


