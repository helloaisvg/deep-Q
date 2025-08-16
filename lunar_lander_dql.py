#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-Learning for Lunar Lander
本项目实现了带有经验回放的Deep Q-Learning算法，用于训练智能体在LunarLander-v2环境中安全着陆
"""

import time
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
import gym
import matplotlib.pyplot as plt
import pickle
import os

# 设置随机种子
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# 超参数
MEMORY_SIZE = 100_000
GAMMA = 0.995
ALPHA = 1e-3
NUM_STEPS_FOR_UPDATE = 4
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 100

class MemoryBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size=BATCH_SIZE):
        batch_size = min(batch_size, len(self.buffer))
        experiences = np.random.choice(self.buffer, batch_size, replace=False)
        return experiences
    
    def __len__(self):
        return len(self.buffer)

def create_q_network(state_size, num_actions):
    """创建Q网络"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(state_size,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA), loss='mse')
    return model

def update_target_network(q_network, target_q_network):
    """更新目标网络权重"""
    for target_weights, q_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(q_weights)

def get_action(q_values, epsilon):
    """使用ε-贪婪策略选择动作"""
    if np.random.random() < epsilon:
        # 随机探索
        action = np.random.randint(0, q_values.shape[1])
    else:
        # 贪婪选择
        action = tf.argmax(q_values[0]).numpy()
    
    return action

def get_new_eps(epsilon):
    """更新ε值"""
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    return epsilon

def check_update_conditions(t, num_steps_upd, memory_buffer):
    """检查是否应该更新网络"""
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > 1000:
        return True
    return False

def get_experiences(memory_buffer):
    """从经验回放缓冲区中随机采样经验"""
    experiences = memory_buffer.sample()
    states = tf.cast([e[0] for e in experiences], tf.float32)
    actions = tf.cast([e[1] for e in experiences], tf.int32)
    rewards = tf.cast([e[2] for e in experiences], tf.float32)
    next_states = tf.cast([e[3] for e in experiences], tf.float32)
    done_vals = tf.cast([e[4] for e in experiences], tf.float32)
    
    return (states, actions, rewards, next_states, done_vals)

def compute_loss(experiences, gamma, q_network, target_q_network):
    """计算损失函数"""
    states, actions, rewards, next_states, done_vals = experiences
    
    # 计算目标Q值
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    y_targets = rewards + (1 - done_vals) * gamma * max_qsa
    
    # 计算当前Q值
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), actions], axis=1))
    
    # 计算MSE损失
    loss = tf.keras.losses.MSE(y_targets, q_values)
    return loss

@tf.function
def agent_learn(experiences, gamma, q_network, target_q_network, optimizer):
    """智能体学习函数"""
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    
    # 计算梯度并应用
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    
    return loss

def plot_history(total_point_history, num_p_av=100):
    """绘制训练历史"""
    # 计算移动平均
    av_latest_points = []
    for idx in range(num_p_av, len(total_point_history) + 1):
        av_latest_points.append(np.mean(total_point_history[idx - num_p_av:idx]))
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制总点数历史
    ax1.plot(total_point_history)
    ax1.set_title('训练奖励历史')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('总奖励')
    ax1.grid(True)
    
    # 绘制移动平均
    ax2.plot(av_latest_points)
    ax2.set_title(f'最后{num_p_av}个episodes的平均奖励')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('平均奖励')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_agent(env, q_network, num_episodes=10):
    """评估智能体性能"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_qn = np.expand_dims(state, axis=0)
            q_values = q_network(state_qn)
            action = get_action(q_values, 0.0)  # 使用贪婪策略
            
            state, reward, done, info = env.step(action)
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"评估结果 ({num_episodes} episodes):")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"最小奖励: {np.min(total_rewards):.2f}")
    print(f"最大奖励: {np.max(total_rewards):.2f}")
    
    return avg_reward, std_reward

def train_agent(env, q_network, target_q_network, memory_buffer, num_episodes=2000, max_steps_per_episode=1000):
    """训练智能体"""
    
    total_point_history = []
    num_p_av = 100
    epsilon = EPSILON_START
    
    print(f"开始训练，目标episodes: {num_episodes}")
    print(f"解决阈值: 200 (平均奖励超过{num_p_av}个episodes)")
    
    # 设置目标网络权重等于Q网络权重
    target_q_network.set_weights(q_network.get_weights())
    
    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()
        total_points = 0
        
        for step in range(max_steps_per_episode):
            # 选择动作
            state_qn = np.expand_dims(state, axis=0)
            q_values = q_network(state_qn)
            action = get_action(q_values, epsilon)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            experience = (state, action, reward, next_state, done)
            memory_buffer.add(experience)
            
            # 检查是否应该更新网络
            update = check_update_conditions(step, NUM_STEPS_FOR_UPDATE, memory_buffer)
            
            if update:
                # 采样经验并训练
                experiences = get_experiences(memory_buffer)
                agent_learn(experiences, GAMMA, q_network, target_q_network, q_network.optimizer)
            
            state = next_state.copy()
            total_points += reward
            
            if done:
                break
        
        # 更新目标网络
        if episode % TARGET_UPDATE_FREQ == 0:
            update_target_network(q_network, target_q_network)
        
        # 衰减探索率
        epsilon = get_new_eps(epsilon)
        
        # 记录历史
        total_point_history.append(total_points)
        
        # 打印进度
        if episode % 100 == 0:
            av_latest_points = np.mean(total_point_history[-num_p_av:])
            print(f"Episode {episode:4d} | 奖励: {total_points:6.1f} | 平均奖励: {av_latest_points:6.1f} | Epsilon: {epsilon:.3f}")
        
        # 检查是否解决环境
        if len(total_point_history) >= num_p_av:
            recent_avg = np.mean(total_point_history[-num_p_av:])
            if recent_avg >= 200.0:
                print(f"\n🎉 环境已解决！Episode {episode} 时平均奖励达到 {recent_avg:.1f}")
                break
    
    print(f"\n训练完成！总共训练了 {len(total_point_history)} 个episodes")
    
    return total_point_history

def main():
    """主函数"""
    print("Deep Q-Learning for Lunar Lander")
    print("=" * 50)
    
    # 创建环境
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    print(f"环境: {env.unwrapped.spec.id}")
    print(f"状态空间维度: {state_size}")
    print(f"动作空间大小: {num_actions}")
    
    # 创建网络
    q_network = create_q_network(state_size, num_actions)
    target_q_network = create_q_network(state_size, num_actions)
    
    print(f"\nQ网络参数数量: {q_network.count_params():,}")
    print("开始训练...")
    
    # 创建经验回放缓冲区
    memory_buffer = MemoryBuffer(MEMORY_SIZE)
    
    # 训练智能体
    start_time = time.time()
    total_point_history = train_agent(env, q_network, target_q_network, memory_buffer, num_episodes=2000)
    training_time = time.time() - start_time
    
    print(f"\n训练完成！总用时: {training_time/60:.1f} 分钟")
    
    # 绘制结果
    plot_history(total_point_history)
    
    # 评估智能体
    print("\n评估训练好的智能体...")
    avg_reward, std_reward = evaluate_agent(env, q_network, num_episodes=10)
    
    # 保存模型
    model_filename = 'lunar_lander_dql_model.h5'
    q_network.save(model_filename)
    print(f"\n模型已保存为: {model_filename}")
    
    # 保存训练历史
    history_filename = 'training_history.pkl'
    with open(history_filename, 'wb') as f:
        pickle.dump({
            'total_point_history': total_point_history
        }, f)
    print(f"训练历史已保存为: {history_filename}")
    
    # 计算最终性能
    final_avg_reward = np.mean(total_point_history[-100:])
    final_std_reward = np.std(total_point_history[-100:])
    
    print(f"\n最终性能 (最后100个episodes):")
    print(f"平均奖励: {final_avg_reward:.2f} ± {final_std_reward:.2f}")
    print(f"最大奖励: {np.max(total_point_history):.2f}")
    print(f"最小奖励: {np.min(total_point_history):.2f}")
    
    env.close()
    print("\n🎉 项目完成！")

if __name__ == "__main__":
    main()
