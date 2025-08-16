import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import base64
import io

# 设置随机种子以确保可重现性
SEED = 42

def update_target_network(q_network, target_q_network):
    """
    更新目标网络权重
    """
    for target_weights, q_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(q_weights)

def get_experiences(memory_buffer):
    """
    从经验回放缓冲区中随机采样经验
    """
    experiences = memory_buffer.sample()
    states = tf.cast([e.state for e in experiences], tf.float32)
    actions = tf.cast([e.action for e in experiences], tf.int32)
    rewards = tf.cast([e.reward for e in experiences], tf.float32)
    next_states = tf.cast([e.next_state for e in experiences], tf.float32)
    done_vals = tf.cast([e.done for e in experiences], tf.float32)
    
    return (states, actions, rewards, next_states, done_vals)

def check_update_conditions(t, num_steps_upd, memory_buffer):
    """
    检查是否应该更新网络
    """
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > 1000:
        return True
    return False

def get_action(q_values, epsilon):
    """
    使用ε-贪婪策略选择动作
    """
    if np.random.random() < epsilon:
        # 随机探索
        action = np.random.randint(0, q_values.shape[1])
    else:
        # 贪婪选择
        action = tf.argmax(q_values[0]).numpy()
    
    return action

def get_new_eps(epsilon):
    """
    更新ε值
    """
    epsilon = max(0.01, epsilon * 0.995)
    return epsilon

def plot_history(total_point_history, num_p_av=100):
    """
    绘制训练历史
    """
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

def display_table(current_state, action, next_state, reward, done):
    """
    显示状态转换表
    """
    print("=" * 50)
    print("状态转换表")
    print("=" * 50)
    print(f"当前状态: {current_state}")
    print(f"动作: {action}")
    print(f"下一个状态: {next_state}")
    print(f"奖励: {reward}")
    print(f"结束: {done}")
    print("=" * 50)

def create_video(filename, env, q_network, num_episodes=1):
    """
    创建智能体行为的视频
    """
    frames = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 渲染环境
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            # 选择动作
            state_qn = np.expand_dims(state, axis=0)
            q_values = q_network(state_qn)
            action = get_action(q_values, 0.0)  # 使用贪婪策略
            
            # 执行动作
            state, reward, done, info = env.step(action)
    
    env.close()
    
    # 保存视频
    if len(frames) > 0:
        try:
            import imageio
            imageio.mimsave(filename, frames, fps=30)
            print(f"视频已保存为: {filename}")
        except ImportError:
            print("需要安装imageio来保存视频: pip install imageio")
        except Exception as e:
            print(f"视频保存失败: {e}")

def embed_mp4(filename):
    """
    在Jupyter notebook中嵌入MP4视频
    """
    try:
        with open(filename, 'rb') as f:
            video = f.read()
        
        data_url = "data:video/mp4;base64," + base64.b64encode(video).decode()
        return HTML(f'''
        <video width="400" controls>
            <source src="{data_url}" type="video/mp4">
        </video>
        ''')
    except Exception as e:
        print(f"视频嵌入失败: {e}")
        return None

def evaluate_agent(env, q_network, num_episodes=10):
    """
    评估智能体性能
    """
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
