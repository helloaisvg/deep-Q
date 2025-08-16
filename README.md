# Deep Q-Learning for Lunar Lander

## 概述
本项目实现了带有经验回放的Deep Q-Learning算法，用于训练智能体在OpenAI Gym的LunarLander-v2环境中安全着陆月球模块。它展示了强化学习技术，如目标网络和ε-贪婪探索。

## 特性
- Q网络和目标网络用于稳定的Q值近似
- 经验回放缓冲区用于非相关学习
- 带有ε衰减和性能可视化的训练循环

## 要求
- Python 3.7+
- TensorFlow 2.x
- Gym (0.24.0)
- NumPy, Collections
- 安装命令: `pip install tensorflow gym numpy`

## 用法
1. 运行Jupyter notebook: `jupyter notebook lunar_lander_dql.ipynb`
2. 训练智能体并生成视频

## 结果
- 在<2000个episode内解决环境，平均奖励>200
- 运行时间: ~10-15分钟

## 许可证
MIT License
