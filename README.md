# RL-DQN: Modular DQN/Double+Dueling with PER for Gym

## 概述
本项目将原有课业式Notebook重构为可复用的模块化Python包，支持在OpenAI Gym环境（默认`LunarLander-v2`）上训练的DQN/Double DQN/Dueling DQN，提供优先经验回放（PER）、TensorBoard日志与可配置的训练脚本。

## 特性
- **算法**: 基线DQN、Double DQN、Dueling DQN（可开关）
- **数据**: 经验回放与优先经验回放（PER）
- **训练**: YAML配置、命令行入口、TensorBoard日志（奖励/长度/损失/ε）
- **结构**: 模块化`rldqn`包，便于扩展到CartPole/Atari等

## 要求
- Python 3.8+
- TensorFlow 2.x
- Gym (0.24.x)
- NumPy, PyYAML, TensorBoard

## 用法
安装依赖：

```bash
pip install -r requirements.txt
```

开始训练：

```bash
python train.py --config configs/default.yaml
```

查看日志：

```bash
tensorboard --logdir runs
```

## 结果
- 在<2000个episode内可解决`LunarLander-v2`（硬件/随机种子相关）
- 典型运行时间: ~10-15分钟（CPU）

## 许可证
MIT License
