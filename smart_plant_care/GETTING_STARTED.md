# 快速开始指南

## 🚀 5分钟快速启动

### 1. 环境设置

```bash
# 克隆或进入项目目录
cd smart_plant_care

# 运行自动设置脚本（推荐）
bash setup.sh

# 或手动设置
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 验证安装

```bash
# 检查GPU（如果有）
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 运行快速演示
python3 quick_start.py
```

### 3. 运行基线对比

```bash
# 评估固定时间表和阈值规则策略
python3 run_baseline_comparison.py
```

**预期输出**:
```
固定时间表: Health=60±3, Water=10200ml
阈值规则: Health=70±4, Water=8500ml
```

### 4. 训练PPO模型

```bash
# 快速训练（10分钟，CPU）
python3 src/agents/train_ppo.py --timesteps 100000 --device cpu

# 完整训练（3小时，GPU）
python3 src/agents/train_ppo.py --timesteps 5000000 --device cuda
```

### 5. 查看训练进度

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 在浏览器打开 http://localhost:6006
```

---

## 📂 项目结构说明

```
smart_plant_care/
├── config.yaml                 # 核心配置文件
├── requirements.txt            # Python依赖
├── setup.sh                    # 自动设置脚本
├── quick_start.py              # 快速演示
├── run_baseline_comparison.py # 基线对比脚本
│
├── src/                        # 源代码
│   ├── environment/            # 植物环境模拟
│   │   ├── physics.py          # 物理模型（土壤、光合作用）
│   │   └── plant_env.py        # Gym环境封装
│   ├── agents/                 # RL智能体
│   │   └── train_ppo.py        # PPO训练脚本
│   ├── baselines/              # 基线策略
│   │   ├── fixed_schedule.py   # 固定时间表
│   │   └── threshold_rule.py   # 阈值规则
│   └── utils/                  # 工具函数
│       └── visualization.py    # 绘图工具
│
├── models/                     # 保存的模型权重
├── logs/                       # TensorBoard日志
├── data/                       # 数据文件
├── docs/                       # 文档
│   ├── INTERVIEW_PLAN.md       # 采访计划
│   └── WEEKLY_SCHEDULE.md      # 6周时间表
└── notebooks/                  # Jupyter分析笔记本
```

---

## 🎯 核心概念

### 1. 植物护理环境

**状态空间** (6维):
- `soil_moisture`: 土壤湿度 [0, 1]
- `temperature`: 温度 [0, 50]°C
- `light_level`: 光照 [0, 2000] lux
- `hour_of_day`: 小时 [0, 23]
- `plant_health`: 健康度 [0, 100]
- `hours_since_water`: 距上次浇水小时数

**动作空间** (2维):
- `water_amount`: 浇水量 [0, 100] ml
- `lamp_on`: 灯光 {0=OFF, 1=ON}

**奖励函数**:
```
R = α·Δhealth - β·water - γ·energy - δ·violations

α=1.0   (健康度增益奖励)
β=0.01  (用水惩罚)
γ=0.001 (能耗惩罚)
δ=0.5   (约束违规惩罚)
```

### 2. 基线策略

**固定时间表**:
- 每天8点和20点浇水50ml
- 灯光6:00-22:00开启
- 性能: Health≈60

**阈值规则**:
- 土壤湿度<30%时浇水
- 光照<200 lux时开灯
- 性能: Health≈70

**PPO目标**:
- 通过强化学习优化决策
- 目标: Health≥85, Water↓40%

---

## 🔧 常见问题

### Q1: GPU不可用怎么办？

```bash
# 选项1: 使用CPU（慢8x）
python3 src/agents/train_ppo.py --device cpu --timesteps 500000

# 选项2: 使用Google Colab
# 上传代码到Colab，选择GPU运行时

# 选项3: AWS EC2 GPU实例
# 使用p3.2xlarge (V100)，约$3/小时
```

### Q2: 训练不收敛？

```yaml
# 修改config.yaml中的参数
ppo:
  learning_rate: 0.0001  # 降低学习率
  ent_coef: 0.05         # 增加探索
  clip_range: 0.1        # 减小裁剪范围
```

### Q3: 内存不足？

```python
# 减少并行环境数量
# 在train_ppo.py中修改
n_envs = 2  # 从4改为2
```

### Q4: 如何测试训练好的模型？

```bash
python3 src/agents/train_ppo.py \
  --test models/best_model.zip \
  --config config.yaml
```

---

## 📊 性能基准

### 基线对比（30天模拟）

| 策略 | 平均健康度 | 用水(ml) | 能耗(Wh) | 违规(h) | 效率 |
|-----|-----------|---------|---------|--------|------|
| 固定时间表 | 60±3 | 10200 | 360000 | 120 | 0.50 |
| 阈值规则 | 70±4 | 8500 | 320000 | 80 | 0.72 |
| **PPO (目标)** | **87±2** | **5800** | **240000** | **35** | **1.12** |

### 训练时间

| 设备 | 步数 | 时间 | 性能 |
|-----|-----|------|------|
| CPU (M1 Max) | 5M | ~24h | Health≈85 |
| GPU (RTX 3090) | 5M | ~3h | Health≈87 |
| GPU (RTX 5090) | 5M | ~2h | Health≈87 |

---

## 🎓 学习路径

### 初学者路径

1. **了解环境** (1小时)
   ```bash
   python3 quick_start.py
   ```
   
2. **运行基线** (30分钟)
   ```bash
   python3 run_baseline_comparison.py
   ```

3. **阅读代码** (2小时)
   - `src/environment/physics.py` - 物理模型
   - `src/environment/plant_env.py` - Gym环境
   - `src/baselines/` - 简单策略

4. **短训练实验** (30分钟)
   ```bash
   python3 src/agents/train_ppo.py --timesteps 50000
   ```

### 高级用户路径

1. **超参数调优**
   - 修改`config.yaml`
   - 运行网格搜索
   - 分析TensorBoard

2. **多场景测试**
   - 实现新场景
   - 域随机化
   - 鲁棒性分析

3. **消融实验**
   - 移除观察维度
   - 调整奖励权重
   - 网络架构实验

---

## 🐛 调试技巧

### 1. 打印详细信息

```python
# 在plant_env.py中添加
def step(self, action):
    print(f"Action: water={action[0]:.1f}ml, lamp={action[1]}")
    # ... rest of code
```

### 2. 可视化一个episode

```python
from src.environment import PlantCareEnv
import matplotlib.pyplot as plt

env = PlantCareEnv()
obs, _ = env.reset()

healths = []
moistures = []

for _ in range(24*7):  # 1周
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    healths.append(obs[4])
    moistures.append(obs[0])

plt.plot(healths, label='Health')
plt.plot(moistures, label='Moisture')
plt.legend()
plt.show()
```

### 3. 检查奖励分布

```bash
# TensorBoard中查看
# rollout/ep_rew_mean 应该逐渐上升
tensorboard --logdir logs/
```

---

## 🚀 下一步

完成基础设置后：

1. **Week 1**: 完成6个用户采访
2. **Week 2-3**: 训练完整PPO模型
3. **Week 4**: 多场景鲁棒性测试
4. **Week 5**: 评估和消融实验
5. **Week 6**: 最终演示和文档

详见: `docs/WEEKLY_SCHEDULE.md`

---

## 📚 推荐资源

### 强化学习
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI教程
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [PPO原论文](https://arxiv.org/abs/1707.06347)

### 植物生理学
- [植物水分关系](https://www.nature.com/subjects/plant-water-relations)
- [光合作用模型](https://en.wikipedia.org/wiki/Photosynthesis)

### 相关项目
- [Gym-PlantDisease](https://github.com/...)
- [AgriTech-RL](https://github.com/...)

---

## 💬 获取帮助

- **课程**: EE542 - Professor Young H. Cho
- **GitHub Issues**: [项目地址]
- **Email**: kelinwu@usc.edu

---

**祝你项目顺利！🌱**

