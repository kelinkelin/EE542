# 🌱 Smart Plant Care System - Reinforcement Learning Project

## 项目概述

这是一个基于深度强化学习的智能植物护理系统，旨在通过自主学习优化灌溉和光照策略，解决传统固定时间表导致的50%水资源浪费问题。

### 核心价值主张

- **问题**：传统灌溉系统浪费50%的水资源，40%的家庭植物在第一年内死亡
- **解决方案**：使用PPO强化学习算法，学习自适应护理策略
- **目标**：植物健康度提升至95%+，同时优化资源使用效率

## 技术路线

### Category 1: Advanced Reinforcement Learning

- **算法**：Proximal Policy Optimization (PPO)
- **环境**：自研植物生长物理模拟器
- **状态空间**：[soil_moisture, temperature, light_level, time_of_day, plant_health]
- **动作空间**：{water_amount: 0-100ml, lamp: ON/OFF}
- **奖励函数**：R = α·Δhealth - β·water_used - γ·energy_used - δ·violations

### GPU加速

- **训练加速**：PyTorch + CUDA 12.1
- **目标硬件**：RTX 5090
- **预期提速**：8x（3小时 vs 24小时）

## 项目结构

```
smart_plant_care/
├── src/
│   ├── environment/       # 植物模拟环境
│   │   ├── plant_env.py   # Gym环境封装
│   │   └── physics.py     # 物理模型（土壤、光合作用）
│   ├── agents/            # RL智能体
│   │   ├── ppo_agent.py   # PPO实现
│   │   └── networks.py    # Actor-Critic网络
│   ├── baselines/         # 基线策略
│   │   ├── fixed_schedule.py
│   │   └── threshold_rule.py
│   └── utils/             # 工具函数
│       ├── visualization.py
│       └── metrics.py
├── data/                  # 训练数据/日志
├── models/                # 保存的模型权重
├── logs/                  # TensorBoard日志
├── notebooks/             # Jupyter分析笔记本
├── tests/                 # 单元测试
└── docs/                  # 文档
```

## 最终项目成果演示 (Final Project Demo)

我们已完成所有算法的实现与对比测试，包括 PPO 智能体与传统基线策略的全面对决。

### 1. 运行最终评估
生成三方对比数据（Fixed vs Threshold vs PPO）和静态分析图表：
```bash
cd smart_plant_care
./venv/bin/python run_final_evaluation.py
```
输出结果位于：
- `docs/images/final_comparison_bars.png` (性能对比柱状图)
- `docs/images/final_comparison_table.png` (详细数据表)

### 2. 查看交互式动画 (演示模式)

我们提供了两种演示模式：标准版和**高科技指挥中心版**。

**推荐：启动炫酷指挥中心演示 (Sci-Fi Dashboard)**
这是为了展示最终效果而专门设计的赛博朋克风格界面，具有动态粒子特效、实时数据流和拟真植物状态反馈。

```bash
# 生成炫酷演示文件
cd smart_plant_care
./venv/bin/python generate_cool_demo.py

# 在浏览器中打开
open docs/cool_demo.html  # MacOS
# start docs/cool_demo.html # Windows
```

**标准版演示**：
```bash
# 生成标准演示文件
./venv/bin/python generate_demo_html.py
open docs/demo.html
```

演示包含：
- **实时环境监控**：类似飞船控制台的仪表盘 (HUD)
- **植物生长模拟**：基于健康度的动态 SVG 渲染
- **AI 决策日志**：实时显示 PPO 智能体的思考过程
- **对比分析**：AI vs 传统方法的实时效率竞赛

### 3. 实验结果摘要
| 策略 | 健康度 (0-100) | 状态 | 评价 |
|-----|--------------|------|------|
| **Fixed Schedule** | ~54 | 濒死 | 严重缺水，策略僵化 |
| **Threshold Rule** | ~71 | 亚健康 | 勉强维持，有些许资源浪费 |
| **PPO Agent (AI)** | **~97** | **茁壮** | **精准灌溉，资源利用率最高** |

## 6周实施计划（已完成）

| 周次 | 里程碑 | 交付物 |
|-----|--------|--------|
| **Week 1** | 环境 + 基线 | 植物模拟器 + 固定时间表/阈值规则基线 |
| **Week 2-3** | PPO实现 | GPU加速训练，达到85+健康度目标 |
| **Week 4** | 多场景训练 | 热浪/阴天/传感器噪声鲁棒性测试 |
| **Week 5** | 评估 + 消融 | 统计显著性测试，超参数敏感性分析 |
| **Week 6** | 可视化 + 演示 | 最终演示视频 + 部署代码 |

## 市场验证

### 真实公司案例

1. **John Deere** - AI精准农业，$52B年收入
2. **Click & Grow** - 智能种植器，估值$50M+
3. **AeroGarden** - 家用水培系统，年销售$100M+
4. **Gardyn** - 垂直农场，B轮融资$20M

### 采访计划（需完成6个）

- [ ] 2名垂直农场运营者
- [ ] 2名智能家居产品经理
- [ ] 2名家庭园艺爱好者

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt

# 验证GPU（如果有）
python -c "import torch; print(torch.cuda.is_available())"
```

### 运行基线测试

```bash
# 固定时间表基线
python src/baselines/fixed_schedule.py

# 阈值规则基线
python src/baselines/threshold_rule.py
```

### 训练PPO智能体

```bash
# CPU训练（慢）
python src/agents/train_ppo.py --device cpu --timesteps 1000000

# GPU训练（快8x）
python src/agents/train_ppo.py --device cuda --timesteps 5000000
```

### 可视化结果

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 生成对比图表
python src/utils/visualization.py --model models/ppo_best.pth
```

## 参考文献

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
2. MarketsandMarkets (2024): "Smart Agriculture Market - $12B growing to $22B by 2028"
3. FAO (2023): "Global irrigation efficiency - 50% water waste"
4. Thornley & Johnson (2000): "Plant and Crop Modelling"
5. Google DeepMind (2016): "Data center cooling optimization with RL"

## 开发者

- **Kelin Wu** - EE542 Fall 2025
- **导师**：Professor Young H. Cho

## License

MIT License - 教育用途
