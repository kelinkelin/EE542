# 🌱 Smart Plant Care System - Reinforcement Learning Project

## Project Overview

A deep reinforcement learning-based intelligent plant care system that optimizes irrigation and lighting strategies through autonomous learning, addressing the 50% water waste problem caused by traditional fixed schedules.

### Core Value Proposition

- **Problem**: Traditional irrigation systems waste 50% of water resources, 40% of household plants die within the first year
- **Solution**: Use PPO reinforcement learning algorithm to learn adaptive care strategies
- **Goal**: Improve plant health to 95%+, while optimizing resource efficiency

## Technical Approach

### Category 1: Advanced Reinforcement Learning

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Environment**: Custom plant growth physics simulator
- **State Space**: [soil_moisture, temperature, light_level, time_of_day, plant_health]
- **Action Space**: {water_amount: 0-100ml, lamp: ON/OFF}
- **Reward Function**: R = α·Δhealth - β·water_used - γ·energy_used - δ·violations

### GPU Acceleration

- **Training Acceleration**: PyTorch + CUDA 12.1
- **Target Hardware**: RTX 5090
- **Expected Speedup**: 8x (3 hours vs 24 hours)

## Project Structure

```
smart_plant_care/
├── src/
│   ├── environment/       # Plant simulation environment
│   │   ├── plant_env.py   # Gym environment wrapper
│   │   └── physics.py     # Physics model (soil, photosynthesis)
│   ├── agents/            # RL agents
│   │   └── train_ppo.py   # PPO implementation
│   ├── baselines/         # Baseline policies
│   │   ├── fixed_schedule.py
│   │   └── threshold_rule.py
│   └── utils/             # Utility functions
│       └── visualization.py
├── config.yaml            # Configuration file
└── requirements.txt       # Dependencies
```

## Quick Start

### Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Verify GPU (if available)
python -c "import torch; print(torch.cuda.is_available())"
```

### Run Baseline Tests

```bash
# Fixed schedule baseline
python src/baselines/fixed_schedule.py

# Threshold rule baseline
python src/baselines/threshold_rule.py
```

### Train PPO Agent

```bash
# CPU training (slow)
python src/agents/train_ppo.py --device cpu --timesteps 1000000

# GPU training (8x faster)
python src/agents/train_ppo.py --device cuda --timesteps 5000000
```

## References

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
2. MarketsandMarkets (2024): "Smart Agriculture Market - $12B growing to $22B by 2028"
3. FAO (2023): "Global irrigation efficiency - 50% water waste"
4. Thornley & Johnson (2000): "Plant and Crop Modelling"

## Author

- **Kelin Wu** - EE542 Fall 2025
- **Advisor**: Professor Young H. Cho

## License

MIT License - Educational Use
