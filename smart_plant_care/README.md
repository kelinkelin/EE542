# 🌱 Smart Plant Care System - Reinforcement Learning Project

## Project Overview

An intelligent plant care system using **Reinforcement Learning (RL)** to optimize irrigation and lighting strategies. The agent learns through trial-and-error interaction with a plant growth simulator, maximizing long-term plant health while minimizing resource consumption.

### Core Value Proposition

- **Problem**: Traditional irrigation systems waste 50% of water resources
- **Solution**: Use PPO reinforcement learning algorithm to learn adaptive care strategies
- **Goal**: Achieve 95%+ plant health with optimal resource efficiency

## Reinforcement Learning Framework

### RL Loop (Core of This Project)

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Training Loop                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌───────────┐    action     ┌─────────────────┐          │
│   │   Agent   │──────────────▶│   Environment   │          │
│   │   (PPO)   │               │ (Plant Simulator)│          │
│   └───────────┘               └─────────────────┘          │
│         ▲                            │                      │
│         │      state, reward         │                      │
│         └────────────────────────────┘                      │
│                                                             │
│   Agent observes state → selects action → receives reward   │
│   → updates policy to maximize cumulative reward            │
└─────────────────────────────────────────────────────────────┘
```

### Why This is Reinforcement Learning (Not Supervised Learning)

| Aspect | Supervised Learning | Our RL Approach |
|--------|--------------------|-----------------|
| Data | Pre-labeled dataset | No labels, learns from interaction |
| Learning | Minimize prediction error | Maximize cumulative reward |
| Feedback | Immediate correct answer | Delayed reward signal |
| Exploration | None | Agent explores action space |

### PPO Algorithm (Proximal Policy Optimization)

PPO is a **policy gradient** method that:
1. Collects trajectories by interacting with environment
2. Computes advantage estimates (GAE)
3. Updates policy using clipped objective to prevent large updates
4. Key hyperparameters: `clip_range=0.2`, `gae_lambda=0.95`

### State-Action-Reward Design

**State Space (6-dim observation):**
- `soil_moisture`: Current soil moisture [0, 1]
- `temperature`: Ambient temperature [0, 50]°C
- `light_level`: Light intensity [0, 2000] lux
- `hour_of_day`: Time of day [0, 23]
- `plant_health`: Current health [0, 100]
- `hours_since_water`: Hours since last watering [0, 24]

**Action Space (2-dim continuous):**
- `water_amount`: Water to dispense [0, 100] ml
- `lamp_on`: Lamp switch [0, 1]

**Reward Function:**
```
R = α·Δhealth - β·water_used - γ·energy_used - δ·violations
```
- Positive reward for health improvement
- Penalties for resource consumption and constraint violations

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
