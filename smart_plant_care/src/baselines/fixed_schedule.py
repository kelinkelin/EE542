"""
固定时间表基线策略
每天固定时间浇水和开灯（非智能）
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.environment import PlantCareEnv
from typing import Dict, List
import yaml


class FixedSchedulePolicy:
    """
    固定时间表策略
    - 每天8点和20点浇水50ml
    - 灯光6点开，22点关
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化固定时间表"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.water_times = config['baselines']['fixed_schedule']['water_times']
        self.water_amount = config['baselines']['fixed_schedule']['water_amount']
        self.lamp_schedule = config['baselines']['fixed_schedule']['lamp_schedule']
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        根据固定时间表决定动作
        
        Args:
            observation: [soil_moisture, temperature, light_level, hour_of_day, plant_health, hours_since_water]
            
        Returns:
            action: [water_amount, lamp_on]
        """
        hour = int(observation[3])
        
        # 判断是否浇水
        water_amount = self.water_amount if hour in self.water_times else 0.0
        
        # 判断是否开灯
        lamp_start, lamp_end = self.lamp_schedule
        lamp_on = 1.0 if lamp_start <= hour < lamp_end else 0.0
        
        return np.array([water_amount, lamp_on], dtype=np.float32)


def evaluate_policy(
    policy: FixedSchedulePolicy,
    env: PlantCareEnv,
    n_episodes: int = 5,
    seed: int = 42
) -> Dict:
    """
    评估策略性能
    
    Returns:
        metrics: 包含平均健康度、用水量、能耗等指标的字典
    """
    results = {
        'avg_health': [],
        'final_health': [],
        'total_water': [],
        'total_energy': [],
        'violations': [],
        'efficiency': []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录指标
        avg_health = info['avg_health']
        final_health = obs[4]  # plant_health
        total_water = info['total_water_used']
        total_energy = info['total_energy_used']
        violations = info['total_violations']
        
        # 效率 = 健康增益 / (用水 + λ·能耗)
        efficiency = avg_health / (total_water + 0.001 * total_energy + 1e-6)
        
        results['avg_health'].append(avg_health)
        results['final_health'].append(final_health)
        results['total_water'].append(total_water)
        results['total_energy'].append(total_energy)
        results['violations'].append(violations)
        results['efficiency'].append(efficiency)
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Avg Health={avg_health:.1f}, "
              f"Water={total_water:.1f}ml, "
              f"Violations={violations}")
    
    # 计算统计量
    summary = {
        'avg_health_mean': np.mean(results['avg_health']),
        'avg_health_std': np.std(results['avg_health']),
        'final_health_mean': np.mean(results['final_health']),
        'total_water_mean': np.mean(results['total_water']),
        'total_energy_mean': np.mean(results['total_energy']),
        'violations_mean': np.mean(results['violations']),
        'efficiency_mean': np.mean(results['efficiency'])
    }
    
    return summary, results


if __name__ == "__main__":
    print("=" * 60)
    print("固定时间表基线策略评估")
    print("=" * 60 + "\n")
    
    # 创建环境
    config_path = "../../config.yaml"
    env = PlantCareEnv(config_path=config_path)
    
    # 创建策略
    policy = FixedSchedulePolicy(config_path=config_path)
    
    print("策略配置：")
    print(f"  浇水时间: {policy.water_times}")
    print(f"  浇水量: {policy.water_amount} ml")
    print(f"  灯光时间: {policy.lamp_schedule[0]}:00 - {policy.lamp_schedule[1]}:00")
    print()
    
    # 评估策略
    print("开始评估（5个episodes，每个30天）...\n")
    summary, results = evaluate_policy(policy, env, n_episodes=5, seed=42)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果（均值 ± 标准差）")
    print("=" * 60)
    print(f"平均健康度: {summary['avg_health_mean']:.1f} ± {summary['avg_health_std']:.1f}")
    print(f"最终健康度: {summary['final_health_mean']:.1f}")
    print(f"总用水量: {summary['total_water_mean']:.1f} ml")
    print(f"总能耗: {summary['total_energy_mean']:.1f} Wh")
    print(f"约束违规: {summary['violations_mean']:.1f} 小时")
    print(f"资源效率: {summary['efficiency_mean']:.3f}")
    print("=" * 60)

