#!/usr/bin/env python3
"""
运行所有基线策略并生成对比报告
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.environment import PlantCareEnv
from src.baselines import FixedSchedulePolicy, ThresholdRulePolicy
from src.utils.visualization import plot_comparison_table, plot_metrics_comparison_bars
import numpy as np


def evaluate_policy(policy, env, n_episodes=5, seed=42, policy_name="Policy"):
    """评估单个策略"""
    print(f"\n{'=' * 60}")
    print(f"评估: {policy_name}")
    print('=' * 60)
    
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
        final_health = obs[4]
        total_water = info['total_water_used']
        total_energy = info['total_energy_used']
        violations = info['total_violations']
        efficiency = avg_health / (total_water + 0.001 * total_energy + 1e-6)
        
        results['avg_health'].append(avg_health)
        results['final_health'].append(final_health)
        results['total_water'].append(total_water)
        results['total_energy'].append(total_energy)
        results['violations'].append(violations)
        results['efficiency'].append(efficiency)
        
        print(f"  Episode {episode + 1}/{n_episodes}: "
              f"Health={avg_health:.1f}, "
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
    
    print(f"\n结果摘要:")
    print(f"  平均健康度: {summary['avg_health_mean']:.1f} ± {summary['avg_health_std']:.1f}")
    print(f"  用水量: {summary['total_water_mean']:.1f} ml")
    print(f"  能耗: {summary['total_energy_mean']:.1f} Wh")
    print(f"  违规: {summary['violations_mean']:.1f} 小时")
    print(f"  效率: {summary['efficiency_mean']:.3f}")
    
    return summary


def main():
    """主函数"""
    print("=" * 60)
    print("智能植物护理系统 - 基线对比实验")
    print("=" * 60)
    
    # 创建环境
    config_path = "config.yaml"
    env = PlantCareEnv(config_path=config_path)
    
    # 评估所有策略
    all_results = {}
    
    # 1. 固定时间表
    policy1 = FixedSchedulePolicy(config_path)
    all_results["固定时间表"] = evaluate_policy(
        policy1, env, n_episodes=5, policy_name="固定时间表基线"
    )
    
    # 2. 阈值规则
    policy2 = ThresholdRulePolicy(config_path)
    all_results["阈值规则"] = evaluate_policy(
        policy2, env, n_episodes=5, policy_name="阈值规则基线"
    )
    
    # 打印最终对比
    print("\n" + "=" * 60)
    print("最终对比结果")
    print("=" * 60)
    print(f"{'策略':<15} {'健康度':<12} {'用水(ml)':<12} {'违规(h)':<10} {'效率':<10}")
    print("-" * 60)
    
    for policy_name, results in all_results.items():
        print(f"{policy_name:<15} "
              f"{results['avg_health_mean']:.1f}±{results['avg_health_std']:.1f}   "
              f"{results['total_water_mean']:.1f}       "
              f"{results['violations_mean']:.1f}       "
              f"{results['efficiency_mean']:.3f}")
    
    print("=" * 60)
    
    # 生成可视化
    print("\n生成可视化图表...")
    os.makedirs("docs/images", exist_ok=True)
    
    plot_comparison_table(
        all_results,
        save_path="docs/images/baseline_comparison_table.png"
    )
    
    plot_metrics_comparison_bars(
        all_results,
        save_path="docs/images/baseline_comparison_bars.png"
    )
    
    # 生成离线数据集 CSV
    print("\n生成离线数据集 CSV...")
    import csv
    os.makedirs("data", exist_ok=True)
    out_path = "data/rollouts.csv"
    fieldnames = [
        "policy","t","soil_moisture","temperature","light_level","hour_of_day",
        "plant_health","hours_since_water","water_amount","lamp_on","reward","next_plant_health"
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy_name, policy in [("fixed", FixedSchedulePolicy(config_path)), ("threshold", ThresholdRulePolicy(config_path))]:
            for seed in [0,1,2,3,4]:
                obs, info = env.reset(seed=seed)
                terminated = False
                truncated = False
                t = 0
                while not (terminated or truncated):
                    action = policy.get_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    writer.writerow({
                        "policy": policy_name,
                        "t": t,
                        "soil_moisture": float(obs[0]),
                        "temperature": float(obs[1]),
                        "light_level": float(obs[2]),
                        "hour_of_day": int(obs[3]),
                        "plant_health": float(obs[4]),
                        "hours_since_water": float(obs[5]),
                        "water_amount": float(action[0]),
                        "lamp_on": int(action[1] > 0.5),
                        "reward": float(reward),
                        "next_plant_health": float(next_obs[4])
                    })
                    obs = next_obs
                    t += 1
    print(f"✅ 数据集已保存到 {out_path}")

    print("\n✅ 实验完成!")
    print("📊 图表已保存在 docs/images/ 目录")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()

