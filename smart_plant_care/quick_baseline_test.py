#!/usr/bin/env python3
"""
快速基线测试 - 生成Week 1演示数据
运行时间：约2-3分钟
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.environment import PlantCareEnv
from src.baselines import FixedSchedulePolicy, ThresholdRulePolicy
import numpy as np
import json

def evaluate_policy(policy, env, n_episodes=3, policy_name="Policy"):
    """快速评估策略（3个episodes）"""
    print(f"\n{'='*60}")
    print(f"评估: {policy_name}")
    print('='*60)
    
    results = {
        'avg_health': [],
        'final_health': [],
        'total_water': [],
        'total_energy': [],
        'violations': [],
        'efficiency': []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=42 + episode)
        terminated = False
        truncated = False
        
        step_count = 0
        while not (terminated or truncated):
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # 每100步打印一次进度
            if step_count % 100 == 0:
                print(f"  Episode {episode+1}, Step {step_count}: Health={obs[4]:.1f}", end='\r')
        
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
              f"Water={total_water:.0f}ml, "
              f"Violations={violations}        ")
    
    # 计算统计量
    summary = {
        'policy_name': policy_name,
        'avg_health_mean': float(np.mean(results['avg_health'])),
        'avg_health_std': float(np.std(results['avg_health'])),
        'final_health_mean': float(np.mean(results['final_health'])),
        'total_water_mean': float(np.mean(results['total_water'])),
        'total_water_std': float(np.std(results['total_water'])),
        'total_energy_mean': float(np.mean(results['total_energy'])),
        'violations_mean': float(np.mean(results['violations'])),
        'violations_std': float(np.std(results['violations'])),
        'efficiency_mean': float(np.mean(results['efficiency']))
    }
    
    return summary, results


def print_comparison_table(all_results):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("Week 1 基线对比结果".center(80))
    print("=" * 80)
    print(f"{'策略':<20} {'平均健康度':<15} {'用水(ml)':<15} {'违规(h)':<12} {'效率':<10}")
    print("-" * 80)
    
    for policy_name, results in all_results.items():
        health_str = f"{results['avg_health_mean']:.1f}±{results['avg_health_std']:.1f}"
        water_str = f"{results['total_water_mean']:.0f}±{results['total_water_std']:.0f}"
        violations_str = f"{results['violations_mean']:.0f}±{results['violations_std']:.0f}"
        eff_str = f"{results['efficiency_mean']:.3f}"
        
        print(f"{policy_name:<20} {health_str:<15} {water_str:<15} {violations_str:<12} {eff_str:<10}")
    
    print("=" * 80)


def save_results_json(all_results, filename="week1_baseline_results.json"):
    """保存结果为JSON"""
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ 结果已保存: {filename}")


def main():
    print("=" * 80)
    print("Week 1 Milestone - 基线性能测试".center(80))
    print("=" * 80)
    print("\n这将运行两个基线策略，每个3 episodes (30天)")
    print("预计时间: 2-3分钟\n")
    
    # 创建环境
    config_path = "config.yaml"
    env = PlantCareEnv(config_path=config_path)
    
    print(f"✅ 环境创建成功")
    print(f"   观察空间: {env.observation_space.shape}")
    print(f"   动作空间: {env.action_space.shape}")
    print(f"   最大步数: {env.max_steps} (30天)")
    
    # 评估所有策略
    all_results = {}
    
    # 1. 固定时间表
    print("\n" + "▶" * 40)
    policy1 = FixedSchedulePolicy(config_path)
    summary1, _ = evaluate_policy(
        policy1, env, n_episodes=3, policy_name="固定时间表 (Fixed Schedule)"
    )
    all_results["固定时间表"] = summary1
    
    # 2. 阈值规则
    print("\n" + "▶" * 40)
    policy2 = ThresholdRulePolicy(config_path)
    summary2, _ = evaluate_policy(
        policy2, env, n_episodes=3, policy_name="阈值规则 (Threshold Rule)"
    )
    all_results["阈值规则"] = summary2
    
    # 打印对比表格
    print_comparison_table(all_results)
    
    # 保存结果
    os.makedirs("data", exist_ok=True)
    save_results_json(all_results, "data/week1_baseline_results.json")
    
    # 生成PPT用的总结
    print("\n" + "=" * 80)
    print("📊 演示PPT数据总结".center(80))
    print("=" * 80)
    
    print("\n**基线性能对比** (用于Slide 6):\n")
    print("| 策略 | 平均健康度 | 用水(L/月) | 违规(小时) | 效率 |")
    print("|------|------------|-----------|-----------|------|")
    
    for policy_name, results in all_results.items():
        print(f"| {policy_name} | "
              f"{results['avg_health_mean']:.1f}±{results['avg_health_std']:.1f} | "
              f"{results['total_water_mean']/1000:.2f} | "
              f"{results['violations_mean']:.0f} | "
              f"{results['efficiency_mean']:.3f} |")
    
    print("\n**关键发现**:")
    health_improvement = (all_results['阈值规则']['avg_health_mean'] - 
                         all_results['固定时间表']['avg_health_mean'])
    water_reduction = ((all_results['固定时间表']['total_water_mean'] - 
                       all_results['阈值规则']['total_water_mean']) / 
                      all_results['固定时间表']['total_water_mean'] * 100)
    
    print(f"- 阈值规则比固定时间表健康度提升: +{health_improvement:.1f} (+{health_improvement/all_results['固定时间表']['avg_health_mean']*100:.1f}%)")
    print(f"- 阈值规则比固定时间表用水减少: {water_reduction:.1f}%")
    print(f"\n- PPO目标: 健康度 ≥85, 用水 <6000ml/月")
    print(f"- 如果达到目标，将比固定时间表提升 ~{(85-all_results['固定时间表']['avg_health_mean'])/all_results['固定时间表']['avg_health_mean']*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ Week 1基线测试完成!".center(80))
    print("=" * 80)
    print("\n下一步:")
    print("  1. 将上述表格数据添加到PPT Slide 6")
    print("  2. 运行可视化生成图表: python generate_visualizations.py")
    print("  3. 准备演讲，参考 docs/Week1_Presentation_Script.md")
    
    env.close()


if __name__ == "__main__":
    main()









