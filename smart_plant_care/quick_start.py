#!/usr/bin/env python3
"""
快速开始演示
演示环境、基线策略和PPO训练
"""

import sys
import os
import time

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60 + "\n")


def demo_environment():
    """演示环境"""
    print_section("1. 环境演示")
    
    from src.environment import PlantCareEnv
    
    print("创建植物护理环境...")
    env = PlantCareEnv(config_path="config.yaml")
    
    print(f"✅ 环境创建成功!")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  最大步数: {env.max_steps} (30天)")
    
    print("\n运行一个随机episode（48小时）...")
    obs, info = env.reset(seed=42)
    
    for step in range(48):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 12 == 0:
            print(f"  Hour {step}: Health={obs[4]:.1f}, Moisture={obs[0]:.2%}, Temp={obs[1]:.1f}°C")
        
        if terminated or truncated:
            break
    
    print(f"\n最终状态:")
    print(f"  平均健康度: {info['avg_health']:.1f}")
    print(f"  总用水量: {info['total_water_used']:.1f} ml")
    
    env.close()
    print("✅ 环境演示完成")
    
    input("\n按Enter继续...")


def demo_baselines():
    """演示基线策略"""
    print_section("2. 基线策略演示")
    
    from src.environment import PlantCareEnv
    from src.baselines import FixedSchedulePolicy, ThresholdRulePolicy
    import numpy as np
    
    env = PlantCareEnv(config_path="config.yaml")
    
    # 固定时间表
    print("📅 固定时间表策略:")
    policy1 = FixedSchedulePolicy(config_path="config.yaml")
    print(f"  浇水时间: {policy1.water_times}")
    print(f"  灯光时间: {policy1.lamp_schedule}")
    
    obs, info = env.reset(seed=42)
    for _ in range(24):
        action = policy1.get_action(obs)
        obs, _, _, _, info = env.step(action)
    
    print(f"  24小时后健康度: {obs[4]:.1f}")
    
    # 阈值规则
    print("\n📊 阈值规则策略:")
    policy2 = ThresholdRulePolicy(config_path="config.yaml")
    print(f"  湿度阈值: {policy2.moisture_threshold:.1%}")
    print(f"  光照阈值: {policy2.light_threshold} lux")
    
    obs, info = env.reset(seed=42)
    for _ in range(24):
        action = policy2.get_action(obs)
        obs, _, _, _, info = env.step(action)
    
    print(f"  24小时后健康度: {obs[4]:.1f}")
    
    env.close()
    print("\n✅ 基线策略演示完成")
    
    input("\n按Enter继续...")


def demo_ppo_training():
    """演示PPO训练（短时间）"""
    print_section("3. PPO训练演示")
    
    print("这将演示PPO训练流程（仅10,000步，约1分钟）")
    print("完整训练需要5,000,000步（GPU下约3小时）")
    print()
    
    response = input("是否继续? (y/n): ")
    if response.lower() != 'y':
        print("跳过训练演示")
        return
    
    from src.agents.train_ppo import train_ppo_agent
    
    print("\n开始训练...")
    
    model = train_ppo_agent(
        config_path="config.yaml",
        total_timesteps=10_000,  # 仅用于演示
        device="auto",
        save_path="./models/demo/",
        log_path="./logs/demo/",
        seed=42
    )
    
    print("\n✅ 训练演示完成")
    print("💡 提示: 完整训练请运行:")
    print("   python src/agents/train_ppo.py --timesteps 5000000")


def demo_visualization():
    """演示可视化"""
    print_section("4. 可视化演示")
    
    print("生成对比图表...")
    
    from src.utils.visualization import plot_comparison_table, plot_metrics_comparison_bars
    
    # 模拟结果数据
    results = {
        "固定时间表": {
            "avg_health_mean": 60,
            "avg_health_std": 3,
            "total_water_mean": 10200,
            "total_energy_mean": 360000,
            "violations_mean": 120,
            "efficiency_mean": 0.50
        },
        "阈值规则": {
            "avg_health_mean": 70,
            "avg_health_std": 4,
            "total_water_mean": 8500,
            "total_energy_mean": 320000,
            "violations_mean": 80,
            "efficiency_mean": 0.72
        }
    }
    
    os.makedirs("docs/images", exist_ok=True)
    
    plot_comparison_table(results, save_path="docs/images/demo_table.png")
    plot_metrics_comparison_bars(results, save_path="docs/images/demo_bars.png")
    
    print("✅ 图表已保存在 docs/images/")
    print("  - demo_table.png")
    print("  - demo_bars.png")


def main():
    """主函数"""
    print("=" * 60)
    print("智能植物护理系统 - 快速开始演示".center(60))
    print("=" * 60)
    print()
    print("这个演示将引导你了解项目的主要功能:")
    print("  1. 植物护理环境")
    print("  2. 基线策略")
    print("  3. PPO训练（可选）")
    print("  4. 可视化工具")
    print()
    
    try:
        demo_environment()
        demo_baselines()
        demo_ppo_training()
        demo_visualization()
        
        print_section("演示完成!")
        print("🎉 恭喜！你已经了解了项目的主要功能")
        print()
        print("下一步:")
        print("  1. 运行完整基线对比: python run_baseline_comparison.py")
        print("  2. 训练完整PPO模型: python src/agents/train_ppo.py --timesteps 5000000")
        print("  3. 查看TensorBoard: tensorboard --logdir logs/")
        print()
        print("详细信息请查看 README.md")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        print("请确保已运行 setup.sh 并激活虚拟环境")


if __name__ == "__main__":
    main()

