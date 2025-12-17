#!/usr/bin/env python3
"""
生成Week 1演示用的可视化图表
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(filename="data/week1_baseline_results.json"):
    """加载测试结果"""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_comparison_bars(results, save_path="docs/images/week1_comparison_bars.png"):
    """生成对比柱状图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    policies = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 1. 平均健康度
    health_means = [results[p]['avg_health_mean'] for p in policies]
    health_stds = [results[p]['avg_health_std'] for p in policies]
    bars1 = axes[0, 0].bar(policies, health_means, yerr=health_stds, 
                           color=colors, alpha=0.8, capsize=5)
    axes[0, 0].set_ylabel('Average Health Score', fontsize=12)
    axes[0, 0].set_title('Plant Health Comparison', fontsize=14, weight='bold')
    axes[0, 0].axhline(y=85, color='green', linestyle='--', linewidth=2, label='PPO Target: 85')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim(0, 100)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, health_means)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 5, 
                       f'{val:.1f}', ha='center', fontsize=11, weight='bold')
    
    # 2. 用水量
    water_means = [results[p]['total_water_mean'] / 1000 for p in policies]  # 转为L
    bars2 = axes[0, 1].bar(policies, water_means, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Water Usage (L/month)', fontsize=12)
    axes[0, 1].set_title('Water Consumption', fontsize=14, weight='bold')
    axes[0, 1].axhline(y=6, color='green', linestyle='--', linewidth=2, label='PPO Target: <6L')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars2, water_means):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                       f'{val:.2f}L', ha='center', fontsize=11, weight='bold')
    
    # 3. 约束违规
    violations = [results[p]['violations_mean'] for p in policies]
    bars3 = axes[1, 0].bar(policies, violations, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Constraint Violations (hours)', fontsize=12)
    axes[1, 0].set_title('Constraint Violations', fontsize=14, weight='bold')
    axes[1, 0].axhline(y=35, color='green', linestyle='--', linewidth=2, label='PPO Target: <35h')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars3, violations):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 5, 
                       f'{int(val)}h', ha='center', fontsize=11, weight='bold')
    
    # 4. 资源效率
    efficiency = [results[p]['efficiency_mean'] for p in policies]
    bars4 = axes[1, 1].bar(policies, efficiency, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Efficiency Score', fontsize=12)
    axes[1, 1].set_title('Resource Efficiency', fontsize=14, weight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars4, efficiency):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 0.002, 
                       f'{val:.3f}', ha='center', fontsize=11, weight='bold')
    
    plt.suptitle('Week 1: Baseline Performance Comparison', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比柱状图已保存: {save_path}")
    plt.close()


def plot_comparison_table(results, save_path="docs/images/week1_comparison_table.png"):
    """生成对比表格图"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    policies = list(results.keys()) + ["PPO Target"]
    table_data = []
    
    for policy in policies:
        if policy == "PPO Target":
            # PPO目标值
            row = ["87 ± 2", "5.8", "240", "35", "1.12"]
        else:
            r = results[policy]
            row = [
                f"{r['avg_health_mean']:.1f} ± {r['avg_health_std']:.1f}",
                f"{r['total_water_mean'] / 1000:.2f}",
                f"{r['total_energy_mean'] / 1000:.1f}",
                f"{r['violations_mean']:.0f}",
                f"{r['efficiency_mean']:.3f}"
            ]
        table_data.append(row)
    
    # 转置表格
    metrics = ['Avg Health', 'Water (L)', 'Energy (kWh)', 'Violations (h)', 'Efficiency']
    table_data_transposed = [
        [metrics[i]] + [table_data[j][i] for j in range(len(policies))]
        for i in range(len(metrics))
    ]
    
    # 创建表格
    table = ax.table(
        cellText=table_data_transposed,
        colLabels=['Metric'] + policies,
        cellLoc='center',
        loc='center',
        colWidths=[0.18] + [0.2] * len(policies)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(len(policies) + 1):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # 设置指标列样式
    for i in range(1, len(metrics) + 1):
        cell = table[(i, 0)]
        cell.set_facecolor('#E8F5E9')
        cell.set_text_props(weight='bold', fontsize=11)
    
    # 高亮PPO目标列
    for i in range(len(metrics) + 1):
        cell = table[(i, len(policies))]
        if i == 0:
            cell.set_facecolor('#2E7D32')
        else:
            cell.set_facecolor('#C8E6C9')
            cell.set_text_props(weight='bold')
    
    plt.title('Week 1: Baseline Performance Metrics', 
              fontsize=16, weight='bold', pad=20)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比表格已保存: {save_path}")
    plt.close()


def plot_improvement_potential(results, save_path="docs/images/week1_improvement_potential.png"):
    """生成改进潜力图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 健康度改进
    baseline_health = results['固定时间表']['avg_health_mean']
    threshold_health = results['阈值规则']['avg_health_mean']
    target_health = 85
    
    health_data = [baseline_health, threshold_health, target_health]
    health_labels = ['Fixed\nSchedule', 'Threshold\nRule', 'PPO\nTarget']
    colors1 = ['#FF6B6B', '#FFB366', '#4CAF50']
    
    bars = ax1.bar(health_labels, health_data, color=colors1, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Plant Health Score', fontsize=13)
    ax1.set_title('Health Improvement Potential', fontsize=15, weight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值和改进百分比
    for i, (bar, val) in enumerate(zip(bars, health_data)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 3, 
                f'{val:.1f}', ha='center', fontsize=12, weight='bold')
        
        if i > 0:
            improvement = ((val - baseline_health) / baseline_health) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, val/2, 
                    f'+{improvement:.0f}%', ha='center', fontsize=11, 
                    weight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    
    # 用水量对比
    baseline_water = results['固定时间表']['total_water_mean'] / 1000
    threshold_water = results['阈值规则']['total_water_mean'] / 1000
    target_water = 5.8
    
    water_data = [baseline_water, threshold_water, target_water]
    water_labels = ['Fixed\nSchedule', 'Threshold\nRule', 'PPO\nTarget']
    
    bars2 = ax2.bar(water_labels, water_data, color=colors1, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Water Usage (L/month)', fontsize=13)
    ax2.set_title('Water Efficiency Improvement', fontsize=15, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值
    for bar, val in zip(bars2, water_data):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                f'{val:.2f}L', ha='center', fontsize=12, weight='bold')
    
    plt.suptitle('Week 1: Why We Need Reinforcement Learning', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 改进潜力图已保存: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Week 1: 生成演示可视化图表".center(80))
    print("=" * 80 + "\n")
    
    # 加载数据
    try:
        results = load_results()
        print("✅ 测试结果加载成功\n")
    except FileNotFoundError:
        print("❌ 错误: 未找到 data/week1_baseline_results.json")
        print("请先运行: python quick_baseline_test.py")
        return
    
    # 生成图表
    print("生成图表中...\n")
    
    plot_comparison_bars(results)
    plot_comparison_table(results)
    plot_improvement_potential(results)
    
    print("\n" + "=" * 80)
    print("✅ 所有图表生成完成!".center(80))
    print("=" * 80)
    print("\n生成的文件:")
    print("  1. docs/images/week1_comparison_bars.png - 4格对比柱状图")
    print("  2. docs/images/week1_comparison_table.png - 性能对比表格")
    print("  3. docs/images/week1_improvement_potential.png - 改进潜力图")
    print("\n这些图表可以直接插入到PPT中！")
    print("=" * 80)


if __name__ == "__main__":
    main()









