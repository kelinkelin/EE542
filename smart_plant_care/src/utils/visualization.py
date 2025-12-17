"""
可视化工具
生成对比图表、训练曲线、动作时间线等
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_comparison_table(results: Dict[str, Dict], save_path: str = None):
    """
    绘制不同策略的对比表格
    
    Args:
        results: 格式为 {"策略名": {"avg_health": 80, "total_water": 10000, ...}}
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    policies = list(results.keys())
    metrics = ['平均健康度', '用水量(L)', '能耗(kWh)', '违规(小时)', '资源效率']
    table_data = []
    
    for policy in policies:
        r = results[policy]
        row = [
            f"{r['avg_health_mean']:.1f} ± {r.get('avg_health_std', 0):.1f}",
            f"{r['total_water_mean'] / 1000:.2f}",  # ml转L
            f"{r['total_energy_mean'] / 1000:.2f}",  # Wh转kWh
            f"{r['violations_mean']:.0f}",
            f"{r['efficiency_mean']:.3f}"
        ]
        table_data.append(row)
    
    # 转置表格（指标为行，策略为列）
    table_data_transposed = [
        [metrics[i]] + [table_data[j][i] for j in range(len(policies))]
        for i in range(len(metrics))
    ]
    
    # 创建表格
    table = ax.table(
        cellText=table_data_transposed,
        colLabels=['指标'] + policies,
        cellLoc='center',
        loc='center',
        colWidths=[0.2] + [0.15] * len(policies)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(policies) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置指标列样式
    for i in range(1, len(metrics) + 1):
        table[(i, 0)].set_facecolor('#E8F5E9')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('植物护理策略性能对比', fontsize=16, weight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比表格已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(log_path: str, save_path: str = None):
    """
    绘制训练曲线（从TensorBoard日志读取）
    
    Args:
        log_path: TensorBoard日志路径
        save_path: 保存路径
    """
    # 注意：这需要从TensorBoard日志解析数据
    # 这里提供一个简化版本
    print("绘制训练曲线需要从TensorBoard日志解析数据")
    print("请使用: tensorboard --logdir logs/")


def plot_action_timeline(
    observations: List[np.ndarray],
    actions: List[np.ndarray],
    save_path: str = None
):
    """
    绘制24小时动作时间线
    
    Args:
        observations: 观察序列
        actions: 动作序列
        save_path: 保存路径
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    hours = [obs[3] for obs in observations]  # hour_of_day
    
    # 1. 健康度曲线
    health = [obs[4] for obs in observations]
    axes[0].plot(hours, health, color='green', linewidth=2, label='植物健康度')
    axes[0].fill_between(hours, health, alpha=0.3, color='green')
    axes[0].set_ylabel('健康度', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 土壤湿度
    moisture = [obs[0] * 100 for obs in observations]  # 转为百分比
    axes[1].plot(hours, moisture, color='blue', linewidth=2, label='土壤湿度')
    axes[1].axhspan(40, 70, alpha=0.2, color='green', label='最优范围')
    axes[1].set_ylabel('湿度 (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 浇水事件
    water_amounts = [act[0] for act in actions]
    water_times = [hours[i] for i in range(len(actions)) if actions[i][0] > 5]
    water_values = [actions[i][0] for i in range(len(actions)) if actions[i][0] > 5]
    
    axes[2].bar(water_times, water_values, width=0.5, color='cyan', 
                edgecolor='blue', linewidth=1.5, label='浇水')
    axes[2].set_ylabel('浇水量 (ml)', fontsize=12)
    axes[2].set_ylim(0, 100)
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    # 4. 灯光状态
    lamp_status = [act[1] for act in actions]
    axes[3].fill_between(hours, lamp_status, step='mid', alpha=0.6, 
                         color='yellow', label='灯光状态')
    axes[3].set_ylabel('灯光', fontsize=12)
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(['OFF', 'ON'])
    axes[3].set_xlabel('小时', fontsize=12)
    axes[3].legend(loc='upper left')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('植物护理24小时动作时间线', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"动作时间线已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison_bars(results: Dict[str, Dict], save_path: str = None):
    """
    绘制不同策略的指标对比柱状图
    
    Args:
        results: 策略结果字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    policies = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # 1. 平均健康度
    health_means = [results[p]['avg_health_mean'] for p in policies]
    health_stds = [results[p].get('avg_health_std', 0) for p in policies]
    axes[0, 0].bar(policies, health_means, yerr=health_stds, color=colors, alpha=0.8, capsize=5)
    axes[0, 0].set_ylabel('平均健康度', fontsize=12)
    axes[0, 0].set_title('平均健康度对比', fontsize=14, weight='bold')
    axes[0, 0].axhline(y=85, color='green', linestyle='--', label='目标: 85')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. 用水量
    water_means = [results[p]['total_water_mean'] / 1000 for p in policies]  # 转为L
    axes[0, 1].bar(policies, water_means, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('用水量 (L)', fontsize=12)
    axes[0, 1].set_title('用水量对比', fontsize=14, weight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. 约束违规
    violations = [results[p]['violations_mean'] for p in policies]
    axes[1, 0].bar(policies, violations, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('违规时长 (小时)', fontsize=12)
    axes[1, 0].set_title('约束违规对比', fontsize=14, weight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 资源效率
    efficiency = [results[p]['efficiency_mean'] for p in policies]
    axes[1, 1].bar(policies, efficiency, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('资源效率', fontsize=12)
    axes[1, 1].set_title('资源效率对比', fontsize=14, weight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('植物护理策略综合对比', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比柱状图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # 示例：生成对比图表
    print("=== 可视化工具示例 ===\n")
    
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
        },
        "PPO (RL)": {
            "avg_health_mean": 87,
            "avg_health_std": 2,
            "total_water_mean": 5800,
            "total_energy_mean": 240000,
            "violations_mean": 35,
            "efficiency_mean": 1.12
        }
    }
    
    # 创建输出目录
    os.makedirs("../../docs/images", exist_ok=True)
    
    # 生成对比表格
    plot_comparison_table(results, save_path="../../docs/images/comparison_table.png")
    
    # 生成对比柱状图
    plot_metrics_comparison_bars(results, save_path="../../docs/images/comparison_bars.png")
    
    print("\n示例图表已生成在 docs/images/ 目录")

