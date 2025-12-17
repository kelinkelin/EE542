"""
Visualization Tools
Generate comparison charts, training curves, action timelines, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_comparison_table(results: Dict[str, Dict], save_path: str = None):
    """
    Draw comparison table for different policies
    
    Args:
        results: Format {"policy_name": {"avg_health": 80, "total_water": 10000, ...}}
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    policies = list(results.keys())
    metrics = ['Avg Health', 'Water (L)', 'Energy (kWh)', 'Violations (h)', 'Efficiency']
    table_data = []
    
    for policy in policies:
        r = results[policy]
        row = [
            f"{r['avg_health_mean']:.1f} ± {r.get('avg_health_std', 0):.1f}",
            f"{r['total_water_mean'] / 1000:.2f}",  # ml to L
            f"{r['total_energy_mean'] / 1000:.2f}",  # Wh to kWh
            f"{r['violations_mean']:.0f}",
            f"{r['efficiency_mean']:.3f}"
        ]
        table_data.append(row)
    
    # Transpose table (metrics as rows, policies as columns)
    table_data_transposed = [
        [metrics[i]] + [table_data[j][i] for j in range(len(policies))]
        for i in range(len(metrics))
    ]
    
    # Create table
    table = ax.table(
        cellText=table_data_transposed,
        colLabels=['Metric'] + policies,
        cellLoc='center',
        loc='center',
        colWidths=[0.2] + [0.15] * len(policies)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Set header style
    for i in range(len(policies) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set metric column style
    for i in range(1, len(metrics) + 1):
        table[(i, 0)].set_facecolor('#E8F5E9')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Plant Care Policy Performance Comparison', fontsize=16, weight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison table saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(log_path: str, save_path: str = None):
    """
    Draw training curves (read from TensorBoard logs)
    
    Args:
        log_path: TensorBoard log path
        save_path: Save path
    """
    # Note: This requires parsing data from TensorBoard logs
    # Providing a simplified version here
    print("Drawing training curves requires parsing data from TensorBoard logs")
    print("Please use: tensorboard --logdir logs/")


def plot_action_timeline(
    observations: List[np.ndarray],
    actions: List[np.ndarray],
    save_path: str = None
):
    """
    Draw 24-hour action timeline
    
    Args:
        observations: Observation sequence
        actions: Action sequence
        save_path: Save path
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    hours = [obs[3] for obs in observations]  # hour_of_day
    
    # 1. Health curve
    health = [obs[4] for obs in observations]
    axes[0].plot(hours, health, color='green', linewidth=2, label='Plant Health')
    axes[0].fill_between(hours, health, alpha=0.3, color='green')
    axes[0].set_ylabel('Health', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Soil moisture
    moisture = [obs[0] * 100 for obs in observations]  # Convert to percentage
    axes[1].plot(hours, moisture, color='blue', linewidth=2, label='Soil Moisture')
    axes[1].axhspan(40, 70, alpha=0.2, color='green', label='Optimal Range')
    axes[1].set_ylabel('Moisture (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Watering events
    water_amounts = [act[0] for act in actions]
    water_times = [hours[i] for i in range(len(actions)) if actions[i][0] > 5]
    water_values = [actions[i][0] for i in range(len(actions)) if actions[i][0] > 5]
    
    axes[2].bar(water_times, water_values, width=0.5, color='cyan', 
                edgecolor='blue', linewidth=1.5, label='Watering')
    axes[2].set_ylabel('Water (ml)', fontsize=12)
    axes[2].set_ylim(0, 100)
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Lamp status
    lamp_status = [act[1] for act in actions]
    axes[3].fill_between(hours, lamp_status, step='mid', alpha=0.6, 
                         color='yellow', label='Lamp Status')
    axes[3].set_ylabel('Lamp', fontsize=12)
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(['OFF', 'ON'])
    axes[3].set_xlabel('Hour', fontsize=12)
    axes[3].legend(loc='upper left')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Plant Care 24-Hour Action Timeline', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action timeline saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison_bars(results: Dict[str, Dict], save_path: str = None):
    """
    Draw metric comparison bar charts for different policies
    
    Args:
        results: Policy results dictionary
        save_path: Save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    policies = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # 1. Average health
    health_means = [results[p]['avg_health_mean'] for p in policies]
    health_stds = [results[p].get('avg_health_std', 0) for p in policies]
    axes[0, 0].bar(policies, health_means, yerr=health_stds, color=colors, alpha=0.8, capsize=5)
    axes[0, 0].set_ylabel('Average Health', fontsize=12)
    axes[0, 0].set_title('Average Health Comparison', fontsize=14, weight='bold')
    axes[0, 0].axhline(y=85, color='green', linestyle='--', label='Target: 85')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Water usage
    water_means = [results[p]['total_water_mean'] / 1000 for p in policies]  # Convert to L
    axes[0, 1].bar(policies, water_means, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Water Usage (L)', fontsize=12)
    axes[0, 1].set_title('Water Usage Comparison', fontsize=14, weight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Violations
    violations = [results[p]['violations_mean'] for p in policies]
    axes[1, 0].bar(policies, violations, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Violation Hours', fontsize=12)
    axes[1, 0].set_title('Constraint Violations Comparison', fontsize=14, weight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Efficiency
    efficiency = [results[p]['efficiency_mean'] for p in policies]
    axes[1, 1].bar(policies, efficiency, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Efficiency', fontsize=12)
    axes[1, 1].set_title('Resource Efficiency Comparison', fontsize=14, weight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Plant Care Policy Comprehensive Comparison', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison bar chart saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example: generate comparison charts
    print("=== Visualization Tools Example ===\n")
    
    # Simulated result data
    results = {
        "Fixed Schedule": {
            "avg_health_mean": 60,
            "avg_health_std": 3,
            "total_water_mean": 10200,
            "total_energy_mean": 360000,
            "violations_mean": 120,
            "efficiency_mean": 0.50
        },
        "Threshold Rule": {
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
    
    # Create output directory
    os.makedirs("../../docs/images", exist_ok=True)
    
    # Generate comparison table
    plot_comparison_table(results, save_path="../../docs/images/comparison_table.png")
    
    # Generate comparison bar chart
    plot_metrics_comparison_bars(results, save_path="../../docs/images/comparison_bars.png")
    
    print("\nExample charts generated in docs/images/ directory")
