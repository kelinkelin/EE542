"""
Threshold Rule Baseline Policy
Simple rule based on sensor readings:
- Water when soil moisture below 30%
- Turn on lamp when light below 200 lux
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.environment import PlantCareEnv
from typing import Dict
import yaml


class ThresholdRulePolicy:
    """
    Threshold Rule Policy (slightly smarter than fixed schedule)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize threshold parameters"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.moisture_threshold = config['baselines']['threshold_rule']['moisture_threshold']
        self.water_amount = config['baselines']['threshold_rule']['water_amount']
        self.light_threshold = config['baselines']['threshold_rule']['light_threshold']
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Decide action based on sensor readings
        
        Args:
            observation: [soil_moisture, temperature, light_level, hour_of_day, plant_health, hours_since_water]
            
        Returns:
            action: [water_amount, lamp_on]
        """
        soil_moisture = observation[0]
        light_level = observation[2]
        
        # Determine watering (moisture below threshold)
        water_amount = self.water_amount if soil_moisture < self.moisture_threshold else 0.0
        
        # Determine lamp (light below threshold)
        lamp_on = 1.0 if light_level < self.light_threshold else 0.0
        
        return np.array([water_amount, lamp_on], dtype=np.float32)


def evaluate_policy(
    policy: ThresholdRulePolicy,
    env: PlantCareEnv,
    n_episodes: int = 5,
    seed: int = 42
) -> Dict:
    """Evaluate policy performance"""
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
        
        # Record metrics
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
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Avg Health={avg_health:.1f}, "
              f"Water={total_water:.1f}ml, "
              f"Violations={violations}")
    
    # Calculate statistics
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
    print("Threshold Rule Baseline Policy Evaluation")
    print("=" * 60 + "\n")
    
    # Create environment
    config_path = "../../config.yaml"
    env = PlantCareEnv(config_path=config_path)
    
    # Create policy
    policy = ThresholdRulePolicy(config_path=config_path)
    
    print("Policy configuration:")
    print(f"  Moisture threshold: {policy.moisture_threshold:.1%}")
    print(f"  Water amount: {policy.water_amount} ml")
    print(f"  Light threshold: {policy.light_threshold} lux")
    print()
    
    # Evaluate policy
    print("Starting evaluation (5 episodes, 30 days each)...\n")
    summary, results = evaluate_policy(policy, env, n_episodes=5, seed=42)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results (mean ± std)")
    print("=" * 60)
    print(f"Average health: {summary['avg_health_mean']:.1f} ± {summary['avg_health_std']:.1f}")
    print(f"Final health: {summary['final_health_mean']:.1f}")
    print(f"Total water: {summary['total_water_mean']:.1f} ml")
    print(f"Total energy: {summary['total_energy_mean']:.1f} Wh")
    print(f"Violations: {summary['violations_mean']:.1f} hours")
    print(f"Resource efficiency: {summary['efficiency_mean']:.3f}")
    print("=" * 60)
