"""
Plant Care Gymnasium Environment
Compliant with OpenAI Gym interface specification
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
import yaml

from .physics import PlantPhysics


class PlantCareEnv(gym.Env):
    """
    Plant Care Reinforcement Learning Environment
    
    State Space:
        - soil_moisture: Soil moisture [0, 1]
        - temperature: Temperature [0, 50]°C
        - light_level: Light intensity [0, 2000] lux
        - time_of_day: Hour of day [0, 23]
        - plant_health: Plant health [0, 100]
        - hours_since_water: Hours since last watering [0, 24]
    
    Action Space:
        - water_amount: Water amount [0, 100] ml (continuous)
        - lamp_on: Lamp switch {0, 1} (discrete)
    
    Reward:
        R = α·Δhealth - β·water_used - γ·energy_used - δ·violations
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config_path: str = "config.yaml", weather_scenario: str = "normal"):
        """
        Initialize environment
        
        Args:
            config_path: Configuration file path
            weather_scenario: Weather scenario ("normal", "hot_dry", "cloudy")
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.weather_scenario = weather_scenario
        self.physics = PlantPhysics(self.config)
        self.weather_hook = None
        
        # Extract key parameters
        self.timestep_hours = self.config['environment']['timestep_hours']
        self.episode_days = self.config['environment']['episode_days']
        self.max_steps = self.episode_days * 24 // self.timestep_hours
        
        # Define state space (Box - continuous space)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # [moisture, temp, light, hour, health, hours_since_water]
            high=np.array([1.0, 50.0, 2000.0, 23.0, 100.0, 24.0]),
            dtype=np.float32
        )
        
        # Define action space (simplified to Box)
        # [water_amount (0-100ml), lamp_on (0-1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([100.0, 1.0]),
            dtype=np.float32
        )
        
        # Reward weights
        self.alpha = self.config['reward']['alpha']
        self.beta = self.config['reward']['beta']
        self.gamma = self.config['reward']['gamma']
        self.delta = self.config['reward']['delta']
        
        # Constraint thresholds
        self.constraints = self.config['reward']['constraints']
        
        # Initialize state variables
        self.current_step = 0
        self.soil_moisture = 0.0
        self.temperature = 0.0
        self.light_level = 0.0
        self.plant_health = 0.0
        self.hour_of_day = 0
        self.hours_since_water = 0
        
        # Statistics
        self.total_water_used = 0.0
        self.total_energy_used = 0.0
        self.total_violations = 0
        self.health_history = []
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state
        
        Returns:
            observation: Initial observation
            info: Additional info dictionary
        """
        super().reset(seed=seed)
        
        # Reset time
        self.current_step = 0
        self.hour_of_day = 0
        self.hours_since_water = 0
        
        # Reset plant state
        self.soil_moisture = self.config['environment']['soil']['initial_moisture']
        self.plant_health = self.config['environment']['plant']['initial_health']
        
        # Get initial environmental conditions
        if hasattr(self, 'weather_hook') and self.weather_hook:
            self.temperature, ambient_light = self.weather_hook(self.hour_of_day, self.weather_scenario)
        else:
            self.temperature, ambient_light = self.physics.get_ambient_conditions(
                self.hour_of_day, self.weather_scenario
            )
        self.light_level = ambient_light  
        
        # Reset statistics
        self.total_water_used = 0.0
        self.total_energy_used = 0.0
        self.total_violations = 0
        self.health_history = [self.plant_health]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one action step
        
        Args:
            action: [water_amount, lamp_on]
            
        Returns:
            observation: New observation
            reward: Reward
            terminated: Whether terminated (plant died)
            truncated: Whether truncated (max steps reached)
            info: Additional info
        """
        # Parse action
        water_amount = np.clip(action[0], 0, 100)  # ml
        lamp_on = 1 if action[1] > 0.5 else 0  # Binarize
        
        # Record previous health
        previous_health = self.plant_health
        
        # Get environmental conditions
        if hasattr(self, 'weather_hook') and self.weather_hook:
            self.temperature, ambient_light = self.weather_hook(self.hour_of_day, self.weather_scenario)
        else:
            self.temperature, ambient_light = self.physics.get_ambient_conditions(
                self.hour_of_day, self.weather_scenario
            )
        
        # Lamp contribution (if on, add 500 lux)
        lamp_contribution = 500 if lamp_on else 0
        self.light_level = ambient_light + lamp_contribution
        
        # Update soil moisture
        self.soil_moisture = self.physics.update_soil_moisture(
            self.soil_moisture,
            water_amount,
            self.temperature,
            self.light_level,
            dt=self.timestep_hours
        )
        
        # Calculate photosynthesis and stress
        photosynthesis = self.physics.calculate_photosynthesis(
            self.light_level, self.soil_moisture, self.temperature
        )
        stress = self.physics.calculate_stress(self.soil_moisture, self.temperature)
        
        # Update plant health
        self.plant_health = self.physics.update_plant_health(
            self.plant_health,
            photosynthesis,
            stress,
            dt=self.timestep_hours
        )
        
        # Update time
        self.current_step += 1
        self.hour_of_day = (self.hour_of_day + self.timestep_hours) % 24
        
        # Update watering timer
        if water_amount > 5:  # Only counts as effective watering if > 5ml
            self.hours_since_water = 0
        else:
            self.hours_since_water = min(self.hours_since_water + self.timestep_hours, 24)
        
        # Calculate reward
        reward = self._calculate_reward(
            previous_health,
            water_amount,
            lamp_on
        )
        
        # Update statistics
        self.total_water_used += water_amount
        self.total_energy_used += lamp_contribution * self.timestep_hours
        self.health_history.append(self.plant_health)
        
        # Check termination conditions
        terminated = self.plant_health < 10.0  # Plant died
        truncated = self.current_step >= self.max_steps  # Max steps reached
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        return np.array([
            self.soil_moisture,
            self.temperature,
            self.light_level,
            float(self.hour_of_day),
            self.plant_health,
            float(self.hours_since_water)
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            'total_water_used': self.total_water_used,
            'total_energy_used': self.total_energy_used,
            'total_violations': self.total_violations,
            'avg_health': np.mean(self.health_history),
            'current_step': self.current_step
        }
        
    def set_weather_provider(self, provider_fn):
        """Optional external weather provider, signature: provider_fn(hour_of_day, weather_scenario) -> (temperature, ambient_light)"""
        self.weather_hook = provider_fn
    
    def _calculate_reward(
        self,
        previous_health: float,
        water_amount: float,
        lamp_on: int
    ) -> float:
        """
        Calculate reward
        
        R = α·Δhealth - β·water_used - γ·energy_used - δ·violations
        """
        # Health change
        health_delta = self.plant_health - previous_health
        
        # Resource consumption
        water_penalty = water_amount
        energy_penalty = 500 * self.timestep_hours if lamp_on else 0
        
        # Constraint violation detection
        violations = 0
        if self.soil_moisture < self.constraints['moisture_min']:
            violations += 1
        if self.soil_moisture > self.constraints['moisture_max']:
            violations += 1
        if self.temperature < self.constraints['temp_min']:
            violations += 1
        if self.temperature > self.constraints['temp_max']:
            violations += 1
        
        self.total_violations += violations
        
        # Calculate total reward
        reward = (
            self.alpha * health_delta
            - self.beta * water_penalty
            - self.gamma * energy_penalty
            - self.delta * violations
        )
        
        return reward
    
    def render(self, mode='human'):
        """Render environment (optional implementation)"""
        if mode == 'human':
            print(f"Step {self.current_step}/{self.max_steps} | "
                  f"Hour {self.hour_of_day} | "
                  f"Health: {self.plant_health:.1f} | "
                  f"Moisture: {self.soil_moisture:.2%} | "
                  f"Temp: {self.temperature:.1f}°C")
        return None


# Register environment
gym.register(
    id='PlantCare-v0',
    entry_point='environment.plant_env:PlantCareEnv',
    max_episode_steps=720,  # 30 days × 24 hours
)


if __name__ == "__main__":
    # Test environment
    print("=== Testing PlantCareEnv ===\n")
    
    env = PlantCareEnv(config_path="../../config.yaml")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max steps: {env.max_steps}\n")
    
    # Run a random episode
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}\n")
    
    total_reward = 0
    for step in range(48):  # Run 48 hours (2 days)
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 12 == 0:  # Print every 12 hours
            env.render()
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Average health: {info['avg_health']:.1f}")
    print(f"Total water used: {info['total_water_used']:.1f} ml")
    print(f"Total energy used: {info['total_energy_used']:.1f} Wh")
