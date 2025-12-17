"""
Plant Growth Physics Model
Includes: Soil moisture dynamics, photosynthesis, health calculation
"""

import numpy as np
from typing import Dict, Tuple


class PlantPhysics:
    """Plant Physics Simulator - Based on real plant physiology"""
    
    def __init__(self, config: Dict):
        """
        Initialize physics model parameters
        
        Args:
            config: Configuration dictionary containing all physics parameters
        """
        self.config = config
        
        # Extract key parameters
        self.soil_capacity = config['environment']['soil']['capacity']
        self.evap_base = config['environment']['soil']['evaporation_rate']
        self.temp_evap_coeff = config['environment']['soil']['temp_evap_coeff']
        
        # Plant optimal ranges
        self.optimal_moisture = (
            config['environment']['plant']['optimal_moisture_min'],
            config['environment']['plant']['optimal_moisture_max']
        )
        self.optimal_temp = (
            config['environment']['plant']['optimal_temp_min'],
            config['environment']['plant']['optimal_temp_max']
        )
        
    def update_soil_moisture(
        self, 
        current_moisture: float, 
        water_added: float,  # ml
        temperature: float,
        light_level: float,
        dt: float = 1.0  # Time step (hours)
    ) -> float:
        """
        Update soil moisture (considering evaporation and watering)
        
        Evaporation model: E = base_rate * (1 + temp_coeff * T) * (1 + 0.0005 * light)
        Absorption model: moisture += water / soil_capacity
        
        Args:
            current_moisture: Current soil moisture [0, 1]
            water_added: Amount of water added (ml)
            temperature: Current temperature (°C)
            light_level: Current light intensity (lux)
            dt: Time step (hours)
            
        Returns:
            New soil moisture [0, 1]
        """
        # Evaporation: higher temp and light = faster evaporation
        evaporation_rate = self.evap_base * (
            1 + self.temp_evap_coeff * (temperature - 20)
        ) * (1 + 0.0005 * light_level)
        
        # Evaporation amount (depends on current moisture)
        evaporation = evaporation_rate * current_moisture * dt
        
        # Watering increases moisture (normalized to [0, 1])
        water_absorption = water_added / (self.soil_capacity * 1000)  # Assume capacity 1L = 1000ml
        
        # Update moisture
        new_moisture = current_moisture - evaporation + water_absorption
        
        # Clip to [0, 1] range
        return np.clip(new_moisture, 0.0, 1.0)
    
    def calculate_photosynthesis(
        self,
        light_level: float,
        moisture: float,
        temperature: float
    ) -> float:
        """
        Calculate photosynthesis efficiency (affects plant growth)
        
        Using Michaelis-Menten kinetics model:
        P = P_max * (L / (K_L + L)) * water_factor * temp_factor
        
        Args:
            light_level: Light intensity (lux)
            moisture: Soil moisture [0, 1]
            temperature: Temperature (°C)
            
        Returns:
            Photosynthesis efficiency [0, 1]
        """
        # Light response curve (saturation curve)
        K_light = 300  # Half-saturation constant
        light_factor = light_level / (K_light + light_level)
        
        # Water limitation factor (linear decrease below threshold)
        if moisture > 0.3:
            water_factor = 1.0
        else:
            water_factor = moisture / 0.3
            
        # Temperature response curve (parabolic, optimal at 23°C)
        optimal_temp = 23.0
        temp_deviation = abs(temperature - optimal_temp)
        temp_factor = np.exp(-0.01 * temp_deviation**2)
        
        # Total photosynthesis efficiency
        photosynthesis = light_factor * water_factor * temp_factor
        
        return np.clip(photosynthesis, 0.0, 1.0)
    
    def calculate_stress(
        self,
        moisture: float,
        temperature: float
    ) -> float:
        """
        Calculate plant stress level (based on deviation from optimal conditions)
        
        Args:
            moisture: Soil moisture [0, 1]
            temperature: Temperature (°C)
            
        Returns:
            Stress level [0, 1], 0 means no stress
        """
        # Moisture stress
        if self.optimal_moisture[0] <= moisture <= self.optimal_moisture[1]:
            moisture_stress = 0.0
        else:
            # Degree of deviation from optimal range
            if moisture < self.optimal_moisture[0]:
                moisture_stress = (self.optimal_moisture[0] - moisture) / self.optimal_moisture[0]
            else:
                moisture_stress = (moisture - self.optimal_moisture[1]) / (1.0 - self.optimal_moisture[1])
        
        # Temperature stress
        if self.optimal_temp[0] <= temperature <= self.optimal_temp[1]:
            temp_stress = 0.0
        else:
            if temperature < self.optimal_temp[0]:
                temp_stress = (self.optimal_temp[0] - temperature) / self.optimal_temp[0]
            else:
                temp_stress = (temperature - self.optimal_temp[1]) / (40 - self.optimal_temp[1])
        
        # Total stress (take max, any extreme causes stress)
        total_stress = max(moisture_stress, temp_stress)
        
        return np.clip(total_stress, 0.0, 1.0)
    
    def update_plant_health(
        self,
        current_health: float,
        photosynthesis: float,
        stress: float,
        dt: float = 1.0  # Time step (hours)
    ) -> float:
        """
        Update plant health
        
        Health change = Photosynthesis gain - Stress decay
        
        Args:
            current_health: Current health [0, 100]
            photosynthesis: Photosynthesis efficiency [0, 1]
            stress: Stress level [0, 1]
            dt: Time step (hours)
            
        Returns:
            New health [0, 100]
        """
        # Health gain (from photosynthesis)
        health_gain = photosynthesis * 0.5 * dt  # Max +0.5 per hour
        
        # Health decay (from stress)
        health_decay = stress * 1.0 * dt  # Max -1.0 per hour under stress
        
        # Natural decay (maintenance metabolism)
        natural_decay = 0.05 * dt
        
        # Update health
        new_health = current_health + health_gain - health_decay - natural_decay
        
        # Clip to [0, 100] range
        return np.clip(new_health, 0.0, 100.0)
    
    def get_ambient_conditions(
        self, 
        hour_of_day: int,
        weather_scenario: str = "normal"
    ) -> Tuple[float, float]:
        """
        Get environmental conditions (temperature, light)
        
        Args:
            hour_of_day: Hour of day [0, 23]
            weather_scenario: Weather scenario ("normal", "hot_dry", "cloudy")
            
        Returns:
            (temperature, ambient_light) Temperature (°C) and ambient light (lux)
        """
        # Base day-night temperature variation (sine wave)
        temp_mean = self.config['environment']['weather']['temp_mean']
        temp_amplitude = self.config['environment']['weather']['temp_day_night_diff'] / 2
        temperature = temp_mean + temp_amplitude * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Base day-night light variation (high during day, low at night)
        light_max = self.config['environment']['weather']['light_max']
        if 6 <= hour_of_day <= 18:
            # Daytime: sine curve, strongest at noon
            ambient_light = light_max * np.sin(np.pi * (hour_of_day - 6) / 12)
        else:
            # Nighttime: near 0
            ambient_light = 0.0
        
        # Apply weather scenario adjustments
        if weather_scenario == "hot_dry":
            temperature += 5.0
            ambient_light *= 1.2
        elif weather_scenario == "cloudy":
            temperature -= 2.0
            ambient_light *= 0.6
            
        # Add random noise (simulate weather fluctuations)
        temperature += np.random.normal(0, 1.0)
        ambient_light = max(0, ambient_light + np.random.normal(0, 50))
        
        return temperature, ambient_light


if __name__ == "__main__":
    # Simple test
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    physics = PlantPhysics(config)
    
    print("=== Plant Physics Model Test ===\n")
    
    # Test scenario: normal day
    moisture = 0.5
    health = 80.0
    
    print("Initial state:")
    print(f"  Soil moisture: {moisture:.2%}")
    print(f"  Plant health: {health:.1f}/100\n")
    
    for hour in range(24):
        temp, light = physics.get_ambient_conditions(hour)
        
        # Water 50ml at 8am
        water_added = 50 if hour == 8 else 0
        
        # Update soil moisture
        moisture = physics.update_soil_moisture(moisture, water_added, temp, light)
        
        # Calculate photosynthesis and stress
        photosynthesis = physics.calculate_photosynthesis(light, moisture, temp)
        stress = physics.calculate_stress(moisture, temp)
        
        # Update health
        health = physics.update_plant_health(health, photosynthesis, stress)
        
        if hour % 6 == 0:  # Print every 6 hours
            print(f"Hour {hour:2d}: Temp={temp:.1f}°C, Light={light:.0f}lux, "
                  f"Moisture={moisture:.2%}, Health={health:.1f}")
    
    print(f"\nFinal health after 24 hours: {health:.1f}/100")
