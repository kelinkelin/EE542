"""
植物生长物理模型
包括：土壤水分动力学、光合作用、健康度计算
"""

import numpy as np
from typing import Dict, Tuple


class PlantPhysics:
    """植物物理模拟器 - 基于真实植物生理学"""
    
    def __init__(self, config: Dict):
        """
        初始化物理模型参数
        
        Args:
            config: 配置字典，包含所有物理参数
        """
        self.config = config
        
        # 提取关键参数
        self.soil_capacity = config['environment']['soil']['capacity']
        self.evap_base = config['environment']['soil']['evaporation_rate']
        self.temp_evap_coeff = config['environment']['soil']['temp_evap_coeff']
        
        # 植物最优范围
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
        dt: float = 1.0  # 时间步长（小时）
    ) -> float:
        """
        更新土壤湿度（考虑蒸发和浇水）
        
        蒸发模型：E = base_rate * (1 + temp_coeff * T) * (1 + 0.0005 * light)
        吸收模型：moisture += water / soil_capacity
        
        Args:
            current_moisture: 当前土壤湿度 [0, 1]
            water_added: 添加的水量（ml）
            temperature: 当前温度（°C）
            light_level: 当前光照强度（lux）
            dt: 时间步长（小时）
            
        Returns:
            新的土壤湿度 [0, 1]
        """
        # 蒸发：温度越高、光照越强，蒸发越快
        evaporation_rate = self.evap_base * (
            1 + self.temp_evap_coeff * (temperature - 20)
        ) * (1 + 0.0005 * light_level)
        
        # 蒸发量（取决于当前湿度，湿度越高蒸发越快）
        evaporation = evaporation_rate * current_moisture * dt
        
        # 浇水增加湿度（归一化到[0, 1]）
        water_absorption = water_added / (self.soil_capacity * 1000)  # 假设容量1L = 1000ml
        
        # 更新湿度
        new_moisture = current_moisture - evaporation + water_absorption
        
        # 限制在[0, 1]范围
        return np.clip(new_moisture, 0.0, 1.0)
    
    def calculate_photosynthesis(
        self,
        light_level: float,
        moisture: float,
        temperature: float
    ) -> float:
        """
        计算光合作用效率（影响植物生长）
        
        使用Michaelis-Menten动力学模型：
        P = P_max * (L / (K_L + L)) * water_factor * temp_factor
        
        Args:
            light_level: 光照强度（lux）
            moisture: 土壤湿度 [0, 1]
            temperature: 温度（°C）
            
        Returns:
            光合作用效率 [0, 1]
        """
        # 光照响应曲线（饱和曲线）
        K_light = 300  # 半饱和常数
        light_factor = light_level / (K_light + light_level)
        
        # 水分限制因子（低于阈值线性下降）
        if moisture > 0.3:
            water_factor = 1.0
        else:
            water_factor = moisture / 0.3
            
        # 温度响应曲线（抛物线，最优在23°C）
        optimal_temp = 23.0
        temp_deviation = abs(temperature - optimal_temp)
        temp_factor = np.exp(-0.01 * temp_deviation**2)
        
        # 总光合作用效率
        photosynthesis = light_factor * water_factor * temp_factor
        
        return np.clip(photosynthesis, 0.0, 1.0)
    
    def calculate_stress(
        self,
        moisture: float,
        temperature: float
    ) -> float:
        """
        计算植物压力水平（基于偏离最优条件的程度）
        
        Args:
            moisture: 土壤湿度 [0, 1]
            temperature: 温度（°C）
            
        Returns:
            压力水平 [0, 1]，0表示无压力
        """
        # 水分压力
        if self.optimal_moisture[0] <= moisture <= self.optimal_moisture[1]:
            moisture_stress = 0.0
        else:
            # 偏离最优范围的程度
            if moisture < self.optimal_moisture[0]:
                moisture_stress = (self.optimal_moisture[0] - moisture) / self.optimal_moisture[0]
            else:
                moisture_stress = (moisture - self.optimal_moisture[1]) / (1.0 - self.optimal_moisture[1])
        
        # 温度压力
        if self.optimal_temp[0] <= temperature <= self.optimal_temp[1]:
            temp_stress = 0.0
        else:
            if temperature < self.optimal_temp[0]:
                temp_stress = (self.optimal_temp[0] - temperature) / self.optimal_temp[0]
            else:
                temp_stress = (temperature - self.optimal_temp[1]) / (40 - self.optimal_temp[1])
        
        # 总压力（取最大值，因为任何一个极端都会造成压力）
        total_stress = max(moisture_stress, temp_stress)
        
        return np.clip(total_stress, 0.0, 1.0)
    
    def update_plant_health(
        self,
        current_health: float,
        photosynthesis: float,
        stress: float,
        dt: float = 1.0  # 时间步长（小时）
    ) -> float:
        """
        更新植物健康度
        
        健康度变化 = 光合作用增益 - 压力衰减
        
        Args:
            current_health: 当前健康度 [0, 100]
            photosynthesis: 光合作用效率 [0, 1]
            stress: 压力水平 [0, 1]
            dt: 时间步长（小时）
            
        Returns:
            新的健康度 [0, 100]
        """
        # 健康度增益（来自光合作用）
        health_gain = photosynthesis * 0.5 * dt  # 每小时最多增加0.5
        
        # 健康度衰减（来自压力）
        health_decay = stress * 1.0 * dt  # 压力下每小时最多减少1.0
        
        # 自然衰减（维持代谢）
        natural_decay = 0.05 * dt
        
        # 更新健康度
        new_health = current_health + health_gain - health_decay - natural_decay
        
        # 限制在[0, 100]范围
        return np.clip(new_health, 0.0, 100.0)
    
    def get_ambient_conditions(
        self, 
        hour_of_day: int,
        weather_scenario: str = "normal"
    ) -> Tuple[float, float]:
        """
        获取环境条件（温度、光照）
        
        Args:
            hour_of_day: 一天中的小时数 [0, 23]
            weather_scenario: 天气场景 ("normal", "hot_dry", "cloudy")
            
        Returns:
            (temperature, ambient_light) 温度（°C）和环境光照（lux）
        """
        # 基础昼夜温度变化（正弦波）
        temp_mean = self.config['environment']['weather']['temp_mean']
        temp_amplitude = self.config['environment']['weather']['temp_day_night_diff'] / 2
        temperature = temp_mean + temp_amplitude * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # 基础昼夜光照变化（白天高，晚上低）
        light_max = self.config['environment']['weather']['light_max']
        if 6 <= hour_of_day <= 18:
            # 白天：正弦曲线，中午最强
            ambient_light = light_max * np.sin(np.pi * (hour_of_day - 6) / 12)
        else:
            # 晚上：接近0
            ambient_light = 0.0
        
        # 应用天气场景修正
        if weather_scenario == "hot_dry":
            temperature += 5.0
            ambient_light *= 1.2
        elif weather_scenario == "cloudy":
            temperature -= 2.0
            ambient_light *= 0.6
            
        # 添加随机噪声（模拟天气波动）
        temperature += np.random.normal(0, 1.0)
        ambient_light = max(0, ambient_light + np.random.normal(0, 50))
        
        return temperature, ambient_light


if __name__ == "__main__":
    # 简单测试
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    physics = PlantPhysics(config)
    
    print("=== 植物物理模型测试 ===\n")
    
    # 测试场景：正常一天
    moisture = 0.5
    health = 80.0
    
    print("初始状态：")
    print(f"  土壤湿度: {moisture:.2%}")
    print(f"  植物健康: {health:.1f}/100\n")
    
    for hour in range(24):
        temp, light = physics.get_ambient_conditions(hour)
        
        # 早上8点浇水50ml
        water_added = 50 if hour == 8 else 0
        
        # 更新土壤湿度
        moisture = physics.update_soil_moisture(moisture, water_added, temp, light)
        
        # 计算光合作用和压力
        photosynthesis = physics.calculate_photosynthesis(light, moisture, temp)
        stress = physics.calculate_stress(moisture, temp)
        
        # 更新健康度
        health = physics.update_plant_health(health, photosynthesis, stress)
        
        if hour % 6 == 0:  # 每6小时打印一次
            print(f"Hour {hour:2d}: Temp={temp:.1f}°C, Light={light:.0f}lux, "
                  f"Moisture={moisture:.2%}, Health={health:.1f}")
    
    print(f"\n24小时后最终健康度: {health:.1f}/100")

