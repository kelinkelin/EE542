"""
植物护理Gymnasium环境
符合OpenAI Gym接口规范
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
import yaml

from .physics import PlantPhysics


class PlantCareEnv(gym.Env):
    """
    植物护理强化学习环境
    
    状态空间：
        - soil_moisture: 土壤湿度 [0, 1]
        - temperature: 温度 [0, 50]°C
        - light_level: 光照强度 [0, 2000] lux
        - time_of_day: 一天中的小时 [0, 23]
        - plant_health: 植物健康度 [0, 100]
        - hours_since_water: 距离上次浇水的小时数 [0, 24]
    
    动作空间：
        - water_amount: 浇水量 [0, 100] ml（连续）
        - lamp_on: 灯光开关 {0, 1}（离散）
    
    奖励：
        R = α·Δhealth - β·water_used - γ·energy_used - δ·violations
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config_path: str = "config.yaml", weather_scenario: str = "normal"):
        """
        初始化环境
        
        Args:
            config_path: 配置文件路径
            weather_scenario: 天气场景 ("normal", "hot_dry", "cloudy")
        """
        super().__init__()
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.weather_scenario = weather_scenario
        self.physics = PlantPhysics(self.config)
        self.weather_hook = None
        
        # 提取关键参数
        self.timestep_hours = self.config['environment']['timestep_hours']
        self.episode_days = self.config['environment']['episode_days']
        self.max_steps = self.episode_days * 24 // self.timestep_hours
        
        # 定义状态空间（Box - 连续空间）
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # [moisture, temp, light, hour, health, hours_since_water]
            high=np.array([1.0, 50.0, 2000.0, 23.0, 100.0, 24.0]),
            dtype=np.float32
        )
        
        # 定义动作空间（MultiDiscrete + Box的混合 - 简化为Box）
        # [water_amount (0-100ml), lamp_on (0-1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([100.0, 1.0]),
            dtype=np.float32
        )
        
        # 奖励权重
        self.alpha = self.config['reward']['alpha']
        self.beta = self.config['reward']['beta']
        self.gamma = self.config['reward']['gamma']
        self.delta = self.config['reward']['delta']
        
        # 约束阈值
        self.constraints = self.config['reward']['constraints']
        
        # 初始化状态变量
        self.current_step = 0
        self.soil_moisture = 0.0
        self.temperature = 0.0
        self.light_level = 0.0
        self.plant_health = 0.0
        self.hour_of_day = 0
        self.hours_since_water = 0
        
        # 统计信息
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
        重置环境到初始状态
        
        Returns:
            observation: 初始观察
            info: 额外信息字典
        """
        super().reset(seed=seed)
        
        # 重置时间
        self.current_step = 0
        self.hour_of_day = 0
        self.hours_since_water = 0
        
        # 重置植物状态
        self.soil_moisture = self.config['environment']['soil']['initial_moisture']
        self.plant_health = self.config['environment']['plant']['initial_health']
        
        # 获取初始环境条件
        if hasattr(self, 'weather_hook') and self.weather_hook:
            self.temperature, ambient_light = self.weather_hook(self.hour_of_day, self.weather_scenario)
        else:
            self.temperature, ambient_light = self.physics.get_ambient_conditions(
                self.hour_of_day, self.weather_scenario
            )
        self.light_level = ambient_light  
        
        # 重置统计
        self.total_water_used = 0.0
        self.total_energy_used = 0.0
        self.total_violations = 0
        self.health_history = [self.plant_health]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: [water_amount, lamp_on]
            
        Returns:
            observation: 新观察
            reward: 奖励
            terminated: 是否终止（植物死亡）
            truncated: 是否截断（达到最大步数）
            info: 额外信息
        """
        # 解析动作
        water_amount = np.clip(action[0], 0, 100)  # ml
        lamp_on = 1 if action[1] > 0.5 else 0  # 二值化
        
        # 记录上一时刻的健康度
        previous_health = self.plant_health
        
        # 获取环境条件
        if hasattr(self, 'weather_hook') and self.weather_hook:
            self.temperature, ambient_light = self.weather_hook(self.hour_of_day, self.weather_scenario)
        else:
            self.temperature, ambient_light = self.physics.get_ambient_conditions(
                self.hour_of_day, self.weather_scenario
            )
        
        # 灯光贡献（如果开灯，增加500 lux）
        lamp_contribution = 500 if lamp_on else 0
        self.light_level = ambient_light + lamp_contribution
        
        # 更新土壤湿度
        self.soil_moisture = self.physics.update_soil_moisture(
            self.soil_moisture,
            water_amount,
            self.temperature,
            self.light_level,
            dt=self.timestep_hours
        )
        
        # 计算光合作用和压力
        photosynthesis = self.physics.calculate_photosynthesis(
            self.light_level, self.soil_moisture, self.temperature
        )
        stress = self.physics.calculate_stress(self.soil_moisture, self.temperature)
        
        # 更新植物健康度
        self.plant_health = self.physics.update_plant_health(
            self.plant_health,
            photosynthesis,
            stress,
            dt=self.timestep_hours
        )
        
        # 更新时间
        self.current_step += 1
        self.hour_of_day = (self.hour_of_day + self.timestep_hours) % 24
        
        # 更新浇水计时器
        if water_amount > 5:  # 浇水量大于5ml才算有效浇水
            self.hours_since_water = 0
        else:
            self.hours_since_water = min(self.hours_since_water + self.timestep_hours, 24)
        
        # 计算奖励
        reward = self._calculate_reward(
            previous_health,
            water_amount,
            lamp_on
        )
        
        # 更新统计
        self.total_water_used += water_amount
        self.total_energy_used += lamp_contribution * self.timestep_hours
        self.health_history.append(self.plant_health)
        
        # 检查终止条件
        terminated = self.plant_health < 10.0  # 植物死亡
        truncated = self.current_step >= self.max_steps  # 达到最大步数
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        return np.array([
            self.soil_moisture,
            self.temperature,
            self.light_level,
            float(self.hour_of_day),
            self.plant_health,
            float(self.hours_since_water)
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            'total_water_used': self.total_water_used,
            'total_energy_used': self.total_energy_used,
            'total_violations': self.total_violations,
            'avg_health': np.mean(self.health_history),
            'current_step': self.current_step
        }
        
    def set_weather_provider(self, provider_fn):
        """可选外部天气提供器，签名: provider_fn(hour_of_day, weather_scenario) -> (temperature, ambient_light)"""
        self.weather_hook = provider_fn
    
    def _calculate_reward(
        self,
        previous_health: float,
        water_amount: float,
        lamp_on: int
    ) -> float:
        """
        计算奖励
        
        R = α·Δhealth - β·water_used - γ·energy_used - δ·violations
        """
        # 健康度变化
        health_delta = self.plant_health - previous_health
        
        # 资源消耗
        water_penalty = water_amount
        energy_penalty = 500 * self.timestep_hours if lamp_on else 0
        
        # 约束违规检测
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
        
        # 计算总奖励
        reward = (
            self.alpha * health_delta
            - self.beta * water_penalty
            - self.gamma * energy_penalty
            - self.delta * violations
        )
        
        return reward
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            print(f"Step {self.current_step}/{self.max_steps} | "
                  f"Hour {self.hour_of_day} | "
                  f"Health: {self.plant_health:.1f} | "
                  f"Moisture: {self.soil_moisture:.2%} | "
                  f"Temp: {self.temperature:.1f}°C")
        return None


# 注册环境
gym.register(
    id='PlantCare-v0',
    entry_point='environment.plant_env:PlantCareEnv',
    max_episode_steps=720,  # 30天 × 24小时
)


if __name__ == "__main__":
    # 测试环境
    print("=== 测试PlantCareEnv ===\n")
    
    env = PlantCareEnv(config_path="../../config.yaml")
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"最大步数: {env.max_steps}\n")
    
    # 运行一个随机episode
    obs, info = env.reset(seed=42)
    print(f"初始观察: {obs}")
    print(f"初始信息: {info}\n")
    
    total_reward = 0
    for step in range(48):  # 运行48小时（2天）
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 12 == 0:  # 每12小时打印一次
            env.render()
        
        if terminated or truncated:
            print(f"\nEpisode结束于step {step}")
            break
    
    print(f"\n总奖励: {total_reward:.2f}")
    print(f"平均健康度: {info['avg_health']:.1f}")
    print(f"总用水量: {info['total_water_used']:.1f} ml")
    print(f"总能耗: {info['total_energy_used']:.1f} Wh")

