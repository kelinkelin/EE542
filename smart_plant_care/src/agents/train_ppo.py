"""
PPO智能体训练脚本
使用Stable-Baselines3库
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import yaml
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import argparse

from src.environment import PlantCareEnv


def train_ppo_agent(
    config_path: str = "config.yaml",
    total_timesteps: int = 5_000_000,
    device: str = "auto",
    save_path: str = "./models/",
    log_path: str = "./logs/",
    seed: int = 42
):
    """
    训练PPO智能体
    
    Args:
        config_path: 配置文件路径
        total_timesteps: 总训练步数
        device: 设备 ("auto", "cuda", "cpu")
        save_path: 模型保存路径
        log_path: 日志保存路径
        seed: 随机种子
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # 检查GPU可用性
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("智能植物护理 - PPO训练")
    print("=" * 60)
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    print(f"总训练步数: {total_timesteps:,}")
    print(f"随机种子: {seed}")
    print("=" * 60 + "\n")
    
    # 创建向量化环境（并行训练）
    print("创建训练环境...")
    n_envs = 4  # 4个并行环境
    env = make_vec_env(
        lambda: PlantCareEnv(config_path=config_path),
        n_envs=n_envs,
        seed=seed
    )
    
    # 创建评估环境
    eval_env = PlantCareEnv(config_path=config_path)
    
    # 配置PPO参数
    ppo_config = config['ppo']
    
    print("初始化PPO智能体...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        policy_kwargs=dict(
            net_arch=dict(
                pi=ppo_config['policy_network']['net_arch'],
                vf=ppo_config['value_network']['net_arch']
            )
        ),
        device=device,
        verbose=1,
        seed=seed,
        tensorboard_log=log_path
    )
    
    # 配置回调
    print("配置训练回调...")
    
    # 评估回调（每10k步评估一次）
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # 检查点回调（每50k步保存一次）
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,  # 除以环境数量
        save_path=save_path,
        name_prefix="ppo_checkpoint"
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    # 开始训练
    print("\n开始训练...\n")
    print("=" * 60)
    
    import time
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    
    elapsed_time = time.time() - start_time
    
    # 保存最终模型
    final_model_path = os.path.join(save_path, "ppo_final_model")
    model.save(final_model_path)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"训练时长: {elapsed_time / 3600:.2f} 小时")
    print(f"最终模型保存于: {final_model_path}.zip")
    print(f"最佳模型保存于: {os.path.join(save_path, 'best_model.zip')}")
    print("=" * 60)
    
    # 清理
    env.close()
    eval_env.close()
    
    return model


def test_trained_model(model_path: str, config_path: str = "config.yaml", n_episodes: int = 5):
    """
    测试训练好的模型
    
    Args:
        model_path: 模型文件路径（.zip）
        config_path: 配置文件路径
        n_episodes: 测试episode数量
    """
    print("=" * 60)
    print("测试训练好的PPO模型")
    print("=" * 60 + "\n")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)
    
    # 创建环境
    env = PlantCareEnv(config_path=config_path)
    
    # 测试
    results = {
        'avg_health': [],
        'total_water': [],
        'total_energy': [],
        'violations': []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=42 + episode)
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        results['avg_health'].append(info['avg_health'])
        results['total_water'].append(info['total_water_used'])
        results['total_energy'].append(info['total_energy_used'])
        results['violations'].append(info['total_violations'])
        
        print(f"Episode {episode + 1}: Health={info['avg_health']:.1f}, "
              f"Water={info['total_water_used']:.1f}ml, Reward={total_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"平均健康度: {np.mean(results['avg_health']):.1f} ± {np.std(results['avg_health']):.1f}")
    print(f"平均用水量: {np.mean(results['total_water']):.1f} ml")
    print(f"平均能耗: {np.mean(results['total_energy']):.1f} Wh")
    print(f"平均违规: {np.mean(results['violations']):.1f} 小时")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练PPO智能体用于植物护理")
    parser.add_argument("--config", type=str, default="../../config.yaml", help="配置文件路径")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="总训练步数")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="训练设备")
    parser.add_argument("--save_path", type=str, default="../../models/", help="模型保存路径")
    parser.add_argument("--log_path", type=str, default="../../logs/", help="日志保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test", type=str, default=None, help="测试模型路径")
    
    args = parser.parse_args()
    
    if args.test:
        # 测试模式
        test_trained_model(args.test, args.config)
    else:
        # 训练模式
        train_ppo_agent(
            config_path=args.config,
            total_timesteps=args.timesteps,
            device=args.device,
            save_path=args.save_path,
            log_path=args.log_path,
            seed=args.seed
        )

