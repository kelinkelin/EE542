"""
PPO Agent Training Script
Using Stable-Baselines3 library
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
    Train PPO agent
    
    Args:
        config_path: Configuration file path
        total_timesteps: Total training steps
        device: Device ("auto", "cuda", "cpu")
        save_path: Model save path
        log_path: Log save path
        seed: Random seed
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create save directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Check GPU availability
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Smart Plant Care - PPO Training")
    print("=" * 60)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Random seed: {seed}")
    print("=" * 60 + "\n")
    
    # Create vectorized environment (parallel training)
    print("Creating training environment...")
    n_envs = 4  # 4 parallel environments
    env = make_vec_env(
        lambda: PlantCareEnv(config_path=config_path),
        n_envs=n_envs,
        seed=seed
    )
    
    # Create evaluation environment
    eval_env = PlantCareEnv(config_path=config_path)
    
    # Configure PPO parameters
    ppo_config = config['ppo']
    
    print("Initializing PPO agent...")
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
    
    # Configure callbacks
    print("Configuring training callbacks...")
    
    # Evaluation callback (evaluate every 10k steps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Checkpoint callback (save every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,  # Divide by number of environments
        save_path=save_path,
        name_prefix="ppo_checkpoint"
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    # Start training
    print("\nStarting training...\n")
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
        print("\nTraining interrupted by user")
    
    elapsed_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(save_path, "ppo_final_model")
    model.save(final_model_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Training duration: {elapsed_time / 3600:.2f} hours")
    print(f"Final model saved at: {final_model_path}.zip")
    print(f"Best model saved at: {os.path.join(save_path, 'best_model.zip')}")
    print("=" * 60)
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


def test_trained_model(model_path: str, config_path: str = "config.yaml", n_episodes: int = 5):
    """
    Test trained model
    
    Args:
        model_path: Model file path (.zip)
        config_path: Configuration file path
        n_episodes: Number of test episodes
    """
    print("=" * 60)
    print("Testing trained PPO model")
    print("=" * 60 + "\n")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = PlantCareEnv(config_path=config_path)
    
    # Test
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
    print("Test Results")
    print("=" * 60)
    print(f"Average health: {np.mean(results['avg_health']):.1f} ± {np.std(results['avg_health']):.1f}")
    print(f"Average water usage: {np.mean(results['total_water']):.1f} ml")
    print(f"Average energy usage: {np.mean(results['total_energy']):.1f} Wh")
    print(f"Average violations: {np.mean(results['violations']):.1f} hours")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for plant care")
    parser.add_argument("--config", type=str, default="../../config.yaml", help="Configuration file path")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Training device")
    parser.add_argument("--save_path", type=str, default="../../models/", help="Model save path")
    parser.add_argument("--log_path", type=str, default="../../logs/", help="Log save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test", type=str, default=None, help="Test model path")
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        test_trained_model(args.test, args.config)
    else:
        # Training mode
        train_ppo_agent(
            config_path=args.config,
            total_timesteps=args.timesteps,
            device=args.device,
            save_path=args.save_path,
            log_path=args.log_path,
            seed=args.seed
        )
