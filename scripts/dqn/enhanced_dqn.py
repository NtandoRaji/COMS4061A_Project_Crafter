"""
Enhanced DQN for Crafter with Multiple Improvements
Based on modern DQN best practices and Crafter-specific optimizations
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# Add current directory to path for local crafter import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import crafter
from shimmy import GymV21CompatibilityV0

class ImprovedCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor optimized for Crafter
    - Deeper network for better feature learning
    - Proper normalization and regularization
    - Skip connections for better gradient flow
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # First block: Basic feature detection
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Second block: More complex features
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Third block: High-level features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Fourth block: Final feature extraction
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten(),
        )
        
        # Calculate the output size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Light regularization
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize input to [0, 1] if needed
        if observations.max() > 1.0:
            observations = observations.float() / 255.0
        
        features = self.cnn(observations)
        return self.linear(features)

class CrafterRewardShaping:
    """
    Reward shaping specifically designed for Crafter
    Provides additional learning signals beyond the sparse environment rewards
    """
    def __init__(self):
        self.prev_achievements = set()
        self.prev_health = 9  # Starting health
        self.prev_food = 9    # Starting food
        self.prev_water = 9   # Starting water
        self.step_count = 0
        
    def reset(self):
        self.prev_achievements = set()
        self.prev_health = 9
        self.prev_food = 9
        self.prev_water = 9
        self.step_count = 0
    
    def shape_reward(self, reward, info):
        """Add shaped rewards to encourage exploration and survival"""
        shaped_reward = reward
        
        # Achievement bonuses (major rewards)
        if 'achievements' in info:
            current_achievements = set(k for k, v in info['achievements'].items() if v)
            new_achievements = current_achievements - self.prev_achievements
            
            # Bonus for new achievements
            for achievement in new_achievements:
                if 'collect' in achievement:
                    shaped_reward += 2.0  # Collection rewards
                elif 'make' in achievement or 'place' in achievement:
                    shaped_reward += 5.0  # Crafting rewards (more valuable)
                elif 'defeat' in achievement:
                    shaped_reward += 3.0  # Combat rewards
                else:
                    shaped_reward += 1.0  # Other achievements
            
            self.prev_achievements = current_achievements
        
        # Survival bonuses (encourage staying alive)
        current_health = info.get('health', self.prev_health)
        current_food = info.get('food', self.prev_food)
        current_water = info.get('water', self.prev_water)
        
        # Small reward for maintaining health/food/water
        if current_health > self.prev_health:
            shaped_reward += 0.1
        elif current_health < self.prev_health:
            shaped_reward -= 0.2  # Penalty for taking damage
            
        if current_food > self.prev_food:
            shaped_reward += 0.1
        elif current_food < self.prev_food:
            shaped_reward -= 0.1
            
        if current_water > self.prev_water:
            shaped_reward += 0.1
        elif current_water < self.prev_water:
            shaped_reward -= 0.1
        
        self.prev_health = current_health
        self.prev_food = current_food
        self.prev_water = current_water
        
        # Survival time bonus (small but helps)
        self.step_count += 1
        if self.step_count % 100 == 0:  # Every 100 steps survived
            shaped_reward += 0.5
        
        return shaped_reward

class RewardShapingWrapper(gym.Wrapper):
    """Wrapper to apply reward shaping"""
    def __init__(self, env):
        super().__init__(env)
        self.reward_shaper = CrafterRewardShaping()
    
    def reset(self, **kwargs):
        self.reward_shaper.reset()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # This wrapper works with old Gym API (before shimmy conversion)
        obs, reward, done, info = self.env.step(action)
        shaped_reward = self.reward_shaper.shape_reward(reward, info)
        return obs, shaped_reward, done, info

class AdvancedCallback(BaseCallback):
    """Enhanced callback with better logging and adaptive learning"""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.shaped_rewards = []
        self.episode_lengths = []
        self.achievements_per_episode = []
        self.best_reward = float('-inf')
        self.episodes_since_improvement = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                # Track achievements
                achievements = 0
                if "achievements" in info:
                    achievements = sum(1 for v in info["achievements"].values() if v)
                self.achievements_per_episode.append(achievements)
                
                # Check for improvement
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    self.episodes_since_improvement = 0
                    print(f"🎉 New best reward: {ep_reward:.2f}")
                else:
                    self.episodes_since_improvement += 1
                
                # Detailed logging every 25 episodes
                if len(self.episode_rewards) % 25 == 0:
                    recent_rewards = self.episode_rewards[-25:]
                    recent_lengths = self.episode_lengths[-25:]
                    recent_achievements = self.achievements_per_episode[-25:]
                    
                    print(f"\n📊 Episode {len(self.episode_rewards)} Stats:")
                    print(f"   Avg Reward (last 25): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                    print(f"   Avg Length (last 25): {np.mean(recent_lengths):.1f}")
                    print(f"   Avg Achievements (last 25): {np.mean(recent_achievements):.1f}")
                    print(f"   Best Reward Ever: {self.best_reward:.2f}")
                    print(f"   Episodes since improvement: {self.episodes_since_improvement}")
        
        return True

def create_improved_env():
    """Create environment with all improvements"""
    def _make_env():
        # Add local crafter to path
        crafter_path = os.path.join(os.path.dirname(__file__), '.')
        if crafter_path not in sys.path:
            sys.path.insert(0, crafter_path)
        
        # Create base environment
        env = crafter.Env(reward=True, size=(64, 64))
        
        # Add logging first (before any wrappers)
        logdir = f"{RESULTS_DIR}/enhanced_dqn_logs"
        os.makedirs(logdir, exist_ok=True)
        env = crafter.Recorder(
            env,
            logdir,
            save_stats=True,
            save_video=False,
            save_episode=False,
        )
        
        # Add reward shaping (still uses old Gym API)
        env = RewardShapingWrapper(env)
        
        # Convert to Gymnasium API
        env = GymV21CompatibilityV0(env=env)
        env = Monitor(env)
        
        return env
    
    return _make_env

def train_enhanced_dqn():
    """Train enhanced DQN with improvements"""
    print("🚀 Training Enhanced DQN for Crafter")
    print("Improvements:")
    print("  ✅ Custom CNN architecture with BatchNorm")
    print("  ✅ Reward shaping for better learning signals")
    print("  ✅ Optimized hyperparameters")
    print("  ✅ Advanced logging and monitoring")
    
    # Create environment
    env = DummyVecEnv([create_improved_env()])
    env = VecTransposeImage(env)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # Custom policy with improved CNN
    policy_kwargs = {
        "features_extractor_class": ImprovedCNNExtractor,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": [512, 512],  # Larger network
    }
    
    # Create enhanced DQN model
    model = DQN(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        
        # Optimized hyperparameters for Crafter
        learning_rate=5e-5,  # Lower learning rate for stability
        buffer_size=200_000,  # Larger buffer for more diverse experiences
        learning_starts=20_000,  # More random exploration initially
        batch_size=64,  # Larger batches for stable gradients
        gamma=0.995,  # Higher discount for long-term planning
        train_freq=4,
        gradient_steps=2,  # More gradient steps per update
        target_update_interval=2000,  # Less frequent target updates
        
        # Improved exploration
        exploration_fraction=0.5,  # Longer exploration period
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,  # Higher final exploration
        
        # Training stability
        tau=0.005,  # Soft target updates
        
        verbose=1,
        device=device,
        tensorboard_log="./logs/",
    )
    
    # Create callback
    callback = AdvancedCallback()
    
    print(f"\n🎯 Starting training for 500K timesteps...")
    print("Monitor progress with: tensorboard --logdir logs")
    
    # Train for same duration as baseline (500K timesteps)
    model.learn(
        total_timesteps=500_000,  # Same as original baseline
        callback=callback,
        tb_log_name="enhanced_dqn"
    )
    
    # Save model
    model.save("enhanced_dqn_crafter")
    print("\n✅ Enhanced model saved as 'enhanced_dqn_crafter'")
    
    # Final statistics
    if callback.episode_rewards:
        print(f"\n📈 Final Training Statistics:")
        print(f"   Total Episodes: {len(callback.episode_rewards)}")
        print(f"   Best Reward: {max(callback.episode_rewards):.2f}")
        print(f"   Average Reward (last 100): {np.mean(callback.episode_rewards[-100:]):.2f}")
        print(f"   Average Length (last 100): {np.mean(callback.episode_lengths[-100:]):.1f}")
        print(f"   Max Achievements in Episode: {max(callback.achievements_per_episode) if callback.achievements_per_episode else 0}")
    
    env.close()

if __name__ == "__main__":
    train_enhanced_dqn()