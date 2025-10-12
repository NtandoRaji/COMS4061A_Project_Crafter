import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

# ------------------------
# Add project root to sys.path
# ------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from scripts.ppo.env.wrappers import CustomCrafterEnv, CrafterStatsWrapper, ResizeForVideoWrapper
from scripts.utilities.evaluate import evaluate

# ------------------------
# Register environments
# ------------------------
gym.register(
    id="CustomCrafterReward-v1",
    entry_point="scripts.ppo.env.wrappers:CustomCrafterEnv",
    kwargs={"reward": True, "render_mode": "rgb_array"},
    max_episode_steps=10_000,
)

gym.register(
    id="CustomCrafterNoReward-v1",
    entry_point="scripts.ppo.env.wrappers:CustomCrafterEnv",
    kwargs={"reward": False, "render_mode": "rgb_array"},
    max_episode_steps=10_000,
)


def make_env():
    """Creates a Crafter environment for SB3."""
    env = gym.make("CustomCrafterReward-v1")
    env = CrafterStatsWrapper(env)
    env = Monitor(env)
    env = ResizeForVideoWrapper(env, 512, 512)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to trained PPO model (.zip)")
    parser.add_argument('--video_path', default="./videos/ppo_baseline_evaluation.mp4", help="Optional path to save video (e.g., ./eval_run.mp4)")
    parser.add_argument('--num_episodes', type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    # --- Create evaluation environment ---
    env = make_env()

    # --- Load PPO model ---
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    # --- Evaluate model ---
    evaluate(model, env, num_episodes=args.num_episodes, video_path=args.video_path)

    env.close()


if __name__ == "__main__":
    main()
