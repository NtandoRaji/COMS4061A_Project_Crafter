import argparse
import os
import sys
import time
import gymnasium as gym
import torch as T
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# ------------------------
# Add project root to sys.path
# ------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from scripts.ppo.environment.wrappers import *
from scripts.ppo.environment.callbacks import CrafterCustomLogger
from scripts.ppo.environment.feature_extractors import CrafterLatentFeatures


device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# ------------------------
# Register environments
# ------------------------
gym.register(
    id="CustomCrafterReward-v1",
    entry_point="scripts.ppo.environment.wrappers:CustomCrafterEnv",
    kwargs={"reward": True, "render_mode": "rgb_array"},
    max_episode_steps=10_000,
)

gym.register(
    id="CustomCrafterNoReward-v1",
    entry_point="scripts.ppo.environment.wrappers:CustomCrafterEnv",
    kwargs={"reward": False, "render_mode": "rgb_array"},
    max_episode_steps=10_000,
)


def make_env(hyper_params: dict):

    """Returns a function that creates a Crafter environment for SB3."""
    def _init():
        env = gym.make("CustomCrafterReward-v1", render_mode="rgb_array")
        env = GrayscaleFrame(env)
        env = ScaledFloatFrame(env)
        env = PyTorchFrame(env)

        if hyper_params["n_stack_frames"] > 1:
            env = FrameStack(env, hyper_params["n_stack_frames"])

        env = CrafterStatsWrapper(env)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='./logs/crafter_reward-ppo_vae/0')
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--model_path', default=f"./scripts/ppo/models/ppo_vae_basline_crafter_{time.time()}.zip")
    parser.add_argument('--video_path', default="crafter_run.mp4")
    parser.add_argument('--log_dir', default="./logs/ppo_vae_baseline_crafter")
    parser.add_argument('--results_dir', default="./results/ppo_vae_basline_training_metrics.csv")
    args = parser.parse_args()

    # --- Training environment ---
    hyper_params = {
        "num_envs": 4,
        "n_stack_frames": 4,
        "verbose": True
    }

    env = SubprocVecEnv([make_env(hyper_params) for _ in range(hyper_params["num_envs"])])

    # --- Initializing PPO model ---
    policy_kwargs = dict(
        features_extractor_class=CrafterLatentFeatures,
        features_extractor_kwargs= {
            "latent_dim": 128,
            "model_path": "./scripts/ppo/models/vae_checkpoint.pth",
            "device": device
        }
    )

    model = PPO(
            "CnnPolicy", 
            env, 
            policy_kwargs=policy_kwargs, 
            verbose=hyper_params["verbose"], 
            tensorboard_log=args.log_dir
        )

    # --- Attaching Custom Logger ---
    logger =  CrafterCustomLogger(log_path=args.results_dir)

    # --- Training Model ---
    model.learn(
        total_timesteps=args.steps, 
        callback=logger
    )

    # --- Saving Model ---
    model.save(args.model_path)


if __name__ == "__main__":
    main()
