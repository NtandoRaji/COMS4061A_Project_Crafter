import argparse
import os
import sys
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

# ------------------------
# Add project root to sys.path
# ------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from scripts.utilities.seed_setter import set_global_seeds
from scripts.ppo.environment.wrappers import CustomCrafterEnv, CrafterStatsWrapper
from scripts.ppo.environment.callbacks import CrafterCustomLogger

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
        env = gym.make("CustomCrafterReward-v1")
        env = CrafterStatsWrapper(env)
        env = Monitor(env)
        env.reset(seed=hyper_params["seed"])
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='./logs/crafter_reward-ppo/0')
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--model_path', default=f"./scripts/ppo/models/ppo_basline_crafter_{time.time()}.zip")
    parser.add_argument('--video_path', default="crafter_run.mp4")
    parser.add_argument('--log_dir', default="./logs/ppo_baseline_crafter")
    parser.add_argument('--results_dir', default="./results/ppo_basline_training_metrics.csv")
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    # --- Training environment ---
    hyper_params = {
        "num_envs": 4,
        "verbose": True,
        "seed": args.seed
    }

    set_global_seeds(hyper_params["seed"])

    env = SubprocVecEnv([make_env(hyper_params) for _ in range(hyper_params["num_envs"])])
    env = VecTransposeImage(env)

    # --- Initializing PPO model ---
    model = PPO("CnnPolicy", env, verbose=hyper_params["verbose"], tensorboard_log=args.log_dir)

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
