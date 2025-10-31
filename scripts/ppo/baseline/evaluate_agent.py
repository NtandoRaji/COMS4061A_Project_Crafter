import argparse
import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ------------------------
# Add project root to sys.path
# ------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from scripts.utilities.seed_setter import set_global_seeds
from scripts.ppo.environment.wrappers import CustomCrafterEnv, CrafterStatsWrapper, ResizeForVideoWrapper
from scripts.utilities.evaluate import evaluate

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
    """Creates a Crafter environment for SB3."""
    env = gym.make("CustomCrafterReward-v1")
    env = CrafterStatsWrapper(env)
    env = Monitor(env)
    env = ResizeForVideoWrapper(env, 512, 512)
    env.reset(seed=hyper_params["seed"])
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to trained PPO model (.zip)")
    parser.add_argument('--video_path', default="./videos/ppo_baseline_evaluation.mp4", help="Optional path to save video (e.g., ./eval_run.mp4)")
    parser.add_argument('--num_episodes', type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    # --- Create evaluation environment ---
    hyper_params = {
        "seed": args.seed
    }

    set_global_seeds(hyper_params["seed"])

    env = make_env(hyper_params)

    # --- Load PPO model ---
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    # --- Evaluate model ---
    evaluate(model, env, num_episodes=args.num_episodes, video_path=args.video_path)

    env.close()


if __name__ == "__main__":
    main()
