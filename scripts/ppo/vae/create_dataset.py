import os
import sys
import gymnasium as gym
import numpy as np

# ------------------------
# Add project root to sys.path
# ------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from scripts.ppo.environment.wrappers import *


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


def make_env(env_name: str, n_stack_frames: int = 4, render_mode: str = "rgb_array"):
    env = gym.make(env_name, render_mode=render_mode)
    env = GrayscaleFrame(env)
    env = ScaledFloatFrame(env)
    env = PyTorchFrame(env)

    if n_stack_frames > 1:
        env = FrameStack(env, n_stack_frames)

    return env


def main():
    env_name = "CustomCrafterReward-v1"
    n_stack_frames = 4

    print(f"[INFO] Creating environment: {env_name}")
    env = make_env(env_name, n_stack_frames)

    data_dir = "./scripts/ppo/vae/data"
    os.makedirs(data_dir, exist_ok=True)

    frames = []
    n_episodes = 100
    total_steps = 0

    print(f"[INFO] Starting data collection for {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_steps = 0

        frames.append(obs)  # first frame

        while not done:
            action = env.action_space.sample()  # random action
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(obs)
            episode_steps += 1
            total_steps += 1

        print(f"[EPISODE {ep+1}/{n_episodes}] Collected {episode_steps} steps")

    frames = np.array(frames)  # shape: (N, C, 64, 64)
    save_path = os.path.join(data_dir, f"{env_name}_dataset.npy")

    print(f"[INFO] Saving dataset with shape {frames.shape} to {save_path}")
    np.save(save_path, frames)

    env.close()
    print(f"[DONE] Data collection complete: {len(frames)} frames saved.")


if __name__ == "__main__":
    main()
