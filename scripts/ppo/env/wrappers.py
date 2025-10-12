import numpy as np
import gymnasium as gym
import cv2

from crafter.env import Env as CrafterBaseEnv
from gymnasium import spaces


class CustomCrafterEnv(CrafterBaseEnv):
    def __init__(self, area=(64,64), view=(9,9), size=(64,64),
                 reward=True, length=10000, seed=None, render_mode=None):
        super().__init__(area=area, view=view, size=size,
                         reward=reward, length=length, seed=seed)
        self.render_mode = render_mode
        self._size = np.array(size)  # keep a local copy for properties
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    @property
    def observation_space(self):
        return spaces.Box(
            low=0, high=255, shape=(self._size[0], self._size[1], 3), dtype=np.uint8
        )

    @property
    def action_space(self):
        return spaces.Discrete(len(self.action_names))

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options or {})
        if self.render_mode == "rgb_array":
            obs = self.render()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.render_mode == "rgb_array":
            obs = self.render()
        return obs, reward, terminated, truncated, info

    def render(self, size=None, mode="rgb_array"):
        mode = mode or self.render_mode
        return super().render(size=size)

    def _obs(self):
        return self.render(None)
    


class ResizeForVideoWrapper(gym.Wrapper):
    """Resize the frames returned by `render()` for video recording only."""
    def __init__(self, env, width=256, height=256):
        super().__init__(env)
        self.width = width
        self.height = height

        self.metadata = getattr(env, "metadata", {})
        if "render_modes" not in self.metadata:
            self.metadata["render_modes"] = ["rgb_array"]

    def render(self, mode="rgb_array", **kwargs):
        frame = self.env.render(**kwargs)
        if frame is not None:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class CrafterStatsWrapper(gym.Wrapper):
    """
    Tracks episode-level statistics for Crafter.
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_achievements = []
        self.current_reward = 0.0
        self.current_achievements = {}

        self.metadata = getattr(env, "metadata", {})
        if "render_modes" not in self.metadata:
            self.metadata["render_modes"] = ["rgb_array"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_reward = 0.0
        self.current_achievements = {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_reward += reward

        # Merge achievements
        for key, val in info["achievements"].items():
            self.current_achievements[key] = self.current_achievements.get(key, 0) + val

        if terminated or truncated:
            # Episode finished → store stats
            success = sum(info["achievements"].values()) > 0
            self.episode_rewards.append(self.current_reward)
            self.episode_successes.append(success)
            self.episode_achievements.append(info["achievements"].copy())

        return obs, reward, terminated, truncated, info

    def get_last_metrics(self):
        """Return rolling metrics over last 100 episodes."""
        if not self.episode_rewards:
            return {}
        return {
            "reward_mean": np.mean(self.episode_rewards[-100:]),
            "success_rate": np.mean(self.episode_successes[-100:]),
            "reward_breakdown": self._aggregate_achievements()
        }

    def _aggregate_achievements(self):
        all_achievements = {}
        for ach in self.episode_achievements[-100:]:
            for key, val in ach.items():
                all_achievements[key] = all_achievements.get(key, 0) + val
        # Normalize to mean per episode
        num_eps = len(self.episode_achievements[-100:])
        return {k: v / num_eps for k, v in all_achievements.items()}