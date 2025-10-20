import os
from collections import deque

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from crafter.env import Env as CrafterBaseEnv

import torch as T
import torch.nn as nn


# ---------------------------
# Environment adapter for Crafter (keeps original behavior)
# ---------------------------
class CustomCrafterEnv(CrafterBaseEnv): 
    def __init__(self, area=(64,64), view=(9,9), size=(64,64), reward=True, length=10000, seed=None, render_mode=None): 
        super().__init__(area=area, view=view, size=size, reward=reward, length=length, seed=seed) 
        self.render_mode = render_mode 
        self._size = np.array(size) 
        # keep a local copy for properties 
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 4} 
        
        @property 
        def observation_space(self): 
            return spaces.Box( low=0, high=255, shape=(self._size[0], self._size[1], 3), dtype=np.uint8 ) 
        
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


# ---------------------------
# Observation wrappers
# ---------------------------
class GrayscaleFrame(gym.ObservationWrapper):
    """
    Convert HxWxC uint8 -> HxW uint8 (single channel).
    Downstream wrappers (PyTorchFrame) will add channel axis.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # new observation space: H x W (single channel)
        old = env.observation_space
        assert len(old.shape) == 3, "GrayscaleFrame expects HxWxC input observation space."
        h, w, _ = old.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w), dtype=np.uint8)

    def observation(self, observation):
        # observation could be HxWxC uint8. Use cv2 conversion robustly.
        obs = np.asarray(observation)
        if obs.ndim == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        elif obs.ndim == 2:
            gray = obs
        else:
            raise ValueError("Unexpected observation shape for GrayscaleFrame: " + str(obs.shape))
        # keep uint8
        return gray.astype(np.uint8)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Convert numeric observation to float32 in [0,1]. Accepts HxW or HxWxC or CxHxW.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old = env.observation_space
        # maintain same shape but float32 in [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=old.shape, dtype=np.float32)

    def _to_float01(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.dtype == np.uint8 or arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return arr

    def observation(self, observation):
        return self._to_float01(observation)


class PyTorchFrame(gym.ObservationWrapper):
    """
    Convert HxW (or HxWxC) or CxHxW to CxHxW float32.
    Sets observation_space accordingly with dtype float32.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old = env.observation_space
        # old may be (H,W) or (H,W,C) or (C,H,W)
        shape = old.shape
        # determine channel-first shape (C,H,W)
        if len(shape) == 2:  # H,W -> 1,H,W
            c, h, w = 1, shape[0], shape[1]
        elif len(shape) == 3:
            # assume H,W,C if last dim in {1,3,4}
            if shape[2] in (1, 3, 4):
                c, h, w = shape[2], shape[0], shape[1]
            else:  # assume C,H,W
                c, h, w = shape[0], shape[1], shape[2]
        else:
            raise ValueError("Unsupported observation shape for PyTorchFrame: " + str(shape))

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)

    def observation(self, observation):
        arr = np.asarray(observation)
        # Cases:
        # H x W      -> (1,H,W)
        # H x W x C  -> (C,H,W)
        # C x H x W  -> (C,H,W)
        if arr.ndim == 2:  # H,W
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            # detect HWC vs CHW
            if arr.shape[2] in (1, 3, 4):  # HWC
                arr = np.transpose(arr, (2, 0, 1))
            else:
                # assume CHW already
                pass
        else:
            raise ValueError("Unsupported observation ndim in PyTorchFrame: " + str(arr.shape))

        # ensure float32 and in [0,1] (if not, keep as-is)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr


# ---------------------------
# Frame stacking & LazyFrames
# ---------------------------
class LazyFrames:
    """
    Memory-efficient container for stacked frames. Converts to numpy on demand.
    Expects frames to be channel-first arrays (C,H,W).
    """
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def __array__(self, dtype=None):
        if self._out is None:
            # concatenate along channel axis (axis=0)
            self._out = np.concatenate([np.asarray(f) for f in self._frames], axis=0)
        if dtype is None:
            return self._out
        return self._out.astype(dtype, copy=False)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]


class FrameStack(gym.Wrapper):
    """
    Stack k last frames. Assumes incoming observations are channel-first (C,H,W).
    The resulting observation_space is (C*k, H, W).
    """
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape  # expected C,H,W
        assert len(shp) == 3, "FrameStack expects channel-first observation space (C,H,W)."
        c, h, w = shp
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(c * k, h, w), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # make sure obs is channel-first array
        arr = np.asarray(obs)
        for _ in range(self.k):
            self.frames.append(arr)
        return LazyFrames(list(self.frames)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        arr = np.asarray(obs)
        self.frames.append(arr)
        return LazyFrames(list(self.frames)), reward, terminated, truncated, info


# ---------------------------
# Video resize wrapper
# ---------------------------
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


# ---------------------------
# Crafter stats wrapper
# ---------------------------
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
        self.current_reward += float(reward)

        # Merge achievements if present
        if isinstance(info, dict) and "achievements" in info:
            for key, val in info["achievements"].items():
                self.current_achievements[key] = self.current_achievements.get(key, 0) + val

        if terminated or truncated:
            success = sum(info.get("achievements", {}).values()) > 0 if isinstance(info, dict) else False
            self.episode_rewards.append(self.current_reward)
            self.episode_successes.append(success)
            self.episode_achievements.append(info.get("achievements", {}).copy() if isinstance(info, dict) else {})

        return obs, reward, terminated, truncated, info

    def get_last_metrics(self):
        """Return rolling metrics over last 100 episodes."""
        if not self.episode_rewards:
            return {}
        num = min(100, len(self.episode_achievements))
        return {
            "reward_mean": float(np.mean(self.episode_rewards[-100:])),
            "success_rate": float(np.mean(self.episode_successes[-100:])),
            "reward_breakdown": self._aggregate_achievements(num)
        }

    def _aggregate_achievements(self, num):
        all_achievements = {}
        for ach in self.episode_achievements[-num:]:
            for key, val in ach.items():
                all_achievements[key] = all_achievements.get(key, 0) + val
        # Normalize to mean per episode
        if num == 0:
            return {}
        return {k: v / num for k, v in all_achievements.items()}