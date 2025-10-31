import csv
import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback


class CrafterCustomLogger(BaseCallback):
    def __init__(self, log_path="training_metrics.csv", verbose=1):
        super().__init__(verbose)
        self.log_path = log_path
        self.headers_written = False
        self.episode_rewards = []
        self.episode_extrinsic = []
        self.episode_intrinsic = []

        self.icm_inv_loss = np.nan
        self.icm_fwd_loss = np.nan


    def _init_callback(self):
        # Initialize CSV with headers once
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "reward_mean",
                "reward_extrinsic_mean",
                "reward_intrinsic_mean",
                "success_rate",
                "value_loss",
                "policy_loss",
                "entropy_loss",
                "loss_moving_avg",
                "icm_inv_loss",
                "icm_fwd_loss",
                "reward_breakdown",
            ])
        self.headers_written = True

    def _on_step(self) -> bool:
        # Track per-step reward if available
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:  # Happens at the end of an episode
                self.episode_rewards.append(info["episode"]["r"])

                ext_r = info.get("extrinsic_reward", info["episode"]["r"])
                int_r = info.get("intrinsic_reward", 0.0)
                self.episode_extrinsic.append(ext_r)
                self.episode_intrinsic.append(int_r)
        return True  # Continue training

    def _on_rollout_end(self):
        logs = self.logger.name_to_value
        step = self.num_timesteps

        # Core training losses
        val_loss = logs.get("train/value_loss", np.nan)
        pol_loss = logs.get("train/policy_loss", np.nan)
        ent_loss = logs.get("train/entropy_loss", np.nan)

        # Moving average of losses
        loss_moving_avg = np.nanmean([v for v in [val_loss, pol_loss] if not np.isnan(v)])

        # Compute mean reward over last 100 episodes
        reward_mean = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else np.nan
        reward_extrinsic_mean = np.mean(self.episode_extrinsic[-100:]) if self.episode_extrinsic else np.nan
        reward_intrinsic_mean = np.mean(self.episode_intrinsic[-100:]) if self.episode_intrinsic else np.nan


        # Crafter-specific stats
        stats = {}
        if hasattr(self.training_env, "get_attr"):
            stats_list = self.training_env.get_attr("get_last_metrics")
            if stats_list and callable(stats_list[0]):
                stats = stats_list[0]() or {}

        # Write to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                reward_mean,
                reward_extrinsic_mean,
                reward_intrinsic_mean,
                stats["success_rate"] if isinstance(stats, dict) and "success_rate" in stats else np.nan,
                val_loss,
                pol_loss,
                ent_loss,
                loss_moving_avg,
                self.icm_inv_loss,
                self.icm_fwd_loss,
                stats["reward_breakdown"] if isinstance(stats, dict) and "reward_breakdown" in stats else {},
            ])

        if self.verbose:
            success_rate = stats["success_rate"] if isinstance(stats, dict) and "success_rate" in stats else np.nan
            print(
                f"[{step}] "
                f"Reward(Total)={reward_mean:.3f} | "
                f"Extrinsic={reward_extrinsic_mean:.3f} | "
                f"Intrinsic={reward_intrinsic_mean:.3f} | "
                f"Success={success_rate:.2f}"
            )
        
    def update_icm_losses(self, inv_loss: float, fwd_loss: float):
        """External hook called by ICM callback to log losses."""
        self.icm_inv_loss = inv_loss
        self.icm_fwd_loss = fwd_loss


class ICMTrainingCallback(BaseCallback):
    def __init__(
            self, icm: nn.Module, logger_callback: CrafterCustomLogger | None = None, 
            lr: float = 1e-4, buffer_size: int = 256, beta: float = 0.2, device: str = "cpu"
        ):
        super().__init__()
        self.icm = icm
        self.buffer = []
        self.buffer_size = buffer_size
        self.beta = beta
        self.device = device
        self.optimizer = optim.Adam(self.icm.parameters(), lr=lr)
        self.logger_callback = logger_callback
        self.last_obs = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        obs_tensor = self.locals.get("obs_tensor", None)
        actions = self.locals.get("actions", None)

        if obs_tensor is None or actions is None:
            return True

        for i, info in enumerate(infos):
            obs = obs_tensor[i].detach().cpu()
            action = actions[i]

            if self.last_obs is not None:
                self.buffer.append((self.last_obs, action, obs))

            self.last_obs = obs

        if len(self.buffer) >= self.buffer_size:
            self._train_icm()

        return True

    def _train_icm(self):
        batch = self.buffer
        self.buffer = []

        states, actions, states_ = zip(*batch)
        states = T.stack(states).float().to(self.device)
        states_ = T.stack(states_).float().to(self.device)
        actions = T.tensor(actions).long().to(self.device)

        r_int, inv_loss, fwd_loss = self.icm(states, actions, states_)
        loss = (1 - self.beta) * inv_loss + self.beta * fwd_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log metrics
        if self.model.logger:
            self.model.logger.record("icm/inv_loss", inv_loss.item())
            self.model.logger.record("icm/fwd_loss", fwd_loss.item())
            self.model.logger.record("icm/reward_intrinsic_mean", r_int.mean().item())

        if self.logger_callback:
            self.logger_callback.update_icm_losses(inv_loss.item(), fwd_loss.item())