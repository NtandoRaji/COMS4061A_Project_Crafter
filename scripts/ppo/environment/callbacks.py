import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CrafterCustomLogger(BaseCallback):
    def __init__(self, log_path="training_metrics.csv", verbose=1):
        super().__init__(verbose)
        self.log_path = log_path
        self.headers_written = False
        self.episode_rewards = []

    def _init_callback(self):
        # Initialize CSV with headers once
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "reward_mean",
                "success_rate",
                "value_loss",
                "policy_loss",
                "entropy_loss",
                "loss_moving_avg",
                "reward_breakdown",  # JSON-style field
            ])
        self.headers_written = True

    def _on_step(self) -> bool:
        # Track per-step reward if available
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:  # Happens at the end of an episode
                self.episode_rewards.append(info["episode"]["r"])
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
                stats["success_rate"] if isinstance(stats, dict) and "success_rate" in stats else np.nan,
                val_loss,
                pol_loss,
                ent_loss,
                loss_moving_avg,
                stats["reward_breakdown"] if isinstance(stats, dict) and "reward_breakdown" in stats else {},
            ])

        if self.verbose:
            success_rate = stats["success_rate"] if isinstance(stats, dict) and "success_rate" in stats else np.nan
            print(
                f"[{step}] Reward={reward_mean:.3f} | "
                f"Success={success_rate:.2f}"
            )
