import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
CSV_PATH = "./results/all_tensorboard_scalars.csv"
FILES = {
    "Baseline": "./results/ppo_basline_training_metrics.csv",
    "VAE": "./results/ppo_vae_training_metrics.csv",
    "ICM": "./results/ppo_icm_training_metrics.csv",
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------------------
# LOAD MAIN CSV (tensorboard scalars)
# -----------------------------------------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
df["run"] = df["run"].str.replace("ppo_", "", regex=False).str.replace("_crafter", "", regex=False).str.upper()

print("Loaded data with columns:")
print(df.columns.tolist())

# -----------------------------------------
# Helper to save and close figures
# -----------------------------------------
def save_plot(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)

# -----------------------------------------
# PLOT 1: Mean Episode Reward
# -----------------------------------------
if "rollout/ep_rew_mean" in df.columns:
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="rollout/ep_rew_mean", hue="run", linewidth=2)
    plt.title("Mean Episode Reward per PPO Variant")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_plot(fig, "mean_episode_reward.png")

# -----------------------------------------
# PLOT 2: Episode Length
# -----------------------------------------
if "rollout/ep_len_mean" in df.columns:
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="rollout/ep_len_mean", hue="run", linewidth=2)
    plt.title("Mean Episode Length per PPO Variant")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Episode Length")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_plot(fig, "mean_episode_length.png")

# -----------------------------------------
# PLOT 3: Training Stability (2x2)
# -----------------------------------------
stability_cols = ["train/value_loss", "train/policy_gradient_loss", "train/entropy_loss", "train/loss"]
available_stability = [c for c in stability_cols if c in df.columns]

if available_stability:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, available_stability):
        sns.lineplot(data=df, x="step", y=col, hue="run", ax=ax, linewidth=2)
        ax.set_title(col.replace("train/", "").replace("_", " ").title())
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
    for ax in axes[len(available_stability):]:
        ax.set_visible(False)
    plt.tight_layout()
    save_plot(fig, "training_stability_losses.png")

# -----------------------------------------
# PLOT 4: ICM Intrinsic Reward and Losses
# -----------------------------------------
icm_cols = ["icm/reward_intrinsic_mean", "icm/fwd_loss", "icm/inv_loss"]
available_icm = [c for c in icm_cols if c in df.columns]

if available_icm:
    icm_df = df[df["run"].str.contains("ICM", case=False, na=False)].copy()
    icm_df = icm_df.melt(
        id_vars=["step", "run"],
        value_vars=available_icm,
        var_name="Metric",
        value_name="Value"
    )
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=icm_df, x="step", y="Value", hue="Metric", linewidth=2)
    plt.title("ICM Intrinsic Reward and Losses Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_plot(fig, "icm_metrics.png")

# -----------------------------------------
# PLOT 5: Reward Breakdown / Task Success Rate
# Using only the original CSV files from FILES
# -----------------------------------------
dfs = []
for label, path in FILES.items():
    if os.path.exists(path):
        df_file = pd.read_csv(path)
        df_file.columns = df_file.columns.str.strip().str.lower().str.replace(" ", "_")
        df_file["run"] = label
        dfs.append(df_file)
    else:
        print(f"Warning: File not found: {path}")

if dfs:
    df_rewards = pd.concat(dfs, ignore_index=True)

    # Identify reward columns
    reward_cols = [c for c in df_rewards.columns if "reward" in c.lower()]
    mean_reward_col = next((c for c in reward_cols if "mean" in c), None)
    breakdown_cols = [c for c in reward_cols if c != mean_reward_col]

    if breakdown_cols:
        def parse_dict(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    return {}
            return x

        for col in breakdown_cols:
            df_rewards[col] = df_rewards[col].apply(parse_dict)

        dict_cols = [c for c in breakdown_cols if isinstance(df_rewards[c].dropna().iloc[0], dict)]
        for col in dict_cols:
            expanded = pd.json_normalize(df_rewards[col])
            expanded = expanded.add_prefix(f"{col}_")
            df_rewards = pd.concat([df_rewards.drop(columns=[col]), expanded], axis=1)

        breakdown_cols = [c for c in df_rewards.columns if "reward_breakdown" in c.lower() and c != mean_reward_col]

        if breakdown_cols:
            avg_rewards = (
                df_rewards.groupby("run")[breakdown_cols]
                .mean()
                .reset_index()
                .melt(id_vars="run", var_name="Task", value_name="MeanReward")
            )
            avg_rewards["Task"] = (
                avg_rewards["Task"].str.replace("_", " ")
                .str.replace("reward breakdown", "")
                .str.strip()
                .str.title()
            )

            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=avg_rewards, x="Task", y="MeanReward", hue="run")
            plt.title("Average Reward Breakdown by Task")
            plt.xlabel("Task")
            plt.ylabel("Average Reward")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Run")
            plt.tight_layout()
            save_plot(fig, "task_success_rate.png")
        else:
            print("⚠️ No reward breakdown columns found after expansion.")
    else:
        print("⚠️ No reward breakdown columns found in CSV files.")
else:
    print("⚠️ No CSV files available for reward breakdown.")

# -----------------------------------------
# Optional: Extra PPO metrics
# -----------------------------------------
extra_cols = ["train/approx_kl", "train/explained_variance"]
extra_available = [c for c in extra_cols if c in df.columns]

if extra_available:
    fig, axes = plt.subplots(len(extra_available), 1, figsize=(10, 5 * len(extra_available)))
    if len(extra_available) == 1:
        axes = [axes]
    for ax, col in zip(axes, extra_available):
        sns.lineplot(data=df, x="step", y=col, hue="run", ax=ax)
        ax.set_title(col.replace("train/", "").replace("_", " ").title())
        ax.set_xlabel("Training Steps")
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_plot(fig, "training_extra_metrics.png")

print("\nAll plots generated and saved to:", PLOT_DIR)
