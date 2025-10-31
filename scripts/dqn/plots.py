"""
Unlock Rate vs Achievements Plot for DQN Models
Compares Baseline, Enhanced, and Advanced DQN
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import math


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Define model info and stats file locations
MODELS = [
    {"name": "DQN Baseline (SB3 Default)", "stats": f"{RESULTS_DIR}/crafter_logs_eval1/stats.jsonl"},
    {"name": "DQN Enhanced (Custom CNN + Shaping)", "stats": f"{RESULTS_DIR}/enhanced_dqn_logs/stats.jsonl"},
    {"name": "DQN Advanced (PER + Noisy Nets + Multi-Step)", "stats": f"{RESULTS_DIR}/advanced_dqn_logs/stats.jsonl"}
]

# Collect unlock rates and achievements
unlock_rates = []
achievement_counts = []
labels = []

for model in MODELS:
    stats_path = model["stats"]
    if not os.path.exists(stats_path):
        print(f"❌ Stats file not found for {model['name']}: {stats_path}")
        continue
    achievement_keys = set()
    achievement_total = 0
    episode_count = 0
    with open(stats_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            # Find all achievement fields
            for k, v in entry.items():
                if k.startswith("achievement_"):
                    achievement_keys.add(k)
                    achievement_total += v
            episode_count += 1
    if achievement_keys and episode_count > 0:
        unlock_rate = achievement_total / episode_count
        unlock_rates.append(unlock_rate)
        achievement_counts.append(len(achievement_keys))
        labels.append(model["name"])
    else:
        print(f"⚠️ No achievement data for {model['name']}")

# Ensure plots directory exists
plots_dir = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(plots_dir, exist_ok=True)

def plot_survival_time(MODELS, model_colors, plots_dir):
    # Survival Time vs Episodes Plot
    # Collect all survival times and determine max length
    all_survival_times = []
    max_len = 0
    for model in MODELS:
        stats_path = model["stats"]
        survival_times = []
        if not os.path.exists(stats_path):
            all_survival_times.append([])
            continue
        with open(stats_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if "episode_length" in entry:
                    survival_times.append(entry["episode_length"])
                elif "length" in entry:
                    survival_times.append(entry["length"])
        all_survival_times.append(survival_times)
        if len(survival_times) > max_len:
            max_len = len(survival_times)
    # Truncate or pad all models to match max_len
    plt.figure(figsize=(10,6))
    for idx, (model, survival_times) in enumerate(zip(MODELS, all_survival_times)):
        if len(survival_times) > 0:
            if len(survival_times) < max_len:
                pad_val = survival_times[-1]
                survival_times = survival_times + [pad_val] * (max_len - len(survival_times))
            elif len(survival_times) > max_len:
                survival_times = survival_times[:max_len]
            plt.plot(range(1, max_len+1), survival_times, label=model["name"], color=model_colors[idx % len(model_colors)])
        else:
            print(f"⚠️ No episode survival data for {model['name']}")
    plt.xlabel("Episode")
    plt.ylabel("Survival Time (steps)")
    plt.title("Survival Time vs Episodes: DQN Model Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    survival_plot_path = os.path.join(plots_dir, "survival_time_vs_episodes_dqn.png")
    plt.savefig(survival_plot_path, dpi=150)
    plt.show()
    print(f"✅ Survival time plot saved as {survival_plot_path}")

    # Individual Survival Time vs Episodes Plots
    for idx, model in enumerate(MODELS):
        stats_path = model["stats"]
        if not os.path.exists(stats_path):
            print(f"❌ Stats file not found for {model['name']}: {stats_path}")
            continue
        survival_times = []
        with open(stats_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if "episode_length" in entry:
                    survival_times.append(entry["episode_length"])
                elif "length" in entry:
                    survival_times.append(entry["length"])
        if survival_times:
            plt.figure(figsize=(10,6))
            plt.plot(range(1, len(survival_times)+1), survival_times, label=model["name"], color=model_colors[idx % len(model_colors)])
            plt.xlabel("Episode")
            plt.ylabel("Survival Time (steps)")
            plt.title(f"Survival Time vs Episodes: {model['name']}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            indiv_plot_path = os.path.join(plots_dir, f"survival_time_vs_episodes_{model['name'].replace(' ', '_').lower()}.png")
            plt.savefig(indiv_plot_path, dpi=150)
            plt.show()
            print(f"✅ Individual survival time plot saved as {indiv_plot_path}")
        else:
            print(f"⚠️ No episode survival data for {model['name']}")

def plot_achievement_histograms(MODELS, model_colors, plots_dir):
    # Individual Achievement Unlock Rate Histograms
    for idx, model in enumerate(MODELS):
        stats_path = model["stats"]
        if not os.path.exists(stats_path):
            print(f"❌ Stats file not found for {model['name']}: {stats_path}")
            continue
        achievement_counts = {}
        episode_count = 0
        with open(stats_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                for k, v in entry.items():
                    if k.startswith("achievement_"):
                        achievement_counts[k] = achievement_counts.get(k, 0) + v
                episode_count += 1
        if achievement_counts and episode_count > 0:
            # Calculate unlock rates
            unlock_rates = {k: v / episode_count for k, v in achievement_counts.items()}
            # Sort by unlock rate descending and select top 11
            sorted_items = sorted(unlock_rates.items(), key=lambda x: x[1], reverse=True)[:11]
            actions = [k.replace("achievement_", "") for k, _ in sorted_items]
            rates = [v for _, v in sorted_items]
            plt.figure(figsize=(12,6))
            bars = plt.bar(actions, rates, color=model_colors[idx % len(model_colors)])
            plt.ylabel("Unlock Rate per Episode")
            plt.xlabel("Achievement")
            plt.title(f"Achievement Unlock Rate Histogram: {model['name']}")
            plt.xticks(rotation=45, ha='right')
            # Show values above each bar
            for bar, rate in zip(bars, rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{rate:.2f}", ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            hist_path = os.path.join(plots_dir, f"achievement_unlock_rate_histogram_{model['name'].replace(' ', '_').lower()}.png")
            plt.savefig(hist_path, dpi=150)
            plt.show()
            print(f"✅ Achievement unlock rate histogram saved as {hist_path}")
        else:
            print(f"⚠️ No achievement data for {model['name']}")

def geometric_mean(rates):
    rates = [r for r in rates if r > 0]
    if not rates:
        return 0.0
    log_rates = [math.log(r) for r in rates]
    return math.exp(sum(log_rates) / len(log_rates))

def plot_geometric_mean_progression(MODELS, model_colors, plots_dir):
    # Geometric Mean Progression Plots
    # Collect all geometric means and determine max length
    all_geom_means = []
    max_len = 0
    for model in MODELS:
        stats_path = model["stats"]
        geom_means = []
        achievement_keys = set()
        if not os.path.exists(stats_path):
            all_geom_means.append([])
            continue
        with open(stats_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                rates = []
                for k, v in entry.items():
                    if k.startswith("achievement_"):
                        achievement_keys.add(k)
                        rates.append(1.0 if v > 0 else 1e-10)
                if rates:
                    gm = geometric_mean(rates)
                    geom_means.append(gm)
        all_geom_means.append(geom_means)
        if len(geom_means) > max_len:
            max_len = len(geom_means)
    for idx, (model, geom_means) in enumerate(zip(MODELS, all_geom_means)):
        if len(geom_means) > 0:
            if len(geom_means) < max_len:
                pad_val = geom_means[-1]
                geom_means = geom_means + [pad_val] * (max_len - len(geom_means))
            elif len(geom_means) > max_len:
                geom_means = geom_means[:max_len]
            plt.figure(figsize=(10,6))
            plt.plot(range(1, max_len+1), geom_means, color=model_colors[idx % len(model_colors)], label=model["name"], marker=None)
            plt.xlabel("Episode")
            plt.ylabel("Geometric Mean of Achievement Rates")
            plt.title(f"Geometric Mean Progression: {model['name']}")
            plt.grid(True)
            plt.tight_layout()
            gm_plot_path = os.path.join(plots_dir, f"geometric_mean_progression_{model['name'].replace(' ', '_').lower()}.png")
            plt.savefig(gm_plot_path, dpi=150)
            plt.show()
            print(f"✅ Geometric mean progression plot saved as {gm_plot_path}")
        else:
            print(f"⚠️ No achievement data for geometric mean plot: {model['name']}")

def plot_cumulative_reward(MODELS, model_colors, plots_dir):
    # Cumulative Reward per Episode (Ascending Line Graph)
    # Collect all rewards and determine max length
    all_rewards = []
    max_len = 0
    for model in MODELS:
        stats_path = model["stats"]
        rewards = []
        if not os.path.exists(stats_path):
            all_rewards.append([])
            continue
        with open(stats_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if "reward" in entry:
                    rewards.append(entry["reward"])
                elif "episode_reward" in entry:
                    rewards.append(entry["episode_reward"])
        all_rewards.append(rewards)
        if len(rewards) > max_len:
            max_len = len(rewards)
    for idx, (model, rewards) in enumerate(zip(MODELS, all_rewards)):
        if len(rewards) > 0:
            if len(rewards) < max_len:
                pad_val = rewards[-1]
                rewards = rewards + [pad_val] * (max_len - len(rewards))
            elif len(rewards) > max_len:
                rewards = rewards[:max_len]
            cumulative_rewards = np.cumsum(rewards)
            plt.figure(figsize=(10,6))
            plt.plot(range(1, max_len+1), cumulative_rewards, color=model_colors[idx % len(model_colors)], label=model["name"])
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward (Sum)")
            plt.title(f"Cumulative Reward Progression: {model['name']}")
            plt.grid(True)
            plt.tight_layout()
            reward_plot_path = os.path.join(plots_dir, f"cumulative_reward_progression_{model['name'].replace(' ', '_').lower()}.png")
            plt.savefig(reward_plot_path, dpi=150)
            plt.show()
            print(f"✅ Cumulative reward progression plot saved as {reward_plot_path}")
        else:
            print(f"⚠️ No reward data for {model['name']}")

def save_summary_table_image(MODELS, plots_dir):
    import pandas as pd
    # Collect summary data
    summary_data = []
    for model in MODELS:
        stats_path = model["stats"]
        if not os.path.exists(stats_path):
            summary_data.append({
                "Model": model["name"],
                "Episodes": "N/A",
                "Total Reward": "N/A",
                "Mean Reward": "N/A",
                "Achievements": "N/A",
                "Mean Survival": "N/A"
            })
            continue
        rewards = []
        achievements = set()
        survival_times = []
        with open(stats_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if "reward" in entry:
                    rewards.append(entry["reward"])
                elif "episode_reward" in entry:
                    rewards.append(entry["episode_reward"])
                for k in entry:
                    if k.startswith("achievement_"):
                        achievements.add(k)
                if "episode_length" in entry:
                    survival_times.append(entry["episode_length"])
                elif "length" in entry:
                    survival_times.append(entry["length"])
        episodes = len(rewards)
        total_reward = np.sum(rewards) if rewards else 0
        mean_reward = np.mean(rewards) if rewards else 0
        num_achievements = len(achievements)
        mean_survival = np.mean(survival_times) if survival_times else 0
        summary_data.append({
            "Model": model["name"],
            "Episodes": episodes,
            "Total Reward": f"{total_reward:.2f}",
            "Mean Reward": f"{mean_reward:.2f}",
            "Achievements": num_achievements,
            "Mean Survival": f"{mean_survival:.2f}"
        })
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    # Plot table as image
    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(df)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.tight_layout()
    table_path = os.path.join(plots_dir, "model_summary_table.png")
    plt.savefig(table_path, dpi=150)
    plt.close(fig)
    print(f"✅ Model summary table saved as {table_path}")

def save_model_description_table_image(plots_dir):
    import pandas as pd
    # Model descriptions
    descriptions = [
        {
            "Model": "DQN Baseline (SB3 Default)",
            "Architecture": "SB3 DQN, default CNN",
            "Features": "Standard DQN, no enhancements",
            "Notes": "Reference baseline"
        },
        {
            "Model": "DQN Enhanced (Custom CNN + Shaping)",
            "Architecture": "Custom CNN, reward shaping",
            "Features": "Improved feature extraction, shaped rewards",
            "Notes": "Better exploration, faster learning"
        },
        {
            "Model": "DQN Advanced (PER + Noisy Nets + Multi-Step)",
            "Architecture": "Custom CNN, PER, Noisy Nets, Multi-Step",
            "Features": "Prioritized replay, stochastic exploration, multi-step returns",
            "Notes": "State-of-the-art enhancements"
        }
    ]
    df = pd.DataFrame(descriptions)
    fig, ax = plt.subplots(figsize=(12, 2 + 0.7 * len(df)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.tight_layout()
    desc_table_path = os.path.join(plots_dir, "model_description_table.png")
    plt.savefig(desc_table_path, dpi=150)
    plt.close(fig)
    print(f"✅ Model description table saved as {desc_table_path}")

def save_model_parameters_table_image(plots_dir):
    import pandas as pd
    # Model parameters (example values, update as needed)
    parameters = [
        {
            "Model": "DQN Baseline (SB3 Default)",
            "Learning Rate": "1e-4",
            "Batch Size": "32",
            "Buffer Size": "100000",
            "Gamma": "0.99",
            "Target Update": "1000",
            "Exploration": "Epsilon-Greedy"
        },
        {
            "Model": "DQN Enhanced (Custom CNN + Shaping)",
            "Learning Rate": "5e-5",
            "Batch Size": "64",
            "Buffer Size": "200000",
            "Gamma": "0.99",
            "Target Update": "500",
            "Exploration": "Epsilon-Greedy, Reward Shaping"
        },
        {
            "Model": "DQN Advanced (PER + Noisy Nets + Multi-Step)",
            "Learning Rate": "2e-4",
            "Batch Size": "128",
            "Buffer Size": "500000",
            "Gamma": "0.99",
            "Target Update": "250",
            "Exploration": "PER, Noisy Nets, Multi-Step"
        }
    ]
    df = pd.DataFrame(parameters)
    fig, ax = plt.subplots(figsize=(14, 2 + 0.7 * len(df)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.tight_layout()
    param_table_path = os.path.join(plots_dir, "model_parameters_table.png")
    plt.savefig(param_table_path, dpi=150)
    plt.close(fig)
    print(f"✅ Model parameters table saved as {param_table_path}")

# --- Main execution ---
if __name__ == "__main__":
    # Survival Time Plots
    model_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    # plot_survival_time(MODELS, model_colors, plots_dir)

    # Achievement Unlock Rate Histograms
    plot_achievement_histograms(MODELS, model_colors, plots_dir)

    # Geometric Mean Progression Plots
    plot_geometric_mean_progression(MODELS, model_colors, plots_dir)

    # Cumulative Reward Plots
    plot_cumulative_reward(MODELS, model_colors, plots_dir)

    # Save summary table image
    save_summary_table_image(MODELS, plots_dir)

    # Save model description table image
    save_model_description_table_image(plots_dir)

    # Save model parameters table image
    save_model_parameters_table_image(plots_dir)
