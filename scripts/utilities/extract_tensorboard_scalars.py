import os
import sys
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# -----------------------------------------
# PROJECT ROOT
# -----------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
# Example structure:
# ./results/
# ├── ppo_baseline/
# │   └── events.out.tfevents...
# ├── ppo_vae/
# │   └── events.out.tfevents...
# └── ppo_icm/
#     └── events.out.tfevents...

RESULTS_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "results", "all_tensorboard_scalars.csv")

# -----------------------------------------
# LOAD ALL RUNS
# -----------------------------------------
runs = {}
for directory in os.listdir(RESULTS_DIR):
    for dirpath, _, filenames in os.walk(os.path.join(RESULTS_DIR, directory)):
        if os.path.isdir(dirpath):
            event_files = [f for f in os.listdir(dirpath) if f.startswith("events.out.tfevents")]
            if event_files:
                runs[directory] = os.path.join(dirpath, event_files[0])

if not runs:
    raise FileNotFoundError(f"No TensorBoard event files found in {RESULTS_DIR}")

print("Found runs:")
for k, v in runs.items():
    print(f" - {k}: {v}")

# -----------------------------------------
# EXTRACT DATA
# -----------------------------------------
all_dfs = []

for run_name, event_file in runs.items():
    print(f"Processing {run_name}...")
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Only scalar tags (rewards, losses, etc.)
    scalar_tags = ea.Tags().get("scalars", [])
    if not scalar_tags:
        print(f"No scalar data in {run_name}, skipping.")
        continue

    dfs = []
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            "step": [e.step for e in events],
            tag: [e.value for e in events]
        })
        dfs.append(df)

    # Merge all tags on step
    if dfs:
        df_run = dfs[0]
        for df in dfs[1:]:
            df_run = pd.merge(df_run, df, on="step", how="outer")
        df_run["run"] = run_name
        all_dfs.append(df_run)

# -----------------------------------------
# SAVE MERGED DATA
# -----------------------------------------
if not all_dfs:
    raise ValueError("No scalar data extracted.")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values(by=["run", "step"])
df_all.to_csv(OUTPUT_CSV, index=False)

print(f"\nExported all runs to {OUTPUT_CSV}")
print("Columns:", list(df_all.columns))
