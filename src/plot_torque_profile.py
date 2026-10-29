
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("results/human_inverse_dynamics.csv")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Keep only samples assigned to a complete gait cycle.
df = df.dropna(subset=["gait_cycle_id", "gait_cycle_pct"])

# Average torque at each gait-cycle percentage across cycles.
df["gait_cycle_pct"] = df["gait_cycle_pct"].round().clip(0, 100)

profile = (
    df.groupby("gait_cycle_pct")
    [["generalized_torque_q1_Nm", "generalized_torque_q2_Nm"]]
    .mean()
    .reindex(range(101))
    .interpolate(limit_direction="both")
)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(
    profile.index,
    profile["generalized_torque_q1_Nm"]
)
axes[0].set_ylabel("Generalized torque q1 (N·m)")
axes[0].set_title("Exploratory torque profile across the gait cycle")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    profile.index,
    profile["generalized_torque_q2_Nm"]
)
axes[1].set_xlabel("Gait cycle (%)")
axes[1].set_ylabel("Generalized torque q2 (N·m)")
axes[1].set_xticks(range(0, 101, 10))
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    "NONAN S034 — Model-based estimates, not validated joint torques",
    fontsize=11
)
fig.tight_layout()

output_path = OUTPUT_DIR / "exploratory_torque_profile.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved plot: {output_path}")

plt.show()
