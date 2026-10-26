#!/usr/bin/env python3
"""
plot_trial.py - quick look at one recorded trial.

    python plot_trial.py data/S01_normal_T01_20261024-110512.csv

Saves <file>.png next to the CSV and opens a window (add --no-show to skip the window).
What to look for: the first and last ~3 s should be flat (standing), the stomps should
show as sharp spikes, and walking should look like a repeating wave in the gyro traces.
Axis note: with the sensors mounted flat on the side of the leg, the gyro axis pointing
out of the board face is the knee-swing axis.
"""
import sys

import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    sys.exit(__doc__)
path = args[0]

df = pd.read_csv(path)
t = (df["t_us"] - df["t_us"].iloc[0]) / 1e6

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 9))
panels = [("Thigh accel (g)", "th", "a", "_g"),
          ("Thigh gyro (deg/s)", "th", "g", "_dps"),
          ("Shank accel (g)", "sh", "a", "_g"),
          ("Shank gyro (deg/s)", "sh", "g", "_dps")]
for ax, (title, seg, kind, unit) in zip(axes, panels):
    for axis in "xyz":
        ax.plot(t, df[f"{seg}_{kind}{axis}{unit}"], label=axis, linewidth=0.8)
    ax.set_ylabel(title)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("time (s)")
fig.suptitle(path.split("/")[-1])
fig.tight_layout()

png = path.rsplit(".", 1)[0] + ".png"
fig.savefig(png, dpi=130)
print("Saved", png)
if "--no-show" not in sys.argv:
    plt.show()
