
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def calculate_kinematics(csv_path, window=31, polyorder=3):
    df = pd.read_csv(
        csv_path,
        usecols=["time", "Knee Flexion RT (deg)"],
    ).dropna()

    time = df["time"].to_numpy(dtype=float)
    angle_deg = df["Knee Flexion RT (deg)"].to_numpy(dtype=float)

    if len(time) < window:
        raise ValueError("Not enough samples for the filter window.")

    dt = np.diff(time)
    if not np.all(dt > 0):
        raise ValueError("Time values must be strictly increasing.")

    fs = 1.0 / np.median(dt)

    # Smooth the angle, then differentiate using the actual sampling rate.
    angle_smooth_deg = savgol_filter(
        angle_deg, window_length=window, polyorder=polyorder
    )
    angle_rad = np.deg2rad(angle_smooth_deg)

    velocity = savgol_filter(
        angle_rad, window_length=window, polyorder=polyorder,
        deriv=1, delta=1.0 / fs
    )
    acceleration = savgol_filter(
        angle_rad, window_length=window, polyorder=polyorder,
        deriv=2, delta=1.0 / fs
    )

    result = pd.DataFrame({
        "time_s": time,
        "knee_angle_deg": angle_deg,
        "knee_angle_smooth_deg": angle_smooth_deg,
        "knee_velocity_rad_s": velocity,
        "knee_acceleration_rad_s2": acceleration,
    })

    return result, fs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input NONAN trial CSV")
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument(
        "--output", default="results/knee_kinematics.csv"
    )
    args = parser.parse_args()

    if args.window % 2 == 0:
        parser.error("--window must be odd")
    if args.polyorder >= args.window:
        parser.error("--polyorder must be smaller than --window")

    result, fs = calculate_kinematics(
        args.csv, args.window, args.polyorder
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f"Sampling frequency: {fs:.1f} Hz")
    print(f"Samples: {len(result)}")
    for col in [
        "knee_angle_smooth_deg",
        "knee_velocity_rad_s",
        "knee_acceleration_rad_s2",
    ]:
        print(
            f"{col}: min={result[col].min():.2f}, "
            f"max={result[col].max():.2f}"
        )
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
