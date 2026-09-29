import pandas as pd


def load_gait_data(path):
    """Load time and thigh/shank IMU signals from a GaitPrint CSV."""

    columns = ["time"]

    for side in ["LT", "RT"]:
        for segment in ["Thigh", "Shank"]:
            columns += [
                f"{segment} Accel Sensor X {side} (mG)",
                f"{segment} Accel Sensor Y {side} (mG)",
                f"{segment} Accel Sensor Z {side} (mG)",
                f"Noraxon MyoMotion-Segments-{segment} {side}-Gyroscope-x (deg/s)",
                f"Noraxon MyoMotion-Segments-{segment} {side}-Gyroscope-y (deg/s)",
                f"Noraxon MyoMotion-Segments-{segment} {side}-Gyroscope-z (deg/s)",
            ]

    df = pd.read_csv(path, usecols=columns)

    return df