
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def segment_gait(csv_path):
    df = pd.read_csv(
        csv_path,
        usecols=["time", "Contact RT"],
    )

    df = df.dropna(subset=["time"]).sort_values("time")
    time = df["time"].to_numpy(dtype=float)
    contact = df["Contact RT"].fillna(0).to_numpy() > 0

    if len(time) < 2:
        raise ValueError("Not enough samples in the recording.")

    if not np.all(np.diff(time) > 0):
        raise ValueError("Time values must be strictly increasing.")

    # Detect transitions in the contact signal.
    starts = np.flatnonzero(contact & ~np.r_[False, contact[:-1]])
    ends = np.flatnonzero(contact & ~np.r_[contact[1:], False])

    # Only use complete stance intervals: ignore intervals
    # already active at the beginning or still active at the end.
    if contact[0]:
        starts = starts[starts > 0]

    if contact[-1]:
        ends = ends[ends < len(contact) - 1]

    # Pair each contact start with its following contact end.
    intervals = []
    j = 0

    for start in starts:
        while j < len(ends) and ends[j] < start:
            j += 1

        if j < len(ends):
            intervals.append((start, ends[j]))
            j += 1

    # Complete stance intervals, ordered by time.
    stance_intervals = [
        (s, e) for s, e in intervals if e > s
    ]

    rows = []

    for i in range(len(stance_intervals) - 1):
        start_idx, end_idx = stance_intervals[i]
        next_start_idx, _ = stance_intervals[i + 1]

        stride_time = time[next_start_idx] - time[start_idx]
        stance_time = time[end_idx] - time[start_idx]

        if stride_time <= 0 or stance_time <= 0:
            continue

        swing_time = stride_time - stance_time

        rows.append({
            "cycle": len(rows) + 1,
            "start_time_s": time[start_idx],
            "next_contact_time_s": time[next_start_idx],
            "stride_time_s": stride_time,
            "stance_time_s": stance_time,
            "swing_time_s": swing_time,
            "stance_pct": 100 * stance_time / stride_time,
            "swing_pct": 100 * swing_time / stride_time,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Segment gait cycles using a contact signal."
    )
    parser.add_argument("csv", help="Input NONAN trial CSV")
    parser.add_argument(
        "--output",
        default="results/gait_cycles.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    cycles = segment_gait(args.csv)

    if cycles.empty:
        print("No complete gait cycles found.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(output, index=False)

    print(f"Complete cycles: {len(cycles)}")
    print("\nMean values:")
    for col in [
        "stride_time_s",
        "stance_time_s",
        "swing_time_s",
        "stance_pct",
    ]:
        print(f"{col}: {cycles[col].mean():.3f}")

    print(f"\nSaved cycle data to: {output}")


if __name__ == "__main__":
    main()
