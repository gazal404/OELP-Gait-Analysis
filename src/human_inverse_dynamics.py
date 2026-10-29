"""Exploratory planar thigh-shank inverse dynamics for NONAN GaitPrint.

IMPORTANT LIMITATIONS
---------------------
* The two-link model omits the foot and all external forces.
* Ground-reaction forces are not available in this dataset. Stance-phase
  torques are therefore not identifiable/validated from this model.
* Segment pitch must be verified as the sagittal-plane angle relative to the
  gravity reference. A good knee-angle correlation alone does not prove this.
* Anthropometric values are generic approximations, not subject-specific data.
* Outputs are exploratory model-based generalized torques, not safe actuator
  commands or clinically validated joint moments.

Run from the repository root, for example:
python src/human_inverse_dynamics.py data/raw/S034_G01_D01_B01_T01.csv \
  --body-mass-kg 70 --thigh-length-m 0.42 --shank-length-m 0.42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


REQUIRED_COLUMNS = [
    "time",
    "Thigh pitch RT (deg)",
    "Shank pitch RT (deg)",
    "Knee Flexion RT (deg)",
    "Contact RT",
]


def anthropometric_parameters(
    body_mass_kg: float,
    thigh_length_m: float,
    shank_length_m: float,
    thigh_mass_fraction: float = 0.10,
    shank_mass_fraction: float = 0.0465,
    thigh_com_fraction: float = 0.433,
    shank_com_fraction: float = 0.433,
    thigh_radius_gyration_fraction: float = 0.323,
    shank_radius_gyration_fraction: float = 0.302,
    gravity: float = 9.81,
) -> dict:
    """Build generic, configurable segment parameters (SI units)."""
    if body_mass_kg <= 0 or thigh_length_m <= 0 or shank_length_m <= 0:
        raise ValueError("Body mass and segment lengths must be positive.")

    m1 = body_mass_kg * thigh_mass_fraction
    m2 = body_mass_kg * shank_mass_fraction
    c1 = thigh_length_m * thigh_com_fraction
    c2 = shank_length_m * shank_com_fraction
    I1 = m1 * (thigh_length_m * thigh_radius_gyration_fraction) ** 2
    I2 = m2 * (shank_length_m * shank_radius_gyration_fraction) ** 2

    return {
        "m1": m1, "m2": m2,
        "l1": thigh_length_m, "c1": c1, "c2": c2,
        "I1": I1, "I2": I2, "g": gravity,
    }


def generalized_torques(q1, q2, q1_dot, q2_dot, q1_ddot, q2_ddot, params):
    """Return generalized torques for absolute segment coordinates.

    q1 = thigh angle; q2 = shank angle. Both must be radians measured from
    the gravity vertical, with a consistent positive direction. The model
    assumes no external forces and omits foot dynamics.
    """
    m1, m2 = params["m1"], params["m2"]
    l1, c1, c2 = params["l1"], params["c1"], params["c2"]
    I1, I2, g = params["I1"], params["I2"], params["g"]

    A = m1 * c1**2 + I1 + m2 * l1**2
    B = m2 * c2**2 + I2
    C = m2 * l1 * c2
    dq = q1 - q2

    tau_q1 = (
        A * q1_ddot
        + C * np.cos(dq) * q2_ddot
        + C * np.sin(dq) * q2_dot**2
        + (m1 * c1 + m2 * l1) * g * np.sin(q1)
    )
    tau_q2 = (
        C * np.cos(dq) * q1_ddot
        + B * q2_ddot
        - C * np.sin(dq) * q1_dot**2
        + m2 * c2 * g * np.sin(q2)
    )
    return tau_q1, tau_q2


def _contact_cycle_percent(time, contact):
    """Assign 0-100% phase between consecutive rising contact edges.

    Samples outside complete contact-to-contact cycles remain NaN. Contact
    is an IMU-derived label, not a force-plate ground-truth measurement.
    """
    active = np.asarray(contact, dtype=bool)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    # If contact is already active at t=0, that onset is truncated; ignore it.
    if active[0]:
        starts = starts[starts > 0]
    if len(starts) < 2:
        raise ValueError("Need at least two contact-onsets to form gait cycles.")

    phase = np.full(len(time), np.nan, dtype=float)
    cycle_id = np.full(len(time), np.nan, dtype=float)
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i + 1]
        if b <= a:
            continue
        idx = np.arange(a, b + 1)
        phase[idx] = 100.0 * (time[idx] - time[a]) / (time[b] - time[a])
        cycle_id[idx] = i + 1
    return phase, cycle_id


def process_trial(
    csv_path: str | Path,
    params: dict,
    window: int = 31,
    polyorder: int = 3,
    thigh_offset_deg: float = 0.0,
    shank_offset_deg: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Process one NONAN trial into kinematics and exploratory torque signals."""
    if window % 2 == 0 or window < 3:
        raise ValueError("Filter window must be an odd integer >= 3.")
    if polyorder < 0 or polyorder >= window:
        raise ValueError("polyorder must be >= 0 and smaller than window.")

    df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS).dropna(subset=REQUIRED_COLUMNS)
    df = df.sort_values("time").reset_index(drop=True)
    time = df["time"].to_numpy(dtype=float)
    if len(time) < window:
        raise ValueError(f"Need at least {window} valid samples.")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time values must be strictly increasing.")
    sample_dt = float(np.median(dt))
    if not np.allclose(dt, sample_dt, rtol=0.02, atol=1e-6):
        raise ValueError("Sampling is not sufficiently uniform for this filter.")

    # Offsets are configurable because the dataset pitch reference convention
    # must be confirmed before interpreting gravity terms physically.
    q1_deg = df["Thigh pitch RT (deg)"].to_numpy() + thigh_offset_deg
    q2_deg = df["Shank pitch RT (deg)"].to_numpy() + shank_offset_deg
    q1 = np.deg2rad(q1_deg)
    q2 = np.deg2rad(q2_deg)

    q1_s = savgol_filter(q1, window, polyorder)
    q2_s = savgol_filter(q2, window, polyorder)
    q1_dot = savgol_filter(q1, window, polyorder, deriv=1, delta=sample_dt)
    q2_dot = savgol_filter(q2, window, polyorder, deriv=1, delta=sample_dt)
    q1_ddot = savgol_filter(q1, window, polyorder, deriv=2, delta=sample_dt)
    q2_ddot = savgol_filter(q2, window, polyorder, deriv=2, delta=sample_dt)

    tau_q1, tau_q2 = generalized_torques(
        q1_s, q2_s, q1_dot, q2_dot, q1_ddot, q2_ddot, params
    )
    phase, cycle_id = _contact_cycle_percent(
        time, df["Contact RT"].to_numpy() > 0
    )

    measured_knee = df["Knee Flexion RT (deg)"].to_numpy()
    reconstructed_knee = np.rad2deg(q1_s - q2_s)
    # Fit an offset for consistency checking only; this is not a calibration
    # of the absolute segment angles against gravity.
    knee_offset = float(np.mean(measured_knee - reconstructed_knee))
    knee_rmse = float(np.sqrt(np.mean((measured_knee - (reconstructed_knee + knee_offset)) ** 2)))

    out = pd.DataFrame({
        "time_s": time,
        "gait_cycle_id": cycle_id,
        "gait_cycle_pct": phase,
        "thigh_angle_smooth_deg": np.rad2deg(q1_s),
        "shank_angle_smooth_deg": np.rad2deg(q2_s),
        "thigh_velocity_rad_s": q1_dot,
        "shank_velocity_rad_s": q2_dot,
        "thigh_acceleration_rad_s2": q1_ddot,
        "shank_acceleration_rad_s2": q2_ddot,
        "knee_flexion_measured_deg": measured_knee,
        "knee_relative_angle_deg": reconstructed_knee,
        "generalized_torque_q1_Nm": tau_q1,
        "generalized_torque_q2_Nm": tau_q2,
        "contact_label_imu_derived": df["Contact RT"].to_numpy() > 0,
        "grf_available": False,
        "torque_validated": False,
    })
    metadata = {
        "sample_rate_hz": 1.0 / sample_dt,
        "samples": len(df),
        "complete_contact_to_contact_cycles": int(np.nanmax(cycle_id)) if np.isfinite(cycle_id).any() else 0,
        "knee_angle_offset_deg_for_check_only": knee_offset,
        "knee_angle_rmse_after_offset_deg": knee_rmse,
        "warning": (
            "Exploratory model only: pitch-to-gravity convention not yet verified; "
            "foot and external forces omitted; GRF unavailable; torques are not validated."
        ),
    }
    return out, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to a NONAN trial CSV")
    parser.add_argument("--body-mass-kg", type=float, required=True)
    parser.add_argument("--thigh-length-m", type=float, required=True)
    parser.add_argument("--shank-length-m", type=float, required=True)
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--thigh-offset-deg", type=float, default=0.0,
                        help="Angle offset relative to gravity vertical; verify before physical use")
    parser.add_argument("--shank-offset-deg", type=float, default=0.0,
                        help="Angle offset relative to gravity vertical; verify before physical use")
    parser.add_argument("--output", default="results/human_inverse_dynamics.csv")
    args = parser.parse_args()

    params = anthropometric_parameters(
        args.body_mass_kg, args.thigh_length_m, args.shank_length_m
    )
    result, metadata = process_trial(
        args.csv, params, args.window, args.polyorder,
        args.thigh_offset_deg, args.shank_offset_deg,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f"Sampling frequency: {metadata['sample_rate_hz']:.1f} Hz")
    print(f"Samples: {metadata['samples']}")
    print(f"Complete cycles: {metadata['complete_contact_to_contact_cycles']}")
    print(f"Knee-angle consistency RMSE after offset: {metadata['knee_angle_rmse_after_offset_deg']:.2f} deg")
    print(f"Saved exploratory output: {output}")
    print("WARNING: " + metadata["warning"])


if __name__ == "__main__":
    main()
