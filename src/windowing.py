import numpy as np


def make_windows(signal, fs=200.0, window_seconds=0.15, step_seconds=0.05):
    """
    Split a continuous signal into overlapping windows.

    Parameters:
        signal: numpy array, shape (samples, channels)
        fs: sampling frequency in Hz
        window_seconds: window length in seconds
        step_seconds: distance between consecutive windows

    Returns:
        windows: shape (num_windows, window_samples, channels)
        centers: center time of each window in seconds
    """

    window_samples = int(window_seconds * fs)
    step_samples = int(step_seconds * fs)

    starts = np.arange(
        0,
        len(signal) - window_samples + 1,
        step_samples
    )

    windows = np.stack([
        signal[start:start + window_samples]
        for start in starts
    ])

    centers = (starts + window_samples / 2) / fs

    return windows, centers