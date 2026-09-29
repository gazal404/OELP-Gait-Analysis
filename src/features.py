import numpy as np


def extract_features(windows):
    """
    Extract statistical features from each window.

    Input:
        windows: shape (num_windows, samples_per_window, channels)

    Returns:
        features: shape (num_windows, channels * 5)
    """

    mean = windows.mean(axis=1)
    variance = windows.var(axis=1)
    maximum = windows.max(axis=1)
    minimum = windows.min(axis=1)

    centered = windows - mean[:, None, :]
    zero_crossings = (
        np.diff(np.sign(centered), axis=1) != 0
    ).sum(axis=1)

    features = np.hstack([
        mean,
        variance,
        maximum,
        minimum,
        zero_crossings
    ])

    return features