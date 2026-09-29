import numpy as np
from scipy.signal import butter, filtfilt


def interpolate_missing(data):
    """
    Linearly interpolate missing values in each channel.
    """
    data = data.copy()

    for i in range(data.shape[1]):
        nans = np.isnan(data[:, i])

        if nans.any():
            valid = ~nans
            data[nans, i] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(valid),
                data[valid, i]
            )

    return data


def lowpass_filter(data, cutoff=10.0, fs=200.0, order=4):
    """
    Apply a Butterworth low-pass filter.
    """
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype="low")

    return filtfilt(b, a, data, axis=0)