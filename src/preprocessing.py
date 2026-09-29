import numpy as np
from scipy.signal import butter, filtfilt


def lowpass_filter(data, cutoff=10.0, fs=200.0, order=4):
    """
    Apply a Butterworth low-pass filter.

    Parameters:
        data: numpy array, shape (samples, channels)
        cutoff: cutoff frequency in Hz
        fs: sampling frequency in Hz
        order: filter order

    Returns:
        Filtered numpy array with the same shape as data.
    """

    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype="low")

    return filtfilt(b, a, data, axis=0)