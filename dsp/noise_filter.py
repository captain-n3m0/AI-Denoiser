import numpy as np
from scipy.signal import butter, lfilter

def butter_lowpass_filter(data, cutoff=3000, fs=16000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def remove_noise(signal, noise_factor=0.02):
    noise = np.random.normal(0, noise_factor, signal.shape)
    return signal - noise
