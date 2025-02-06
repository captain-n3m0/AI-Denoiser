import numpy as np

def apply_fft(signal):
    fft_signal = np.fft.fft(signal)
    return np.abs(fft_signal)

def apply_ifft(signal):
    return np.fft.ifft(signal).real
