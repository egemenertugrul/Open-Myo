from scipy.signal import butter, filtfilt

def process_signal(signal, order=4, low_pass=10, sfreq=200, high_band=10, low_band=90):
    emg_filtered = filter_signal(signal, order, sfreq, high_band, low_band)
    emg_rectified = rectify_signal(emg_filtered)
    emg_envelope = envelope_signal(emg_rectified, order, low_pass, sfreq)
    return emg_filtered, emg_rectified, emg_envelope

def filter_signal(signal, order=4, sfreq=200, high_band=10, low_band=90):
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)
    b1, a1 = butter(order, [high_band,low_band], btype='bandpass')
    emg_filtered = filtfilt(b1, a1, signal)
    return emg_filtered

def rectify_signal(signal):
    emg_rectified = abs(signal)
    return emg_rectified

def envelope_signal(signal, order=4, low_pass=10, sfreq=1000):
    low_pass = low_pass / (sfreq / 2)
    b2, a2 = butter(order, low_pass, btype='lowpass')
    emg_envelope = filtfilt(b2, a2, signal)
    return emg_envelope