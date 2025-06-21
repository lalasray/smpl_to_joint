import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data, axis=0)

def add_imu_noise(
    signal,
    fs=100,
    noise_type='full',
    relative_noise_level=0.05,
    bias_drift_ratio=0.01,
    misalignment_deg=0.5,
    cutoff=20
):
    """
    Flexible IMU noise injection:
    Choose noise_type:
        - 'none' : return input unchanged
        - 'lpf'  : only low-pass filter (simulate bandwidth limit)
        - 'white' : add zero-mean white noise only
        - 'bias' : add drifting bias only
        - 'misalign' : add axis misalignment only
        - 'full' : add ALL effects: LPF + noise + bias + misalignment

    signal: np.array (N, 3)
    fs: sampling frequency (Hz)
    relative_noise_level: % of RMS for measurement noise
    bias_drift_ratio: % of RMS for bias drift per second
    misalignment_deg: axis cross-coupling
    cutoff: LPF cutoff frequency in Hz

    Returns:
        noisy_signal (N, 3)
    """
    if noise_type == 'none':
        return signal.copy()

    N = len(signal)

    #Compute RMS for scaling
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-8:
        rms = np.max(np.abs(signal))
    if rms < 1e-8:
        rms = 1.0  # fallback for zero signal

    #Low-pass filter if needed
    if noise_type in ['lpf', 'full']:
        processed = butter_lowpass_filter(signal, cutoff, fs)
    else:
        processed = signal.copy()

    #Add white noise if needed
    if noise_type in ['white', 'full']:
        noise_std = relative_noise_level * rms
        white_noise = np.random.normal(0, noise_std, size=signal.shape)
        processed += white_noise

    #Add bias drift if needed
    if noise_type in ['bias', 'full']:
        bias_per_sample = bias_drift_ratio * rms / fs
        bias = np.cumsum(np.random.normal(0, bias_per_sample, size=(N, 3)), axis=0)
        processed += bias

    #Add misalignment if needed
    if noise_type in ['misalign', 'full']:
        angle_rad = np.deg2rad(misalignment_deg)
        misalign_matrix = np.array([
            [1, -angle_rad, angle_rad],
            [angle_rad, 1, -angle_rad],
            [-angle_rad, angle_rad, 1]
        ])
        processed = processed @ misalign_matrix.T

    return processed
