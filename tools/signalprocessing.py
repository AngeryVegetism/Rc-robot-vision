import numpy as np
try:
    import scipy.signal as scipy_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(data: list[float], lowcut: float, highcut: float,
                    fs: float = 50.0, order: int = 4) -> list[float]:
    """Butterworth bandpass filter for breathing / heartbeat extraction."""
    if not SCIPY_AVAILABLE or len(data) < 10:
        return data
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if low <= 0 or high >= 1:
        return data
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, data).tolist()


def estimate_breathing_rate(acoustic_data: list[float], fs: float = 50.0) -> float:
    """Estimate breathing rate (breaths/min) from acoustic signal via FFT."""
    if len(acoustic_data) < 30:
        return 0.0
    arr = np.array(acoustic_data) - np.mean(acoustic_data)
    # Breathing band: 0.1–0.5 Hz (6–30 breaths/min)
    filtered = bandpass_filter(list(arr), 0.1, 0.5, fs)
    fft_vals = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    mask = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(mask):
        return 0.0
    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    return round(peak_freq * 60, 1)  # Hz → breaths/min


def estimate_heart_rate(radar_data: list[float], fs: float = 50.0) -> float:
    """Estimate heart rate (BPM) from micro-Doppler radar via FFT."""
    if len(radar_data) < 30:
        return 0.0
    arr = np.array(radar_data) - np.mean(radar_data)
    # Heartbeat band: 0.8–3.0 Hz (48–180 BPM)
    filtered = bandpass_filter(list(arr), 0.8, 3.0, fs)
    fft_vals = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    mask = (freqs >= 0.8) & (freqs <= 3.0)
    if not np.any(mask):
        return 0.0
    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    return round(peak_freq * 60, 1)  # Hz → BPM


def signal_energy(data: list[float]) -> float:
    """RMS energy of a signal window."""
    if not data:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))
