import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator


# ===== USER PARAMETERS =====
EVENT_FILE = "./fan_const_rpm.raw"   # or .dat
ROI = (0, 0, 160, 90)      # (x_min, y_min, x_max, y_max) -> adjust!
BLADE_COUNT = 2                # number of blades on the propeller
DELTA_T_US = 200               # time bin width in microseconds (~0.2 ms)
F_MIN_HZ = 10                  # ignore very low frequencies
F_MAX_HZ = None                # set e.g. 2000 if you know the range
# ============================


def build_roi_timeseries(path, roi, delta_t_us):
    """
    Build a 1D signal from event counts in an ROI, using EventsIterator
    with constant time bins of delta_t_us microseconds.
    """
    x_min, y_min, x_max, y_max = roi

    it = EventsIterator(input_path=path, mode="delta_t", delta_t=delta_t_us)
    height, width = it.get_size()
    print(f"Sensor size: {width}x{height}")

    samples = []

    for events in it:
        if events.size == 0:
            samples.append(0.0)
            continue

        # ROI mask
        xs = events["x"]
        ys = events["y"]
        mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)

        if not np.any(mask):
            samples.append(0.0)
            continue

        # Option 1: plain event count
        # value = np.count_nonzero(mask)

        # Option 2: polarity-weighted count (often a bit nicer)
        pol = events["p"][mask].astype(np.int8)  # 0 or 1
        pol = 2 * pol - 1                        # -> -1 or +1
        value = np.sum(pol)

        samples.append(float(value))

    samples = np.asarray(samples, dtype=np.float64)

    dt = delta_t_us * 1e-6  # seconds per sample
    t = np.arange(len(samples)) * dt

    return t, samples


def estimate_frequency_fft(t, signal, fmin=0.0, fmax=None, plot=True):
    """
    Estimate dominant frequency in 'signal' using FFT.

    Returns:
        f_peak: dominant frequency in Hz
    """
    # Remove DC
    sig = signal - np.mean(signal)

    # Apply window to reduce leakage
    window = np.hanning(len(sig))
    sig_win = sig * window

    dt = t[1] - t[0]  # assuming constant sampling spacing

    # Real FFT
    fft_vals = np.fft.rfft(sig_win)
    freqs = np.fft.rfftfreq(len(sig_win), d=dt)
    mag = np.abs(fft_vals)

    # Limit frequency range if requested
    valid = freqs > 0  # drop DC

    if fmin is not None and fmin > 0:
        valid &= freqs >= fmin
    if fmax is not None:
        valid &= freqs <= fmax

    if not np.any(valid):
        raise RuntimeError("No valid frequencies in the chosen range.")

    # Find peak
    idx_peak = np.argmax(mag[valid])
    f_peak = freqs[valid][idx_peak]

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        ax1.plot(t, sig)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Event signal")
        ax1.set_title("ROI event time series (DC removed)")

        ax2.plot(freqs, mag)
        ax2.axvline(f_peak, linestyle="--")
        ax2.set_xlim(left=0)
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("|FFT|")
        ax2.set_title(f"FFT magnitude (peak at {f_peak:.1f} Hz)")

        plt.tight_layout()
        plt.show()

    return f_peak


def main():
    print(f"Reading events from: {EVENT_FILE}")
    t, s = build_roi_timeseries(EVENT_FILE, ROI, DELTA_T_US)
    print(f"Built time series with {len(s)} samples, duration {t[-1]:.3f} s")

    f_peak = estimate_frequency_fft(t, s, fmin=F_MIN_HZ, fmax=F_MAX_HZ, plot=True)
    print(f"Blade-passing frequency ≈ {f_peak:.2f} Hz")

    if BLADE_COUNT > 0:
        f_rot = f_peak / BLADE_COUNT
        rpm = 60.0 * f_rot
        print(f"Estimated rotation frequency ≈ {f_rot:.2f} Hz")
        print(f"Estimated RPM ≈ {rpm:.1f} rpm")


if __name__ == "__main__":
    main()
