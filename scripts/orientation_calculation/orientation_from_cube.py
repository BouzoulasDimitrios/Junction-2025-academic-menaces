import numpy as np
import matplotlib.pyplot as plt
import os

"""
Orientation tracking on top of an existing event cube (.npz).

This script:
  1) Loads `event_cube.npz` (produced by event_cube_and_frequency_visualization.py).
  2) Computes the global dominant frequency (excluding DC).
  3) For a series of time windows, builds a frequency-filtered amplitude map
     at that frequency.
  4) From each amplitude map, computes a principal-axis orientation angle.

Result:
  - A timeseries of orientation angles (deg) vs time (s).
  - Optional save of results to `orientation_results.npz`.
"""

# ===== USER PARAMETERS =====
CUBE_FILE = "event_cube.npz"   # this is the file from build_cube_and_global_fft.py

# If None: automatically pick the strongest global frequency (>= F_MIN_HZ).
TARGET_FREQ_HZ = None

# Minimum frequency for global peak search (Hz)
F_MIN_HZ = 50.0    # or whatever lower cutoff makes sense for your props

# Sliding window settings (in "bins", i.e. time steps of the cube).
WINDOW_SIZE_BINS = 100      # e.g. 100 bins * (dt) seconds per bin
WINDOW_STRIDE_BINS = 20     # hop size between consecutive windows

# Threshold to select the high-frequency region in each amplitude map.
AMP_THRESHOLD_FRACTION = 0.5  # keep pixels with amp >= 0.5 * max_amp

# Minimum number of pixels required to accept an orientation estimate.
MIN_PIXELS = 10

# Save results?
SAVE_RESULTS = True
RESULTS_FILE = "orientation_results.npz"
# ===========================


def load_cube(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cube file not found: {path}")
    data = np.load(path)
    cube = data["cube"]          # shape (T, H, W)
    dt = float(data["dt"])       # sampling interval [s]
    print(f"Loaded cube: T={cube.shape[0]}, H={cube.shape[1]}, W={cube.shape[2]}, dt={dt:.3e}s")
    return cube, dt


def global_dominant_frequency(cube, dt, f_min_hz=None):
    s = cube.sum(axis=(1, 2)).astype(np.float64)
    s -= s.mean()

    freqs = np.fft.rfftfreq(len(s), d=dt)
    spec = np.fft.rfft(s)
    mag = np.abs(spec)

    if len(mag) <= 1:
        raise RuntimeError("Signal too short to estimate frequency.")

    # mask: ignore DC and < f_min_hz
    mask = np.ones_like(freqs, dtype=bool)
    mask[0] = False
    if f_min_hz is not None:
        mask &= freqs >= f_min_hz
        
    if not np.any(mask):
        raise RuntimeError("No frequencies above f_min_hz found.")

    mag_masked = mag[mask]
    freq_idx = np.where(mask)[0]
    idx_peak_local = int(np.argmax(mag_masked))
    idx_peak = freq_idx[idx_peak_local]

    f_peak = freqs[idx_peak]
    print(f"Global dominant frequency ≈ {f_peak:.2f} Hz (bin {idx_peak}) with f_min_hz={f_min_hz}")
    return f_peak, freqs, mag



def amplitude_map_at_frequency(cube_window, dt, f_hz):
    """
    Compute a frequency-specific amplitude map for a given time window.

    Uses correlation with cos/sin at frequency f_hz:
      A(x,y) = sqrt( (sum_t x_t cos)^2 + (sum_t x_t sin)^2 )
    """
    T_win = cube_window.shape[0]
    t_indices = np.arange(T_win, dtype=np.float64)
    t = t_indices * dt  # time for each bin in this window

    # Precompute cos/sin wave at target frequency
    omega = 2.0 * np.pi * f_hz
    cos_vec = np.cos(omega * t)   # shape (T_win,)
    sin_vec = np.sin(omega * t)   # shape (T_win,)

    # Project cube_window (T_win, H, W) onto cos and sin
    # result shapes: (H, W)
    proj_cos = np.tensordot(cos_vec, cube_window, axes=(0, 0))
    proj_sin = np.tensordot(sin_vec, cube_window, axes=(0, 0))

    amp = np.sqrt(proj_cos**2 + proj_sin**2)
    return amp


def orientation_from_amplitude_map(amp, threshold_fraction, min_pixels):
    """
    Estimate orientation angle (in radians) from amplitude map using
    weighted covariance of pixel coordinates.

    Returns:
        theta (float): angle in radians, measured w.r.t. x-axis.
        valid (bool):   whether estimation was successful.
    """
    if amp.size == 0:
        return 0.0, False

    max_amp = amp.max()
    if max_amp <= 0:
        return 0.0, False

    thresh = threshold_fraction * max_amp
    mask = amp >= thresh

    if mask.sum() < min_pixels:
        return 0.0, False

    # Coordinates
    H, W = amp.shape
    ys, xs = np.indices((H, W))  # ys: row index, xs: col index

    # Weights
    w = amp * mask

    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0, False

    # Weighted centroid
    x_mean = (w * xs).sum() / w_sum
    y_mean = (w * ys).sum() / w_sum

    # Weighted second moments
    x_centered = xs - x_mean
    y_centered = ys - y_mean

    sigma_xx = (w * x_centered * x_centered).sum() / w_sum
    sigma_yy = (w * y_centered * y_centered).sum() / w_sum
    sigma_xy = (w * x_centered * y_centered).sum() / w_sum

    # Orientation of principal axis
    theta = 0.5 * np.arctan2(2.0 * sigma_xy, (sigma_xx - sigma_yy))
    return float(theta), True


def sliding_window_orientation(cube, dt, f_hz,
                               window_size_bins,
                               window_stride_bins,
                               threshold_fraction,
                               min_pixels):
    """
    Compute orientation over time using sliding windows and a fixed frequency f_hz.
    """
    T = cube.shape[0]
    times = []
    angles = []

    start = 0
    while start + window_size_bins <= T:
        end = start + window_size_bins
        cube_win = cube[start:end]  # shape (window_size_bins, H, W)

        t_center = dt * (start + 0.5 * window_size_bins)

        amp = amplitude_map_at_frequency(cube_win, dt, f_hz)
        theta, valid = orientation_from_amplitude_map(
            amp,
            threshold_fraction=threshold_fraction,
            min_pixels=min_pixels,
        )

        if valid:
            times.append(t_center)
            angles.append(np.degrees(theta))  # store in degrees
        else:
            # Use NaN to indicate failure
            times.append(t_center)
            angles.append(np.nan)

        start += window_stride_bins

    return np.array(times), np.array(angles)


def main():
    cube, dt = load_cube(CUBE_FILE)

    if TARGET_FREQ_HZ is None:
        f_peak, freqs, mag = global_dominant_frequency(
            cube,
            dt,
            f_min_hz=F_MIN_HZ,   # <-- use the cutoff here
        )
        target_freq = f_peak
    else:
        target_freq = float(TARGET_FREQ_HZ)

    times, angles_deg = sliding_window_orientation(
        cube,
        dt,
        f_hz=target_freq,
        window_size_bins=WINDOW_SIZE_BINS,
        window_stride_bins=WINDOW_STRIDE_BINS,
        threshold_fraction=AMP_THRESHOLD_FRACTION,
        min_pixels=MIN_PIXELS,
    )

    # ---- Plot orientation vs time ----
    plt.figure(figsize=(8, 4))
    plt.plot(times, angles_deg, marker="o")
    plt.xlabel("Time [s]")
    plt.ylabel("Orientation [deg]")
    plt.title(f"High-frequency region orientation @ {target_freq:.1f} Hz")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally save results
    if SAVE_RESULTS:
        np.savez(
            RESULTS_FILE,
            times=times,
            angles_deg=angles_deg,
            target_freq_hz=target_freq,
            dt=dt,
        )
        print(f"Saved orientation results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
