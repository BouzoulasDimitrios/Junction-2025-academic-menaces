# import numpy as np
# from metavision_core.event_io import EventsIterator
# import matplotlib.pyplot as plt
# import os

# # ===== USER PARAMS =====
# EVENT_FILE = "./datasets/fan_const_rpm.raw"         # or .dat

import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator
import os

"""
Build a spatio-temporal event cube from an event camera file and analyze
its dominant temporal frequencies, with memory-safe defaults.

Usage:
    python event_cube_and_frequency_visualization.py

This script:
  1) Loads an event-based .raw/.dat file via Metavision/OpenEB.
  2) Bins events in time and space into a 3D tensor (T, H_bin, W_bin).
  3) Saves that cube to disk as a compressed .npz for reuse.
  4) Computes a global FFT over time to find dominant frequencies.

You can then use a separate script (e.g. a small viewer with a slider)
that loads `event_cube.npz` and visualizes per-frequency amplitude maps.
"""

# ===== USER PARAMETERS =====
EVENT_FILE = "./datasets/fan_const_rpm.raw"         # or .dat
OUTPUT_CUBE = "event_cube.npz"       # intermediate storage

# Time bin size in microseconds.
# 2000 us = 2 ms -> max frequency ~ 1 / (2 * 2 ms) = 250 Hz (Nyquist)
# Decrease for higher max frequency (at the cost of more memory).
DELTA_T_US = 2000

# Spatial binning factor. 8 means 8x8 sensor pixels are aggregated into
# a single grid cell. Increase to reduce memory usage.
GRID_DOWNSAMPLE = 4

# Limit how much of the file we process to avoid huge cubes.
# If both are set, the earliest stopping criterion triggers first.
MAX_TIME_BINS = None        # e.g. 6000 -> ~ 6000 * DELTA_T_US
MAX_DURATION_S = 2.0        # stop after ~2 seconds of data

# Optional region of interest (x_min, y_min, x_max, y_max).
# Use None to process the full sensor.
ROI = None

# Number of top global frequency peaks to print
TOP_K_PEAKS = 5
# ===========================


def build_event_cube(path,
                     delta_t_us,
                     grid_downsample,
                     roi=None,
                     max_time_bins=None,
                     max_duration_s=None):
    """Build a 3D event cube (T, H_bin, W_bin) from an event file.

    Args:
        path: Path to .raw / .dat event file.
        delta_t_us: Time bin size in microseconds.
        grid_downsample: Spatial binning factor.
        roi: (x_min, y_min, x_max, y_max) or None for full frame.
        max_time_bins: Stop after this many time bins (int) or None.
        max_duration_s: Stop after this many seconds (float) or None.

    Returns:
        cube: np.ndarray of shape (T, H_bin, W_bin) with float32 counts.
        dt:   Sampling interval in seconds (delta_t_us * 1e-6).
    """
    it = EventsIterator(input_path=path, mode="delta_t", delta_t=delta_t_us)
    H, W = it.get_size()

    if roi is None:
        x_min, y_min, x_max, y_max = 0, 0, W, H
    else:
        x_min, y_min, x_max, y_max = roi

    W_roi = x_max - x_min
    H_roi = y_max - y_min

    H_bin = H_roi // grid_downsample
    W_bin = W_roi // grid_downsample

    print(f"Sensor: {W}x{H}, ROI: {W_roi}x{H_roi}, grid: {W_bin}x{H_bin}")

    frames = []
    total_events = 0
    dt = delta_t_us * 1e-6

    for i, events in enumerate(it):
        # Stopping criteria based on bin index and elapsed time
        t_now = (i + 1) * dt
        if max_time_bins is not None and i >= max_time_bins:
            print(f"Stopping: reached max_time_bins = {max_time_bins}")
            break
        if max_duration_s is not None and t_now >= max_duration_s:
            print(f"Stopping: reached max_duration_s = {max_duration_s:.3f}s")
            break

        frame = np.zeros((H_bin, W_bin), dtype=np.float32)

        if events.size > 0:
            x = events["x"]
            y = events["y"]
            p = events["p"].astype(np.int8)  # 0/1 polarity

            # ROI mask
            m = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
            x = x[m] - x_min
            y = y[m] - y_min
            p = p[m]

            if x.size > 0:
                # Map polarity -> -1 / +1
                p = 2 * p - 1

                # Spatial binning
                xb = x // grid_downsample
                yb = y // grid_downsample

                np.add.at(frame, (yb, xb), p)
                total_events += x.size

        frames.append(frame)

    if not frames:
        raise RuntimeError("No frames were built; check ROI and file contents.")

    cube = np.stack(frames, axis=0)  # (T, H_bin, W_bin)
    print(f"Built cube: {cube.shape[0]} time bins, duration ≈ {cube.shape[0] * dt:.3f}s")
    print(f"Total events used (in ROI): {total_events}")

    return cube, dt


def analyze_global_frequency(cube, dt, top_k=5):
    """Compute a global temporal spectrum by summing over space.

    Args:
        cube: Array of shape (T, H, W).
        dt:   Sampling interval (seconds).
        top_k: Number of strongest frequency peaks to print.

    Returns:
        freqs: Frequencies (Hz) corresponding to rFFT bins.
        spec:  Complex spectrum values for the global time series.
    """
    # Global time series: sum over spatial dimensions
    s = cube.sum(axis=(1, 2)).astype(np.float64)
    s -= s.mean()

    freqs = np.fft.rfftfreq(len(s), d=dt)
    spec = np.fft.rfft(s)
    mag = np.abs(spec)

    # Ignore DC (index 0) when searching for peaks
    if len(mag) > 1:
        idx_sorted = np.argsort(mag[1:])[::-1] + 1
        idx_sorted = idx_sorted[: min(top_k, len(idx_sorted))]

        print("Top global frequency peaks:")
        for i in idx_sorted:
            print(f"  f ≈ {freqs[i]:.1f} Hz, magnitude = {mag[i]:.3g}")
    else:
        print("Spectrum too short to analyze peaks.")

    # Plot spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, mag)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|FFT|")
    plt.title("Global spectrum from event cube")
    plt.xlim(0, freqs.max() if freqs.size > 0 else 1)
    plt.tight_layout()
    plt.show()

    return freqs, spec


def main():
    if os.path.exists(OUTPUT_CUBE):
        print(f"Loading existing cube from {OUTPUT_CUBE}")
        data = np.load(OUTPUT_CUBE)
        cube = data["cube"]
        dt = float(data["dt"])
    else:
        cube, dt = build_event_cube(
            EVENT_FILE,
            DELTA_T_US,
            GRID_DOWNSAMPLE,
            ROI,
            max_time_bins=MAX_TIME_BINS,
            max_duration_s=MAX_DURATION_S,
        )
        np.savez_compressed(OUTPUT_CUBE, cube=cube.astype(np.float32), dt=dt)
        print(f"Saved cube to {OUTPUT_CUBE}")

    _freqs, _spec = analyze_global_frequency(cube, dt, top_k=TOP_K_PEAKS)


if __name__ == "__main__":
    main()
