import numpy as np
import time 
from numba import njit
import argparse
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass 
from sklearn.cluster import KMeans  # <-- NEW IMPORT

# --- Assume necessary imports are available ---
from evio.core.mmap import DatMemmap
from evio.core.index_scheduler import build_windows 
from evio.core.recording import Recording

# ==============================================================================
# 1. NUMBA CORE FUNCTION (Unchanged)
# ==============================================================================

@njit
def process_events_window(
    timestamps, x_coords, y_coords, polarities,
    WIDTH, NUM_PIXELS, NUM_BINS, decay_time_us, bin_width_us, min_detections, 
    t_on_prev, t_off_prev, period_histogram,
    fifo_bins, fifo_times, q_head, q_tail,
    pixel_last_period, 
) -> tuple[float, int, int]: 
    # ... (This function is correct and unchanged) ...
    max_period_us = NUM_BINS * bin_width_us
    WIDTH_64 = np.int64(WIDTH) 
    
    for i in range(len(timestamps)):
        t, x, y, p = timestamps[i], x_coords[i], y_coords[i], polarities[i]
        pixel_idx = y * WIDTH_64 + x 

        if p == 1:
            if t_off_prev[pixel_idx] > t_on_prev[pixel_idx]:
                delta_t = t - t_on_prev[pixel_idx]
                pixel_last_period[pixel_idx] = delta_t
                
                if 0 < delta_t < max_period_us:
                    bin_index = delta_t // bin_width_us
                    if bin_index >= NUM_BINS: bin_index = NUM_BINS - 1
                    period_histogram[bin_index] += 1
                    
                    if q_tail < len(fifo_bins):
                        fifo_bins[q_tail] = bin_index
                        fifo_times[q_tail] = t
                        q_tail += 1
            t_on_prev[pixel_idx] = t
        else:
            t_off_prev[pixel_idx] = t

        while q_head < q_tail:
            bin_old, t_old = fifo_bins[q_head], fifo_times[q_head]
            if t - t_old > decay_time_us:
                period_histogram[bin_old] -= 1
                q_head += 1
            else:
                break

    total = period_histogram.sum()
    freq = 0.0
    if total < min_detections:
        return 0.0, q_head, q_tail

    peak_bin = period_histogram.argmax()
    est_period_us = (peak_bin + 0.5) * bin_width_us
    freq = 1_000_000.0 / est_period_us

    return freq, q_head, q_tail


# ==============================================================================
# 2. PLOTTING FUNCTIONS
# ==============================================================================

def plot_histogram(histogram, bin_width_us, window_index, dominant_rpm, save_dir="plots"):
    # ... (This function is correct and unchanged) ...
    total_count = np.sum(histogram)
    bin_centers_us = (np.arange(len(histogram)) + 0.5) * bin_width_us
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f"histogram_window_{window_index:04d}.png")

    plt.figure(figsize=(12, 7))
    plt.bar(bin_centers_us, histogram, width=bin_width_us * 0.9, color='skyblue', edgecolor='darkblue')
    
    if dominant_rpm > 0: 
        peak_bin = np.argmax(histogram)
        peak_period = bin_centers_us[peak_bin]
        peak_count = histogram[peak_bin]
        plt.axvline(peak_period, color='red', linestyle='--', label=f'Dominant Peak ({dominant_rpm:.0f} RPM)')
        plt.text(peak_period, peak_count * 1.05, f'Peak\n{peak_period:.0f} µs\n{dominant_rpm:.0f} RPM', 
                 color='red', ha='center', fontsize=9)
    
    plt.title(f"Full Period Histogram (Window {window_index}) - Total Measurements: {total_count}")
    plt.xlabel(f"Period ($\mu$s) - Bin Width: {bin_width_us} $\mu$s")
    plt.ylabel("Measurement Count")
    
    def period_to_freq(p):
        out = np.full_like(p, np.inf)
        return np.divide(1_000_000.0, p, out=out, where=p>0)
    def freq_to_period(f):
        out = np.full_like(f, np.inf)
        return np.divide(1_000_000.0, f, out=out, where=f>0)

    ax = plt.gca()
    ax2 = ax.secondary_xaxis('top', functions=(period_to_freq, freq_to_period))
    ax2.set_xlabel("Approximate Frequency (Hz)")
    
    valid_indices = np.where(histogram > 0)[0]
    if valid_indices.size > 0:
        min_p = bin_centers_us[valid_indices.min()] - bin_width_us
        max_p = bin_centers_us[valid_indices.max()] + bin_width_us
        plt.xlim(max(0, min_p), max_p)
    
    if dominant_rpm > 0: plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    return file_path


def plot_highlight_frame(frame_data, window_index, dominant_rpm, save_dir="plots"):
    # ... (This function is correct and unchanged) ...
    if np.sum(frame_data) == 0: return None
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f"frame_window_{window_index:04d}.png")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(frame_data, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
    plt.title(f"Peak Frequency Pixel Highlights (Window {window_index}) - {dominant_rpm:.0f} RPM")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.colorbar(label="Closeness to Peak Frequency (Gaussian Score)")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    return file_path


# NEW IMPORT (at the top of your script)
from sklearn.cluster import DBSCAN

# --- REPLACE your old plot_rotor_clusters function with this ---

def plot_rotor_clusters(highlight_frame, window_index, save_dir="plots"):
    """
    Applies a >0.1 threshold to the hotness map, finds clusters (rotors) 
    using DBSCAN, and plots them with different colors.
    """
    # 1. Apply threshold to get binary mask
    binary_mask = highlight_frame > 0.001
    
    # 2. Get (y, x) coordinates of all active pixels
    y_coords, x_coords = np.where(binary_mask)
    
    # 3. Check if we have enough points to cluster
    if len(y_coords) < 5: # Need at least 5 points for DBSCAN to start
        return None
        
    # 4. Format data for DBSCAN: (n_samples, n_features)
    points = np.stack([x_coords, y_coords], axis=1)
    
    # 5. Run DBSCAN
    # eps=50: Max distance (in pixels) between points to be in the same cluster.
    # min_samples=5: Minimum number of points to form a cluster.
    # *** You may need to TUNE the 'eps' value! ***
    try:
        dbscan = DBSCAN(eps=5, min_samples=10)
        dbscan.fit(points)
        labels = dbscan.labels_ # labels will be 0, 1, 2, 3... (-1 is noise)
    except Exception as e:
        print(f"  [DBSCAN Error: {e}]")
        return None

    # 6. Create the output cluster image
    cluster_image = np.zeros(highlight_frame.shape, dtype=np.int8)
    # DBSCAN uses -1 for noise, so we only plot non-noise points
    mask_valid_labels = labels != -1
    valid_y = y_coords[mask_valid_labels]
    valid_x = x_coords[mask_valid_labels]
    valid_labels = labels[mask_valid_labels]
    
    # Assign cluster labels (1, 2, 3, 4) to the active pixels
    cluster_image[valid_y, valid_x] = valid_labels + 1
    
    num_found_clusters = len(np.unique(valid_labels))
    
    # 7. Plot the cluster image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f"clusters_window_{window_index:04d}.png")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_image, cmap='tab10', interpolation='nearest', vmin=0, vmax=9)
    plt.title(f"Rotor Clusters (Found {num_found_clusters}) - Window {window_index}")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.colorbar(label="Cluster ID (0=Background)")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    return file_path


# ==============================================================================
# 3. MAIN CONTINUOUS PRINTING LOGIC (Modified)
# ==============================================================================

def main_frequency_estimation() -> None:
    parser = argparse.ArgumentParser(description="Estimate dominant frequency from a .dat event file.")
    # (Add parser arguments as before)
    parser.add_argument("dat_path", help="Path to the .dat file.")
    parser.add_argument("--window-ms", type=float, default=10)
    parser.add_argument("--decay-ms", type=float, default=100)
    parser.add_argument("--bin-us", type=int, default=100)
    parser.add_argument("--min-detections", type=int, default=1000) 
    parser.add_argument("--sigma-factor", type=float, default=10.0)
    
    args = parser.parse_args()
    
    window_duration_us = int(args.window_ms * 1000)
    decay_time_us = int(args.decay_ms * 1000)
    bin_width_us = args.bin_us
    min_detections = args.min_detections
    sigma_factor = args.sigma_factor 

    try:
        # 1. Load data and setup
        print(f"Loading and decoding events from: {args.dat_path}...")
        dat_reader = DatMemmap.open(args.dat_path)
        
        timestamps_full = dat_reader.timestamps
        x_coords_full = dat_reader.x_coords
        y_coords_full = dat_reader.y_coords
        polarities_full = dat_reader.polarities
        
        sensor_size = (dat_reader.width, dat_reader.height)
        WIDTH, HEIGHT = sensor_size
        NUM_PIXELS = WIDTH * HEIGHT
        NUM_BINS = 256
        
        print(f"✅ Loaded {dat_reader.event_count} events. Sensor Size: {WIDTH}x{HEIGHT}")
        print(f"Window: {args.window_ms}ms | Decay: {args.decay_ms}ms | Min Detections: {min_detections} | Sigma Factor: {sigma_factor}")

        # 2. Initialize Continuous State
        t_on_prev = np.full(NUM_PIXELS, -1, dtype=np.int64)
        t_off_prev = np.full(NUM_PIXELS, -1, dtype=np.int64)
        period_histogram = np.zeros(NUM_BINS, dtype=np.int32)
        
        fifo_size = dat_reader.event_count
        fifo_bins = np.zeros(fifo_size, np.int32)
        fifo_times = np.zeros(fifo_size, np.int64)
        q_head = 0
        q_tail = 0
        
        # 3. Use index_scheduler logic to chunk the data
        rec = Recording(
            width=WIDTH,
            height=HEIGHT,
            timestamps=timestamps_full,
            event_words=np.empty(0, dtype=np.uint32), 
            order=np.empty(0, dtype=np.int32)
        )
        time_windows = build_windows(rec, window_duration_us)
        WIDTH_numba = np.int64(WIDTH) 

        # 4. Iterate over windows and print RPM
        print("\n" + "=" * 80)
        print(f"Starting Continuous Analysis ({len(time_windows)} windows found)")
        print("=" * 80)
        
        start_time = time.perf_counter()
        
        for i, (start_idx, stop_idx) in enumerate(time_windows):
            if start_idx == stop_idx:
                continue
            
            pixel_last_period = np.zeros(NUM_PIXELS, dtype=np.int64)
            
            ts_window = timestamps_full[start_idx:stop_idx]
            x_window = x_coords_full[start_idx:stop_idx]
            y_window = y_coords_full[start_idx:stop_idx]
            p_window = polarities_full[start_idx:stop_idx]

            freq, q_head, q_tail = process_events_window(
                ts_window, x_window, y_window, p_window,
                WIDTH_numba, NUM_PIXELS, NUM_BINS, decay_time_us, bin_width_us,
                min_detections, 
                t_on_prev, t_off_prev, period_histogram,
                fifo_bins, fifo_times, q_head, q_tail,
                pixel_last_period 
            )
            
            if freq == 0.0:
                continue 
            
            rpm = freq * 60.0

            print(f"\n--- Window {i+1} --- (Time: {ts_window[0]}µs - {ts_window[-1]}µs)")
            print(f"-> Dominant RPM: {rpm:.2f}")

            plot_file_hist = plot_histogram(period_histogram, bin_width_us, i + 1, rpm)
            if plot_file_hist:
                 print(f"-> Histogram saved to: {plot_file_hist}")

            # --- GRADED HIGHLIGHT FRAME LOGIC ---
            peak_bin = np.argmax(period_histogram)
            peak_period_us = (peak_bin + 0.5) * bin_width_us
            sigma_us = bin_width_us * sigma_factor 
            scores_1d = np.zeros(NUM_PIXELS, dtype=float)
            valid_pixels_mask = pixel_last_period > 0
            pixel_error_us = np.abs(pixel_last_period[valid_pixels_mask] - peak_period_us)
            scores = np.exp(-0.5 * (pixel_error_us / sigma_us)**2)
            scores_1d[valid_pixels_mask] = scores
            highlight_frame = scores_1d.reshape((HEIGHT, WIDTH))
            
            plot_file_frame = plot_highlight_frame(highlight_frame, i + 1, rpm)
            if plot_file_frame:
                print(f"-> Highlight frame saved to: {plot_file_frame}")
            
            # --- NEW: CALL CLUSTER PLOTTING ---
            plot_file_cluster = plot_rotor_clusters(highlight_frame, i + 1)
            if plot_file_cluster:
                print(f"-> Rotor clusters saved to: {plot_file_cluster}")

        end_time = time.perf_counter()
        print("=" * 80)
        print(f"End of analysis. Processed {len(time_windows)} windows in {end_time - start_time:.2f} seconds.")
        
    except FileNotFoundError:
        print(f"\nError: File not found at '{args.dat_path}'.")
    except ImportError:
        print("\nError: `scikit-learn` library not found. Please install it with `pip install scikit-learn` to use the clustering feature.")
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")

if __name__ == "__main__":
    main_frequency_estimation()