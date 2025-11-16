import numpy as np
import time 
from numba import njit
import argparse
import os
import cv2
from sklearn.cluster import DBSCAN
import math
from collections import deque

from evio.core.mmap import DatMemmap
from evio.core.index_scheduler import build_windows 
from evio.core.recording import Recording


# --- plot_future_pose_linear function removed ---

@njit
def process_events_window(
    timestamps, x_coords, y_coords, polarities,
    WIDTH, NUM_BINS, decay_time_us, bin_width_us,
    t_on_prev, t_off_prev,
    pixel_last_period,
    period_histogram,
    fifo_bins, fifo_times, q_head, q_tail,
):
    # ... (function content is unchanged)
    WIDTH_64 = np.int64(WIDTH)
    max_period_us = NUM_BINS * bin_width_us
    
    for i in range(len(timestamps)):
        t, x, y, p = timestamps[i], x_coords[i], y_coords[i], polarities[i]
        pixel_idx = y * WIDTH_64 + x

        if p == 1:  # ON event
            if t_off_prev[pixel_idx] > t_on_prev[pixel_idx]:
                delta_t = t - t_on_prev[pixel_idx]
                pixel_last_period[pixel_idx] = delta_t
                
                if 0 < delta_t < max_period_us:
                    bin_index = delta_t // bin_width_us
                    if bin_index >= NUM_BINS:
                        bin_index = NUM_BINS - 1
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
    
    return q_head, q_tail


def visualize_rotors_opencv(
    pixel_last_period_1d, WIDTH, HEIGHT, window_index,
    min_detections, global_period_us, similarity_threshold,
    num_blades=1, max_rpm_display=10000
):
    """
    Processes rotor data and returns visualization frames.
    Returns:
        (frame, bar_frame) if drone is detected.
        (None, None) if no drone is detected.
    """

    # Identify valid pixels
    valid_mask = pixel_last_period_1d > 0
    if valid_mask.sum() < min_detections:
        return None, None

    pixel_freqs = 1_000_000.0 / pixel_last_period_1d[valid_mask]
    global_freq = 1_000_000.0 / global_period_us
    freq_diff_percent = np.abs(pixel_freqs - global_freq) / global_freq * 100
    similarity_mask = freq_diff_percent <= similarity_threshold
    if similarity_mask.sum() < min_detections:
        return None, None

    valid_indices = np.where(valid_mask)[0]
    similar_indices = valid_indices[similarity_mask]
    similar_periods = pixel_last_period_1d[similar_indices]
    y_coords = similar_indices // WIDTH
    x_coords = similar_indices % WIDTH
    points = np.stack([x_coords, y_coords], axis=1)

    # Cluster pixels using DBSCAN
    dbscan = DBSCAN(eps=5, min_samples=10)
    dbscan.fit(points)
    labels = dbscan.labels_

    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if not unique_labels:
        return None, None

    # Predefined colors for clusters (BGR)
    colors = [
        (0, 255, 0),     # green
        (0, 255, 255),   # yellow
        (255, 0, 0),     # blue
        (255, 0, 255)    # magenta
    ]

    # Compute cluster data
    cluster_data = []
    for idx, k in enumerate(unique_labels[:4]):  # limit to 4 clusters
        mask = labels == k
        cluster_points = points[mask]
        cluster_periods = similar_periods[mask]
        mean_period_us = np.mean(cluster_periods)
        mean_rpm = (1_000_000.0 / mean_period_us) * 60.0 / num_blades

        x_min, y_min = np.min(cluster_points, axis=0)
        x_max, y_max = np.max(cluster_points, axis=0)
        width = x_max - x_min
        height = y_max - y_min

        cluster_data.append({
            'rpm': mean_rpm,
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'width': width, 'height': height,
            'color': colors[idx]
        })

    # Active pixel frame
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[y_coords, x_coords] = [255, 255, 255]  # red active pixels

    # Draw cluster bounding boxes
    for cluster in cluster_data:
        cv2.rectangle(
            frame,
            (cluster['x_min'], cluster['y_min']),
            (cluster['x_max'], cluster['y_max']),
            cluster['color'], 1
        )

    # Draw bounding box for the whole drone
    if cluster_data:
        x_min_total = min(c['x_min'] for c in cluster_data)
        y_min_total = min(c['y_min'] for c in cluster_data)
        x_max_total = max(c['x_max'] for c in cluster_data)
        y_max_total = max(c['y_max'] for c in cluster_data)
        cv2.rectangle(
            frame,
            (x_min_total, y_min_total),
            (x_max_total, y_max_total),
            (255, 255, 255),   # white for drone bounding box
        )

        # compute center
        cx = (x_min_total + x_max_total) / 2.0
        cy = (y_min_total + y_max_total) / 2.0
        
        # use last event timestamp as timestamp_us
        timestamp_us = int(time.time() * 1e6)
        
        # append to queue
        drone_centers_queue.append((timestamp_us, cx, cy))

    # --- MODIFIED: Bar plot layout ---
    bar_draw_height = HEIGHT // 8   # Increased from // 10
    text_space = 20                 # Increased from 15
    bar_frame_height = bar_draw_height + text_space # Total height of the bar window

    bar_width = 40                  # Increased from 30
    spacing = 25                    # Increased from 20
    num_bars = 4
    bar_frame_width = num_bars * bar_width + (num_bars + 1) * spacing
    
    # Create the bar_frame with the new total height
    bar_frame = np.zeros((bar_frame_height, bar_frame_width, 3), dtype=np.uint8)

    # Fill RPMs for bars in fixed order
    rpms = [c['rpm'] if i < len(cluster_data) else 0 for i, c in enumerate(cluster_data)]
    while len(rpms) < num_bars:
        rpms.append(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5                # Increased from 0.4
    font_thickness = 1
    
    # Draw bars
    for i in range(num_bars):
        rpm = rpms[i]
        
        # Bar height is calculated based on the drawing area
        height = int(np.clip(rpm / max_rpm_display, 0, 1) * bar_draw_height)
        
        x1 = spacing + i * (bar_width + spacing)
        
        # Bar top Y-coordinate
        y1 = bar_draw_height - height 
        
        # Bar bottom Y-coordinate is now shifted up
        y2 = bar_draw_height 
        
        x2 = x1 + bar_width
        
        color = cluster_data[i]['color'] if i < len(cluster_data) else (100, 100, 100)
        cv2.rectangle(bar_frame, (x1, y1), (x2, y2), color, -1)
        
        # Draw RPM value *below* the bar
        text = f"{int(rpm)}"
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x1 + (bar_width - text_width) // 2  # Center text
        
        # Place text in the 'text_space' area
        text_y = bar_frame_height - 7 # 7px from the bottom edge
        
        cv2.putText(bar_frame, text, (text_x, text_y),
                    font, font_scale, (255, 255, 255), font_thickness)
    # --- END MODIFICATION ---

    return frame, bar_frame


def main():
    parser = argparse.ArgumentParser(description="Real-time rotor RPM visualization using OpenCV")
    parser.add_argument("dat_path", help="Path to the .dat file")
    parser.add_argument("--window-ms", type=float, default=10)
    parser.add_argument("--decay-ms", type=float, default=100)
    parser.add_argument("--min-detections", type=int, default=100)
    parser.add_argument("--bin-us", type=int, default=100)
    parser.add_argument("--similarity-threshold", type=float, default=70.0)
    parser.add_argument("--num-blades", type=int, default=2)
    args = parser.parse_args()

    window_duration_us = int(args.window_ms * 1000)
    decay_time_us = int(args.decay_ms * 1000)
    num_blades = args.num_blades

    dat_reader = DatMemmap.open(args.dat_path)
    timestamps_full = dat_reader.timestamps
    x_coords_full = dat_reader.x_coords
    y_coords_full = dat_reader.y_coords
    polarities_full = dat_reader.polarities
    WIDTH = dat_reader.width
    HEIGHT = dat_reader.height
    NUM_PIXELS = WIDTH * HEIGHT
    NUM_BINS = 256

    t_on_prev = np.full(NUM_PIXELS, -1, dtype=np.int64)
    t_off_prev = np.full(NUM_PIXELS, -1, dtype=np.int64)
    period_histogram = np.zeros(NUM_BINS, dtype=np.int32)
    fifo_size = dat_reader.event_count
    fifo_bins = np.zeros(fifo_size, dtype=np.int32)
    fifo_times = np.zeros(fifo_size, dtype=np.int64)
    q_head = 0
    q_tail = 0

    rec = Recording(
        width=WIDTH,
        height=HEIGHT,
        timestamps=timestamps_full,
        event_words=np.empty(0, dtype=np.uint32), 
        order=np.empty(0, dtype=np.int32)
    )
    time_windows = build_windows(rec, window_duration_us)


    # Output video parameters
    output_file = "drone_visualization.mp4"
    fps = 1000.0 / args.window_ms 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (WIDTH, HEIGHT))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_file}")
        return

    print(f"Writing video to {output_file} at {fps:.2f} FPS... Press 'q' to stop.")

    for i, (start_idx, stop_idx) in enumerate(time_windows):
        
        video_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        if start_idx == stop_idx:
            video_writer.write(video_frame)
            cv2.imshow("Drone Visualization", video_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping...")
                break
            continue

        pixel_last_period = np.zeros(NUM_PIXELS, dtype=np.int64)

        ts_window = timestamps_full[start_idx:stop_idx]
        x_window = x_coords_full[start_idx:stop_idx]
        y_window = y_coords_full[start_idx:stop_idx]
        p_window = polarities_full[start_idx:stop_idx]

        q_head, q_tail = process_events_window(
            ts_window, x_window, y_window, p_window,
            np.int64(WIDTH), NUM_BINS, decay_time_us, args.bin_us,
            t_on_prev, t_off_prev,
            pixel_last_period,
            period_histogram,
            fifo_bins, fifo_times, q_head, q_tail,
        )

        if period_histogram.sum() < args.min_detections:
            video_writer.write(video_frame)
            cv2.imshow("Drone Visualization", video_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping...")
                break
            continue

        peak_bin = np.argmax(period_histogram)
        global_period_us = (peak_bin + 0.5) * args.bin_us

        frame, bar_frame = visualize_rotors_opencv(
            pixel_last_period,
            WIDTH, HEIGHT,
            i + 1,
            args.min_detections,
            global_period_us,
            args.similarity_threshold,
            num_blades=num_blades
        )
        
        if frame is not None:
            video_frame = frame
            
            if bar_frame is not None:
                bar_height, bar_frame_width = bar_frame.shape[:2]
                
                margin = 10
                y_offset = HEIGHT - bar_height - margin
                x_offset = WIDTH - bar_frame_width - margin

                # Check bounds
                if (y_offset >= 0) and (x_offset >= 0):
                    video_frame[y_offset : y_offset + bar_height, 
                                x_offset : x_offset + bar_frame_width] = bar_frame
        
        video_writer.write(video_frame)
        cv2.imshow("Drone Visualization", video_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping...")
            break

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Finished writing video to {output_file}")


if __name__ == "__main__":
    main()