from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
from numba import njit
from sklearn.cluster import DBSCAN

# --- Assume necessary imports are available ---
from evio.core.mmap import DatMemmap
from evio.core.index_scheduler import build_windows
from evio.core.recording import Recording


# ==============================================================================
# 1. NUMBA CORE FUNCTION
# ==============================================================================

@njit
def process_events_window(
    timestamps, x_coords, y_coords, polarities,
    WIDTH, NUM_PIXELS, NUM_BINS, decay_time_us, bin_width_us, min_detections,
    t_on_prev, t_off_prev, period_histogram,
    fifo_bins, fifo_times, q_head, q_tail,
    pixel_last_period,
) -> tuple[float, int, int]:
    """
    Process one window of events and update the global period_histogram as well
    as per-pixel last period estimates. Returns (freq, q_head, q_tail).
    """
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

        # decay old histogram contributions
        while q_head < q_tail:
            bin_old, t_old = fifo_bins[q_head], fifo_times[q_head]
            if t - t_old > decay_time_us:
                period_histogram[bin_old] -= 1
                q_head += 1
            else:
                break

    total = period_histogram.sum()
    if total < min_detections:
        return 0.0, q_head, q_tail

    peak_bin = period_histogram.argmax()
    est_period_us = (peak_bin + 0.5) * bin_width_us
    freq = 1_000_000.0 / est_period_us

    return freq, q_head, q_tail


# ==============================================================================
# 2. CLUSTERING: ROTOR & DRONE BOUNDING BOXES
# ==============================================================================

def compute_rotor_and_drone_bboxes(
    highlight_frame: np.ndarray,
    eps: float = 5.0,
    min_samples: int = 10,
    threshold: float = 0.001,
):
    """
    From the 'hotness' map (highlight_frame), find rotor clusters via DBSCAN and
    return:
      - rotor_boxes: list of (x_min, y_min, width, height)
      - drone_box: single (x_min, y_min, width, height) covering all rotors
    """
    binary_mask = highlight_frame > threshold
    y_coords, x_coords = np.where(binary_mask)

    if len(y_coords) < min_samples:
        return [], None

    points = np.stack([x_coords, y_coords], axis=1)

    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(points)
        labels = dbscan.labels_
    except Exception as e:
        print(f"[DBSCAN Error: {e}]")
        return [], None

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # remove noise

    rotor_boxes: list[tuple[int, int, int, int]] = []
    all_x_min, all_y_min = np.inf, np.inf
    all_x_max, all_y_max = -np.inf, -np.inf

    for lab in unique_labels:
        cluster_points = points[labels == lab]
        if cluster_points.size == 0:
            continue

        xs = cluster_points[:, 0]
        ys = cluster_points[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        rotor_boxes.append(
            (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
        )

        all_x_min = min(all_x_min, x_min)
        all_y_min = min(all_y_min, y_min)
        all_x_max = max(all_x_max, x_max)
        all_y_max = max(all_y_max, y_max)

    if len(rotor_boxes) == 0:
        return [], None

    drone_box = (
        int(all_x_min),
        int(all_y_min),
        int(all_x_max - all_x_min + 1),
        int(all_y_max - all_y_min + 1),
    )
    return rotor_boxes, drone_box


# ==============================================================================
# 3. KALMAN FILTER TRACKER (RPM-AWARE)
# ==============================================================================

class BoxKalmanTracker:
    """
    Kalman filter for a single bounding box.

    State: [cx, cy, vx, vy, w, h]^T
    Measurement: [cx, cy, w, h]^T

    RPM-aware:
    - RPM level & trend modulate process noise (expected motion).
    - Low/flat RPM -> tracker tends to keep velocity near zero.
    - High/increasing RPM -> tracker allows larger velocity changes.
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.initialized = False
        self.last_t = None

        # Recent predicted centers for extra motion sensitivity (optional)
        self.recent_states: list[tuple[float, float, float]] = []

        # RPM state
        self.current_rpm: float | None = None
        self.rpm_history: list[tuple[float, float]] = []  # (t, rpm)

        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # cx
                [0, 1, 0, 0, 0, 0],  # cy
                [0, 0, 0, 0, 1, 0],  # w
                [0, 0, 0, 0, 0, 1],  # h
            ],
            dtype=np.float32,
        )

        # Base process noise (will be modulated by RPM)
        self.base_process_noise = np.diag(
            [1e-3, 1e-3, 1e-2, 1e-2, 1e-4, 1e-4]
        ).astype(np.float32)
        self.kf.processNoiseCov = self.base_process_noise.copy()

        self.kf.measurementNoiseCov = np.diag(
            [1e-1, 1e-1, 1e-1, 1e-1]
        ).astype(np.float32)

        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    # -------- RPM interface -------------------------------------------------
    def update_rpm(self, rpm: float, t: float):
        """
        Call this each frame BEFORE predict():

        tracker.update_rpm(rpm_estimate, t_rec)
        """
        self.current_rpm = float(rpm)
        self.rpm_history.append((t, self.current_rpm))
        if len(self.rpm_history) > 20:
            self.rpm_history = self.rpm_history[-20:]

    def _rpm_stats(self):
        """
        Compute a simple RPM level and derivative (trend) from history.

        Returns:
            rpm_level: float   (current rpm or 0 if unknown)
            rpm_deriv: float   (approx dRPM/dt, rpm per second)
        """
        if self.current_rpm is None or len(self.rpm_history) < 2:
            return 0.0, 0.0

        rpm_level = self.current_rpm
        (t0, r0), (t1, r1) = self.rpm_history[-2], self.rpm_history[-1]
        dt = max(1e-3, t1 - t0)
        rpm_deriv = (r1 - r0) / dt
        return rpm_level, rpm_deriv

    # -------- Kalman core ---------------------------------------------------
    def init(self, bbox, t: float):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        self.kf.statePost = np.array(
            [[cx], [cy], [0.0], [0.0], [w], [h]], dtype=np.float32
        )
        self.last_t = t
        self.initialized = True
        self.recent_states = [(t, cx, cy)]

    def _update_transition(self, dt: float):
        A = np.eye(6, dtype=np.float32)
        A[0, 2] = dt  # cx += vx * dt
        A[1, 3] = dt  # cy += vy * dt
        self.kf.transitionMatrix = A

    def _update_process_noise_with_rpm(self):
        """
        Adjust processNoiseCov depending on RPM level and trend.

        Intuition:
        - When RPM low & not changing: small velocity noise => motion stays near zero.
        - When RPM high or increasing rapidly: large velocity noise => motion can change quickly.
        """
        rpm_level, rpm_deriv = self._rpm_stats()

        MAX_RPM = 10000.0  # approximate normalization
        rpm_norm = np.clip(rpm_level / MAX_RPM, 0.0, 1.0)

        RPM_DERIV_REF = 5000.0  # rpm/s considered "big change"
        trend_norm = np.clip(rpm_deriv / RPM_DERIV_REF, -1.0, 1.0)

        base_vel_scale = 0.2
        k_level = 1.0
        k_trend = 1.0

        vel_scale = base_vel_scale + k_level * rpm_norm + k_trend * max(0.0, trend_norm)
        vel_scale = float(np.clip(vel_scale, 0.1, 5.0))

        Q = self.base_process_noise.copy()
        # Scale velocity noise
        Q[2, 2] *= vel_scale**2
        Q[3, 3] *= vel_scale**2

        # Slightly scale position noise with rpm to allow small moves at high RPM
        pos_scale = 0.5 + 0.5 * rpm_norm  # 0.5–1.0
        Q[0, 0] *= pos_scale**2
        Q[1, 1] *= pos_scale**2

        self.kf.processNoiseCov = Q

        # If rpm is very low and trend small -> damp velocity
        if rpm_level < 500.0 and abs(rpm_deriv) < 200.0:
            state = self.kf.statePost
            state[2, 0] *= 0.7  # vx
            state[3, 0] *= 0.7  # vy
            self.kf.statePost = state

    def predict(self, t: float):
        if not self.initialized:
            return None

        dt = 0.0
        if self.last_t is not None:
            dt = max(1e-3, t - self.last_t)
        self.last_t = t

        self._update_transition(dt)
        self._update_process_noise_with_rpm()

        pred = self.kf.predict()

        cx, cy, vx, vy, w, h = pred.flatten()
        self.recent_states.append((t, float(cx), float(cy)))
        if len(self.recent_states) > 10:
            self.recent_states = self.recent_states[-10:]

        return self._state_to_bbox(pred)

    def correct(self, bbox):
        if not self.initialized or bbox is None:
            return
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(meas)

    def get_state_bbox(self):
        if not self.initialized:
            return None
        return self._state_to_bbox(self.kf.statePost)

    def predict_future(self, horizon_s: float, frame_shape):
        """
        Optional discrete future bbox prediction using current state + recent velocity.
        """
        if not self.initialized:
            return None

        state = self.kf.statePost.flatten()
        cx, cy, vx_kf, vy_kf, w, h = state

        if len(self.recent_states) >= 2:
            (t0, cx0, cy0), (t1, cx1, cy1) = self.recent_states[-2], self.recent_states[-1]
            dt = max(1e-3, t1 - t0)
            vx_recent = (cx1 - cx0) / dt
            vy_recent = (cy1 - cy0) / dt

            alpha_v = 0.7
            vx = alpha_v * vx_recent + (1.0 - alpha_v) * vx_kf
            vy = alpha_v * vy_recent + (1.0 - alpha_v) * vy_kf
        else:
            vx = vx_kf
            vy = vy_kf

        cx_future = cx + vx * horizon_s
        cy_future = cy + vy * horizon_s

        H, W, _ = frame_shape
        x_future = int(round(cx_future - w / 2.0))
        y_future = int(round(cy_future - h / 2.0))

        x_future = max(0, min(W - int(w), x_future))
        y_future = max(0, min(H - int(h), y_future))

        return (x_future, y_future, int(w), int(h))

    @staticmethod
    def _state_to_bbox(state_vec):
        cx, cy, vx, vy, w, h = state_vec.flatten()
        x = int(round(cx - w / 2.0))
        y = int(round(cy - h / 2.0))
        return (x, y, int(round(w)), int(round(h)))


# ==============================================================================
# 4. FUTURE GAUSSIAN HEAT VISUALIZATION
# ==============================================================================

def draw_future_gaussian_distribution(
    tracker: BoxKalmanTracker,
    frame: np.ndarray,
    max_horizon: float = 2.0,
    steps: int = 20,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Draw a Gaussian-like probability distribution of where the drone can be
    from t=0s to t=max_horizon seconds into the future.
    """
    if not getattr(tracker, "initialized", False):
        return frame

    H, W, _ = frame.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    state = tracker.kf.statePost.flatten()
    cx, cy, vx_kf, vy_kf, w, h = state

    vx, vy = vx_kf, vy_kf
    if hasattr(tracker, "recent_states") and len(tracker.recent_states) >= 2:
        (t0, cx0, cy0), (t1, cx1, cy1) = tracker.recent_states[-2], tracker.recent_states[-1]
        dt = max(1e-3, t1 - t0)
        vx_recent = (cx1 - cx0) / dt
        vy_recent = (cy1 - cy0) / dt
        alpha_v = 0.7
        vx = alpha_v * vx_recent + (1.0 - alpha_v) * vx_kf
        vy = alpha_v * vy_recent + (1.0 - alpha_v) * vy_kf

    times = np.linspace(0.0, max_horizon, steps)

    base_sigma = max(w, h) * 0.25
    sigma_growth = max(w, h) * 0.5 / max_horizon if max_horizon > 0 else 0.0

    for t in times:
        cx_t = cx + vx * t
        cy_t = cy + vy * t

        sigma_t = base_sigma + sigma_growth * t
        sigma_x = sigma_t
        sigma_y = sigma_t

        x_min = int(max(0, np.floor(cx_t - 3 * sigma_x)))
        x_max = int(min(W - 1, np.ceil(cx_t + 3 * sigma_x)))
        y_min = int(max(0, np.floor(cy_t - 3 * sigma_y)))
        y_max = int(min(H - 1, np.ceil(cy_t + 3 * sigma_y)))

        if x_max <= x_min or y_max <= y_min:
            continue

        xs = np.arange(x_min, x_max + 1, dtype=np.float32)
        ys = np.arange(y_min, y_max + 1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        g = np.exp(
            -(((X - cx_t) ** 2) / (2 * sigma_x**2)
              + ((Y - cy_t) ** 2) / (2 * sigma_y**2))
        )

        weight = 1.0 - (t / max_horizon) * 0.3 if max_horizon > 0 else 1.0
        g *= weight

        heatmap[y_min:y_max + 1, x_min:x_max + 1] += g.astype(np.float32)

    if heatmap.max() > 0:
        heatmap_norm = heatmap / heatmap.max()
    else:
        heatmap_norm = heatmap

    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(color_map, alpha, frame, 1 - alpha, 0)

    mask = heatmap_uint8 > 0
    out = frame.copy()
    out[mask] = overlay[mask]

    cv2.putText(
        out,
        f"Gaussian future distribution (0–{max_horizon:.1f}s)",
        (10, H - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


# ==============================================================================
# 5. HUD / BOUNDING BOX DRAWING
# ==============================================================================

@dataclass
class BatchRange:
    end_ts_us: int


class DummyPacer:
    """
    Minimal pacer-like object just to drive the HUD if you don't have your own
    Pacer implementation handy.
    """
    def __init__(self, speed: float, e_start: int):
        self._t_start = time.perf_counter()
        self._e_start = e_start
        self.speed = speed
        self.force_speed = False
        self.instantaneous_drop_rate = 0.0
        self.average_drop_rate = 0.0


def draw_drone_hud(
    frame: np.ndarray,
    pacer,
    batch_range: BatchRange,
    rotor_boxes: list[tuple[int, int, int, int]],
    drone_box: tuple[int, int, int, int] | None,
    rpm: float,
    *,
    predicted_box: tuple[int, int, int, int] | None = None,
    hud_color: tuple[int, int, int] = (0, 0, 0),      # text color (BGR)
    rotor_color: tuple[int, int, int] = (0, 255, 0),  # propellers (green)
    drone_color: tuple[int, int, int] = (0, 0, 255),  # measurement drone (red)
    pred_color: tuple[int, int, int] = (255, 0, 0),   # predicted box (blue)
) -> None:
    """
    Visualization using the same style as your draw_hud template:

      - Timing HUD (wall, rec time, speed info)
      - RPM on a third line
      - Green bounding boxes: propellers
      - Red bounding box: measured drone box
      - Blue bounding box: Kalman predicted box
    """
    # ---------------- HUD PART (from your template) ----------------
    if pacer is not None and pacer._t_start is not None and pacer._e_start is not None:
        wall_time_s = time.perf_counter() - pacer._t_start
        rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

        if getattr(pacer, "force_speed", False):
            first_row_str = (
                f"speed={pacer.speed:.2f}x"
                f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
                f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
            )
        else:
            first_row_str = (
                f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
            )

        second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"
        third_row_str = f"RPM={rpm:8.1f}"

        # first row
        cv2.putText(
            frame,
            first_row_str,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            hud_color,
            1,
            cv2.LINE_AA,
        )

        # second row
        cv2.putText(
            frame,
            second_row_str,
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            hud_color,
            1,
            cv2.LINE_AA,
        )

        # third row (RPM)
        cv2.putText(
            frame,
            third_row_str,
            (8, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            hud_color,
            1,
            cv2.LINE_AA,
        )

    # ---------------- BOUNDING BOXES PART ----------------
    # Rotor / propeller boxes
    for (x, y, w, h) in rotor_boxes:
        cv2.rectangle(
            frame,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            rotor_color,
            2,
        )

    # Measured whole drone box
    if drone_box is not None:
        x, y, w, h = drone_box
        cv2.rectangle(
            frame,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            drone_color,
            2,
        )
        cv2.putText(
            frame,
            "DRONE (meas)",
            (int(x), max(0, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            drone_color,
            1,
            cv2.LINE_AA,
        )

    # Predicted box from Kalman
    if predicted_box is not None:
        px, py, pw, ph = predicted_box
        cv2.rectangle(
            frame,
            (int(px), int(py)),
            (int(px + pw), int(py + ph)),
            pred_color,
            2,
        )
        cv2.putText(
            frame,
            "PRED",
            (int(px), max(0, int(py) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            pred_color,
            1,
            cv2.LINE_AA,
        )


# ==============================================================================
# 6. MAIN CONTINUOUS LOGIC WITH REAL-TIME VISUALIZATION
# ==============================================================================

def main_frequency_estimation() -> None:
    parser = argparse.ArgumentParser(description="Estimate dominant frequency from a .dat event file.")
    parser.add_argument("dat_path", help="Path to the .dat file.")
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--decay-ms", type=float, default=100.0)
    parser.add_argument("--blade", type=float, default=2.0)
    parser.add_argument("--bin-us", type=int, default=100)
    parser.add_argument("--min-detections", type=int, default=1000)
    parser.add_argument("--sigma-factor", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (for HUD display only).")

    args = parser.parse_args()

    window_duration_us = int(args.window_ms * 1000)
    decay_time_us = int(args.decay_ms * 1000)
    bin_width_us = args.bin_us
    min_detections = args.min_detections
    sigma_factor = args.sigma_factor
    blade =  args.blade 

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
        print(f"Window: {args.window_ms}ms | Decay: {args.decay_ms}ms | "
              f"Min Detections: {min_detections} | Sigma Factor: {sigma_factor}")

        # 2. Initialize Continuous State
        t_on_prev = np.full(NUM_PIXELS, -1, dtype=np.int64)
        t_off_prev = np.full(NUM_PIXELS, -1, dtype=np.int64)
        period_histogram = np.zeros(NUM_BINS, dtype=np.int32)

        fifo_size = dat_reader.event_count
        fifo_bins = np.zeros(fifo_size, np.int32)
        fifo_times = np.zeros(fifo_size, np.int64)
        q_head = 0
        q_tail = 0

        rec = Recording(
            width=WIDTH,
            height=HEIGHT,
            timestamps=timestamps_full,
            event_words=np.empty(0, dtype=np.uint32),
            order=np.empty(0, dtype=np.int32),
        )
        time_windows = build_windows(rec, window_duration_us)
        WIDTH_numba = np.int64(WIDTH)

        # Simple Pacer-like object for HUD timing
        if len(timestamps_full) > 0:
            pacer = DummyPacer(speed=args.speed, e_start=int(timestamps_full[0]))
        else:
            pacer = DummyPacer(speed=args.speed, e_start=0)

        # RPM-aware Kalman tracker for the whole drone box
        tracker = BoxKalmanTracker()

        print("\n" + "=" * 80)
        print(f"Starting Continuous Analysis ({len(time_windows)} windows found)")
        print("=" * 80)

        start_time = time.perf_counter()

        # 3. Real-time visualization loop
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
                WIDTH_numba, NUM_PIXELS, NUM_BINS,
                decay_time_us, bin_width_us,
                min_detections,
                t_on_prev, t_off_prev, period_histogram,
                fifo_bins, fifo_times, q_head, q_tail,
                pixel_last_period,
            )

            if freq == 0.0:
                # Not enough detections yet, skip this window
                continue

            rpm = freq * 60.0 / blade 

            print(f"\n--- Window {i+1} --- (Time: {ts_window[0]}µs - {ts_window[-1]}µs)")
            print(f"-> Dominant RPM: {rpm:.2f}")

            # Highlight-frame logic
            peak_bin = np.argmax(period_histogram)
            peak_period_us = (peak_bin + 0.5) * bin_width_us
            sigma_us = bin_width_us * sigma_factor

            scores_1d = np.zeros(NUM_PIXELS, dtype=float)
            valid_pixels_mask = pixel_last_period > 0
            pixel_error_us = np.abs(pixel_last_period[valid_pixels_mask] - peak_period_us)
            scores = np.exp(-0.5 * (pixel_error_us / sigma_us) ** 2)
            scores_1d[valid_pixels_mask] = scores
            highlight_frame = scores_1d.reshape((HEIGHT, WIDTH))

            # Normalize to [0, 1] for visualization
            max_v = np.max(highlight_frame)
            if max_v > 0:
                vis = highlight_frame / max_v
            else:
                vis = highlight_frame

            vis = np.clip(vis, 0.0, 1.0)
            vis_u8 = (vis * 255).astype(np.uint8)        # (H, W)
            frame = cv2.cvtColor(vis_u8, cv2.COLOR_GRAY2BGR)  # (H, W, 3) BGR

            # Compute bounding boxes for propellers & drone
            rotor_boxes, drone_box = compute_rotor_and_drone_bboxes(
                highlight_frame,
                eps=5.0,
                min_samples=10,
                threshold=0.001,
            )

            # Time in seconds (recording time) for Kalman / RPM
            t_rec_s = float(ts_window[-1]) / 1e6

            # Update RPM in tracker (RPM-aware process noise)
            tracker.update_rpm(rpm, t_rec_s)

            predicted_box = None

            # Initialize tracker with first drone_box
            if drone_box is not None and not tracker.initialized:
                tracker.init(drone_box, t_rec_s)

            if tracker.initialized:
                # Predict at current time
                predicted_box = tracker.predict(t_rec_s)
                # Correct with measurement if we have one
                if drone_box is not None:
                    tracker.correct(drone_box)

            # BatchRange for HUD rec_time
            batch_range = BatchRange(end_ts_us=int(ts_window[-1]))

            # Draw HUD + bounding boxes + predicted box
            draw_drone_hud(
                frame,
                pacer=pacer,
                batch_range=batch_range,
                rotor_boxes=rotor_boxes,
                drone_box=drone_box,
                rpm=rpm,
                predicted_box=predicted_box,
            )

            # Draw future Gaussian distribution of possible drone locations
            frame_gauss = draw_future_gaussian_distribution(
                tracker,
                frame,
                max_horizon=2.0,
                steps=20,
                alpha=0.6,
            )

            # Real-time display
            cv2.imshow("Drone Propeller Detection (HUD + Future Gaussian)", frame_gauss)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break

        end_time = time.perf_counter()
        print("=" * 80)
        print(f"End of analysis. Processed {len(time_windows)} windows in {end_time - start_time:.2f} seconds.")

        cv2.destroyAllWindows()

    except FileNotFoundError:
        print(f"\nError: File not found at '{args.dat_path}'.")
    except ImportError as e:
        print("\nError: Missing dependency.")
        print(e)
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")


if __name__ == "__main__":
    main_frequency_estimation()
