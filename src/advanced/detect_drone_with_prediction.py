"""Drone detection and behavioral analysis from event camera data with bounding boxes and trajectory prediction."""

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from numba import njit

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from evio.core.pacer import Pacer
from evio.core.recording import open_dat
from evio.source.dat_file import BatchRange, DatFileSource


@dataclass
class BoundingBox:
    """Bounding box with timestamp."""

    timestamp_s: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class DronePosition:
    """Drone position at a specific time."""

    timestamp_s: float
    center_x: float
    center_y: float


@dataclass
class MotionAnalysis:
    """Motion analysis results."""

    speed_px_s: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    acceleration: float = 0.0
    direction_deg: float = 0.0
    pattern: str = "UNKNOWN"
    yaw_deg: float = 0.0
    rpm: float = 0.0  # Overall RPM
    rpm_propeller_1: float = 0.0
    rpm_propeller_2: float = 0.0
    rpm_propeller_3: float = 0.0
    rpm_propeller_4: float = 0.0


# ==============================================================================
# DRONE DETECTION FROM EVENT DATA (FROM FRIEND'S CODE)
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
# KALMAN FILTER TRACKER FOR TRAJECTORY PREDICTION
# ==============================================================================

class DroneKalmanTracker:
    """
    Kalman filter for drone trajectory prediction.
    
    State: [cx, cy, vx, vy, w, h]^T
    Measurement: [cx, cy, w, h]^T
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.initialized = False
        self.last_t = None

        # Recent predicted centers for velocity estimation
        self.recent_states: list[tuple[float, float, float]] = []

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

        # Process noise
        self.kf.processNoiseCov = np.diag(
            [1e-3, 1e-3, 1e-2, 1e-2, 1e-4, 1e-4]
        ).astype(np.float32)

        # Measurement noise
        self.kf.measurementNoiseCov = np.diag(
            [1e-1, 1e-1, 1e-1, 1e-1]
        ).astype(np.float32)

        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        # Add RPM tracking
        self.current_rpm: float | None = None
        self.rpm_history: list[tuple[float, float]] = []  # (time, rpm)
        
        # Store base process noise for scaling
        self.base_process_noise = np.diag(
            [1e-3, 1e-3, 1e-2, 1e-2, 1e-4, 1e-4]
        ).astype(np.float32)
        self.kf.processNoiseCov = self.base_process_noise.copy()

    def update_rpm(self, rpm: float, t: float):
        """Update RPM before prediction to adjust motion model."""
        self.current_rpm = float(rpm)
        self.rpm_history.append((t, self.current_rpm))
        if len(self.rpm_history) > 20:
            self.rpm_history = self.rpm_history[-20:]

    def _rpm_stats(self):
        """Compute RPM level and derivative (trend)."""
        if self.current_rpm is None or len(self.rpm_history) < 2:
            return 0.0, 0.0
        
        rpm_level = self.current_rpm
        (t0, r0), (t1, r1) = self.rpm_history[-2], self.rpm_history[-1]
        dt = max(1e-3, t1 - t0)
        rpm_deriv = (r1 - r0) / dt
        return rpm_level, rpm_deriv

    def _update_process_noise_with_rpm(self):
        """Adjust process noise based on RPM - high RPM = more motion."""
        rpm_level, rpm_deriv = self._rpm_stats()
        
        # Normalize RPM (assume max ~10000)
        MAX_RPM = 10000.0
        rpm_norm = np.clip(rpm_level / MAX_RPM, 0.0, 1.0)
        
        # Normalize trend (big change = 5000 rpm/s)
        RPM_DERIV_REF = 5000.0
        trend_norm = np.clip(rpm_deriv / RPM_DERIV_REF, -1.0, 1.0)
        
        # Calculate velocity noise scaling
        base_vel_scale = 0.2
        k_level = 1.0  # Weight for RPM level
        k_trend = 1.0  # Weight for RPM change rate
        
        vel_scale = base_vel_scale + k_level * rpm_norm + k_trend * max(0.0, trend_norm)
        vel_scale = float(np.clip(vel_scale, 0.1, 5.0))
        
        # Create scaled process noise
        Q = self.base_process_noise.copy()
        Q[2, 2] *= vel_scale**2  # vx noise
        Q[3, 3] *= vel_scale**2  # vy noise
        
        # Also scale position noise slightly
        pos_scale = 0.5 + 0.5 * rpm_norm
        Q[0, 0] *= pos_scale**2
        Q[1, 1] *= pos_scale**2
        
        self.kf.processNoiseCov = Q
        
        # If hovering (low RPM, no trend), damp velocity
        if rpm_level < 500.0 and abs(rpm_deriv) < 200.0:
            state = self.kf.statePost
            state[2, 0] *= 0.7  # Reduce vx
            state[3, 0] *= 0.7  # Reduce vy
            self.kf.statePost = state

    def init(self, bbox: BoundingBox, t: float):
        """Initialize tracker with first bounding box."""
        cx = bbox.center_x
        cy = bbox.center_y
        w = bbox.width
        h = bbox.height
        
        self.kf.statePost = np.array(
            [[cx], [cy], [0.0], [0.0], [w], [h]], dtype=np.float32
        )
        self.last_t = t
        self.initialized = True
        self.recent_states = [(t, cx, cy)]

    def _update_transition(self, dt: float):
        """Update transition matrix based on time delta."""
        A = np.eye(6, dtype=np.float32)
        A[0, 2] = dt  # cx += vx * dt
        A[1, 3] = dt  # cy += vy * dt
        self.kf.transitionMatrix = A

    def predict(self, t: float):
        """Predict state at time t."""
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

        return pred

    def correct(self, bbox: BoundingBox):
        """Correct prediction with measurement."""
        if not self.initialized or bbox is None:
            return
        
        cx = bbox.center_x
        cy = bbox.center_y
        w = bbox.width
        h = bbox.height
        
        meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(meas)

    def get_velocity(self) -> tuple[float, float]:
        """Get current velocity estimate combining Kalman and recent history."""
        if not self.initialized:
            return 0.0, 0.0

        state = self.kf.statePost.flatten()
        vx_kf, vy_kf = state[2], state[3]

        # Blend Kalman velocity with recent velocity
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

        return vx, vy


def draw_future_gaussian_distribution(
    tracker: DroneKalmanTracker,
    frame: np.ndarray,
    max_horizon: float = 1.0,
    steps: int = 20,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draw a Gaussian-like probability distribution of where the drone can be
    from t=0s to t=max_horizon seconds into the future.
    
    Args:
        tracker: Kalman tracker with current state
        frame: Frame to overlay on
        max_horizon: Maximum time horizon in seconds
        steps: Number of time steps to visualize
        alpha: Transparency of overlay (0=transparent, 1=opaque)
    
    Returns:
        Frame with Gaussian distribution overlay
    """
    if not tracker.initialized:
        return frame

    H, W, _ = frame.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    state = tracker.kf.statePost.flatten()
    cx, cy, vx_kf, vy_kf, w, h = state

    # Get blended velocity
    vx, vy = tracker.get_velocity()

    times = np.linspace(0.0, max_horizon, steps)

    # Uncertainty grows over time
    base_sigma = max(w, h) * 0.25
    sigma_growth = max(w, h) * 0.5 / max_horizon if max_horizon > 0 else 0.0

    for t in times:
        # Future center position
        cx_t = cx + vx * t
        cy_t = cy + vy * t

        # Growing uncertainty
        sigma_t = base_sigma + sigma_growth * t
        sigma_x = sigma_t
        sigma_y = sigma_t

        # Region to update (±3σ)
        x_min = int(max(0, np.floor(cx_t - 3 * sigma_x)))
        x_max = int(min(W - 1, np.ceil(cx_t + 3 * sigma_x)))
        y_min = int(max(0, np.floor(cy_t - 3 * sigma_y)))
        y_max = int(min(H - 1, np.ceil(cy_t + 3 * sigma_y)))

        if x_max <= x_min or y_max <= y_min:
            continue

        xs = np.arange(x_min, x_max + 1, dtype=np.float32)
        ys = np.arange(y_min, y_max + 1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        # 2D Gaussian
        g = np.exp(
            -(((X - cx_t) ** 2) / (2 * sigma_x**2)
              + ((Y - cy_t) ** 2) / (2 * sigma_y**2))
        )

        # Weight decreases for far future (less certain)
        weight = 1.0 - (t / max_horizon) * 0.3 if max_horizon > 0 else 1.0
        g *= weight

        heatmap[y_min:y_max + 1, x_min:x_max + 1] += g.astype(np.float32)

    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap_norm = heatmap / heatmap.max()
    else:
        heatmap_norm = heatmap

    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # Apply colormap (JET: blue=low, red=high)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Blend with original frame
    overlay = cv2.addWeighted(color_map, alpha, frame, 1 - alpha, 0)

    # Only apply where heatmap is non-zero
    mask = heatmap_uint8 > 0
    out = frame.copy()
    out[mask] = overlay[mask]

    # Add text label
    cv2.putText(
        out,
        f"Predicted trajectory (0-{max_horizon:.1f}s)",
        (10, H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


# ==============================================================================
# EXISTING CODE FROM PREVIOUS SOLUTION
# ==============================================================================

@njit
def process_propeller_events_numba(
    x_coords,
    y_coords, 
    timestamps_us,
    polarities,
    last_on,
    last_off,
    width,
    height,
    min_period_us,
    max_period_us,
):
    """JIT-compiled event processing for RPM estimation."""
    n_events = len(x_coords)
    
    # Pre-allocate output arrays (max size = n_events)
    periods = np.zeros(n_events, dtype=np.int64)
    period_times = np.zeros(n_events, dtype=np.int64)
    count = 0
    
    for i in range(n_events):
        x = int(x_coords[i])
        y = int(y_coords[i])
        t_us = int(timestamps_us[i])
        pol = polarities[i]
        
        # Bounds check
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        
        prev_on = last_on[y, x]
        prev_off = last_off[y, x]
        
        if pol:  # ON event
            # Check for complete period: ON -> OFF -> ON
            if prev_off > prev_on and prev_on > 0:
                period_us = t_us - prev_on
                
                # Only accept reasonable periods
                if min_period_us < period_us < max_period_us:
                    periods[count] = period_us
                    period_times[count] = t_us
                    count += 1
            
            # Update last ON
            last_on[y, x] = t_us
        else:  # OFF event
            # Update last OFF
            last_off[y, x] = t_us
    
    # Return only the filled portion
    return periods[:count], period_times[:count], last_on, last_off


class PropellerRPMEstimator:
    """Estimate propeller RPM using period detection from ON->OFF->ON events.
    
    Optimized with Numba for real-time performance.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        num_bins: int = 256,
        bin_width_us: int = 100,
        window_duration_s: float = 0.1,
        num_blades: int = 2,
    ) -> None:
        """Initialize RPM estimator."""
        self.width = width
        self.height = height
        self.num_bins = num_bins
        self.bin_width_us = 100  # Must be 32 since we use >> 5 (divide by 2^5 = 32)
        self.window_duration_us = int(window_duration_s * 1e6)
        self.num_blades = num_blades
        
        # Per-pixel state: 2D arrays for fast access
        # Using int64 to store timestamps in microseconds
        self.last_on = np.zeros((height, width), dtype=np.int64)
        self.last_off = np.zeros((height, width), dtype=np.int64)
        
        # Histogram of periods (counts per bin)
        self.histogram = np.zeros(num_bins, dtype=np.int32)
        
        # FIFO queue of (timestamp_us, period_us) tuples
        self.period_queue: deque[tuple[int, int]] = deque()
        
        # Period constraints
        self.min_period_us = 1000
        self.max_period_us = 30000

        self.last_valid_rpm = 0.0
        self.ema_alpha = 0.3  # Exponential moving average factor
    
    def add_events(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        timestamps_us: np.ndarray,
        polarities: np.ndarray,
    ) -> None:
        """Process a batch of events and update period histogram."""
        if len(x_coords) == 0:
            return
        
        # Call JIT-compiled function
        periods, period_times, self.last_on, self.last_off = process_propeller_events_numba(
            x_coords.astype(np.int32),
            y_coords.astype(np.int32),
            timestamps_us.astype(np.int64),
            polarities,
            self.last_on,
            self.last_off,
            self.width,
            self.height,
            self.min_period_us,
            self.max_period_us,
        )
        
        # Add to histogram and queue
        for period_us, t_us in zip(periods, period_times):
            bin_idx = int(period_us) // 100  # Divide by 32 (≈27) using bit shift
            bin_idx = min(bin_idx, self.num_bins - 1)
            
            self.histogram[bin_idx] += 1
            self.period_queue.append((int(t_us), int(period_us)))
        
        # Remove old measurements outside time window (FIFO)
        if len(timestamps_us) > 0:
            current_time = int(timestamps_us[-1])
            cutoff_time = current_time - self.window_duration_us
            
            while self.period_queue and self.period_queue[0][0] < cutoff_time:
                _, old_period = self.period_queue.popleft()
                # Remove from histogram
                bin_idx = old_period // 100
                bin_idx = min(bin_idx, self.num_bins - 1)
                self.histogram[bin_idx] = max(0, self.histogram[bin_idx] - 1)
    
    def get_rpm(self) -> float:
        """Calculate RPM from histogram peak with smoothing."""
        total_measurements = np.sum(self.histogram)
        
        # Relaxed threshold: only need 3 measurements instead of 10
        if total_measurements < 3:
            # Return last valid value instead of 0
            return self.last_valid_rpm
        
        # Find dominant peak
        peak_bin = np.argmax(self.histogram)
        peak_count = self.histogram[peak_bin]
        
        # Much more relaxed significance threshold: 10% instead of 20%
        if peak_count < total_measurements * 0.10:
            # Still return last valid value
            return self.last_valid_rpm
        
        # Convert bin to period
        period_us = (peak_bin + 0.5) * self.bin_width_us
        period_s = period_us / 1e6
        
        # Convert to RPM
        rpm = 60.0 / (period_s * self.num_blades)
        
        # Apply exponential moving average for smoothing
        if self.last_valid_rpm > 0:
            # Smooth transition
            rpm = self.ema_alpha * rpm + (1 - self.ema_alpha) * self.last_valid_rpm
        
        # Update last valid RPM
        self.last_valid_rpm = rpm
        
        return rpm
    
    def reset(self) -> None:
        """Reset estimator state."""
        self.last_on.fill(0)
        self.last_off.fill(0)
        self.histogram.fill(0)
        self.period_queue.clear()


class DroneAnalyzer:
    """Analyze drone motion and detect behavioral patterns."""

    def __init__(
        self,
        time_window_s: float = 1.0,
        trajectory_history_s: float = 3.0,
        hover_threshold_px_s: float = 50.0,
        straight_direction_variance_deg: float = 15.0,
        aggressive_accel_threshold: float = 500.0,
    ) -> None:
        self.time_window_s = time_window_s
        self.trajectory_history_s = trajectory_history_s
        self.hover_threshold = hover_threshold_px_s
        self.straight_variance_deg = straight_direction_variance_deg
        self.aggressive_threshold = aggressive_accel_threshold

        self.position_history: deque[DronePosition] = deque()
        self.trajectory_history: deque[DronePosition] = deque()
        
        self.current_yaw: float = 0.0

    def add_position(self, timestamp_s: float, center_x: float, center_y: float) -> None:
        """Add new drone position to history."""
        position = DronePosition(timestamp_s, center_x, center_y)
        
        # Add to position history for analysis
        self.position_history.append(position)

        # Remove old positions outside analysis time window
        cutoff_time = timestamp_s - self.time_window_s
        while (
            self.position_history
            and self.position_history[0].timestamp_s < cutoff_time
        ):
            self.position_history.popleft()

        # Add to trajectory history for visualization
        self.trajectory_history.append(position)

        # Remove old positions outside trajectory time window
        trajectory_cutoff = timestamp_s - self.trajectory_history_s
        while (
            self.trajectory_history
            and self.trajectory_history[0].timestamp_s < trajectory_cutoff
        ):
            self.trajectory_history.popleft()

    def calculate_yaw(self, propeller_centers: list[tuple[float, float]]) -> float:
        """Calculate drone yaw angle from propeller positions (X-configuration).
        
        Args:
            propeller_centers: List of (x, y) coordinates for 4 propeller centers
            
        Returns:
            Yaw angle in degrees (0° = right, 90° = up, counterclockwise positive)
        """
        if len(propeller_centers) != 4:
            return self.current_yaw
        
        # Calculate centroid (drone center)
        centers_array = np.array(propeller_centers)
        centroid = centers_array.mean(axis=0)
        
        # Calculate vectors from centroid to each propeller
        vectors = centers_array - centroid
        
        # For X-configuration, find the principal axis using PCA
        # The two principal directions should be roughly 45° and 135° (or -45° and 45°)
        cov_matrix = np.cov(vectors.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Get the principal axis (largest eigenvalue)
        principal_idx = np.argmax(eigenvalues)
        principal_axis = eigenvectors[:, principal_idx]
        
        # Calculate angle of principal axis (this gives us one arm of the X)
        angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
        angle_deg = np.degrees(angle_rad)
        
        # For X-configuration, the yaw is the angle of the forward axis
        # We need to determine which quadrant is "forward"
        # Assuming forward is the direction that makes sense from propeller arrangement
        
        # Normalize to 0-360 range
        yaw = angle_deg % 360
        
        self.current_yaw = yaw
        return yaw

    def get_trajectory(self) -> list[DronePosition]:
        """Get current trajectory history for visualization."""
        return list(self.trajectory_history)

    def analyze(self) -> MotionAnalysis:
        """Analyze recent motion and return analysis results."""
        positions = list(self.position_history)

        if len(positions) < 2:
            return MotionAnalysis()

        # Calculate velocity from most recent two positions
        pos_current = positions[-1]
        pos_prev = positions[-2]
        dt = pos_current.timestamp_s - pos_prev.timestamp_s

        if dt < 1e-6:
            return MotionAnalysis()

        velocity_x = (pos_current.center_x - pos_prev.center_x) / dt
        velocity_y = (pos_current.center_y - pos_prev.center_y) / dt
        speed = np.sqrt(velocity_x**2 + velocity_y**2)

        # Calculate direction
        direction_deg = np.degrees(np.arctan2(velocity_y, velocity_x))

        # Calculate acceleration (if we have enough history)
        acceleration = 0.0
        if len(positions) >= 3:
            pos_prev2 = positions[-3]
            dt2 = pos_prev.timestamp_s - pos_prev2.timestamp_s

            if dt2 > 1e-6:
                vel_x_prev = (pos_prev.center_x - pos_prev2.center_x) / dt2
                vel_y_prev = (pos_prev.center_y - pos_prev2.center_y) / dt2

                accel_x = (velocity_x - vel_x_prev) / dt
                accel_y = (velocity_y - vel_y_prev) / dt
                acceleration = np.sqrt(accel_x**2 + accel_y**2)

        # Classify motion pattern
        pattern = self._classify_pattern(
            speed, acceleration, positions
        )

        return MotionAnalysis(
            speed_px_s=speed,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            acceleration=acceleration,
            direction_deg=direction_deg,
            pattern=pattern,
        )

    def _classify_pattern(
        self,
        speed: float,
        acceleration: float,
        positions: list[DronePosition],
    ) -> str:
        """Classify motion pattern based on kinematics."""
        # Hovering: very low speed
        if speed < self.hover_threshold:
            return "HOVERING"

        # Aggressive maneuver: high acceleration
        if acceleration > self.aggressive_threshold:
            return "AGGRESSIVE"

        # Straight line motion: consistent direction
        if len(positions) >= 5:
            # Calculate direction variance
            directions = []
            for i in range(len(positions) - 1):
                dx = positions[i + 1].center_x - positions[i].center_x
                dy = positions[i + 1].center_y - positions[i].center_y
                direction = np.degrees(np.arctan2(dy, dx))
                directions.append(direction)

            if len(directions) >= 2:
                direction_variance = np.std(directions)
                if direction_variance < self.straight_variance_deg:
                    return "STRAIGHT"

        # Default: general cruising
        return "CRUISING"


def get_window(
    event_words: np.ndarray, order: np.ndarray, start: int, stop: int
) -> np.ndarray:
    """Get a time window of events."""
    event_indexes = order[start:stop]
    return event_words[event_indexes]


def get_frame(
    window: np.ndarray, width: int = 1280, height: int = 720
) -> np.ndarray:
    """Convert event window to BGR frame."""
    frame = np.full((height, width, 3), (0, 0, 0), dtype=np.uint8)

    if len(window) == 0:
        return frame

    words = window.astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities = ((words >> 28) & 0xF) > 0

    # Bounds checking
    valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    polarities = polarities[valid_mask]

    # Draw ON events (white) and OFF events (black)
    if len(x_coords) > 0:
        on_mask = polarities
        off_mask = ~polarities
        
        if np.any(on_mask):
            frame[y_coords[on_mask], x_coords[on_mask]] = (255, 255, 255)
        if np.any(off_mask):
            frame[y_coords[off_mask], x_coords[off_mask]] = (0, 0, 0)

    return frame


def denoise_frame(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Simple spatial denoising on the event frame using median blur.
    
    Args:
        frame: BGR frame to denoise
        kernel_size: Median blur kernel size (3 or 5 recommended)
    
    Returns:
        Denoised BGR frame
    """
    return cv2.medianBlur(frame, kernel_size)


def draw_bounding_box(
    frame: np.ndarray,
    bbox: BoundingBox,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw bounding box on frame."""
    x1, y1 = int(bbox.x1), int(bbox.y1)
    x2, y2 = int(bbox.x2), int(bbox.y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw center point
    center_x, center_y = int(bbox.center_x), int(bbox.center_y)
    cv2.circle(frame, (center_x, center_y), 4, color, -1)


def draw_analysis_info(
    frame: np.ndarray,
    analysis: MotionAnalysis,
    bbox: BoundingBox | None,
    position: tuple[int, int] = (10, 680),
    *,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw motion analysis information on frame (bottom-left)."""
    x, y = position
    line_height = 20

    # info_lines = [
    #     f"Speed: {analysis.speed_px_s:.1f} px/s",
    #     f"Direction: {analysis.direction_deg:.1f} deg",
    #     f"Acceleration: {analysis.acceleration:.1f} px/s^2",
    #     f"Pattern: {analysis.pattern}",
    # ]

    # for i, line in enumerate(info_lines):
    #     cv2.putText(
    #         frame,
    #         line,
    #         (x, y + i * line_height),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         color,
    #         1,
    #         cv2.LINE_AA,
    #     )


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    denoise_enabled: bool = False,
    denoise_kernel: int = 3,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw playback HUD (top-left)."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row = (
            f"speed={pacer.speed:.2f}x  "
            f"drops/ms={pacer.instantaneous_drop_rate:.2f}  "
            f"avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row = f"speed={pacer.speed:.2f}x  force_speed=False"

    second_row = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    cv2.putText(
        frame,
        first_row,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        second_row,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # Show denoising status
    if denoise_enabled:
        denoise_text = f"Denoising: ON (kernel={denoise_kernel})"
        denoise_color = (0, 255, 0)  # Green when enabled
    else:
        denoise_text = "Denoising: OFF"
        denoise_color = (0, 0, 255)  # Red when disabled

    cv2.putText(
        frame,
        denoise_text,
        (8, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        denoise_color,
        1,
        cv2.LINE_AA,
    )


def draw_trajectory(
    frame: np.ndarray,
    trajectory: list[DronePosition],
    color: tuple[int, int, int] = (0, 255, 255),  # Cyan
) -> None:
    """Draw drone trajectory trail on frame."""
    if len(trajectory) < 2:
        return

    # Draw lines connecting trajectory points
    for i in range(len(trajectory) - 1):
        pt1 = (int(trajectory[i].center_x), int(trajectory[i].center_y))
        pt2 = (int(trajectory[i + 1].center_x), int(trajectory[i + 1].center_y))
        
        # Fade older points (gradient from darker to brighter)
        alpha = (i + 1) / len(trajectory)
        point_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
        
        thickness = max(1, int(1 + alpha * 2))
        cv2.line(frame, pt1, pt2, point_color, thickness, cv2.LINE_AA)

    # Draw small circles at each point
    for i, pos in enumerate(trajectory):
        pt = (int(pos.center_x), int(pos.center_y))
        alpha = (i + 1) / len(trajectory)
        point_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
        radius = max(2, int(2 + alpha * 2))
        cv2.circle(frame, pt, radius, point_color, -1, cv2.LINE_AA)


def create_drone_subplot(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
    bbox: BoundingBox,
    subplot_size: tuple[int, int] = (213, 240),
    denoise: bool = False,
    denoise_kernel: int = 3,
    padding: int = 20,  # ADD THIS
) -> np.ndarray:
    """Create a cropped view of drone events within bounding box.
    
    Args:
        event_words: Raw event data
        time_order: Time-sorted indices
        win_start: Window start index
        win_stop: Window stop index
        bbox: Bounding box of drone
        subplot_size: Output size for subplot
        denoise: Whether to apply denoising
        denoise_kernel: Kernel size for median blur denoising
        padding: Padding in pixels to add around bbox (default: 20)  # ADD THIS
    
    Returns:
        Cropped and optionally denoised subplot
    """
    # Get all events in window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities = ((words >> 28) & 0xF) > 0

    # Filter events within bounding box (with padding)  
    x1, y1 = max(0, int(bbox.x1) - padding), max(0, int(bbox.y1) - padding)  
    x2, y2 = min(1279, int(bbox.x2) + padding), min(719, int(bbox.y2) + padding)  
    
    mask = (
        (x_coords >= x1)
        & (x_coords <= x2)
        & (y_coords >= y1)
        & (y_coords <= y2)
    )

    x_cropped = x_coords[mask] - x1
    y_cropped = y_coords[mask] - y1
    pol_cropped = polarities[mask]

    # Create cropped frame (add 1 to include boundary pixels)
    crop_width = x2 - x1 + 1
    crop_height = y2 - y1 + 1

    if crop_width <= 0 or crop_height <= 0:
        # Invalid bbox, return black image
        return np.zeros((subplot_size[1], subplot_size[0], 3), dtype=np.uint8)

    cropped_frame = np.full(
        (crop_height, crop_width, 3), (0, 0, 0), dtype=np.uint8
    )

    # Draw events with bounds checking
    if len(x_cropped) > 0:
        # Filter out any coordinates that are still out of bounds (safety check)
        valid_mask_on = pol_cropped & (x_cropped < crop_width) & (y_cropped < crop_height)
        valid_mask_off = (~pol_cropped) & (x_cropped < crop_width) & (y_cropped < crop_height)
        
        if np.any(valid_mask_on):
            cropped_frame[y_cropped[valid_mask_on], x_cropped[valid_mask_on]] = (255, 255, 255)
        if np.any(valid_mask_off):
            cropped_frame[y_cropped[valid_mask_off], x_cropped[valid_mask_off]] = (0, 0, 0)

    # Apply denoising if requested
    if denoise:
        cropped_frame = denoise_frame(cropped_frame, kernel_size=denoise_kernel)

    # Resize to subplot size
    subplot = cv2.resize(cropped_frame, subplot_size, interpolation=cv2.INTER_NEAREST)

    return subplot


def classify_drone_events(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
    bbox: BoundingBox,
    timestamps_sorted: np.ndarray,
    frequency_threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify events within bbox into propeller and skeleton based on frequency.
    
    Propeller pixels have high event frequency due to fast rotation.
    Skeleton pixels have lower frequency (structural edges).
    
    Args:
        event_words: Raw event data
        time_order: Time-sorted indices
        win_start: Window start index
        win_stop: Window stop index
        bbox: Bounding box of drone
        timestamps_sorted: Time-sorted timestamps array
        frequency_threshold: Percentile threshold to separate propeller from skeleton (0-1)
    
    Returns:
        (x_propeller, y_propeller, t_propeller, x_skeleton, y_skeleton, t_skeleton)
    """
    # Get all events in window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    
    # Get timestamps for these events
    timestamps = timestamps_sorted[win_start:win_stop]
    
    # Filter events within bounding box
    x1, y1 = int(bbox.x1), int(bbox.y1)
    x2, y2 = int(bbox.x2), int(bbox.y2)
    
    mask = (
        (x_coords >= x1)
        & (x_coords <= x2)
        & (y_coords >= y1)
        & (y_coords <= y2)
    )
    
    x_bbox = x_coords[mask]
    y_bbox = y_coords[mask]
    t_bbox = timestamps[mask]
    
    if len(x_bbox) == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int64),
        )
    
    # Create frequency map (count events per pixel within bbox)
    crop_width = x2 - x1 + 1
    crop_height = y2 - y1 + 1
    frequency_map = np.zeros((crop_height, crop_width), dtype=np.float32)
    
    # Convert to relative coordinates
    x_rel = x_bbox - x1
    y_rel = y_bbox - y1
    
    # Count events per pixel
    for x, y in zip(x_rel, y_rel):
        if 0 <= x < crop_width and 0 <= y < crop_height:
            frequency_map[y, x] += 1
    
    # Normalize frequency map
    if frequency_map.max() > 0:
        frequency_map = frequency_map / frequency_map.max()
    
    # Classify pixels based on frequency
    # High frequency = propeller (fast rotation creates many events)
    # Low frequency = skeleton (structural edges, slower changes)
    threshold_value = np.percentile(frequency_map[frequency_map > 0], frequency_threshold * 100)
    
    # Separate into propeller and skeleton
    propeller_mask = frequency_map[y_rel, x_rel] > threshold_value
    skeleton_mask = ~propeller_mask
    
    x_propeller = x_bbox[propeller_mask]
    y_propeller = y_bbox[propeller_mask]
    t_propeller = t_bbox[propeller_mask]
    
    x_skeleton = x_bbox[skeleton_mask]
    y_skeleton = y_bbox[skeleton_mask]
    t_skeleton = t_bbox[skeleton_mask]
    
    return x_propeller, y_propeller, t_propeller, x_skeleton, y_skeleton, t_skeleton


def cluster_propeller_centers(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_clusters: int = 4,
    min_events_for_clustering: int = 20,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Cluster propeller events to find individual propeller centers.
    
    Args:
        x_coords: X coordinates of propeller events
        y_coords: Y coordinates of propeller events
        n_clusters: Number of propellers/clusters to find
        min_events_for_clustering: Minimum events needed for clustering
    
    Returns:
        (propeller_centers, event_counts)
        - propeller_centers: List of (x, y) tuples for each propeller center
        - event_counts: List of event counts per cluster
    """
    if len(x_coords) < min_events_for_clustering:
        return [], []
    
    # Prepare data for K-means
    points = np.column_stack([x_coords, y_coords]).astype(np.float32)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        points,
        n_clusters,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    
    # Count events per cluster
    event_counts = []
    for i in range(n_clusters):
        count = np.sum(labels.flatten() == i)
        event_counts.append(int(count))
    
    # Convert centers to list of tuples
    propeller_centers = [(float(c[0]), float(c[1])) for c in centers]
    
    return propeller_centers, event_counts


def assign_events_to_propellers(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    timestamps: np.ndarray,
    polarities: np.ndarray,
    propeller_centers: list[tuple[float, float]],
    max_distance: float = 50.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Assign each propeller event to its nearest propeller center.
    
    Args:
        x_coords: X coordinates of propeller events
        y_coords: Y coordinates of propeller events
        timestamps: Event timestamps
        polarities: Event polarities
        propeller_centers: List of (x, y) propeller center coordinates
        max_distance: Maximum distance to assign event to propeller
    
    Returns:
        List of 4 tuples, each containing (x, y, t, pol) arrays for one propeller
    """
    n_propellers = len(propeller_centers)
    if n_propellers == 0:
        return [(np.array([]), np.array([]), np.array([]), np.array([]))] * 4
    
    # Initialize lists for each propeller
    propeller_events = [[] for _ in range(n_propellers)]
    
    for i in range(len(x_coords)):
        x, y, t, pol = x_coords[i], y_coords[i], timestamps[i], polarities[i]
        
        # Find closest propeller center
        min_dist = float('inf')
        closest_idx = -1
        
        for j, (cx, cy) in enumerate(propeller_centers):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
        
        # Assign to closest propeller if within max_distance
        if min_dist < max_distance and closest_idx >= 0:
            propeller_events[closest_idx].append((x, y, t, pol))
    
    # Convert to numpy arrays
    result = []
    for events in propeller_events:
        if len(events) > 0:
            events_array = np.array(events)
            result.append((
                events_array[:, 0].astype(np.int32),
                events_array[:, 1].astype(np.int32),
                events_array[:, 2].astype(np.int64),
                events_array[:, 3].astype(bool),
            ))
        else:
            result.append((np.array([]), np.array([]), np.array([]), np.array([])))
    
    # Ensure we always return exactly 4 propellers (pad if needed)
    while len(result) < 4:
        result.append((np.array([]), np.array([]), np.array([]), np.array([])))
    
    return result[:4]

def create_classified_subplot(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    bbox: BoundingBox,
    subplot_size: tuple[int, int] = (213, 240),
    event_color: tuple[int, int, int] = (255, 255, 255),
    base_color: tuple[int, int, int] = (0, 0, 0),
    denoise: bool = False,
    denoise_kernel: int = 3,
    show_propeller_centers: bool = False,
    n_propellers: int = 4,
    padding: int = 20,  # ADD THIS
) -> np.ndarray:
    """Create subplot for classified events (propeller or skeleton).
    
    Args:
        x_coords: X coordinates of events
        y_coords: Y coordinates of events
        bbox: Bounding box for cropping
        subplot_size: Output size
        event_color: Color for events
        base_color: Background color
        denoise: Whether to apply denoising
        denoise_kernel: Kernel size for denoising
        show_propeller_centers: Whether to draw propeller center dots
        n_propellers: Number of propellers for clustering
        padding: Padding in pixels to add around bbox (default: 20)  # ADD THIS
    
    Returns:
        Subplot image
    """
    x1, y1 = max(0, int(bbox.x1) - padding), max(0, int(bbox.y1) - padding) 
    x2, y2 = min(1279, int(bbox.x2) + padding), min(719, int(bbox.y2) + padding) 
    
    crop_width = x2 - x1 + 1
    crop_height = y2 - y1 + 1
    
    if crop_width <= 0 or crop_height <= 0:
        return np.zeros((subplot_size[1], subplot_size[0], 3), dtype=np.uint8)
    
    # Create frame
    frame = np.full((crop_height, crop_width, 3), base_color, dtype=np.uint8)
    
    # Convert to relative coordinates and draw events
    if len(x_coords) > 0:
        x_rel = x_coords - x1
        y_rel = y_coords - y1
        
        # Bounds checking
        valid_mask = (
            (x_rel >= 0) & (x_rel < crop_width) &
            (y_rel >= 0) & (y_rel < crop_height)
        )
        
        x_rel = x_rel[valid_mask]
        y_rel = y_rel[valid_mask]
        
        if len(x_rel) > 0:
            frame[y_rel, x_rel] = event_color
    
    # Optionally show propeller centers
    if show_propeller_centers and len(x_coords) > 0:
        propeller_centers, _ = cluster_propeller_centers(
            x_coords, y_coords, n_clusters=n_propellers
        )
        
        for cx, cy in propeller_centers:
            # Convert to relative coordinates
            cx_rel = int(cx - x1)
            cy_rel = int(cy - y1)
            
            # Draw if within bounds
            if 0 <= cx_rel < crop_width and 0 <= cy_rel < crop_height:
                cv2.circle(frame, (cx_rel, cy_rel), 3, (0, 255, 0), -1)
                cv2.circle(frame, (cx_rel, cy_rel), 4, (255, 255, 0), 1)
    
    # Apply denoising if requested
    if denoise:
        frame = denoise_frame(frame, kernel_size=denoise_kernel)
    
    # Resize to subplot size
    subplot = cv2.resize(frame, subplot_size, interpolation=cv2.INTER_NEAREST)
    
    return subplot


def create_yaw_subplot(
    propeller_centers: list[tuple[float, float]],
    yaw_deg: float,
    subplot_size: tuple[int, int] = (213, 240),
) -> np.ndarray:
    """Create a subplot showing drone orientation and yaw angle.
    
    Args:
        propeller_centers: List of (x, y) coordinates for 4 propeller centers
        yaw_deg: Yaw angle in degrees
        subplot_size: Output size
        
    Returns:
        Subplot image with yaw visualization
    """
    frame = np.zeros((subplot_size[1], subplot_size[0], 3), dtype=np.uint8)
    
    # Draw compass rose
    center_x = subplot_size[0] // 2
    center_y = subplot_size[1] // 2
    radius = min(subplot_size) // 2 - 20
    
    # Draw outer circle
    cv2.circle(frame, (center_x, center_y), radius, (80, 80, 80), 1)
    
    # Draw cardinal directions
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # North (up)
    cv2.putText(frame, "|", (center_x - 7, center_y - radius - 5), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    # East (right)
    cv2.putText(frame, "_", (center_x + radius + 5, center_y + 5), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    # South (down)
    cv2.putText(frame, "|", (center_x - 7, center_y + radius + 15), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    # West (left)
    cv2.putText(frame, "_", (center_x - radius - 15, center_y + 5), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    
    # Draw drone in X-configuration if we have propeller centers
    if len(propeller_centers) == 4:
        centers_array = np.array(propeller_centers)
        centroid = centers_array.mean(axis=0)
        
        # Calculate maximum distance from centroid to normalize
        max_dist = 0
        for cx, cy in propeller_centers:
            dist = np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2)
            max_dist = max(max_dist, dist)
        
        if max_dist > 0:
            scale = (radius * 0.6) / max_dist
            
            # Draw propellers as dots in X-configuration
            for i, (cx, cy) in enumerate(propeller_centers):
                dx = (cx - centroid[0]) * scale
                dy = (cy - centroid[1]) * scale
                
                prop_x = int(center_x + dx)
                prop_y = int(center_y + dy)
                
                # Draw propeller dot
                cv2.circle(frame, (prop_x, prop_y), 6, (255, 255, 0), -1)  # Yellow
                cv2.circle(frame, (prop_x, prop_y), 6, (0, 255, 255), 1)  # Cyan outline
            
            # Draw X-configuration lines
            if len(propeller_centers) == 4:
                # Find pairs (opposite propellers) - sort by distance to form X
                distances = []
                for i in range(4):
                    for j in range(i+1, 4):
                        dist = np.sqrt((centers_array[i][0] - centers_array[j][0])**2 + 
                                     (centers_array[i][1] - centers_array[j][1])**2)
                        distances.append((dist, i, j))
                
                # The two longest distances are the diagonals of the X
                distances.sort(reverse=True)
                
                for k in range(2):  # Draw two diagonals
                    _, i, j = distances[k]
                    dx1 = (centers_array[i][0] - centroid[0]) * scale
                    dy1 = (centers_array[i][1] - centroid[1]) * scale
                    dx2 = (centers_array[j][0] - centroid[0]) * scale
                    dy2 = (centers_array[j][1] - centroid[1]) * scale
                    
                    pt1 = (int(center_x + dx1), int(center_y + dy1))
                    pt2 = (int(center_x + dx2), int(center_y + dy2))
                    
                    cv2.line(frame, pt1, pt2, (100, 100, 100), 1, cv2.LINE_AA)
    
    # Draw yaw arrow
    arrow_length = int(radius * 0.7)
    if yaw_deg > 0:
        yaw_rad = np.radians(yaw_deg)
        
        # In image coordinates: 0° should point right, 90° should point down
        # But we want 0° to point up (North), so rotate by 90°
        adjusted_angle = yaw_rad - np.pi/2
        
        arrow_end_x = int(center_x + arrow_length * np.cos(adjusted_angle))
        arrow_end_y = int(center_y + arrow_length * np.sin(adjusted_angle))
        
        cv2.arrowedLine(
            frame,
            (center_x, center_y),
            (arrow_end_x, arrow_end_y),
            (0, 0, 255),  # Red arrow
            2,
            cv2.LINE_AA,
            tipLength=0.3
        )
    
    # Draw yaw value
    yaw_text = f"Yaw: {yaw_deg:.1f} deg"
    text_size = cv2.getTextSize(yaw_text, font, 0.5, 1)[0]
    text_x = (subplot_size[0] - text_size[0]) // 2
    text_y = subplot_size[1] - 10
    
    cv2.putText(
        frame,
        yaw_text,
        (text_x, text_y),
        font,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )
    
    return frame


def create_rpm_subplot(
    rpm: float,
    subplot_size: tuple[int, int] = (213, 240),
) -> np.ndarray:
    """Create a subplot showing overall propeller RPM.
    
    Args:
        rpm: Estimated RPM value
        subplot_size: Output size
        
    Returns:
        Subplot image with RPM visualization
    """
    frame = np.zeros((subplot_size[1], subplot_size[0], 3), dtype=np.uint8)
    
    # Title
    cv2.putText(
        frame,
        "PROPELLER RPM",
        (25, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    
    # Draw large RPM value in center
    if rpm > 0:
        rpm_text = f"{int(rpm)}"
        
        # Calculate font size to fit well
        font_scale = 1.5
        thickness = 3
        
        # Get text size
        text_size = cv2.getTextSize(
            rpm_text, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            thickness
        )[0]
        
        # Center the text
        text_x = (subplot_size[0] - text_size[0]) // 2
        text_y = (subplot_size[1] + text_size[1]) // 2
        
        # Color based on RPM level
        if rpm < 2000:
            color = (0, 100, 255)  # Orange (low)
        elif rpm < 5000:
            color = (0, 255, 255)  # Yellow (medium)
        else:
            color = (0, 255, 0)  # Green (high)
        
        # Draw RPM value
        cv2.putText(
            frame,
            rpm_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        # Draw "RPM" label below
        label = "RPM"
        label_size = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            1
        )[0]
        label_x = (subplot_size[0] - label_size[0]) // 2
        label_y = text_y + 30
        
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        # Draw circular gauge as background
        center_x = subplot_size[0] // 2
        center_y = subplot_size[1] // 2 - 10
        gauge_radius = 85
        
        # Background circle
        cv2.circle(frame, (center_x, center_y), gauge_radius, (50, 50, 50), 2)
        
        # Draw arc based on RPM (0-10000 RPM mapped to 0-270 degrees)
        max_rpm = 10000
        rpm_angle = int((min(rpm, max_rpm) / max_rpm) * 270)
        start_angle = 135  # Start from bottom-left
        end_angle = start_angle + rpm_angle
        
        cv2.ellipse(
            frame,
            (center_x, center_y),
            (gauge_radius, gauge_radius),
            0,
            start_angle,
            end_angle,
            color,
            3
        )
    else:
        # No data
        cv2.putText(
            frame,
            "NO DATA",
            (55, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 100),
            1,
            cv2.LINE_AA
        )
    
    # Draw measurement note at bottom
    note = "Period-based"
    note_size = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    note_x = (subplot_size[0] - note_size[0]) // 2
    
    cv2.putText(
        frame,
        note,
        (note_x, subplot_size[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (150, 150, 150),
        1,
        cv2.LINE_AA
    )
    
    return frame


def create_per_propeller_rpm_subplot(
    rpm_values: list[float],
    propeller_centers: list[tuple[float, float]],
    subplot_size: tuple[int, int] = (213, 240),
) -> np.ndarray:
    """Create a subplot showing RPM for each individual propeller.
    
    Args:
        rpm_values: List of 4 RPM values (one per propeller)
        propeller_centers: List of (x, y) propeller center coordinates
        subplot_size: Output size
        
    Returns:
        Subplot image with per-propeller RPM bars
    """
    frame = np.zeros((subplot_size[1], subplot_size[0], 3), dtype=np.uint8)
    
    # Title
    cv2.putText(
        frame,
        "PER-PROPELLER RPM",
        (15, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    
    if len(rpm_values) != 4:
        # No data
        cv2.putText(
            frame,
            "NO DATA",
            (55, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            1,
            cv2.LINE_AA
        )
        return frame
    
    # Identify propeller positions and sort
    if len(propeller_centers) == 4:
        centers_with_rpm = list(zip(propeller_centers, rpm_values))
        # Sort by y (top to bottom), then by x (left to right)
        centers_with_rpm.sort(key=lambda item: (item[0][1], item[0][0]))
        
        # Get top 2 and bottom 2
        top_2 = sorted(centers_with_rpm[:2], key=lambda item: item[0][0])
        bottom_2 = sorted(centers_with_rpm[2:], key=lambda item: item[0][0])
        
        # Assign labels: FL, FR, BL, BR
        labels = ["FL", "FR", "BL", "BR"]
        sorted_rpm = [top_2[0][1], top_2[1][1], bottom_2[0][1], bottom_2[1][1]]
    else:
        labels = ["P1", "P2", "P3", "P4"]
        sorted_rpm = rpm_values
    
    # Draw RPM bars
    bar_width = 35
    bar_spacing = 10
    max_bar_height = 140
    start_x = 20
    start_y = subplot_size[1] - 30
    
    max_rpm = max(max(sorted_rpm) if any(r > 0 for r in sorted_rpm) else 1000, 1000)
    
    for i, (label, rpm) in enumerate(zip(labels, sorted_rpm)):
        x = start_x + i * (bar_width + bar_spacing)
        
        # Calculate bar height
        bar_height = int((rpm / max_rpm) * max_bar_height) if rpm > 0 else 0
        bar_height = min(bar_height, max_bar_height)
        
        # Color based on RPM level
        if rpm < 2000:
            color = (0, 100, 255)  # Orange (low)
        elif rpm < 5000:
            color = (0, 255, 255)  # Yellow (medium)
        else:
            color = (0, 255, 0)  # Green (high)
        
        # Draw bar
        if bar_height > 0:
            cv2.rectangle(
                frame,
                (x, start_y - bar_height),
                (x + bar_width, start_y),
                color,
                -1
            )
            
            # Draw outline
            cv2.rectangle(
                frame,
                (x, start_y - bar_height),
                (x + bar_width, start_y),
                (255, 255, 255),
                1
            )
        
        # Draw label
        cv2.putText(
            frame,
            label,
            (x + 8, start_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        # Draw RPM value
        if rpm > 0:
            rpm_text = f"{int(rpm)}"
            text_size = cv2.getTextSize(rpm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            text_x = x + (bar_width - text_size[0]) // 2
            text_y = start_y - bar_height - 5
            
            cv2.putText(
                frame,
                rpm_text,
                (text_x, max(text_y, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    return frame


def overlay_multiple_subplots(
    frame: np.ndarray,
    drone_subplot: np.ndarray,
    propeller_subplot: np.ndarray,
    skeleton_subplot: np.ndarray,
    rpm_subplot: np.ndarray,
    per_propeller_rpm_subplot: np.ndarray,
    yaw_subplot: np.ndarray,
    bottom_start_y: int = 720,
    title_height: int = 40,
) -> None:
    """Overlay six subplots in a 1x6 grid at the bottom of the frame.
    
    Args:
        frame: Main canvas (1280 x 1000)
        drone_subplot, propeller_subplot, skeleton_subplot: Event visualizations
        rpm_subplot, per_propeller_rpm_subplot, yaw_subplot: Analysis visualizations
        bottom_start_y: Y position where bottom section starts (default 720)
        title_height: Height reserved for subplot titles (default 40)
    """
    # Subplot dimensions
    subplot_h, subplot_w = drone_subplot.shape[:2]
    
    # Titles and colors for each subplot (in order)
    subplots = [
        (drone_subplot, "DRONE VIEW", (0, 255, 0)),
        (propeller_subplot, "PROPELLERS", (0, 0, 255)),
        (skeleton_subplot, "SKELETON", (255, 0, 255)),
        (rpm_subplot, "TOTAL RPM", (255, 255, 0)),
        (per_propeller_rpm_subplot, "INDIVIDUAL RPM", (255, 150, 100)),
        (yaw_subplot, "YAW ORIENTATION", (255, 100, 0)),
    ]
    
    # Position each subplot in the 1x6 grid
    for i, (subplot, title, color) in enumerate(subplots):
        # Calculate x position for this subplot
        x = i * subplot_w
        y = bottom_start_y + title_height
        
        # Draw title above subplot
        cv2.putText(
            frame,
            title,
            (x + 5, bottom_start_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        
        # Draw border around subplot
        border_thickness = 2
        cv2.rectangle(
            frame,
            (x, y),
            (x + subplot_w, y + subplot_h),
            color,
            border_thickness,
        )
        
        # Place subplot on canvas
        frame[y:y + subplot_h, x:x + subplot_w] = subplot


class DroneDataLogger:
    """Log drone analysis data to CSV file."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.file = output_path.open("w", newline="")
        self.writer = csv.writer(self.file)
        
        # Write header
        self.writer.writerow([
            "timestamp_s",
            "center_x",
            "center_y",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_width",
            "bbox_height",
            "speed_px_s",
            "velocity_x",
            "velocity_y",
            "acceleration",
            "direction_deg",
            "yaw_deg",
            "rpm",
            "pattern",
            "drone_detected",
        ])
        
        print(f"Logging data to: {output_path}")

    def log(
        self,
        timestamp_s: float,
        bbox: BoundingBox | None,
        analysis: MotionAnalysis,
    ) -> None:
        """Log current frame data to CSV."""
        if bbox is None:
            # Drone not detected
            self.writer.writerow([
                timestamp_s,
                "", "", "", "", "", "",  # Empty bbox data
                "", "",  # Empty width/height
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Zero analysis values
                "NO_DRONE",
                False,
            ])
        else:
            self.writer.writerow([
                timestamp_s,
                bbox.center_x,
                bbox.center_y,
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
                bbox.width,
                bbox.height,
                analysis.speed_px_s,
                analysis.velocity_x,
                analysis.velocity_y,
                analysis.acceleration,
                analysis.direction_deg,
                analysis.yaw_deg,
                analysis.rpm,
                analysis.pattern,
                True,
            ])

    def close(self) -> None:
        """Close the CSV file."""
        self.file.close()
        print(f"Logged data saved to: {self.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze drone motion and behavioral patterns from event camera data with trajectory prediction."
    )
    parser.add_argument(
        "dat",
        type=Path,
        help="Path to .dat file with event camera data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for CSV log (default: <dat_filename>_analysis.csv)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=10,
        help="Window length in milliseconds (default: 10ms)",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force playback speed by dropping events if needed",
    )
    parser.add_argument(
        "--analysis-window",
        type=float,
        default=1.0,
        help="Time window for motion analysis in seconds (default: 1.0s)",
    )
    parser.add_argument(
        "--trajectory-history",
        type=float,
        default=3.0,
        help="Duration of trajectory trail in seconds (default: 3.0s)",
    )
    parser.add_argument(
        "--hover-threshold",
        type=float,
        default=50.0,
        help="Speed threshold for hover detection in px/s (default: 50.0)",
    )
    parser.add_argument(
        "--straight-variance",
        type=float,
        default=15.0,
        help="Direction variance threshold for straight motion in degrees (default: 15.0)",
    )
    parser.add_argument(
        "--aggressive-threshold",
        type=float,
        default=500.0,
        help="Acceleration threshold for aggressive maneuvers in px/s^2 (default: 500.0)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable spatial denoising using median blur",
    )
    parser.add_argument(
        "--denoise-kernel",
        type=int,
        default=3,
        choices=[3, 5],
        help="Kernel size for median blur denoising (default: 3)",
    )
    parser.add_argument(
        "--propeller-threshold",
        type=float,
        default=0.7,
        help="Frequency percentile threshold for propeller classification (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--num-propellers",
        type=int,
        default=4,
        help="Number of propellers to detect (default: 4)",
    )
    parser.add_argument(
        "--num-blades",
        type=int,
        default=2,
        help="Number of blades per propeller for RPM calculation (default: 2)",
    )
    parser.add_argument(
        "--prediction-history",
        type=float,
        default=0.5,
        help="Time window for velocity estimation in trajectory prediction (default: 0.5s)",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=float,
        default=1.3,
        help="Future time horizon for trajectory prediction in seconds (default: 1.0s)",
    )
    parser.add_argument(
        "--prediction-alpha",
        type=float,
        default=0.5,
        help="Transparency of trajectory prediction overlay (0-1, default: 0.5)",
    )
    
    # Drone detection parameters (from friend's code)
    parser.add_argument(
        "--window-ms",
        type=float,
        default=10.0,
        help="Window duration in ms for drone detection (default: 10.0)",
    )
    parser.add_argument(
        "--decay-ms",
        type=float,
        default=100.0,
        help="Histogram decay time in ms (default: 100.0)",
    )
    parser.add_argument(
        "--blade-count",
        type=int,
        default=2,
        help="Number of blades per propeller for detection (default: 2)",
    )
    parser.add_argument(
        "--bin-us",
        type=int,
        default=100,
        help="Histogram bin width in microseconds (default: 100)",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=100,
        help="Minimum detections for frequency estimate (default: 100)",
    )
    parser.add_argument(
        "--sigma-factor",
        type=float,
        default=10.0,
        help="Sigma factor for Gaussian scoring (default: 10.0)",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=5.0,
        help="DBSCAN epsilon parameter (default: 5.0)",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=10,
        help="DBSCAN min samples (default: 10)",
    )
    parser.add_argument(
        "--dbscan-threshold",
        type=float,
        default=0.001,
        help="DBSCAN highlight threshold (default: 0.001)",
    )

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        output_path = args.dat.parent / f"{args.dat.stem}_analysis.csv"
    else:
        output_path = args.output

    print(f"Input DAT file: {args.dat}")
    print(f"Output CSV: {output_path}")
    print(f"Window length: {args.window}ms")
    print(f"Playback speed: {args.speed}x")
    print(f"Denoising: {'ON' if args.denoise else 'OFF'}")
    if args.denoise:
        print(f"Denoise kernel: {args.denoise_kernel}")
    print(f"Prediction history: {args.prediction_history}s")
    print(f"Prediction horizon: {args.prediction_horizon}s")

    # Drone detection initialization
    WIDTH = 1280
    HEIGHT = 720
    NUM_PIXELS = WIDTH * HEIGHT
    NUM_BINS = 256
    
    window_duration_us = int(args.window_ms * 1000)
    decay_time_us = int(args.decay_ms * 1000)
    bin_width_us = args.bin_us
    min_detections = args.min_detections
    sigma_factor = args.sigma_factor
    blade_count = args.blade_count
    
    # Initialize histogram state
    t_on_prev = np.full(NUM_PIXELS, -1, dtype=np.int64)
    t_off_prev = np.full(NUM_PIXELS, -1, dtype=np.int64)
    period_histogram = np.zeros(NUM_BINS, dtype=np.int32)
    
    # FIFO queue for histogram decay
    fifo_size = 1000000
    fifo_bins = np.zeros(fifo_size, dtype=np.int32)
    fifo_times = np.zeros(fifo_size, dtype=np.int64)
    q_head = 0
    q_tail = 0

    # Create analyzer
    analyzer = DroneAnalyzer(
        time_window_s=args.analysis_window,
        trajectory_history_s=args.trajectory_history,
        hover_threshold_px_s=args.hover_threshold,
        straight_direction_variance_deg=args.straight_variance,
        aggressive_accel_threshold=args.aggressive_threshold,
    )

    # Create Kalman tracker for trajectory prediction
    kalman_tracker = DroneKalmanTracker()

    # Load event data
    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )
    
    # Load recording to get timestamps
    recording = open_dat(args.dat, width=1280, height=720)
    timestamps_sorted = recording.timestamps

    # Create RPM estimators (one overall + 4 per-propeller)
    rpm_estimator = PropellerRPMEstimator(
        width=1280,
        height=720,
        num_blades=args.num_blades,
    )

    # Per-propeller estimators
    rpm_estimators_per_prop = [
        PropellerRPMEstimator(width=1280, height=720, num_blades=args.num_blades)
        for _ in range(4)
    ]

    # Create pacer
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    # Create data logger
    logger = DroneDataLogger(output_path)

    cv2.namedWindow("Drone Analysis", cv2.WINDOW_NORMAL)

    try:
        for batch_range in pacer.pace(src.ranges()):
            # Get current timestamp in seconds
            current_time_s = batch_range.end_ts_us / 1e6

            # Extract events for drone detection
            event_indexes = src.order[batch_range.start:batch_range.stop]
            words = src.event_words[event_indexes].astype(np.uint32, copy=False)
            
            x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
            y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
            polarities = ((words >> 28) & 0xF) > 0
            timestamps = timestamps_sorted[event_indexes]
            
            # Initialize pixel_last_period for this window
            pixel_last_period = np.zeros(NUM_PIXELS, dtype=np.int64)
            
            # Detect drone using period histogram
            WIDTH_numba = np.int64(WIDTH)
            freq, q_head, q_tail = process_events_window(
                timestamps, x_coords, y_coords, polarities,
                WIDTH_numba, NUM_PIXELS, NUM_BINS,
                decay_time_us, bin_width_us,
                min_detections,
                t_on_prev, t_off_prev, period_histogram,
                fifo_bins, fifo_times, q_head, q_tail,
                pixel_last_period,
            )
            
            bbox = None
            
            if freq > 0.0:
                # Generate highlight frame
                peak_bin = np.argmax(period_histogram)
                peak_period_us = (peak_bin + 0.5) * bin_width_us
                sigma_us = bin_width_us * sigma_factor
                
                scores_1d = np.zeros(NUM_PIXELS, dtype=float)
                valid_pixels_mask = pixel_last_period > 0
                pixel_error_us = np.abs(pixel_last_period[valid_pixels_mask] - peak_period_us)
                scores = np.exp(-0.5 * (pixel_error_us / sigma_us) ** 2)
                scores_1d[valid_pixels_mask] = scores
                highlight_frame = scores_1d.reshape((HEIGHT, WIDTH))
                
                # Detect drone using DBSCAN
                rotor_boxes, drone_box = compute_rotor_and_drone_bboxes(
                    highlight_frame,
                    eps=args.dbscan_eps,
                    min_samples=args.dbscan_min_samples,
                    threshold=args.dbscan_threshold,
                )
                
                # Convert to BoundingBox if detected
                if drone_box is not None:
                    x, y, w, h = drone_box
                    bbox = BoundingBox(
                        timestamp_s=current_time_s,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                    )

            # Update analyzer if bbox available
            if bbox:
                analyzer.add_position(
                    current_time_s, bbox.center_x, bbox.center_y
                )

            # Render frame
            window = get_window(
                src.event_words,
                src.order,
                batch_range.start,
                batch_range.stop,
            )
            main_frame = get_frame(window)

            # Apply denoising if enabled
            if args.denoise:
                main_frame = denoise_frame(main_frame, kernel_size=args.denoise_kernel)

            # Draw detection visualization on main frame (ADD THIS ENTIRE SECTION HERE)
            if bbox is not None and 'rotor_boxes' in locals():
                # Draw rotor boxes
                for rx, ry, rw, rh in rotor_boxes:
                    cv2.rectangle(
                        main_frame,
                        (rx, ry),
                        (rx + rw, ry + rh),
                        (255, 0, 255),  # Magenta for rotors
                        1
                    )
                
                # Draw centroid
                centroid_x = int(bbox.center_x)
                centroid_y = int(bbox.center_y)
                cv2.circle(main_frame, (centroid_x, centroid_y), 8, (0, 255, 255), -1)  # Cyan filled circle
                cv2.circle(main_frame, (centroid_x, centroid_y), 10, (255, 255, 255), 2)  # White outline
                cv2.putText(
                    main_frame,
                    "CENTROID",
                    (centroid_x + 15, centroid_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

            # Create larger canvas (1280 x 1000) for main view + subplots
            canvas = np.zeros((1000, 1280, 3), dtype=np.uint8)
            # Place the main 720p frame in the top section
            canvas[0:720, 0:1280] = main_frame
            # Update frame reference to use canvas
            frame = canvas

            # Draw trajectory trail
            trajectory = analyzer.get_trajectory()
            if len(trajectory) >= 2:
                draw_trajectory(frame, trajectory)

            # Initialize variables for yaw and RPM
            yaw_deg = 0.0
            rpm = 0.0
            propeller_centers = []

            # Draw bounding box and analyze if available
            if bbox:
                draw_bounding_box(frame, bbox)

                # Update Kalman tracker
                kalman_tracker.update_rpm(rpm, current_time_s)  

                if not kalman_tracker.initialized:
                    kalman_tracker.init(bbox, current_time_s)
                else:
                    kalman_tracker.predict(current_time_s)
                    kalman_tracker.correct(bbox)

                # Classify events into propeller and skeleton (with timestamps)
                x_prop, y_prop, t_prop, x_skel, y_skel, t_skel = classify_drone_events(
                    src.event_words,
                    src.order,
                    batch_range.start,
                    batch_range.stop,
                    bbox,
                    timestamps_sorted,
                    frequency_threshold=args.propeller_threshold,
                )

                # Feed propeller events to RPM estimator
                if len(x_prop) > 0:
                    # Get polarities for propeller events
                    event_indexes = src.order[batch_range.start:batch_range.stop]
                    words = src.event_words[event_indexes].astype(np.uint32, copy=False)
                    x_all = (words & 0x3FFF).astype(np.int32, copy=False)
                    y_all = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
                    pol_all = ((words >> 28) & 0xF) > 0
                    
                    # Match propeller events to get polarities
                    # Create a set of (x, y) tuples for quick lookup
                    prop_coords = set(zip(x_prop, y_prop))
                    prop_mask = np.array([
                        (int(x), int(y)) in prop_coords 
                        for x, y in zip(x_all, y_all)
                    ])
                    pol_prop = pol_all[prop_mask]
                    
                    # Add events to RPM estimator
                    rpm_estimator.add_events(x_prop, y_prop, t_prop, pol_prop)
                    
                    # Get current RPM estimate
                    rpm = rpm_estimator.get_rpm()

                # Cluster propellers and get centers
                propeller_centers, event_counts = cluster_propeller_centers(
                    x_prop,
                    y_prop,
                    n_clusters=args.num_propellers,
                )

                # Calculate yaw if we have propeller centers
                if len(propeller_centers) == 4:
                    yaw_deg = analyzer.calculate_yaw(propeller_centers)

                # Create subplots
                drone_subplot = create_drone_subplot(
                    src.event_words,
                    src.order,
                    batch_range.start,
                    batch_range.stop,
                    bbox,
                    subplot_size=(213, 240),
                    denoise=args.denoise,
                    denoise_kernel=args.denoise_kernel,
                )

                propeller_subplot = create_classified_subplot(
                    x_prop,
                    y_prop,
                    bbox,
                    subplot_size=(213, 240),
                    event_color=(255, 100, 100),
                    base_color=(0, 0, 0),
                    denoise=args.denoise,
                    denoise_kernel=args.denoise_kernel,
                    show_propeller_centers=True,
                    n_propellers=args.num_propellers,
                )

                skeleton_subplot = create_classified_subplot(
                    x_skel,
                    y_skel,
                    bbox,
                    subplot_size=(213, 240),
                    event_color=(255, 100, 255),
                    base_color=(0, 0, 0),
                    denoise=args.denoise,
                    denoise_kernel=args.denoise_kernel,
                )

                # Create yaw and RPM subplots
                yaw_subplot = create_yaw_subplot(propeller_centers, yaw_deg, subplot_size=(213, 240))
                rpm_subplot = create_rpm_subplot(rpm, subplot_size=(213, 240))

                # After clustering propellers:
                if len(propeller_centers) == 4:
                    yaw_deg = analyzer.calculate_yaw(propeller_centers)
                    
                    # Assign events to individual propellers
                    propeller_event_groups = assign_events_to_propellers(
                        x_prop, y_prop, t_prop, pol_prop, propeller_centers
                    )
                    
                    # Process each propeller's events
                    rpm_per_prop = []
                    for i, (x_p, y_p, t_p, pol_p) in enumerate(propeller_event_groups):
                        if len(x_p) > 0:
                            rpm_estimators_per_prop[i].add_events(x_p, y_p, t_p, pol_p)
                        rpm_per_prop.append(rpm_estimators_per_prop[i].get_rpm())

                # Create the per-propeller RPM subplot
                per_propeller_rpm_subplot = create_per_propeller_rpm_subplot(
                    rpm_per_prop if len(propeller_centers) == 4 else [0.0, 0.0, 0.0, 0.0],
                    propeller_centers,
                    subplot_size=(213, 240)
                )

                # Update overlay call to include 6 subplots in new order:
                overlay_multiple_subplots(
                    frame,
                    drone_subplot,
                    propeller_subplot,
                    skeleton_subplot,
                    rpm_subplot,
                    per_propeller_rpm_subplot,
                    yaw_subplot,
                )

                # Draw future trajectory prediction as Gaussian distribution
                # Only draw on the top main frame section (first 720 rows)
                frame_top = frame[0:720, 0:1280]
                frame_with_prediction = draw_future_gaussian_distribution(
                    kalman_tracker,
                    frame_top,
                    max_horizon=args.prediction_horizon,
                    steps=20,
                    alpha=args.prediction_alpha,
                )
                frame[0:720, 0:1280] = frame_with_prediction

            else:
                # Drone not detected - skip frame (no processing needed)
                pass
                    
            # Analyze motion
            analysis = analyzer.analyze()
            
            # Update analysis with current yaw and RPM
            analysis.yaw_deg = yaw_deg
            analysis.rpm = rpm

            # Log data to CSV
            logger.log(current_time_s, bbox, analysis)

            # Draw analysis info
            draw_analysis_info(frame, analysis, bbox)

            # Draw HUD
            draw_hud(frame, pacer, batch_range, args.denoise, args.denoise_kernel)

            cv2.imshow("Drone Analysis", frame)

            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        logger.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()