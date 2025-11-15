import argparse  # noqa: INP001
import time

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
):
    # get indexes corresponding to events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window,
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),  # gray
    on_color: tuple[int, int, int] = (255, 255, 255),  # white
    off_color: tuple[int, int, int] = (0, 0, 0),  # black
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color
    return frame


def denoise_frame(frame: np.ndarray) -> np.ndarray:
    """Simple spatial denoising on the event frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.medianBlur(gray, 3)  # try 5 for stronger smoothing
    denoised_bgr = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)
    return denoised_bgr


def detect_drone_via_fft(frame: np.ndarray, top_k_spikes: int = 4):
    """
    Use a global 2D FFT on the denoised frame to detect:
    - Propeller regions -> red bounding boxes
    - Drone envelope (frequency-based) -> green bounding box (union of props)
    Also compute the top-K magnitude spikes in the 2D FFT.

    Returns:
        annotated_frame: BGR image with drawn boxes
        prop_boxes: list of (x, y, w, h)
        freq_drone_box: (x, y, w, h) or None
        top_spikes: list of dicts with keys:
            {'fx': float, 'fy': float, 'mag': float, 'ky': int, 'kx': int}
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2D FFT and shift
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # High-pass mask: keep high spatial frequencies (propellers / edges)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    radius = 10  # tune this
    high_pass_mask = (dist > radius).astype(np.float32)

    f_hp = fshift * high_pass_mask

    # ---- TOP SPIKES IN FREQUENCY DOMAIN ------------------------------------
    # Use magnitude of original shifted FFT, but ignore low frequencies
    mag = np.abs(fshift)
    spike_mask = (dist > radius * 0.75).astype(np.float32)  # ignore center
    mag_spike = mag * spike_mask

    flat = mag_spike.ravel()
    top_spikes = []
    if flat.size > 0:
        k = min(top_k_spikes, flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        idx_sorted = idx[np.argsort(flat[idx])[::-1]]

        for i in idx_sorted:
            ky = int(i // w)
            kx = int(i % w)
            mval = float(mag[ky, kx])

            # Map indices to normalized spatial frequencies in cycles/pixel
            fx = (kx - cx) / float(w)  # approx -0.5 .. 0.5
            fy = (ky - cy) / float(h)

            top_spikes.append(
                {"fx": fx, "fy": fy, "mag": mval, "kx": kx, "ky": ky}
            )

    # Back to spatial domain with high-pass content
    f_ishift = np.fft.ifftshift(f_hp)
    img_hp = np.fft.ifft2(f_ishift)
    img_hp = np.abs(img_hp)

    # Normalize to 0–1
    hp_norm = cv2.normalize(img_hp, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Slight gamma to emphasize strong responses
    hp_norm = np.power(hp_norm, 0.5)

    # Threshold -> candidate high-frequency regions (likely props / thin edges)
    threshold = 0.6  # tune: 0.4–0.8
    mask = (hp_norm > threshold).astype(np.uint8) * 255  # 0 or 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find connected blobs -> potential propeller regions
    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    prop_boxes = []
    h_img, w_img, _ = frame.shape
    img_area = h_img * w_img

    # heuristic area constraints: props are not tiny noise, not whole image
    min_area = img_area * 0.0001   # 0.01% of frame
    max_area = img_area * 0.05     # 5% of frame

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        prop_boxes.append((x, y, bw, bh))

    # Build frequency-based drone box as union of all prop boxes (current frame)
    freq_drone_box = None
    if prop_boxes:
        x_min = min(b[0] for b in prop_boxes)
        y_min = min(b[1] for b in prop_boxes)
        x_max = max(b[0] + b[2] for b in prop_boxes)
        y_max = max(b[1] + b[3] for b in prop_boxes)
        freq_drone_box = (x_min, y_min, x_max - x_min, y_max - y_min)

    annotated = frame.copy()

    # Draw propeller boxes in RED
    for (x, y, bw, bh) in prop_boxes:
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            "Propeller",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw frequency-based drone box in GREEN (per-frame FFT box)
    if freq_drone_box is not None:
        x, y, bw, bh = freq_drone_box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "Drone (FFT frame)",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated, prop_boxes, freq_drone_box, top_spikes


def estimate_rpm_from_fft_spikes(top_spikes) -> float:
    """
    Heuristic: convert dominant spatial frequency spike into an RPM-like value.

    - Uses radial spatial frequency of the strongest spike.
    - Maps radial freq in [0, 0.5] (cycles/pixel) to [0, MAX_RPM] linearly.
    - This is a *calibration point* you should tune with real data.

    Returns:
        rpm_estimate (float)
    """
    if not top_spikes:
        return 0.0

    # Use strongest spike
    sp = top_spikes[0]
    fx = sp["fx"]
    fy = sp["fy"]

    radial = float(np.hypot(fx, fy))  # 0..~0.7

    MAX_RPM = 10000.0  # tune based on real rotor speeds
    # assume radial=0.5 -> MAX_RPM
    rpm = (radial / 0.5) * MAX_RPM
    rpm = max(0.0, min(MAX_RPM, rpm))

    return rpm


def cluster_accumulated_box(
    accumulated_boxes: list[tuple[int, int, int, int]],
    frame_shape,
) -> tuple[int, int, int, int] | None:
    """
    K=1 clustering over propeller boxes accumulated from several frames.

    accumulated_boxes: list of (x, y, w, h) from LAST N FRAMES
    Returns (x, y, w, h) or None.
    """
    if not accumulated_boxes:
        return None

    H, W, _ = frame_shape

    # Centers of all boxes across multiple frames
    pts = np.float32(
        [[x + w / 2.0, y + h / 2.0] for (x, y, w, h) in accumulated_boxes]
    )

    K = 1  # single cluster
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compact, labels, centers = cv2.kmeans(
        pts,
        K,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    # For K=1, union bbox of all accumulated boxes
    xs, ys, x2s, y2s = [], [], [], []
    for (x, y, w, h) in accumulated_boxes:
        xs.append(x)
        ys.append(y)
        x2s.append(x + w)
        y2s.append(y + h)

    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(W - 1, max(x2s))
    y_max = min(H - 1, max(y2s))

    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def combine_boxes(
    freq_box,
    cluster_box,
    frame_shape,
) -> tuple[int, int, int, int] | None:
    """
    Combine per-frame FFT box and multi-frame cluster box.

    Strategy:
    - If both exist: try intersection; else average box.
    - If only one exists: return that.
    """
    if freq_box is None and cluster_box is None:
        return None
    if freq_box is None:
        return cluster_box
    if cluster_box is None:
        return freq_box

    fx, fy, fw, fh = freq_box
    cx, cy, cw, ch = cluster_box

    fx2 = fx + fw
    fy2 = fy + fh
    cx2 = cx + cw
    cy2 = cy + ch

    H, W, _ = frame_shape

    # Intersection
    ix1 = max(fx, cx)
    iy1 = max(fy, cy)
    ix2 = min(fx2, cx2)
    iy2 = min(fy2, cy2)

    if ix2 > ix1 and iy2 > iy1:
        ix1 = max(0, min(W - 1, ix1))
        iy1 = max(0, min(H - 1, iy1))
        ix2 = max(1, min(W, ix2))
        iy2 = max(1, min(H, iy2))
        return (ix1, iy1, ix2 - ix1, iy2 - iy1)

    # Fallback: average
    ax1 = int(round((fx + cx) / 2.0))
    ay1 = int(round((fy + cy) / 2.0))
    ax2 = int(round((fx2 + cx2) / 2.0))
    ay2 = int(round((fy2 + cy2) / 2.0))

    ax1 = max(0, min(W - 1, ax1))
    ay1 = max(0, min(H - 1, ay1))
    ax2 = max(ax1 + 1, min(W, ax2))
    ay2 = max(ay1 + 1, min(H, ay2))

    return (ax1, ay1, ax2 - ax1, ay2 - ay1)


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

        MAX_RPM = 10000.0  # should match mapping in estimate_rpm_from_fft_spikes
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
        Optional future bbox prediction using current state + recent velocity.
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
    sigma_growth = max(w, h) * 0.5 / max_horizon

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

        weight = 1.0 - (t / max_horizon) * 0.3
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


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
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

    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window", type=float, default=10, help="Windows duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    args = parser.parse_args()

    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    tracker = BoxKalmanTracker()
    trajectory: list[tuple[float, float]] = []

    ACCUM_FRAMES = 5
    prop_history_frames: list[list[tuple[int, int, int, int]]] = []

    cv2.namedWindow(
        "Evio Player (orig | denoised | Gaussian future + RPM)", cv2.WINDOW_NORMAL
    )

    frame_idx = 0

    for batch_range in pacer.pace(src.ranges()):
        frame_idx += 1
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        frame = get_frame(window)
        denoised_frame = denoise_frame(frame)

        draw_hud(frame, pacer, batch_range)
        draw_hud(denoised_frame, pacer, batch_range)

        # --- FFT + spike analysis -----------------------------------------
        t_fft_start = time.perf_counter()
        detected_frame, prop_boxes, freq_box, top_spikes = detect_drone_via_fft(
            denoised_frame, top_k_spikes=4
        )
        t_fft = (time.perf_counter() - t_fft_start) * 1000.0  # ms

        # Estimate RPM from FFT spikes
        rpm_estimate = estimate_rpm_from_fft_spikes(top_spikes)

        # Print spikes + RPM to console
        if top_spikes:
            print(f"\nFrame {frame_idx}: Top {len(top_spikes)} FFT spikes, RPM≈{rpm_estimate:.1f}")
            for i, sp in enumerate(top_spikes):
                print(
                    f"  #{i+1}: fx={sp['fx']:+.4f}, fy={sp['fy']:+.4f}, "
                    f"mag={sp['mag']:.2f}, idx=({sp['ky']},{sp['kx']})"
                )

        # Update propeller history
        if prop_boxes:
            prop_history_frames.append(prop_boxes)
            if len(prop_history_frames) > ACCUM_FRAMES:
                prop_history_frames.pop(0)

        accumulated_boxes = [
            box for frame_boxes in prop_history_frames for box in frame_boxes
        ]

        # --- clustering + combination --------------------------------------
        t_clust_start = time.perf_counter()
        cluster_box = cluster_accumulated_box(
            accumulated_boxes, detected_frame.shape
        )
        combined_box = combine_boxes(freq_box, cluster_box, detected_frame.shape)
        t_cluster = (time.perf_counter() - t_clust_start) * 1000.0  # ms

        if cluster_box is not None:
            x, y, bw, bh = cluster_box
            cv2.rectangle(detected_frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
            cv2.putText(
                detected_frame,
                f"Drone (cluster accum {ACCUM_FRAMES}f)",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if combined_box is not None:
            x, y, bw, bh = combined_box
            cv2.rectangle(detected_frame, (x, y), (x + bw, y + bh), (0, 165, 255), 3)
            cv2.putText(
                detected_frame,
                "Drone (combined FFT+accum cluster)",
                (x, max(0, y - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        t_rec = batch_range.end_ts_us / 1e6

        # --- RPM-aware Kalman tracking -------------------------------------
        t_track_start = time.perf_counter()

        # Feed RPM into tracker before predicting
        tracker.update_rpm(rpm_estimate, t_rec)

        if not tracker.initialized:
            if combined_box is not None:
                tracker.init(combined_box, t_rec)
        else:
            tracker.predict(t_rec)
            if combined_box is not None:
                tracker.correct(combined_box)

        t_track = (time.perf_counter() - t_track_start) * 1000.0  # ms

        tracked_box = tracker.get_state_bbox()

        if tracked_box is not None:
            tx, ty, tw, th = tracked_box
            cv2.rectangle(
                detected_frame, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 2
            )
            cv2.putText(
                detected_frame,
                "Drone (tracked KF+RPM)",
                (tx, max(0, ty - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cx = tx + tw / 2.0
            cy = ty + th / 2.0
            trajectory.append((cx, cy))
            if len(trajectory) > 500:
                trajectory = trajectory[-500:]

        # Trajectory
        if len(trajectory) >= 2:
            for i in range(1, len(trajectory)):
                x1, y1 = trajectory[i - 1]
                x2, y2 = trajectory[i]
                cv2.line(
                    detected_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    2,
                )

        # Gaussian future
        t_future_start = time.perf_counter()
        detected_frame = draw_future_gaussian_distribution(
            tracker,
            detected_frame,
            max_horizon=2.0,
            steps=20,
            alpha=0.6,
        )
        t_future = (time.perf_counter() - t_future_start) * 1000.0  # ms

        # Print processing times + RPM
        print(
            f"Frame {frame_idx}: "
            f"FFT={t_fft:.2f} ms, "
            f"Cluster+Combine={t_cluster:.2f} ms, "
            f"Track+RPM={t_track:.2f} ms, "
            f"FutureGauss={t_future:.2f} ms, "
            f"RPM≈{rpm_estimate:.1f}"
        )

        # Overlay timings and RPM
        timing_str1 = (
            f"FFT={t_fft:.1f}ms  Clust+Comb={t_cluster:.1f}ms"
        )
        timing_str2 = (
            f"Track+RPM={t_track:.1f}ms  FutureGauss={t_future:.1f}ms"
        )
        cv2.putText(
            detected_frame,
            timing_str1,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            detected_frame,
            timing_str2,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            detected_frame,
            f"RPM≈{rpm_estimate:.0f}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            detected_frame,
            f"FFT(green), cluster_accum(magenta,{ACCUM_FRAMES}f), "
            f"combined(orange), tracked(cyan,RPM-aware), future=Gaussian",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combined_view = np.hstack((frame, denoised_frame, detected_frame))
        cv2.imshow(
            "Evio Player (orig | denoised | Gaussian future + RPM)", combined_view
        )

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
