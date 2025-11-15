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


def detect_drone_via_fft(frame: np.ndarray):
    """
    Use a global 2D FFT on the denoised frame to detect:
    - Propeller regions -> red bounding boxes
    - Drone envelope (frequency-based) -> green bounding box (union of props)

    Returns:
        annotated_frame: BGR image with drawn boxes
        prop_boxes: list of (x, y, w, h)
        freq_drone_box: (x, y, w, h) or None
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

    # Back to spatial domain
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

    return annotated, prop_boxes, freq_drone_box


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
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.initialized = False
        self.last_t = None

        # Keep a short history of recent predicted states for
        # more responsive velocity estimation
        # entries: (t, cx, cy)
        self.recent_states: list[tuple[float, float, float]] = []

        # Transition matrix (dt will be updated each step)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)

        # Measurement matrix
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # cx
                [0, 1, 0, 0, 0, 0],  # cy
                [0, 0, 0, 0, 1, 0],  # w
                [0, 0, 0, 0, 0, 1],  # h
            ],
            dtype=np.float32,
        )

        # Process noise (how much we trust the motion model)
        self.kf.processNoiseCov = np.diag(
            [1e-3, 1e-3, 1e-2, 1e-2, 1e-4, 1e-4]
        ).astype(np.float32)

        # Measurement noise (how noisy detections are)
        self.kf.measurementNoiseCov = np.diag(
            [1e-1, 1e-1, 1e-1, 1e-1]
        ).astype(np.float32)

        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

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

    def predict(self, t: float):
        if not self.initialized:
            return None

        dt = 0.0
        if self.last_t is not None:
            dt = max(1e-3, t - self.last_t)
        self.last_t = t

        self._update_transition(dt)
        pred = self.kf.predict()

        # Store recent predicted center for more responsive velocity
        cx, cy, vx, vy, w, h = pred.flatten()
        self.recent_states.append((t, float(cx), float(cy)))
        if len(self.recent_states) > 10:  # keep last N predictions
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
        Predict bbox after `horizon_s` seconds.

        NOW more sensitive to immediate changes:
        - Prefer velocity estimated from the last two predicted states.
        - Fall back to Kalman velocity if history is too short.
        """
        if not self.initialized:
            return None

        state = self.kf.statePost.flatten()
        cx, cy, vx_kf, vy_kf, w, h = state

        # --- NEW: recent-velocity estimate from last two predictions ---
        if len(self.recent_states) >= 2:
            (t0, cx0, cy0), (t1, cx1, cy1) = self.recent_states[-2], self.recent_states[-1]
            dt = max(1e-3, t1 - t0)
            vx_recent = (cx1 - cx0) / dt
            vy_recent = (cy1 - cy0) / dt

            # Blend recent velocity and Kalman velocity for stability
            alpha = 0.7  # closer to 1.0 = more sensitive to recent motion
            vx = alpha * vx_recent + (1.0 - alpha) * vx_kf
            vy = alpha * vy_recent + (1.0 - alpha) * vy_kf
        else:
            # Not enough history, use Kalman velocity
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

    # NEW: store propeller boxes for last N frames (for accumulation clustering)
    ACCUM_FRAMES = 5
    prop_history_frames: list[list[tuple[int, int, int, int]]] = []

    cv2.namedWindow(
        "Evio Player (orig | denoised | accumulated-cluster-tracked)", cv2.WINDOW_NORMAL
    )

    for batch_range in pacer.pace(src.ranges()):
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

        # FFT-based detection for THIS frame
        detected_frame, prop_boxes, freq_box = detect_drone_via_fft(denoised_frame)

        # Update propeller history for accumulation
        if prop_boxes:
            prop_history_frames.append(prop_boxes)
            if len(prop_history_frames) > ACCUM_FRAMES:
                prop_history_frames.pop(0)

        # Flatten accumulated prop boxes from last N frames
        accumulated_boxes = [
            box for frame_boxes in prop_history_frames for box in frame_boxes
        ]

        # Clustering based on accumulated boxes (multi-frame clustering)
        cluster_box = cluster_accumulated_box(
            accumulated_boxes, detected_frame.shape
        )

        # Draw cluster box (magenta) – this is the *temporal* cluster
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

        # Combine per-frame FFT box with multi-frame cluster box
        combined_box = combine_boxes(freq_box, cluster_box, detected_frame.shape)

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

        # Time in seconds
        t_rec = batch_range.end_ts_us / 1e6

        # Kalman tracking on combined box
        if not tracker.initialized:
            if combined_box is not None:
                tracker.init(combined_box, t_rec)
        else:
            tracker.predict(t_rec)
            # Correct only when we have a combined detection
            if combined_box is not None:
                tracker.correct(combined_box)

        tracked_box = tracker.get_state_bbox()

        # Draw tracked box (cyan)
        if tracked_box is not None:
            tx, ty, tw, th = tracked_box
            cv2.rectangle(
                detected_frame, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 2
            )
            cv2.putText(
                detected_frame,
                "Drone (tracked KF)",
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

        # Draw trajectory (yellow)
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

        # 2-second future prediction from tracker
        future_box = tracker.predict_future(2.0, detected_frame.shape)
        if future_box is not None:
            px, py, pw, ph = future_box
            cv2.rectangle(
                detected_frame, (px, py), (px + pw, py + ph), (255, 0, 0), 2
            )
            cv2.putText(
                detected_frame,
                "Predicted (2s, KF)",
                (px, max(0, py - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            detected_frame,
            f"FFT(green), cluster_accum(magenta,{ACCUM_FRAMES}f), combined(orange), tracked(cyan)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combined_view = np.hstack((frame, denoised_frame, detected_frame))
        cv2.imshow(
            "Evio Player (orig | denoised | accumulated-cluster-tracked)", combined_view
        )

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

