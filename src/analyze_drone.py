"""Drone detection and behavioral analysis from event camera data with bounding boxes."""

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from evio.core.pacer import Pacer
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


class BoundingBoxLoader:
    """Load and interpolate bounding box detections."""

    def __init__(self, bbox_file: Path, interpolation_threshold_s: float = 0.5) -> None:
        self.boxes: list[BoundingBox] = []
        self.interpolation_threshold_s = interpolation_threshold_s
        self._load_file(bbox_file)

    def _load_file(self, bbox_file: Path) -> None:
        """Load bounding boxes from file."""
        with bbox_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse: "13.533198: 1253.0, 413.0, 1280.0, 455.0"
                parts = line.split(":")
                if len(parts) != 2:
                    continue

                timestamp_s = float(parts[0].strip())
                coords = [float(x.strip()) for x in parts[1].split(",")]

                if len(coords) == 4:
                    self.boxes.append(
                        BoundingBox(timestamp_s, *coords)
                    )

        self.boxes.sort(key=lambda b: b.timestamp_s)
        print(f"Loaded {len(self.boxes)} bounding box detections")

    def get_box_at_time(self, timestamp_s: float) -> BoundingBox | None:
        """Get interpolated bounding box at given time.
        
        Returns None if:
        - No boxes available
        - Timestamp is before first or after last detection
        - Time gap to nearest detection exceeds interpolation_threshold_s
        """
        if not self.boxes:
            return None

        # Before first detection
        if timestamp_s < self.boxes[0].timestamp_s:
            # Check if within threshold of first box
            if self.boxes[0].timestamp_s - timestamp_s > self.interpolation_threshold_s:
                return None
            return self.boxes[0]

        # After last detection
        if timestamp_s > self.boxes[-1].timestamp_s:
            # Check if within threshold of last box
            if timestamp_s - self.boxes[-1].timestamp_s > self.interpolation_threshold_s:
                return None
            return self.boxes[-1]

        # Find surrounding boxes
        idx = 0
        for i, box in enumerate(self.boxes):
            if box.timestamp_s <= timestamp_s:
                idx = i
            else:
                break

        # Exact match or last box
        if (
            idx == len(self.boxes) - 1
            or self.boxes[idx].timestamp_s == timestamp_s
        ):
            return self.boxes[idx]

        # Check if time gap exceeds interpolation threshold
        box1 = self.boxes[idx]
        box2 = self.boxes[idx + 1]
        
        time_gap = box2.timestamp_s - box1.timestamp_s
        if time_gap > self.interpolation_threshold_s:
            # Gap too large, don't interpolate - return None (no drone detected)
            return None

        # Interpolate between boxes
        alpha = (timestamp_s - box1.timestamp_s) / time_gap

        return BoundingBox(
            timestamp_s=timestamp_s,
            x1=box1.x1 + alpha * (box2.x1 - box1.x1),
            y1=box1.y1 + alpha * (box2.y1 - box1.y1),
            x2=box1.x2 + alpha * (box2.x2 - box1.x2),
            y2=box1.y2 + alpha * (box2.y2 - box1.y2),
        )


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

    def get_trajectory(self) -> list[DronePosition]:
        """Get trajectory history for visualization."""
        return list(self.trajectory_history)

    def analyze(self) -> MotionAnalysis:
        """Analyze motion based on position history."""
        if len(self.position_history) < 2:
            return MotionAnalysis()

        positions = list(self.position_history)

        # Calculate velocity (using last two positions)
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

        # Detect behavioral pattern
        pattern = self._detect_pattern(speed, acceleration, positions)

        return MotionAnalysis(
            speed_px_s=speed,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            acceleration=acceleration,
            direction_deg=direction_deg,
            pattern=pattern,
        )

    def _detect_pattern(
        self,
        speed: float,
        acceleration: float,
        positions: list[DronePosition],
    ) -> str:
        """Detect behavioral pattern from motion data."""
        # Hovering: low speed
        if speed < self.hover_threshold:
            return "HOVERING"

        # Aggressive maneuvering: high acceleration
        if acceleration > self.aggressive_threshold:
            return "AGGRESSIVE MANEUVER"

        # Check for straight flight: consistent direction
        if len(positions) >= 4:
            directions = []
            for i in range(len(positions) - 1):
                dx = positions[i + 1].center_x - positions[i].center_x
                dy = positions[i + 1].center_y - positions[i].center_y
                if abs(dx) > 1 or abs(dy) > 1:  # Avoid noise
                    directions.append(np.degrees(np.arctan2(dy, dx)))

            if len(directions) >= 3:
                # Normalize angles to [-180, 180]
                directions = np.array(directions)
                directions = (directions + 180) % 360 - 180

                variance = np.std(directions)
                if variance < self.straight_variance_deg:
                    return "STRAIGHT FLIGHT"

        # Default: cruising
        return "CRUISING"


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract event coordinates and polarities for a time window."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),
    on_color: tuple[int, int, int] = (255, 255, 255),
    off_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render events to frame."""
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def denoise_frame(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Simple spatial denoising on the event frame using median blur.
    
    Args:
        frame: BGR frame to denoise
        kernel_size: Median blur kernel size (3 or 5 recommended)
    
    Returns:
        Denoised BGR frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.medianBlur(gray, kernel_size)
    denoised_bgr = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)
    return denoised_bgr


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
    cx, cy = int(bbox.center_x), int(bbox.center_y)
    cv2.circle(frame, (cx, cy), 4, color, -1)


def draw_analysis_info(
    frame: np.ndarray,
    analysis: MotionAnalysis,
    bbox: BoundingBox | None,
    *,
    position: tuple[int, int] = (10, 600),
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw motion analysis info on frame."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20

    # Background rectangle for better readability
    if bbox:
        info_lines = [
            f"Pattern: {analysis.pattern}",
            f"Speed: {analysis.speed_px_s:.1f} px/s",
            f"Velocity: ({analysis.velocity_x:.1f}, {analysis.velocity_y:.1f})",
            f"Acceleration: {analysis.acceleration:.1f} px/s^2",
            f"Direction: {analysis.direction_deg:.1f} deg",
            f"Position: ({bbox.center_x:.1f}, {bbox.center_y:.1f})",
            f"Box size: {bbox.width:.1f} x {bbox.height:.1f}",
        ]
    else:
        info_lines = [
            "*** NO DRONE DETECTED ***",
            f"Pattern: {analysis.pattern}",
            f"Speed: {analysis.speed_px_s:.1f} px/s",
            f"Acceleration: {analysis.acceleration:.1f} px/s^2",
        ]
        color = (0, 0, 255)  # Red color when no drone

    # Draw semi-transparent background
    bg_height = len(info_lines) * line_height + 10
    bg_width = 350
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 5, y - 15),
        (x + bg_width, y + bg_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text
    for i, line in enumerate(info_lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    denoise_enabled: bool = False,
    denoise_kernel: int = 3,
    *,
    color: tuple[int, int, int] = (0, 0, 0),
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
    subplot_size: tuple[int, int] = (200, 200),
    denoise: bool = False,
    denoise_kernel: int = 3,
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
    
    Returns:
        Cropped and optionally denoised subplot
    """
    # Get all events in window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities = ((words >> 28) & 0xF) > 0

    # Filter events within bounding box
    x1, y1 = int(bbox.x1), int(bbox.y1)
    x2, y2 = int(bbox.x2), int(bbox.y2)
    
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
        return np.zeros((*subplot_size, 3), dtype=np.uint8)

    cropped_frame = np.full(
        (crop_height, crop_width, 3), (127, 127, 127), dtype=np.uint8
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
    frequency_threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify events within bbox into propeller and skeleton based on frequency.
    
    Propeller pixels have high event frequency due to fast rotation.
    Skeleton pixels have lower frequency (structural edges).
    
    Args:
        event_words: Raw event data
        time_order: Time-sorted indices
        win_start: Window start index
        win_stop: Window stop index
        bbox: Bounding box of drone
        frequency_threshold: Percentile threshold to separate propeller from skeleton (0-1)
    
    Returns:
        (x_propeller, y_propeller, x_skeleton, y_skeleton) coordinate arrays
    """
    # Get all events in window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    
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
    
    if len(x_bbox) == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
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
    x_skeleton = x_bbox[skeleton_mask]
    y_skeleton = y_bbox[skeleton_mask]
    
    return x_propeller, y_propeller, x_skeleton, y_skeleton


def create_classified_subplot(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    bbox: BoundingBox,
    subplot_size: tuple[int, int] = (200, 200),
    event_color: tuple[int, int, int] = (255, 255, 255),
    base_color: tuple[int, int, int] = (0, 0, 0),
    denoise: bool = False,
    denoise_kernel: int = 3,
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
    
    Returns:
        Subplot image
    """
    x1, y1 = int(bbox.x1), int(bbox.y1)
    x2, y2 = int(bbox.x2), int(bbox.y2)
    
    crop_width = x2 - x1 + 1
    crop_height = y2 - y1 + 1
    
    if crop_width <= 0 or crop_height <= 0:
        return np.zeros((*subplot_size, 3), dtype=np.uint8)
    
    # Create frame
    frame = np.full((crop_height, crop_width, 3), base_color, dtype=np.uint8)
    
    # Draw events
    if len(x_coords) > 0:
        x_rel = x_coords - x1
        y_rel = y_coords - y1
        
        # Bounds checking
        valid_mask = (x_rel >= 0) & (x_rel < crop_width) & (y_rel >= 0) & (y_rel < crop_height)
        x_valid = x_rel[valid_mask]
        y_valid = y_rel[valid_mask]
        
        if len(x_valid) > 0:
            frame[y_valid, x_valid] = event_color
    
    # Apply denoising if requested
    if denoise:
        frame = denoise_frame(frame, kernel_size=denoise_kernel)
    
    # Resize to subplot size
    subplot = cv2.resize(frame, subplot_size, interpolation=cv2.INTER_NEAREST)
    
    return subplot


def overlay_multiple_subplots(
    frame: np.ndarray,
    drone_subplot: np.ndarray,
    propeller_subplot: np.ndarray,
    skeleton_subplot: np.ndarray,
    start_position: tuple[int, int] = (950, 10),
    border_color: tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 2,
) -> None:
    """Overlay three subplots vertically on main frame.
    
    Args:
        frame: Main frame to overlay on
        drone_subplot: Full drone view
        propeller_subplot: Propeller events only
        skeleton_subplot: Skeleton events only
        start_position: Top-left position for first subplot
        border_color: Border color
        border_thickness: Border thickness
    """
    x, y = start_position
    h, w = drone_subplot.shape[:2]
    spacing = 10
    
    # Subplot 1: Full Drone View
    cv2.rectangle(
        frame,
        (x - border_thickness, y - border_thickness),
        (x + w + border_thickness, y + h + border_thickness),
        border_color,
        border_thickness,
    )
    frame[y : y + h, x : x + w] = drone_subplot
    cv2.putText(
        frame,
        "DRONE VIEW",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        border_color,
        1,
        cv2.LINE_AA,
    )
    
    # Subplot 2: Propeller/Fan Events
    y2 = y + h + spacing
    propeller_color = (0, 0, 255)  # Red for propellers
    cv2.rectangle(
        frame,
        (x - border_thickness, y2 - border_thickness),
        (x + w + border_thickness, y2 + h + border_thickness),
        propeller_color,
        border_thickness,
    )
    frame[y2 : y2 + h, x : x + w] = propeller_subplot
    cv2.putText(
        frame,
        "PROPELLERS",
        (x, y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        propeller_color,
        1,
        cv2.LINE_AA,
    )
    
    # Subplot 3: Skeleton/Body Events
    y3 = y2 + h + spacing
    skeleton_color = (255, 0, 255)  # Magenta for skeleton
    cv2.rectangle(
        frame,
        (x - border_thickness, y3 - border_thickness),
        (x + w + border_thickness, y3 + h + border_thickness),
        skeleton_color,
        border_thickness,
    )
    frame[y3 : y3 + h, x : x + w] = skeleton_subplot
    cv2.putText(
        frame,
        "SKELETON",
        (x, y3 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        skeleton_color,
        1,
        cv2.LINE_AA,
    )


def overlay_subplot(
    frame: np.ndarray,
    subplot: np.ndarray,
    position: tuple[int, int] = (950, 10),
    border_color: tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 2,
) -> None:
    """Overlay subplot on main frame at specified position."""
    x, y = position
    h, w = subplot.shape[:2]

    # Add border
    cv2.rectangle(
        frame,
        (x - border_thickness, y - border_thickness),
        (x + w + border_thickness, y + h + border_thickness),
        border_color,
        border_thickness,
    )

    # Overlay subplot
    frame[y : y + h, x : x + w] = subplot

    # Add label
    cv2.putText(
        frame,
        "DRONE VIEW",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        border_color,
        1,
        cv2.LINE_AA,
    )


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
        """Log a single frame's data."""
        if bbox:
            self.writer.writerow([
                f"{timestamp_s:.6f}",
                f"{bbox.center_x:.2f}",
                f"{bbox.center_y:.2f}",
                f"{bbox.x1:.2f}",
                f"{bbox.y1:.2f}",
                f"{bbox.x2:.2f}",
                f"{bbox.y2:.2f}",
                f"{bbox.width:.2f}",
                f"{bbox.height:.2f}",
                f"{analysis.speed_px_s:.2f}",
                f"{analysis.velocity_x:.2f}",
                f"{analysis.velocity_y:.2f}",
                f"{analysis.acceleration:.2f}",
                f"{analysis.direction_deg:.2f}",
                analysis.pattern,
                "True",
            ])
        else:
            # No drone detected
            self.writer.writerow([
                f"{timestamp_s:.6f}",
                "",  # center_x
                "",  # center_y
                "",  # bbox_x1
                "",  # bbox_y1
                "",  # bbox_x2
                "",  # bbox_y2
                "",  # bbox_width
                "",  # bbox_height
                f"{analysis.speed_px_s:.2f}",
                f"{analysis.velocity_x:.2f}",
                f"{analysis.velocity_y:.2f}",
                f"{analysis.acceleration:.2f}",
                f"{analysis.direction_deg:.2f}",
                analysis.pattern,
                "False",
            ])

    def close(self) -> None:
        """Close the CSV file."""
        self.file.close()
        print(f"Data logged successfully to: {self.output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze drone behavior from event camera data"
    )
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("bbox", help="Path to bounding box file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: auto-generated based on dat filename)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=10,
        help="Window duration in ms (default: 10)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1,
        help="Playback speed (default: 1.0)",
    )
    parser.add_argument(
        "--force-speed",
        type=bool,
        default=True,
        help="Force playback speed by dropping windows",
    )
    parser.add_argument(
        "--interpolation-threshold",
        type=float,
        default=0.5,
        help="Max time gap for bbox interpolation in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--analysis-window",
        type=float,
        default=1.0,
        help="Time window for motion analysis in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--trajectory-history",
        type=float,
        default=3.0,
        help="Time window for trajectory trail visualization in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--hover-threshold",
        type=float,
        default=50.0,
        help="Speed threshold for hovering in px/s (default: 50.0)",
    )
    parser.add_argument(
        "--straight-variance",
        type=float,
        default=15.0,
        help="Direction variance threshold for straight flight in degrees (default: 15.0)",
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
        help="Apply median blur denoising to frames before analysis and visualization",
    )
    parser.add_argument(
        "--denoise-kernel",
        type=int,
        default=3,
        choices=[3, 5],
        help="Median blur kernel size for denoising (default: 3, use 5 for stronger smoothing)",
    )
    parser.add_argument(
        "--propeller-threshold",
        type=float,
        default=0.7,
        help="Frequency percentile threshold to separate propeller from skeleton (0-1, default: 0.7)",
    )

    args = parser.parse_args()

    # Determine output CSV path
    if args.output:
        output_path = Path(args.output)
    else:
        dat_path = Path(args.dat)
        output_path = dat_path.parent / f"{dat_path.stem}_analysis.csv"

    # Load bounding boxes
    bbox_loader = BoundingBoxLoader(
        Path(args.bbox), 
        interpolation_threshold_s=args.interpolation_threshold
    )

    # Create analyzer
    analyzer = DroneAnalyzer(
        time_window_s=args.analysis_window,
        trajectory_history_s=args.trajectory_history,
        hover_threshold_px_s=args.hover_threshold,
        straight_direction_variance_deg=args.straight_variance,
        aggressive_accel_threshold=args.aggressive_threshold,
    )

    # Load event data
    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    # Create pacer
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    # Create data logger
    logger = DroneDataLogger(output_path)

    cv2.namedWindow("Drone Analysis", cv2.WINDOW_NORMAL)

    try:
        for batch_range in pacer.pace(src.ranges()):
            # Get current timestamp in seconds
            current_time_s = batch_range.end_ts_us / 1e6

            # Get bounding box at current time
            bbox = bbox_loader.get_box_at_time(current_time_s)

            # Update analyzer if bbox available
            if bbox:
                analyzer.add_position(
                    current_time_s, bbox.center_x, bbox.center_y
                )

            # Analyze motion
            analysis = analyzer.analyze()

            # Log data to CSV
            logger.log(current_time_s, bbox, analysis)

            # Render frame
            window = get_window(
                src.event_words,
                src.order,
                batch_range.start,
                batch_range.stop,
            )
            frame = get_frame(window)

            # Apply denoising if enabled
            if args.denoise:
                frame = denoise_frame(frame, kernel_size=args.denoise_kernel)

            # Draw trajectory trail
            trajectory = analyzer.get_trajectory()
            if len(trajectory) >= 2:
                draw_trajectory(frame, trajectory)

            # Draw bounding box if available
            if bbox:
                draw_bounding_box(frame, bbox)

                # Classify events into propeller and skeleton
                x_prop, y_prop, x_skel, y_skel = classify_drone_events(
                    src.event_words,
                    src.order,
                    batch_range.start,
                    batch_range.stop,
                    bbox,
                    frequency_threshold=args.propeller_threshold,
                )

                # Create full drone subplot
                drone_subplot = create_drone_subplot(
                    src.event_words,
                    src.order,
                    batch_range.start,
                    batch_range.stop,
                    bbox,
                    denoise=args.denoise,
                    denoise_kernel=args.denoise_kernel,
                )

                # Create propeller subplot (red/white on black)
                propeller_subplot = create_classified_subplot(
                    x_prop,
                    y_prop,
                    bbox,
                    subplot_size=(200, 200),
                    event_color=(255, 100, 100),  # Light red
                    base_color=(0, 0, 0),
                    denoise=args.denoise,
                    denoise_kernel=args.denoise_kernel,
                )

                # Create skeleton subplot (magenta/white on black)
                skeleton_subplot = create_classified_subplot(
                    x_skel,
                    y_skel,
                    bbox,
                    subplot_size=(200, 200),
                    event_color=(255, 100, 255),  # Light magenta
                    base_color=(0, 0, 0),
                    denoise=args.denoise,
                    denoise_kernel=args.denoise_kernel,
                )

                # Overlay all three subplots
                overlay_multiple_subplots(
                    frame,
                    drone_subplot,
                    propeller_subplot,
                    skeleton_subplot,
                )

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