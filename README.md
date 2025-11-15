# RPM-Based Drone Pose Estimation

This README explains three key parts of the provided code:

1. How the event data is read and organized
2. How the dominant RPM is calculated from the events
3. How the RPM is used to predict the drone pose via an RPM-aware Kalman filter

---

## 1. How the data is read

### 1.1 Loading the .dat event file

In main_frequency_estimation(), the code starts by loading an event-based recording using DatMemmap:
timestamps, x, y and polarity arrays are read from the .dat file, as well as sensor width and height.
Each event is effectively: (time, x, y, polarity).

### 1.2 Building time windows

The full event stream is divided into fixed-duration time windows using build_windows(rec, window_duration_us).
Each window is represented as an index range (start_idx, stop_idx). For each range, the code slices:
- ts_window = timestamps_full[start_idx:stop_idx]
- x_window = x_coords_full[start_idx:stop_idx]
- y_window = y_coords_full[start_idx:stop_idx]
- p_window = polarities_full[start_idx:stop_idx]

So each window is processed independently as a batch of events in a given time interval.

---

## 2. How the RPM is calculated

### 2.1 Per-pixel period estimation

The function process_events_window() is the core of the frequency estimation. It receives the events of one
window together with per-pixel state and a global period_histogram.

For each event (t, x, y, p):
- A linear pixel index is computed as pixel_idx = y * WIDTH + x.
- If the event is ON (p == 1) and there was a previous OFF after the last ON,
  the time difference delta_t = t - t_on_prev[pixel_idx] is interpreted as the period for that pixel.
- If 0 < delta_t < max_period_us, delta_t is quantized into a bin using bin_width_us and the bin count in
  period_histogram is incremented. The contribution is also pushed into a FIFO (fifo_bins, fifo_times) so that
  old contributions can later be decayed.
- The last ON timestamp t_on_prev[pixel_idx] is updated.
- For OFF events (p == 0), t_off_prev[pixel_idx] is updated.

After processing events, a temporal decay loop removes old histogram contributions whose age exceeds
decay_time_us. This creates a sliding temporal window of period evidence in period_histogram.

### 2.2 Frequency from the histogram

Once the histogram has been updated, the code checks:
- total = period_histogram.sum()
- If total < min_detections, the function returns freq = 0.0 (not enough evidence).

Otherwise, the dominant period bin is found as:
- peak_bin = period_histogram.argmax()
- est_period_us = (peak_bin + 0.5) * bin_width_us
- freq = 1_000_000.0 / est_period_us (frequency in Hz, since timestamps are in microseconds)

### 2.3 Converting frequency to RPM

In main_frequency_estimation(), if freq is non-zero, it is converted to RPM:
- rpm = freq * 60.0

This dominant RPM is printed and then used for two tasks:
1) Building a highlight frame that scores each pixel based on how close its last period is to the dominant period.
2) Feeding an RPM estimate into the Kalman tracker so that its motion model depends on the propeller speed.

---

## 3. How RPM is used for pose prediction

### 3.1 Kalman filter state (pose representation)

The class BoxKalmanTracker maintains a Kalman filter for a single bounding box representing the drone pose.
The state vector is: [cx, cy, vx, vy, w, h]^T, where:
- cx, cy: center of the bounding box in image coordinates
- vx, vy: velocities in x and y
- w, h: width and height of the box

The measurement vector is [cx, cy, w, h]^T, i.e. the observed center and size of the drone bounding box.
The tracker is initialized when the first drone_box (cluster-based bounding box of all rotors) is available.

### 3.2 Feeding RPM into the tracker

For each processed window, once RPM is computed, it is passed into the tracker via:
- tracker.update_rpm(rpm, t_rec_s)

The tracker stores a short history of (time, rpm) pairs. From the last two entries it computes:
- rpm_level: current RPM value
- rpm_deriv: approximate derivative of RPM (dRPM/dt), i.e. the trend

### 3.3 RPM-aware process noise

Before each prediction, the tracker calls _update_process_noise_with_rpm(). This function:
- Normalizes rpm_level to a 0-1 range using a MAX_RPM constant.
- Normalizes rpm_deriv to a bounded range using RPM_DERIV_REF.
- Computes a velocity scale factor vel_scale that increases with RPM level and positive RPM trend.
- Scales the processNoiseCov entries that correspond to vx and vy by vel_scale^2.
- Slightly scales the process noise on positions (cx, cy) according to RPM as well.

Intuitively:
- When RPM is low and not changing much, the expected motion is small, so the Kalman filter uses small
  velocity noise. The pose is encouraged to remain stable.
- When RPM is high or rapidly increasing, the drone can move more aggressively, so velocity noise is increased,
  allowing the predicted pose to change more quickly.

In addition, when rpm_level is very low and rpm_deriv is small, the code explicitly damps the velocity states:
- vx and vy are multiplied by a factor (e.g. 0.7), further pulling the model toward a nearly static drone.

### 3.4 Pose prediction and correction

In the main loop:
- If the tracker is initialized, it first calls predicted_box = tracker.predict(t_rec_s).
- The predicted bounding box is drawn on the frame as the current estimated pose.
- If a new drone_box measurement is available for this window, tracker.correct(drone_box) is called,
  which aligns the Kalman state with the observed bounding box.

Thus, the drone pose is continuously estimated from a combination of:
- The RPM-aware motion model (prediction step)
- The rotor-based bounding box detections (correction step)

### 3.5 Future pose distribution (visualization)

The helper function draw_future_gaussian_distribution() uses the current Kalman state and recent velocities to
generate a probabilistic heatmap of possible future drone positions over a chosen time horizon. It repeatedly:
- Extrapolates the center forward in time using the velocity state.
- Draws Gaussian-like blobs around the predicted centers.
- Accumulates these blobs into a heatmap that is overlayed on the frame.

Although RPM is not used directly inside this function, the velocities and state it uses come from the
RPM-aware Kalman filter, so the spread and direction of the future distribution implicitly depend on RPM.

---

## Summary

- Data is read from a .dat event file and split into time windows. Each window provides a batch of events
  (time, x, y, polarity) for analysis.
- RPM is estimated by measuring per-pixel event periods, filling a decaying histogram of periods, and taking
  the peak as the dominant period, which is converted to frequency and then RPM.
- The estimated RPM drives an RPM-aware Kalman filter by modulating the process noise, controlling how much
  motion the model expects. High RPM allows more dynamic motion, while low RPM enforces stability.
- The Kalman filter predicts and corrects a bounding box that represents the drone pose, and an additional
  visualization can display a future probability distribution of possible poses.
