import sys
import numpy as np
import matplotlib.pyplot as plt

from metavision_core.event_io import EventsIterator  # from Metavision/OpenEB


def events_to_frame(events, height, width):
    """
    Turn a batch of events into a simple grayscale frame.
    Positive events -> brighter, negative -> darker.
    """
    # Start from mid gray
    frame = np.zeros((height, width), dtype=np.int16)

    xs = events["x"]
    ys = events["y"]
    ps = events["p"]  # 0 or 1

    # Map polarity {0,1} -> {-1,+1}
    ps = ps * 2 - 1

    # Accumulate at pixel locations
    np.add.at(frame, (ys, xs), ps)

    # Clip / normalize to 0–255
    if frame.max() == frame.min():
        return np.full((height, width), 127, dtype=np.uint8)

    frame = np.clip(frame, -5, 5)  # avoid huge ranges for dense bursts
    frame = (frame - frame.min()) / (frame.max() - frame.min())
    frame = (frame * 255).astype(np.uint8)
    return frame


def play_event_file(path, delta_t_us=50000):
    """
    Visualize events from a RAW or DAT file.

    delta_t_us: time slice in microseconds (50 000 µs = 50 ms per frame)
    """
    mv_it = EventsIterator(input_path=path, mode="delta_t", delta_t=delta_t_us)
    height, width = mv_it.get_size()

    if height is None or width is None:
        raise RuntimeError("Could not get sensor size from file; "
                           "check that this is a valid event file.")

    plt.ion()
    fig, ax = plt.subplots()
    img_disp = ax.imshow(
        np.zeros((height, width), dtype=np.uint8),
        cmap="gray",
        vmin=0,
        vmax=255,
        animated=True,
    )
    ax.set_title(f"{path} (Δt = {delta_t_us/1e3:.1f} ms)")
    ax.axis("off")

    for events in mv_it:
        if events.size == 0:
            continue

        frame = events_to_frame(events, height, width)
        img_disp.set_data(frame)
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_events.py path/to/file.raw|file.dat")
        sys.exit(1)

    play_event_file(sys.argv[1])
