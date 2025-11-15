import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

CUBE_FILE = "event_cube.npz"


def main():
    data = np.load(CUBE_FILE)
    cube = data["cube"]      # shape (T, H, W)
    dt = float(data["dt"])

    T, H, W = cube.shape
    print(f"Loaded cube: T={T}, H={H}, W={W}, dt={dt:.2e}s")

    # remove per-pixel DC to emphasize oscillations
    cube_zero_mean = cube - cube.mean(axis=0, keepdims=True)

    # FFT along time axis: result shape (F, H, W)
    spec = np.fft.rfft(cube_zero_mean, axis=0)
    freqs = np.fft.rfftfreq(T, d=dt)
    amp = np.abs(spec)

    # initial frequency: pick max global response (excluding DC)
    power_per_freq = amp.sum(axis=(1, 2))
    power_per_freq[0] = 0.0
    init_idx = int(np.argmax(power_per_freq))
    print(f"Initial peak frequency ≈ {freqs[init_idx]:.1f} Hz")

    # --- Matplotlib UI ---
    fig, (ax_spec, ax_img) = plt.subplots(1, 2, figsize=(10, 4))

    # spectrum plot
    ax_spec.plot(freqs, power_per_freq)
    ax_spec.set_xlabel("Frequency [Hz]")
    ax_spec.set_ylabel("Global power")
    ax_spec.set_title("Global spatio-temporal spectrum")

    # initial amplitude image
    im = ax_img.imshow(amp[init_idx], cmap="inferno")
    ax_img.set_title(f"Amplitude map at ≈ {freqs[init_idx]:.1f} Hz")
    ax_img.axis("off")
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    # slider
    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax,
                    label="Frequency [Hz]",
                    valmin=freqs[1],
                    valmax=freqs[-1],
                    valinit=freqs[init_idx])

    def update(val):
        f_target = slider.val
        idx = int(np.argmin(np.abs(freqs - f_target)))
        im.set_data(amp[idx])
        ax_img.set_title(f"Amplitude map at ≈ {freqs[idx]:.1f} Hz")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
