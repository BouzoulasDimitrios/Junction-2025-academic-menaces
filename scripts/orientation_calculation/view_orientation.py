import numpy as np
import matplotlib.pyplot as plt

RESULTS_FILE = "orientation_results.npz"

def main():
    data = np.load(RESULTS_FILE)
    times = data["times"]
    angles_deg = data["angles_deg"]
    target_freq_hz = float(data["target_freq_hz"])

    print(f"Loaded {RESULTS_FILE}")
    print(f"Target frequency: {target_freq_hz:.2f} Hz")
    print(f"Number of samples: {len(times)}")

    plt.figure(figsize=(8, 4))
    plt.plot(times, angles_deg, marker="o")
    plt.xlabel("Time [s]")
    plt.ylabel("Orientation [deg]")
    plt.title(f"Orientation vs time @ {target_freq_hz:.1f} Hz")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


