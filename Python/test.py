import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Force_reconstruction_algo_PVDF import ForceReconstructor   

from pathlib import Path

file_path = Path(__file__).parent.parent / "Dataset" / "usecase_exp_peach.csv"

# --------------------------------------------------
# Read data
# --------------------------------------------------
df = pd.read_csv(file_path)

timestamp_col = df.columns[0]
df["Time"] = pd.to_datetime(df[timestamp_col], unit="s", errors="coerce")
df = df.dropna(subset=["Time"]).reset_index(drop=True)

df["Time_sec"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
time_axis = df["Time_sec"].values

# --------------------------------------------------
# Detect sensor columns
# --------------------------------------------------
raw_cols = [c for c in df.columns if c.lower().startswith("raw")]
raw_data = df[raw_cols].values   # shape: (N_samples, N_sensors)

n_samples, n_sensors = raw_data.shape
print(f"Detected {n_sensors} sensors")

# --------------------------------------------------
# Apply force reconstruction to ALL sensors
# --------------------------------------------------
Thr_samples = 1500
fr = ForceReconstructor(NW=30, Thr_samples=Thr_samples, press_sigma=10, alpha=0.05, reset_band_scale=1, slope_multiplier=0.3,
                        nSamples_adaptive_offset=50, press_confirm=5, reset_confirm=5, samples_artifact=2000, debug=True, signal2noise_ratio=10)

n_samples = n_samples - Thr_samples
time_axis = time_axis[Thr_samples:]

integrals = np.zeros((n_samples, n_sensors))
thr_upper = np.zeros(n_sensors)
thr_lower = np.zeros(n_sensors)
signal_smooth_all = np.zeros((n_samples, n_sensors))

ssss = 30

for sensor_idx in range(n_sensors):
    if sensor_idx != ssss and ssss >= 0:
        continue  # test only sensor 8
    res = fr.integral(raw_data, sensor_idx)
    integrals[:, sensor_idx] = res["integral"]
    signal_smooth_all[:, sensor_idx] = res["smoothed_signal"]
    thr_upper[sensor_idx] = res["thr_upper"]
    thr_lower[sensor_idx] = res["thr_lower"]

print("Force reconstruction applied to all sensors")


# ---------- Raw signals ----------
plt.figure(figsize=(14, 8))


# ---------- Raw signals + thresholds ----------
plt.subplot(2, 1, 1)
for i in range(n_sensors):
    if i != ssss and ssss >= 0:
        continue  # test only sensor 8
    raw_centered = signal_smooth_all[:, i] - np.mean(signal_smooth_all[:, i])

    # Raw signal
    plt.plot(raw_centered, alpha=0.5, label=f"S{i}")

    # Thresholds for this sensor
    plt.axhline(thr_upper[i], linestyle="--", alpha=0.4)
    plt.axhline(thr_lower[i], linestyle="--", alpha=0.4)
    plt.grid(True)

plt.title("PVDF Raw Signals with Thresholds (All Sensors)")
plt.ylabel("Voltage (v)")
plt.legend(ncol=4, fontsize=8)

# ---------- Integral signals (unchanged) ----------
plt.subplot(2, 1, 2)
for i in range(n_sensors):
    if i != ssss and ssss >= 0:
        continue  # test only sensor 8
    plt.plot(integrals[:, i], alpha=0.7, label=f"Integral{i}")

plt.title("Reconstructed Integrals (All Sensors)")
plt.xlabel("Time (seconds)")
plt.ylabel("Integral output")
plt.legend(ncol=4, fontsize=8, loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.show()