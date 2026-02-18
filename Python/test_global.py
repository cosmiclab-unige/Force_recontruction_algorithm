import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from Force_reconstruction_algo_PVDF_v4_global import ForceReconstructor


# ==================================================
# CONFIG
# ==================================================
file_path = Path(__file__).parent.parent / "Dataset" / "usecase_exp2_spoon.csv"

sensor_to_process = -1    # -1 = tutti i sensori
Thr_samples = 1500


# ==================================================
# READ DATA
# ==================================================
df = pd.read_csv(file_path)

timestamp_col = df.columns[0]
df["Time"] = pd.to_datetime(df[timestamp_col], unit="s", errors="coerce")
df = df.dropna(subset=["Time"]).reset_index(drop=True)

df["Time_sec"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
time_axis = df["Time_sec"].values

raw_cols = [c for c in df.columns if c.lower().startswith("raw")]
raw_data = df[raw_cols].values

n_samples, n_sensors_dataset = raw_data.shape
print(f"Dataset sensors: {n_sensors_dataset}")


# ==================================================
# SENSOR MODE
# ==================================================
single_sensor = sensor_to_process >= 0
n_sensors = 1 if single_sensor else n_sensors_dataset

print(f"Processing {'single sensor' if single_sensor else 'all sensors'}")


# ==================================================
# FORCE RECONSTRUCTOR (REAL-TIME)
# ==================================================
fr = ForceReconstructor(
    n_sensors=n_sensors,
    NW=100,
    Thr_samples=Thr_samples,
    fifo_buffer_length=20,
    press_sigma=10,
    alpha=1,
    slope_multiplier=0.5,
    nSamples_adaptive_offset=50,
    press_confirm=5,
    samples_after_release=500,
    debug=True,
    min_press_sensors=4,
    release_ratio=0.5,
    signal2noise_ratio=10,
)


# ==================================================
# RUN REAL-TIME RECONSTRUCTION
# ==================================================
for k in range(n_samples):
    if single_sensor:
        fr.integral_step(np.array([raw_data[k, sensor_to_process]]))
    else:
        fr.integral_step(raw_data[k])


# ==================================================
# COLLECT OUTPUTS (NaN padded)
# ==================================================

# ---------- Integrals ----------
integrals = np.full((n_samples, n_sensors), np.nan)
for ns in range(n_sensors):
    integ = np.asarray(fr.integral_out[ns])
    integrals[-len(integ):, ns] = integ

# ---------- Smoothed signals ----------
signal_smooth_all = np.full((n_samples, n_sensors), np.nan)
for ns in range(n_sensors):
    sm = np.asarray(fr.smoothed_signal[ns])
    signal_smooth_all[-len(sm):, ns] = sm

# ---------- Thresholds ----------
thr_upper = fr.thr_press
thr_lower = -fr.thr_press

print("Force reconstruction completed")


# ==================================================
# PLOT
# ==================================================
plt.figure(figsize=(14, 8))

# ---------- Smoothed + thresholds ----------
plt.subplot(2, 1, 1)

if single_sensor:
    sm = signal_smooth_all[:, 0]
    sm_centered = sm - np.nanmean(sm)

    plt.plot(
        sm_centered,
        alpha=0.7,
        label=f"Smoothed S{sensor_to_process}",
    )
    plt.axhline(thr_upper[0], linestyle="--", alpha=0.5)
    plt.axhline(thr_lower[0], linestyle="--", alpha=0.5)

else:
    for i in range(n_sensors):
        sm = signal_smooth_all[:, i]
        sm_centered = sm - np.nanmean(sm)

        plt.plot(
            sm_centered,
            alpha=0.4,
            label=f"S{i}",
        )
        plt.axhline(thr_upper[i], linestyle="--", alpha=0.3)
        plt.axhline(thr_lower[i], linestyle="--", alpha=0.3)

plt.title("PVDF Smoothed Signals with Thresholds")
plt.ylabel("Voltage (v)")
plt.grid(True)
plt.legend(ncol=4, fontsize=8)


# ---------- Integrals ----------
plt.subplot(2, 1, 2)

if single_sensor:
    plt.plot(
        integrals[:, 0],
        alpha=0.8,
        label=f"Integral S{sensor_to_process}",
    )
else:
    for i in range(n_sensors):
        plt.plot(
            integrals[:, i],
            alpha=0.6,
            label=f"Integral {i}",
        )

plt.title("Reconstructed Integrals")
plt.xlabel("Time (seconds)")
plt.ylabel("Integral output")
plt.grid(True)
plt.legend(ncol=4, fontsize=8, loc="upper left")

plt.tight_layout()
plt.show()
