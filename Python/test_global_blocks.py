import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from force_reconstructor_using_blocks import ForceReconstructor


# ==================================================
# CONFIG
# ==================================================
file_path = Path(__file__).parent.parent / "Dataset" / "sensor_force_data.csv"

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
    press_sigma=20,
    alpha=1,
    slope_multiplier=1,
    nSamples_adaptive_offset=50,
    press_confirm=5,
    samples_after_release=500,
    debug=True,
    min_press_sensors=6,
    release_ratio=0.5,
    signal2noise_ratio=10,
)


# ==================================================
# RUN BLOCK PROCESSING (OFFLINE TEST)
# ==================================================

BLOCK_SIZE = 100  # prova 20, 50, 100

print(f"Processing in blocks of {BLOCK_SIZE} samples")

integrals = np.zeros((n_samples, n_sensors))

for start in range(0, n_samples, BLOCK_SIZE):
    stop = min(start + BLOCK_SIZE, n_samples)
    block = raw_data[start:stop]

    if single_sensor:
        block = block[:, sensor_to_process:sensor_to_process+1]

    out = fr.process_block(block)

    integrals[start:stop] = out


# ==================================================
# COLLECT OUTPUTS (NaN padded)
# ==================================================

print("Force reconstruction completed")


# ==================================================
# PLOT (ONLY RECONSTRUCTED OUTPUT)
# ==================================================

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# ---------- Top subplot: RAW ----------
if single_sensor:
    axs[0].plot(
        raw_data[:, 0]-np.mean(raw_data[:,0]),
        linewidth=1.5,
        label=f"Raw S{sensor_to_process}",
    )
else:
    for i in range(n_sensors):
        axs[0].plot(
            raw_data[:, i]-np.mean(raw_data[:,i]),
            linewidth=1.0,
            alpha=0.7,
            label=f"S{i}",
        )

axs[0].set_title("Raw Data")
axs[0].set_ylabel("Raw output")
axs[0].grid(True)
axs[0].legend(ncol=4, fontsize=8)

# ---------- Bottom subplot: INTEGRAL ----------
if single_sensor:
    axs[1].plot(
        integrals[:, 0],
        linewidth=1.5,
        label=f"Integral S{sensor_to_process}",
    )
else:
    for i in range(n_sensors):
        axs[1].plot(
            integrals[:, i],
            linewidth=1.0,
            alpha=0.7,
            label=f"S{i}",
        )

axs[1].set_title("Reconstructed Force (Block Processing)")
axs[1].set_xlabel("Sample index")
axs[1].set_ylabel("Integral output")
axs[1].grid(True)
axs[1].legend(ncol=4, fontsize=8)

plt.tight_layout()
plt.show()