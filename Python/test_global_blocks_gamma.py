import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from force_reconstructor_using_blocks_Hannes import ForceReconstructor


# ==================================================
# CONFIG
# ==================================================
file_path = Path(__file__).parent.parent / "Hannes_Dataset" / "pvdf_unified"

sensor_to_process = -1    # -1 = tutti i sensori
Thr_samples = 1500
n_sensors_dataset = 8

# ==================================================
# READ DATA
# ==================================================
df = pd.read_csv(
    file_path,
    sep=r"\t",        # whitespace separator
    header=None,        # no header in file
    decimal=","        # comma as decimal separator
)
# --------------------------------------------------
# Timestamp handling
# --------------------------------------------------
timestamp = df.iloc[:, 0].astype(float).values   # first column
timestamp = timestamp - timestamp[0]             # subtract first value
time_axis = timestamp / 1e6                 # optional: ms -> seconds

raw_data = df.iloc[:, 1:n_sensors_dataset+1].astype(float).values  # columns 32..47 (16 sensors)

n_samples, n_sensors = raw_data.shape
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
    NW=500,
    Thr_samples=Thr_samples,
    press_sigma=10,
    alpha=0.1,
    nSamples_adaptive_offset=50,
    press_confirm=2,
    samples_after_release=500,
    min_press_sensors=3,
    release_ratio=0.5,
    signal2noise_ratio=10,
)


# ==================================================
# RUN BLOCK PROCESSING (OFFLINE TEST)
# ==================================================

BLOCK_SIZE = 100  # prova 20, 50, 100

print(f"Processing in blocks of {BLOCK_SIZE} samples")

integrals = np.zeros((n_samples, n_sensors))
measurements = np.zeros((n_samples, n_sensors))

for start in range(0, n_samples, BLOCK_SIZE):
    stop = min(start + BLOCK_SIZE, n_samples)
    block = raw_data[start:stop]

    if single_sensor:
        block = block[:, sensor_to_process:sensor_to_process+1]

    out, meas = fr.process_block(block)

    integrals[start:stop] = out
    measurements[start:stop] = meas


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
        measurements[Thr_samples:, 0],
        linewidth=1.5,
        label=f"Raw S{sensor_to_process}",
    )
else:
    for i in range(n_sensors):
        axs[0].plot(
            measurements[Thr_samples:, i],
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
        integrals[Thr_samples:, 0],
        linewidth=1.5,
        label=f"Integral S{sensor_to_process}",
    )
else:
    for i in range(n_sensors):
        axs[1].plot(
            integrals[Thr_samples:, i],
            linewidth=1.0,
            alpha=0.7,
            label=f"S{i}",
        )

# ---------- Plot thresholds ----------
if single_sensor:
    axs[0].axhline(
        fr.thr_press[0],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )
else:
    for i in range(n_sensors):
        axs[0].axhline(
            fr.thr_press[i],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )
        axs[0].axhline(
            -fr.thr_press[i],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )


axs[1].set_title("Reconstructed Force (Block Processing)")
axs[1].set_xlabel("Sample index")
axs[1].set_ylabel("Integral output")
axs[1].grid(True)
axs[1].legend(ncol=4, fontsize=8)

plt.tight_layout()
plt.show()