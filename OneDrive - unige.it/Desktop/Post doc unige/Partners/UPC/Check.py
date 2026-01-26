import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "test_noise_grounded.csv"

# Read and skip the header comment
df = pd.read_csv(file_path, skiprows=1)

# Convert UNIX timestamp to datetime
timestamp_col = df.columns[0]
df['Time'] = pd.to_datetime(df[timestamp_col], unit='s')

# --- Remove the first 40 samples ---
df = df.iloc[100:]   # keep rows from index 40 onward

# Count samples for each second
samples_per_second = df.groupby(df['Time'].dt.floor('S')).size()

print("\nSamples in each second:")
for t, count in samples_per_second.items():
    print(f"{t} -> {count} samples")

# Average sample rate
average_samples_per_second = samples_per_second.mean()
print(f"\nAverage samples per second (Hz): {average_samples_per_second:.2f}")

# --- Create time axis in seconds ---
df['Time_sec'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

# --- Plot sensors ---
time_axis = df['Time_sec']

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- First subplot: first 16 sensors ---
for col in df.columns[1:17]:  # first 16 sensor columns
    signal = df[col] - df[col].mean()  # baseline removal (optional)
    axs[0].plot( signal, label=col)

axs[0].set_title("PVDF Tactile Sensor Readings (Sensors 1–16)")
axs[0].set_ylabel("Sensor Output (V)")
axs[0].grid(True)
axs[0].legend(loc='upper right', ncol=4, fontsize=8)

# --- Second subplot: remaining sensors ---
for col in df.columns[17:-2]:  # remaining sensor columns
    signal = df[col] - df[col].mean()
    axs[1].plot( signal, label=col)
    

axs[1].set_title("PVDF Tactile Sensor Readings (Sensors 17–32)")
axs[1].set_xlabel("Time (seconds)")
axs[1].set_ylabel("Sensor Output (V)")
axs[1].grid(True)
axs[1].legend(loc='upper right', ncol=4, fontsize=8)

plt.tight_layout()
plt.show()
