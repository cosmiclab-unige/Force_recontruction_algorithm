import pandas as pd
import matplotlib.pyplot as plt

file_path = "softobj_exp_closeUntil.csv"

# --- Read file (this format already has proper headers on row 1) ---
df = pd.read_csv(file_path)

# --- Timestamp column (first column) ---
timestamp_col = df.columns[0]  # e.g., "Timestamp(s)"

# Convert UNIX timestamp to datetime (assumes seconds)
df["Time"] = pd.to_datetime(df[timestamp_col], unit="s", errors="coerce")

# Drop rows where timestamp couldn't be parsed (just in case)
df = df.dropna(subset=["Time"]).reset_index(drop=True)

# --- Sample rate info ---
samples_per_second = df.groupby(df["Time"].dt.floor("S")).size()

print("\nSamples in each second:")
for t, count in samples_per_second.items():
    print(f"{t} -> {count} samples")

average_samples_per_second = samples_per_second.mean()
print(f"\nAverage samples per second (Hz): {average_samples_per_second:.2f}")

# --- Time axis in seconds relative to start ---
df["Time_sec"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
time_axis = df["Time_sec"]

# --- Pick columns by name pattern ---
raw_cols = [c for c in df.columns if c.lower().startswith("raw")]
integral_cols = [c for c in df.columns if c.lower().startswith("integral")]

# --- Plot: 2 subplots (Raw on top, Integral on bottom) ---
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Raw subplot (all raws on same plot)
for col in raw_cols:
    signal = df[col] - df[col].mean()   # baseline removal (optional)
    axs[0].plot( signal, label=col)

axs[0].set_title("Raw Signals")
axs[0].set_ylabel("volage(V)")
axs[0].grid(True)
axs[0].legend(loc="lower left", ncol=6, fontsize=8)

# Integral subplot 
for col in integral_cols:
    axs[1].plot(df[col], label=col)

axs[1].set_title("Integral Signals ")
axs[1].set_xlabel("Time (seconds)")
axs[1].set_ylabel("Integral (baseline-removed)")
axs[1].grid(True)
axs[1].legend(loc="upper left", ncol=6, fontsize=8)

plt.tight_layout()
plt.show()
