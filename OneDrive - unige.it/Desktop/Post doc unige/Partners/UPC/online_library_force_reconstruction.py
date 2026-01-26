import numpy as np
import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===========================================================
#                     SERIAL CONFIG
# ===========================================================

SERIAL_PORT = "COM4"
BAUDRATE = 1000000

HEADER = b'\x3C\x3E\x00'

NUM_SENSORS = 8
BYTES_PER_SENSOR = 2
PAYLOAD_LEN = NUM_SENSORS * BYTES_PER_SENSOR

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.05)
time.sleep(0.05)

def send_cmd(cmd: str):
    msg = (cmd + "\n").encode('ascii')
    ser.write(msg)
    ser.flush()
    time.sleep(0.05)

send_cmd("num1,9")
send_cmd("filtoff")

print("Collecting baseline noise, DO NOT touch the sensor yet...")

# ===========================================================
#                     SENSOR READER
# ===========================================================

def read_from_sensor(chunk_size=20, sensor_index=0):
    data = np.zeros((chunk_size, 1), dtype=float)

    for i in range(chunk_size):
        # sync to header
        while True:
            if ser.read(1) != HEADER[0:1]: continue
            if ser.read(1) != HEADER[1:2]: continue
            if ser.read(1) != HEADER[2:3]: continue
            break

        payload = ser.read(PAYLOAD_LEN)
        if len(payload) != PAYLOAD_LEN:
            return None

        raw16 = np.frombuffer(payload, dtype="<u2")
        mV = raw16 * (5000.0 / 65535.0)
        data[i, 0] = mV[sensor_index]

    return data

# ===========================================================
#                     FORCE DETECTOR
# ===========================================================

def detect_force_events(
    data_raw: np.ndarray,
    sensor_idx: int,
    win_len: int = 200,
    baseline_len: int = 1500,
    n_sigma: float = 4.0,
    min_consecutive_neg: int = 3,
    min_consecutive_pos: int = 3,
    eps: float = 1e-4,
):

    signal_raw = data_raw[:, sensor_idx].astype(float)

    N = len(signal_raw)
    if N < baseline_len:
        return None

    baseline_raw = signal_raw[:baseline_len]
    mu0 = np.mean(baseline_raw)
    sigma0 = np.std(baseline_raw, ddof=1)

    thr_sig_upper = mu0 + n_sigma * sigma0
    thr_sig_lower = mu0 - n_sigma * sigma0

    in_band = (signal_raw >= thr_sig_lower) & (signal_raw <= thr_sig_upper)

    # mean-removed signal (what we want to plot)
    signal = signal_raw - mu0

    mask_cross = (signal_raw > thr_sig_upper) | (signal_raw < thr_sig_lower)
    first_cross_idx = int(np.argmax(mask_cross)) if np.any(mask_cross) else None

    cum_signal = np.zeros(N)
    if first_cross_idx is not None:
        cum_signal[first_cross_idx:] = np.cumsum(signal[first_cross_idx:])
    else:
        cum_signal[:] = np.cumsum(signal)

    windows = []
    for start in range(0, N - win_len + 1, win_len):
        end = start + win_len
        slope = (cum_signal[end - 1] - cum_signal[start]) / (win_len - 1)
        windows.append((start, end, slope))

    slopes = np.array([w[2] for w in windows])
    neg_flags = slopes < -eps
    pos_flags = slopes > eps

    def find_first_pos_run_start(after_idx):
        for i, (ws, _, _) in enumerate(windows):
            if ws >= after_idx:
                break
        else:
            return None

        while i < len(windows):
            if pos_flags[i]:
                j = i
                while j < len(windows) and pos_flags[j]:
                    j += 1
                if (j - i) >= min_consecutive_pos:
                    return windows[i][0]
                i = j
            else:
                i += 1
        return None

    events = []
    i = 0
    nW = len(windows)

    while i < nW:
        if neg_flags[i]:
            j = i
            while j < nW and neg_flags[j]:
                j += 1

            if (j - i) >= min_consecutive_neg:
                run_start = windows[i][0]
                run_end = windows[j - 1][1] - 1

                last_out_idx = None
                reentry_idx = None

                for k in range(run_start + 1, run_end + 1):
                    if in_band[k] and not in_band[k - 1]:
                        last_out_idx = k - 1
                        reentry_idx = k
                        break

                if last_out_idx is not None:
                    events.append({
                        "run_start": run_start,
                        "run_end": run_end,
                        "last_out_idx": last_out_idx,
                        "reentry_idx": reentry_idx,
                    })

            i = j
        else:
            i += 1

    zero_mask = np.zeros(N, dtype=bool)
    out_of_band = ~in_band
    if np.any(out_of_band):
        first_out = np.where(out_of_band)[0][0]
        zero_mask[:first_out] = True

    for ev in events:
        last_out = ev["last_out_idx"]
        pos_start = find_first_pos_run_start(last_out + 1)

        if pos_start is None:
            zero_mask[last_out + 1:] = True
        else:
            zero_mask[last_out + 1:pos_start] = True

    cum_signal_reset = np.zeros_like(signal)
    cs = 0.0

    touch_active = False
    force_end_detected = False

    for k in range(N):

        raw_in_band = bool(in_band[k])

        if not touch_active:
            if raw_in_band:
                cs = 0
                cum_signal_reset[k] = 0
                continue
            else:
                touch_active = True
                force_end_detected = False
                cs += signal[k]
                cum_signal_reset[k] = max(cs, 0)
                continue

        if zero_mask[k]:
            force_end_detected = True

        if force_end_detected and raw_in_band:
            touch_active = False
            force_end_detected = False
            cs = 0
            cum_signal_reset[k] = 0
            continue

        cs += signal[k]
        cum_signal_reset[k] = max(cs, 0)

    # ---- UPDATED RETURN ----
    return {
        "signal_raw": signal_raw,
        "signal_demeaned": signal,   # <-- new
        "thr_upper": thr_sig_upper,
        "thr_lower": thr_sig_lower,
        "first_cross_idx": first_cross_idx,
        "cum_signal": cum_signal,
        "cum_signal_reset": cum_signal_reset,
        "events": events,
        "zero_mask": zero_mask,
    }


# ===========================================================
#                     ONLINE WRAPPER
# ===========================================================

BUFFER_LEN = 10000

class OnlineWrapper:
    def __init__(self, baseline_len=1500):
        self.buffer = None
        self.baseline_len = baseline_len
        self.baseline_done = False

    def update(self, chunk):
        if chunk is None:
            return None

        if self.buffer is None:
            self.buffer = chunk.copy()
        else:
            self.buffer = np.vstack([self.buffer, chunk])
            if len(self.buffer) > BUFFER_LEN:
                self.buffer = self.buffer[-BUFFER_LEN:]

        if (not self.baseline_done) and (len(self.buffer) >= self.baseline_len):
            print("Baseline done – NOW you can touch the sensor.")
            self.baseline_done = True

        return detect_force_events(self.buffer, 0, baseline_len=self.baseline_len)

det = OnlineWrapper(baseline_len=1500)

# ===========================================================
#                     PLOTTING
# ===========================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))

raw_line,      = ax1.plot([], [], 'b')
thr_hi_line,   = ax1.plot([], [], 'r--')
thr_lo_line,   = ax1.plot([], [], 'g--')
cum_line,      = ax2.plot([], [], 'r')

ax1.set_title("DEMEANED SENSOR SIGNAL + THRESHOLDS")
ax1.set_ylabel("Signal [mV - mean]")
ax2.set_title("RESET CUMSUM")
ax2.set_ylabel("CUMSUM")
ax2.set_xlabel("Samples")

def init():
    raw_line.set_data([], [])
    thr_hi_line.set_data([], [])
    thr_lo_line.set_data([], [])
    cum_line.set_data([], [])
    return raw_line, thr_hi_line, thr_lo_line, cum_line

def update(frame):
    chunk = read_from_sensor(50)
    res = det.update(chunk)

    if res is None:
        return raw_line, thr_hi_line, thr_lo_line, cum_line

    raw = res["signal_demeaned"]          # <-- plot mean-removed

    cum = res["cum_signal_reset"]

    # shift thresholds by mean
    base_mean = np.mean(res["signal_raw"][:det.baseline_len])
    thr_hi = res["thr_upper"] - base_mean
    thr_lo = res["thr_lower"] - base_mean

    x = np.arange(len(raw))

    raw_line.set_data(x, raw)
    thr_hi_line.set_data(x, np.full_like(x, thr_hi))
    thr_lo_line.set_data(x, np.full_like(x, thr_lo))
    cum_line.set_data(x, cum)

    ax1.set_xlim(0, len(raw))
    ax2.set_xlim(0, len(cum))

    ax1.relim()
    ax1.autoscale_view(scalex=False, scaley=True)

    ax2.relim()
    ax2.autoscale_view(scalex=False, scaley=True)

    return raw_line, thr_hi_line, thr_lo_line, cum_line

ani = FuncAnimation(
    fig, update, init_func=init,
    interval=150, blit=False
)

plt.tight_layout()
plt.show()
