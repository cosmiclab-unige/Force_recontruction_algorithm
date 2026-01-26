import numpy as np
 
 
class ForceReconstructor:
    """
    Class to perform force reconstruction from raw PVDF sensor data.
    """
 
    def __init__(
        self,
        NW: int = 20, # window size for Average calculation
        Thr_samples: int = 1500, # number of samples to compute initial threshold
        press_sigma: float = 7.0, # multiplier for standard deviation to set press threshold
        reset_confirm: int = 5, # number of samples to confirm reset
        press_confirm: int = 5, # number of samples to confirm press
        reset_band_scale: float = 1.0, # scale for reset band
        avg_multiplier: float = 1.0, # multiplier for average to determine states
        samples_artifact: int = 0, # number of samples to hold after peak detection
        alpha: float = 1.0, # smoothing factor (1.0 = no smoothing)
        nSamples_adaptive_offset: int = 50, # number of samples for adaptive offset calculation
        debug: bool = False, 
        signal2noise_ratio : float = 10.0, # minimum signal to noise ratio for valid event
    ):
        self.NW = NW
        self.Thr_samples = Thr_samples
        self.press_sigma = press_sigma
        self.reset_confirm = reset_confirm
        self.press_confirm = press_confirm
        self.reset_band_scale = reset_band_scale
        self.avg_multiplier = avg_multiplier
        self.samples_artifact = samples_artifact
        self.alpha = alpha
        self.debug = debug
        self.signal2noise_ratio = signal2noise_ratio
 
        self.pre_trigger_len = self.press_confirm + 5
        self.pre_trigger_buf = np.zeros(self.pre_trigger_len)
 
        self.thr_press = None
        self.max_noise = None
 
        self.averagetouch = None
        self.avergeplateau = None
 
        self.nSamples_adaptive_offset = nSamples_adaptive_offset
        self.buffer_offset = np.zeros(nSamples_adaptive_offset)
        self.adaptive_offset = 0.0
 
    # --------------------------------------------------------
    def smooth_signal(self, signal_raw):
        if self.alpha >= 1.0:
            return signal_raw.copy()
 
        out = np.empty_like(signal_raw)
        out[0] = signal_raw[0]
        a = self.alpha
        for i in range(1, len(signal_raw)):
            out[i] = a * signal_raw[i] + (1 - a) * out[i - 1]
        return out
 
    # --------------------------------------------------------
    def compute_thresholds(self, signal_raw):
        thr_samples = signal_raw[: self.Thr_samples]
        sigma0 = np.std(thr_samples, ddof=1)
 
        self.max_noise = np.max(thr_samples)
        self.thr_press = self.press_sigma * sigma0
 
        self.buffer_offset[:] = thr_samples[-self.nSamples_adaptive_offset :]
        self.adaptive_offset = np.mean(self.buffer_offset)
 
    # --------------------------------------------------------
    def set_label(self, avg):
        if avg >= self.avg_multiplier * self.averagetouch:
            return "press"
        elif avg <= -self.avg_multiplier * self.averagetouch:
            return "release"
        elif abs(avg) <= self.avergeplateau:
            return "plateau"
        else:
            return "transition"
 
    # --------------------------------------------------------
    def integral(self, data_raw: np.ndarray, sensor_idx: int):
 
        # ---------- local bindings (speed) ----------
        NW = self.NW
        thr_press = self.thr_press
        reset_confirm = self.reset_confirm
        press_confirm = self.press_confirm
        pre_len = self.pre_trigger_len
        reset_band_scale = self.reset_band_scale
 
        # ---------- input ----------
        signal_raw = data_raw[:, sensor_idx].astype(float)
        N = len(signal_raw) - self.Thr_samples
        if N < NW:
            raise ValueError("Not enough samples")
 
        signal_raw = self.smooth_signal(signal_raw)
        self.compute_thresholds(signal_raw)
        signal = signal_raw[self.Thr_samples :]
 
        thr_press = self.thr_press
        num_of_samples_avg = int(round(0.6 * NW))
        half = NW // 2
        half_avg = num_of_samples_avg // 2
        idx1 = 0
        idx2 = half_avg

        # ---------- buffers ----------
        raw_signal_buffer = np.zeros(NW)
        buffer = np.zeros(NW)
        buffer_integral_pre_trigger = np.zeros(pre_len)
        integral_out = np.zeros(N)
 
        # ---------- state ----------
        integral = 0.0
        counter = 0
        confirm_press_count = 0
        previous_polarity = 0
 
        first_cross = False
        sign = 1
        first_window_done = False
 
        states = ["", "", ""]
        idx_states = 0
 
        reset_integral = False
        in_release_mode = False
        quiet_count = 0
 
        # ---------- HOLD ----------
        hold_duration = self.samples_artifact
        hold_counter = 0
        held_integral_value = 0.0
        last_integral = 0.0
        peak_detected = False
 
        # --------- SN ----------
        max_post_trigger = 0.0
        valid_event = True
        noise_level = 0.0
 
        # ---------- loop ----------
        for k in range(N):
            x_raw = signal[k] - self.adaptive_offset
            noise_level = abs(self.max_noise - self.adaptive_offset)
 
            self.pre_trigger_buf[:-1] = self.pre_trigger_buf[1:]
            self.pre_trigger_buf[-1] = x_raw
 
            raw_signal_buffer[:-1] = raw_signal_buffer[1:]
            raw_signal_buffer[-1] = x_raw
 
            # ---------- adaptive offset ----------
            if abs(x_raw) < thr_press:
                previous_polarity = 0
                x_raw_not_norm = x_raw + self.adaptive_offset
                old = self.buffer_offset[0]
                self.buffer_offset[:-1] = self.buffer_offset[1:]
                self.buffer_offset[-1] = x_raw_not_norm
                self.adaptive_offset += (x_raw_not_norm - old) / self.nSamples_adaptive_offset
 
            # ---------- reset ----------
            if reset_integral:
                if x_raw > self.thr_press:
                    previous_polarity = 1
                elif x_raw < -self.thr_press:
                    previous_polarity = -1
                else:
                    previous_polarity = 0
                integral = 0.0
                integral_out[k] = 0.0
                buffer[:] = 0.0
                counter = 0
                idx_states = 0
                states[:] = ["", "", ""]
                first_cross = False
                sign = 1
                first_window_done = False
                in_release_mode = False
                quiet_count = 0
                hold_counter = 0
                confirm_press_count = 0
                held_integral_value = 0.0
                last_integral = 0.0
                peak_detected = False
                reset_integral = False
                valid_event = True
                max_post_trigger = 0.0
                continue
 
            # ---------- threshold crossing ----------
            if not first_cross:
                if x_raw > thr_press and previous_polarity != 1:
                    confirm_press_count += 1
                    if confirm_press_count >= press_confirm:
                        first_cross = True
                        sign = 1
                        quiet_band = reset_band_scale * thr_press
                        integral = np.sum(self.pre_trigger_buf)
                        buffer_integral_pre_trigger = np.cumsum(self.pre_trigger_buf)
                        confirm_press_count = 0
 
                elif x_raw < -thr_press and previous_polarity != -1:
                    confirm_press_count += 1
                    if confirm_press_count >= press_confirm:
                        first_cross = True
                        sign = -1
                        quiet_band = -reset_band_scale * thr_press
                        integral = np.sum(-self.pre_trigger_buf)
                        buffer_integral_pre_trigger = np.cumsum(-self.pre_trigger_buf)
                        confirm_press_count = 0
                else:
                    confirm_press_count = 0
 
            if not first_cross:
                integral_out[k] = 0.0
                integral = 0.0
                counter = 0
                continue
 
            if abs(x_raw) > max_post_trigger:
                max_post_trigger = abs(x_raw)
 
            # ---------- HOLD ----------
            if hold_counter > 0:
                integral_out[k] = held_integral_value
                hold_counter -= 1
                continue
 
            # ---------- integration ----------
            if valid_event:
                x_int = sign * x_raw
                integral += x_int
                if integral < 0:
                    integral = 0.0
            else:
                integral = 0.0
 
            integral_out[k] = integral
 
 
            # ---------- HOLD trigger ----------
            if not peak_detected:
                current_state = states[idx_states] if idx_states > 0 else ""
                if current_state in ("press", "plateau"):
                    if integral <= last_integral + 1e-6:
                        hold_counter = hold_duration
                        held_integral_value = integral
                        peak_detected = True
 
            last_integral = integral
 
            # ---------- buffer ----------
            if first_window_done:
                buffer[counter] = integral
            else:
                if counter == 0:
                    buffer[:pre_len] = buffer_integral_pre_trigger
                else:
                    buffer[counter + pre_len - 1] = integral
 
            counter += 1
 
            # ---------- release ----------
            if in_release_mode:
                if (x_raw - quiet_band) * sign < 0:
                    quiet_count += 1
                else:
                    quiet_count = 0
 
                if quiet_count >= reset_confirm or integral <= 0:
                    reset_integral = True
                    continue
 
            if first_cross and integral <= 0.0:
                reset_integral = True
                continue
 
            # ---------- window ----------
            cond1 = (counter == NW - pre_len + 1 if not first_window_done else counter == NW)
 
            if cond1:
                avg = (-buffer[idx1] + buffer[idx2]) / num_of_samples_avg
                counter = 0
 
                if not first_window_done:
                    if max_post_trigger < self.signal2noise_ratio * noise_level:
                        valid_event = False
                        integral = 0.0
                        integral_out[:k+1] = 0.0
                        continue
 
                    a = abs(avg) if abs(avg) > 1e-12 else 1e-12
                    scale = 1
                    while round(a) == 0:
                        a *= 10
                        scale *= 10
 
                    self.averagetouch = round(avg * scale) / scale
                    self.avergeplateau = self.averagetouch / 10
                    first_window_done = True
                    states[0] = self.set_label(avg)
 
                    if self.debug:
                        print(f"Press detected @ {k}, Average {avg}")
                    continue
 
                st = self.set_label(avg)
 
                if idx_states < 2:
                    idx_states += 1
                    states[idx_states] = st
                else:
                    states[0], states[1], states[2] = states[1], states[2], st
 
                if st == "release":
                    in_release_mode = True
                    if self.debug:
                        print(f"Release detected @ {k}")
 
        return {
            "smoothed_signal": signal,
            "integral": integral_out,
            "thr_upper": self.thr_press,
            "thr_lower": -self.thr_press,
            "averagetouch": self.averagetouch,
            "avergeplateau": self.avergeplateau,
            "reset_confirm": self.reset_confirm,
            "reset_band_scale": self.reset_band_scale,
        }
 
 