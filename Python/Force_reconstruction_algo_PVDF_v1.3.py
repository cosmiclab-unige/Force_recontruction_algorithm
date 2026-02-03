import numpy as np


class ForceReconstructor:
    """
    Real-time force reconstruction from PVDF signals.
    Zero-latency integration with rollback, multi-event support
    and post-release guard time.
    """

    def __init__(
        self,
        NW: int = 200,
        fifo_buffer_lenght: int = 50,
        Thr_samples: int = 1500,
        press_sigma: float = 7.0,
        press_confirm: int = 5,
        slope_multiplier: float = 0.15,
        alpha: float = 0.1,
        nSamples_adaptive_offset: int = 50,
        samples_after_release: int = 100,   # 👈 GUARDIA
        debug: bool = False,
        signal2noise_ratio: float = 10.0,
    ):
        self.NW = NW
        self.fifo_buffer_lenght = fifo_buffer_lenght
        self.Thr_samples = Thr_samples
        self.press_sigma = press_sigma
        self.press_confirm = press_confirm
        self.slope_multiplier = slope_multiplier
        self.alpha = alpha
        self.debug = debug
        self.signal2noise_ratio = signal2noise_ratio

        self.pre_trigger_len = press_confirm + 5
        self.nSamples_adaptive_offset = nSamples_adaptive_offset
        self.samples_after_release = samples_after_release

    # --------------------------------------------------
    def smooth_signal(self, x):
        if self.alpha >= 1.0:
            return x.copy()

        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = self.alpha * x[i] + (1 - self.alpha) * y[i - 1]
        return y

    # --------------------------------------------------
    def compute_thresholds(self, signal):
        noise = signal[: self.Thr_samples]
        sigma = np.std(noise, ddof=1)

        self.thr_press = self.press_sigma * sigma
        self.max_noise = np.mean(np.sort(noise)[int(0.95 * len(noise)) :])

        self.offset_buffer = noise[-self.nSamples_adaptive_offset :].copy()
        self.adaptive_offset = np.mean(self.offset_buffer)

    # --------------------------------------------------
    def integral(self, data_raw: np.ndarray, sensor_idx: int):

        # ---------- INPUT ----------
        signal_raw = data_raw[:, sensor_idx].astype(float)
        signal_raw = self.smooth_signal(signal_raw)
        self.compute_thresholds(signal_raw)

        signal = signal_raw[self.Thr_samples :]
        N = len(signal)

        if N < self.NW:
            raise ValueError("Not enough samples")

        # ---------- BUFFERS ----------
        pre_buf = np.zeros(self.pre_trigger_len)
        press_buf = np.zeros(self.NW)
        integral_out = np.zeros(N)
        fifo = np.zeros(self.fifo_buffer_lenght)

        # ---------- STATE ----------
        integral = 0.0
        counter = 0
        confirm = 0
        fifo_idx = 0
        idx_second_cross = 0
        second_cross = False

        triggered = False
        validated = False
        armed = True

        press_sign = 1
        max_post = 0.0
        event_start_idx = None

        guard_counter = 0   # 👈 GUARD TIMER

        # ---------- LOOP ----------
        for k in range(N):
            x = signal[k] - self.adaptive_offset

            # ======================================================
            # POST-RELEASE GUARD
            # ======================================================
            if guard_counter > 0:
                guard_counter -= 1
                integral_out[k] = 0.0
                continue

            # --- adaptive offset ---
            if abs(x) < self.thr_press:
                old = self.offset_buffer[0]
                self.offset_buffer[:-1] = self.offset_buffer[1:]
                self.offset_buffer[-1] = x + self.adaptive_offset
                self.adaptive_offset += (self.offset_buffer[-1] - old) / self.nSamples_adaptive_offset

            # --- pre-trigger buffer ---
            pre_buf[:-1] = pre_buf[1:]
            pre_buf[-1] = x

            # ======================================================
            # TRIGGER DETECTION
            # ======================================================
            if not triggered:
                if armed and abs(x) > self.thr_press:
                    confirm += 1
                    if confirm >= self.press_confirm:
                        triggered = True
                        armed = False
                        confirm = 0

                        press_sign = np.sign(np.sum(pre_buf))
                        if press_sign == 0:
                            press_sign = np.sign(x)

                        integral = np.sum(press_sign * pre_buf)
                        press_buf[: self.pre_trigger_len] = np.cumsum(press_sign * pre_buf)
                        counter = self.pre_trigger_len

                        event_start_idx = k - self.pre_trigger_len + 1
                        if event_start_idx < 0:
                            event_start_idx = 0
                else:
                    confirm = 0

                integral_out[k] = 0.0
                continue

            # ======================================================
            # INTEGRATION (REAL-TIME)
            # ======================================================
            integral += press_sign * x
            if integral < 0:
                integral = 0.0

            integral_out[k] = integral
            max_post = max(max_post, abs(x))

            if counter < self.NW:
                press_buf[counter] = integral
                counter += 1

            if np.abs(x) < self.thr_press and not second_cross:
                idx_second_cross = np.mod(k, self.NW)
                second_cross = True

            # ======================================================
            # PRESS VALIDATION
            # ======================================================
            if not validated and counter == self.NW:
                idx = np.argmax(press_buf[:idx_second_cross])
                idx_second_cross = 0
                second_cross = False
                avg = (-press_buf[0] + press_buf[idx]) / (idx + 1)

                if self.debug:
                    print(f"Slope PRESS @ {k} -- slope {avg:.6f}")

                noise = abs(self.max_noise - self.adaptive_offset)
                if max_post < self.signal2noise_ratio * noise:
                    if self.debug:
                        print("Event discarded (low S/N) -- max value:", max_post, "noise level:", noise)

                    integral_out[event_start_idx : k + 1] = 0.0

                    triggered = False
                    validated = False
                    integral = 0.0
                    counter = 0
                    confirm = 0
                    fifo_idx = 0
                    fifo[:] = 0.0
                    max_post = 0.0
                    event_start_idx = None

                    guard_counter = self.samples_after_release
                    continue

                self.averagetouch = avg
                self.avergeplateau = avg / 10
                validated = True

            # ======================================================
            # RELEASE DETECTION (SLOPE ON INTEGRAL)
            # ======================================================
            if validated:
                if fifo_idx < self.fifo_buffer_lenght:
                    fifo[fifo_idx] = integral
                    fifo_idx += 1
                else:
                    fifo[:-1] = fifo[1:]
                    fifo[-1] = integral

                if fifo_idx == self.fifo_buffer_lenght or integral == 0.0:
                    avg = (fifo[-1] - fifo[0]) / self.fifo_buffer_lenght
                    if avg < -self.slope_multiplier * abs(self.averagetouch) or integral == 0.0:
                        if self.debug:
                            print(f"Release detected @ {k} -- slope {avg:.6f}")

                        triggered = False
                        validated = False
                        armed = True

                        integral = 0.0
                        counter = 0
                        confirm = 0
                        fifo_idx = 0
                        fifo[:] = 0.0
                        max_post = 0.0
                        event_start_idx = None

                        guard_counter = self.samples_after_release
                        continue

        return {
            "smoothed_signal": signal,
            "integral": integral_out,
            "thr_upper": self.thr_press,
            "thr_lower": -self.thr_press,
            "averagetouch": getattr(self, "averagetouch", None),
            "avergeplateau": getattr(self, "avergeplateau", None),
        }
