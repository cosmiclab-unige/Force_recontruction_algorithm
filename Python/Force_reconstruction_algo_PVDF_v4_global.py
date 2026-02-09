import numpy as np


class ForceReconstructor:
    """
    REAL-TIME, MULTI-SENSOR
    LOGICALLY IDENTICAL to v3 (offline).
    """

    def __init__(
        self,
        n_sensors,
        NW=200,
        fifo_buffer_length=50,
        Thr_samples=1500,
        press_sigma=7.0,
        press_confirm=5,
        slope_multiplier=0.15,
        alpha=0.1,
        nSamples_adaptive_offset=50,
        samples_after_release=100,
        debug=False,
        signal2noise_ratio=10.0,
    ):
        self.n_sensors = n_sensors
        self.NW = NW
        self.fifo_buffer_length = fifo_buffer_length
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

        # ---------------- NOISE ----------------
        self.noise_counter = 0
        self.noise_buffer = np.zeros((Thr_samples, n_sensors))

        # ---------------- BUFFERS ----------------
        self.offset_buffer = np.zeros((nSamples_adaptive_offset, n_sensors))
        self.adaptive_offset = np.zeros(n_sensors)

        self.pre_buf = np.zeros((self.pre_trigger_len, n_sensors))
        self.press_buf = np.zeros((NW, n_sensors))
        self.fifo = np.zeros((fifo_buffer_length, n_sensors))

        # ---------------- STATE ----------------
        self.integral = np.zeros(n_sensors)
        self.counter = np.zeros(n_sensors, dtype=int)
        self.confirm = np.zeros(n_sensors, dtype=int)
        self.fifo_idx = np.zeros(n_sensors, dtype=int)

        self.idx_second_cross = np.zeros(n_sensors, dtype=int)
        self.second_cross = np.zeros(n_sensors, dtype=bool)

        self.triggered = np.zeros(n_sensors, dtype=bool)
        self.validated = np.zeros(n_sensors, dtype=bool)
        self.event_discarded = np.zeros(n_sensors, dtype=bool)
        self.armed = np.ones(n_sensors, dtype=bool)

        self.press_sign = np.ones(n_sensors)
        self.max_post = np.zeros(n_sensors)
        self.press_integral = np.zeros(n_sensors)
        self.averagetouch = np.zeros(n_sensors)

        self.guard_counter = np.zeros(n_sensors, dtype=int)
        self.event_start_idx = np.full(n_sensors, None)
        self.previous_event_polarity = np.zeros(n_sensors)

        self.data_raw_prec = np.zeros(n_sensors)
        self.sample_idx = 0

        # ---------------- OUTPUT ----------------
        self.integral_out = [[] for _ in range(n_sensors)]
        self.smoothed_signal = [[] for _ in range(n_sensors)]

    # --------------------------------------------------
    def compute_thresholds(self):
        noise = self.noise_buffer
        sigma = np.std(noise, axis=0, ddof=1)
        mean = np.median(noise, axis=0)

        self.thr_press = self.press_sigma * sigma
        self.max_noise = np.mean(
            np.sort(np.abs(noise - mean), axis=0)[int(0.99 * len(noise)) :],
            axis=0,
        )

        self.offset_buffer[:] = noise[-self.nSamples_adaptive_offset :]
        self.adaptive_offset = np.mean(self.offset_buffer, axis=0)

    # --------------------------------------------------
    def integral_step(self, x_raw):
        """
        x_raw: array shape (n_sensors,)
        """
        self.sample_idx += 1

        # ---------- NOISE LEARNING ----------
        if self.noise_counter < self.Thr_samples:
            self.noise_buffer[self.noise_counter] = x_raw
            self.noise_counter += 1

            if self.noise_counter == self.Thr_samples:
                self.compute_thresholds()
                self.data_raw_prec = x_raw.copy()

            for ns in range(self.n_sensors):
                self.integral_out[ns].append(0.0)
            return

        # ---------- MAIN LOOP ----------
        for ns in range(self.n_sensors):

            # ---- GUARD ----
            if self.guard_counter[ns] > 0:
                self.guard_counter[ns] -= 1
                self.integral_out[ns].append(0.0)
                continue

            # ---- SMOOTH ----
            sm = self.alpha * x_raw[ns] + (1 - self.alpha) * self.data_raw_prec[ns]
            self.data_raw_prec[ns] = sm
            x = sm - self.adaptive_offset[ns]
            self.smoothed_signal[ns].append(x)

            # ---- ADAPTIVE OFFSET ----
            thr = self.thr_press[ns] * 10 / self.press_sigma
            if abs(x) < thr:
                old = self.offset_buffer[0, ns]
                self.offset_buffer[:-1, ns] = self.offset_buffer[1:, ns]
                self.offset_buffer[-1, ns] = x + self.adaptive_offset[ns]
                self.adaptive_offset[ns] += (self.offset_buffer[-1, ns] - old) / self.nSamples_adaptive_offset

            # ---- PRE BUFFER ----
            self.pre_buf[:-1, ns] = self.pre_buf[1:, ns]
            self.pre_buf[-1, ns] = x

            # ================= TRIGGER =================
            if not self.triggered[ns]:
                if self.armed[ns] and abs(x) > self.thr_press[ns]:
                    self.confirm[ns] += 1
                    if self.confirm[ns] >= self.press_confirm:
                        self.triggered[ns] = True
                        self.armed[ns] = False
                        self.confirm[ns] = 0

                        self.second_cross[ns] = False
                        self.idx_second_cross[ns] = 0

                        self.press_sign[ns] = np.sign(np.sum(self.pre_buf[:, ns]))
                        if self.press_sign[ns] == 0:
                            self.press_sign[ns] = np.sign(x)

                        if (
                            self.event_discarded[ns]
                            and self.press_sign[ns] != self.previous_event_polarity[ns]
                        ):
                            self._reset(ns, guard=True)
                            self.integral_out[ns].append(0.0)
                            continue

                        self.event_discarded[ns] = False
                        self.integral[ns] = np.sum(self.press_sign[ns] * self.pre_buf[:, ns])
                        self.press_buf[: self.pre_trigger_len, ns] = np.cumsum(
                            self.press_sign[ns] * self.pre_buf[:, ns]
                        )
                        self.counter[ns] = self.pre_trigger_len
                        self.previous_event_polarity[ns] = self.press_sign[ns]
                        self.event_start_idx[ns] = self.sample_idx
                else:
                    self.confirm[ns] = 0

                self.integral_out[ns].append(0.0)
                continue

            # ================= INTEGRATION =================
            self.integral[ns] += self.press_sign[ns] * x

            if self.integral[ns] < 0:
                self.event_discarded[ns] = True
                self.previous_event_polarity[ns] *= -1
                self._reset(ns, guard=True)
                self.integral_out[ns].append(0.0)
                if self.debug:
                    print(
                        f"[ROLLBACK] t={self.sample_idx:6d} "
                        f"S{ns:02d} integral<0"
                    )
                continue

            self.integral_out[ns].append(self.integral[ns])
            self.max_post[ns] = max(self.max_post[ns], abs(x))

            if self.counter[ns] < self.NW:
                self.press_buf[self.counter[ns], ns] = self.integral[ns]
                self.counter[ns] += 1

            # ================= SECOND CROSS =================
            if abs(x) < self.thr_press[ns] and not self.second_cross[ns]:
                self.idx_second_cross[ns] = self.counter[ns]
                self.second_cross[ns] = True
                self.press_integral[ns] = self.integral[ns]

            # ================= VALIDATION =================
            if not self.validated[ns] and self.counter[ns] == self.NW:
                idx = np.argmax(self.press_buf[: self.idx_second_cross[ns], ns])
                avg = (-self.press_buf[0, ns] + self.press_buf[idx, ns]) / (idx + 1)

                if self.max_post[ns] < self.signal2noise_ratio * self.max_noise[ns]:
                    self.event_discarded[ns] = True
                    self.integral_out[ns][-self.counter[ns] :] = [0.0] * self.counter[ns]
                    self._reset(ns, guard=False)
                    if self.debug:
                        print(
                            f"[DISCARD] t={self.sample_idx:6d} "
                            f"S{ns:02d} LOW SNR"
                        )
                    continue

                self.averagetouch[ns] = avg
                self.validated[ns] = True
                if self.debug:
                    print(
                        f"[VALIDATED] t={self.sample_idx:6d} "
                        f"S{ns:02d} avg_touch={avg:.6f} "
                    )

            # ================= RELEASE =================
            if self.validated[ns]:
                if self.fifo_idx[ns] < self.fifo_buffer_length:
                    self.fifo[self.fifo_idx[ns], ns] = self.integral[ns]
                    self.fifo_idx[ns] += 1
                else:
                    self.fifo[:-1, ns] = self.fifo[1:, ns]
                    self.fifo[-1, ns] = self.integral[ns]

                if self.fifo_idx[ns] == self.fifo_buffer_length or self.integral[ns] == 0:
                    avg = (self.fifo[-1, ns] - self.fifo[0, ns]) / self.fifo_buffer_length
                    if (
                        avg < -self.slope_multiplier * abs(self.averagetouch[ns])
                        or self.integral[ns] > self.press_integral[ns] * 1.5
                        or self.integral[ns] < self.press_integral[ns] * 0.5
                    ):
                        self._reset(ns, guard=True)
                        if self.debug:
                            print(
                                f"[RELEASE] t={self.sample_idx:6d} "
                                f"S{ns:02d} avg_slope={avg:.6f} "
                            )

    # --------------------------------------------------
    def _reset(self, ns, guard):
        self.integral[ns] = 0.0
        self.counter[ns] = 0
        self.confirm[ns] = 0
        self.fifo_idx[ns] = 0
        self.fifo[:, ns] = 0.0
        self.max_post[ns] = 0.0
        self.press_integral[ns] = 0.0
        self.triggered[ns] = False
        self.validated[ns] = False
        self.armed[ns] = True
        self.event_start_idx[ns] = None
        self.second_cross[ns] = False
        self.idx_second_cross[ns] = 0

        self.guard_counter[ns] = self.samples_after_release if guard else 0
