import numpy as np


class ForceReconstructor:

    def __init__(
        self,
        n_sensors,
        NW=1000,
        Thr_samples=1500,
        warmup_samples=500,
        press_sigma=10.0,
        press_confirm=5,
        alpha=1.0,
        nSamples_adaptive_offset=50,
        samples_after_release=500,
        signal2noise_ratio=10.0,
        min_press_sensors=4,
        release_ratio=0.5,
    ):

        self.n_sensors = n_sensors
        self.NW = NW
        self.Thr_samples = Thr_samples
        self.press_sigma = press_sigma
        self.press_confirm = press_confirm
        self.alpha = alpha
        self.nSamples_adaptive_offset = nSamples_adaptive_offset
        self.samples_after_release = samples_after_release
        self.signal2noise_ratio = signal2noise_ratio
        self.min_press_sensors = min_press_sensors
        self.release_ratio = release_ratio

        # ---------------- Noise ----------------
        self.warmup_samples = warmup_samples
        self.warmup_counter = 0
        self.noise_counter = 0
        self.noise_buffer = np.zeros((Thr_samples, n_sensors))

        # ---------------- Adaptive offset ----------------
        self.offset_buffer = np.zeros((nSamples_adaptive_offset, n_sensors))
        self.adaptive_offset = np.zeros(n_sensors)

        # ---------------- Local state ----------------
        self.phase = np.zeros(n_sensors, dtype=int)  # 0 idle,1 press,2 hold,3 release
        self.integral = np.zeros(n_sensors)
        self.max_integral = np.zeros(n_sensors)
        self.counter = np.zeros(n_sensors, dtype=int)
        self.confirm = np.zeros(n_sensors, dtype=int)
        self.guard_counter = np.zeros(n_sensors, dtype=int)

        # ---------------- Global state ----------------
        self.global_press_active = False
        self.global_phase = 0
        self.press_local_mask = np.zeros(n_sensors, dtype=bool)
        self.release_local_mask = np.zeros(n_sensors, dtype=bool)

        # ---------------- Signal memory ----------------
        self.data_raw_prec = np.zeros(n_sensors)

        # ---------------- Output ----------------
        self.current_output = np.zeros(n_sensors)

    # ==================================================
    def compute_thresholds(self):

        noise = self.noise_buffer
        mean = np.median(noise, axis=0)
        sigma = np.std(noise, axis=0, ddof=1)

        self.thr_press = self.press_sigma * sigma

        self.max_noise = np.mean(
            np.sort(np.abs(noise - mean), axis=0)[int(0.99 * len(noise)):],
            axis=0,
        )

        self.offset_buffer[:] = noise[-self.nSamples_adaptive_offset:]
        self.adaptive_offset = np.mean(self.offset_buffer, axis=0)

    # ==================================================
    def integral_step(self, x_raw):

        self.current_output[:] = 0.0
        # ---------------- Warm-up (scarto iniziale) ----------------
        if self.warmup_counter < self.warmup_samples:
            self.warmup_counter += 1
            self.data_raw_prec = x_raw.copy()
            return self.current_output.copy()

        # ---------------- Noise learning ----------------
        if self.noise_counter < self.Thr_samples:

            self.noise_buffer[self.noise_counter] = x_raw
            self.noise_counter += 1

            if self.noise_counter == self.Thr_samples:
                self.compute_thresholds()
                self.data_raw_prec = x_raw.copy()

            return self.current_output.copy()

        # ==================================================
        for ns in range(self.n_sensors):

            # -------- Filtering --------
            sm = self.alpha * x_raw[ns] + (1 - self.alpha) * self.data_raw_prec[ns]
            self.data_raw_prec[ns] = sm
            x = sm - self.adaptive_offset[ns]

            # -------- Guard --------
            if self.guard_counter[ns] > 0:
                self.guard_counter[ns] -= 1
                continue

            # ================= IDLE =================
            if self.phase[ns] == 0:

                if self.global_phase == 1:
                    continue

                if abs(x) > self.thr_press[ns]:
                    self.confirm[ns] += 1
                else:
                    self.confirm[ns] = 0

                if self.confirm[ns] >= self.press_confirm:
                    self.phase[ns] = 1
                    self.integral[ns] = 0.0
                    self.counter[ns] = 0
                    self.confirm[ns] = 0

                continue

            # ================= PRESS =================
            if self.phase[ns] == 1:

                self.integral[ns] += abs(x)

                # aggiorna massimo in tempo reale
                self.max_integral[ns] = max(self.integral[ns], self.max_integral[ns])

                self.counter[ns] += 1

                if self.counter[ns] >= self.NW:

                    if self.integral[ns] > self.signal2noise_ratio * self.max_noise[ns]:

                        self.phase[ns] = 2
                        self.press_local_mask[ns] = True

                        if (not self.global_press_active and
                            np.count_nonzero(self.press_local_mask) >= self.min_press_sensors):

                            self.global_press_active = True
                            self.global_phase = 1
                    else:
                        self._reset_local(ns)

                continue

            # ================= HOLD =================
            if self.phase[ns] == 2:

                if abs(x) > self.thr_press[ns]:
                    self.confirm[ns] += 1
                else:
                    self.confirm[ns] = 0

                if self.confirm[ns] >= self.press_confirm:
                    self.phase[ns] = 3
                    self.confirm[ns] = 0

                continue

            # ================= RELEASE =================
            if self.phase[ns] == 3:

                self.integral[ns] -= abs(x)

                if self.integral[ns] <= 0.5 * self.max_integral[ns]:
                    self.release_local_mask[ns] = True
                    self._reset_local(ns)

                # ---- Global release ----
                n_press = np.count_nonzero(self.press_local_mask)
                n_release = np.count_nonzero(self.release_local_mask)

                if n_press > 0 and n_release / n_press >= self.release_ratio:
                    self._reset_all_global()

                continue

        # ================= GLOBAL OUTPUT =================
        if self.global_press_active:

            mask = self.max_integral > 1e-9
            self.current_output[mask] = (
                self.integral[mask] / self.max_integral[mask]
            )

        return self.current_output.copy()
        
    # ==================================================    
    def recalibrate_thresholds(self):
        print(">>> Ricalibrazione soglie avviata")

        # reset noise learning
        self.noise_counter = 0
        self.noise_buffer[:] = 0.0

        # reset stati globali e locali
        self._reset_all_global()

        # reset offset
        self.offset_buffer[:] = 0.0
        self.adaptive_offset[:] = 0.0

        # reset memoria filtro
        self.data_raw_prec[:] = 0.0

    # ==================================================
    def process_block(self, block):

        block = np.asarray(block)
        N = block.shape[0]

        out = np.zeros((N, self.n_sensors))

        for i in range(N):
            out[i, :] = self.integral_step(block[i])

        return out

    # ==================================================
    def _reset_local(self, ns):

        self.phase[ns] = 0
        self.integral[ns] = 0.0
        self.counter[ns] = 0
        self.confirm[ns] = 0
        self.guard_counter[ns] = self.samples_after_release

    # ==================================================
    def _reset_all_global(self):

        for ns in range(self.n_sensors):
            self._reset_local(ns)

        self.press_local_mask[:] = False
        self.release_local_mask[:] = False
        self.global_press_active = False
        self.global_phase = 0