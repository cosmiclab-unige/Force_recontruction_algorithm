import numpy as np

class ForceReconstructor:
    def __init__(
        self,
        NW: int = 200,             # Sliding window length for average estimation
        Thr_samples: int = 1500,  # Number of initial samples for thresholding
        n_sigma: float = 7.0,     # Threshold multiplier (N·σ above/below mean)
        reset_confirm: int = 5,  # Consecutive "quiet raw" samples required to reset
        reset_band_scale: float = 1.0,  # Quiet band = reset_band_scale * thr_upper
    ):
        # Configuration
        self.NW = NW
        self.Thr_samples = Thr_samples
        self.n_sigma = n_sigma
        self.reset_confirm = reset_confirm
        self.reset_band_scale = reset_band_scale

        # Threshold variables
        self.mu0 = None
        self.thr_upper = None
        self.thr_lower = None

        # Global scale parameters inferred from the first window AFTER touch
        self.averagetouch = None
        self.avergeplateau = None

    # ---------------------------------------------------------
    def compute_thresholds(self, signal_raw):
        """Compute mean and +/- n_sigma thresholds from the initial samples."""
        Thr_samples = signal_raw[: self.Thr_samples]
        self.mu0 = np.mean(Thr_samples)
        sigma0 = np.std(Thr_samples, ddof=1)

        # thresholds in the offset-removed domain (since we subtract mu0 later)
        self.thr_upper = self.n_sigma * sigma0
        self.thr_lower = -self.n_sigma * sigma0

    # ---------------------------------------------------------
    def set_label(self, avg, averagetouch, avergeplateau):
        """Convert avg slope into a symbolic state label."""
        if avg >= averagetouch:
            return "press"
        elif avg <= -averagetouch:
            return "release"
        elif abs(avg) <= avergeplateau:
            return "plateau"
        else:
            return "transition"

    # ---------------------------------------------------------
    def check_states(self, states):
        """Original reset condition based on last 3 window labels."""
        return (
            states[0] == "release"
            and states[1] == "release"
            and (states[2] in ["release", "transition", "plateau"])
        )

    # ---------------------------------------------------------
    def integral(self, data_raw: np.ndarray, sensor_idx: int):
        """
        Modified reconstruction:
        - Detect first threshold crossing, start integrating.
        - Label windows from integral slope.
        - When release is detected, ONLY reset once the RAW (offset-removed) signal
          stays inside the threshold band for `reset_confirm` consecutive samples.
        """
        signal_raw = data_raw[:, sensor_idx].astype(float)
        N = len(signal_raw)

        if N < self.Thr_samples + self.NW:
            raise ValueError("Not enough samples")

        # thresholds + offset removal
        self.compute_thresholds(signal_raw)
        signal = signal_raw - self.mu0

        integral = 0.0
        integral_out = np.zeros(N)

        buffer = np.zeros(self.NW)
        counter = 0

        first_cross = False
        sign = +1

        first_window_done = False
        states = ["", "", ""]
        idx_states = 0

        reset_integral = False

        # NEW: release gating + raw-quiet confirmation
        in_release_mode = False
        quiet_count = 0
        quiet_band = self.reset_band_scale * self.thr_upper  # symmetric band +/-quiet_band

        for k in range(N):
            x_raw = signal[k]  # offset-removed raw sample (DO NOT apply sign here)

            # -------------------------
            # RESET BLOCK
            # -------------------------
            if reset_integral:
                integral = 0.0
                integral_out[k] = 0.0

                counter = 0
                idx_states = 0
                states = ["", "", ""]
                first_window_done = False

                first_cross = False
                sign = +1
                buffer[:] = 0.0

                # reset new gating vars
                in_release_mode = False
                quiet_count = 0

                reset_integral = False
                continue

            # -------------------------
            # THRESHOLD CROSS DETECTION
            # -------------------------
            if not first_cross:
                if x_raw > self.thr_upper or x_raw < self.thr_lower:
                    first_cross = True
                    sign = -1 if x_raw < self.thr_lower else +1

            if not first_cross:
                integral_out[k] = 0.0
                integral = 0.0
                counter = 0
                continue

            # -------------------------
            # AFTER FIRST CROSS: integrate with sign aligned
            # -------------------------
            x_int = sign * x_raw
            integral += x_int
            if integral < 0:
                integral = 0.0
            integral_out[k] = integral

            buffer[counter] = integral
            counter += 1

            # -------------------------
            # If we're in release mode, confirm raw is quiet before resetting
            # -------------------------
            if in_release_mode:
                if abs(x_raw) < quiet_band:
                    quiet_count += 1
                else:
                    quiet_count = 0

                if quiet_count >= self.reset_confirm:
                    reset_integral = True
                    continue  # next loop will execute reset block

            # -------------------------
            # PROCESS FULL SLIDING WINDOW
            # -------------------------
            if counter == self.NW:
                avg = (buffer[-1] - buffer[0]) / self.NW
                counter = 0

                if not first_window_done:
                    # robust: avoid log10 issues if avg <= 0
                    if self.averagetouch is None:
                        a = abs(avg) if abs(avg) > 1e-12 else 1e-12
                        self.averagetouch = 10 ** np.floor(np.log10(a))
                        self.avergeplateau = self.averagetouch / 10

                    first_window_done = True
                    states[0] = self.set_label(avg, self.averagetouch, self.avergeplateau)

                    print(
                        f"FIRST WINDOW  avg={avg:.6f}, "
                        f"averagetouch={self.averagetouch:.6f}, avergeplateau={self.avergeplateau:.6f}, "
                        f"states={states}"
                    )
                    continue

                st = self.set_label(avg, self.averagetouch, self.avergeplateau)

                # update 3-state history
                if idx_states < 2:
                    idx_states += 1
                    states[idx_states] = st
                else:
                    states[0] = states[1]
                    states[1] = states[2]
                    states[2] = st

                # NEW: enter release mode as soon as we label "release"
                if st == "release":
                    in_release_mode = True

                # We do NOT hard-reset immediately from labels anymore.
                # Instead, we wait for raw-quiet confirmation (in_release_mode block above).
                # You can still *optionally* require your old pattern to START release mode:
                # if self.check_states(states): in_release_mode = True

                print(
                    f"AVG={avg:.6f}, averagetouch={self.averagetouch:.6f}, "
                    f"avergeplateau={self.avergeplateau:.6f}, "
                    f"label={st}, states={states}, "
                    f"in_release_mode={in_release_mode}, quiet_count={quiet_count}"
                )

        return {
            "integral": integral_out,
            "thr_upper": self.thr_upper,
            "thr_lower": self.thr_lower,
            "averagetouch": self.averagetouch,
            "avergeplateau": self.avergeplateau,
            "reset_confirm": self.reset_confirm,
            "reset_band_scale": self.reset_band_scale,
        }
