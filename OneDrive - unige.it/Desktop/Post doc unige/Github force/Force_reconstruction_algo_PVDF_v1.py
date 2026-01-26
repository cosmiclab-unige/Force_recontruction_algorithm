import numpy as np

class ForceReconstructor:
    def __init__(
        self,
        NW: int = 200,          # Sliding window length for average estimation
        Thr_samples: int = 1500, # Number of initial samples for thresholding
        n_sigma: float = 7.0,     # Threshold multiplier (N·σ above/below mean)
    ):
        # Configuration
        self.NW = NW
        self.Thr_samples = Thr_samples
        self.n_sigma = n_sigma

        # Threshold variables 
        self.mu0 = None
        self.thr_upper = None
        self.thr_lower = None

        # Global scale parameters inferred from the first window AFTER touch
        self.averagetouch = None     # Typical integration during touch (press or release)
        self.avergeplateau = None    # Tolerance region around plateau

    # ---------------------------------------------------------
    def compute_thresholds(self, signal_raw):
        """
        Compute mean and +/- n_sigma thresholds from the initial 1500 samples.
        These are used to detect the first touch.
        """
        Thr_samples = signal_raw[: self.Thr_samples]
        self.mu0 = np.mean(Thr_samples)
        sigma0 = np.std(Thr_samples, ddof=1)

        # Upper/ thresholds
        self.thr_upper = self.n_sigma * sigma0
        self.thr_lower = -self.n_sigma * sigma0

    # ---------------------------------------------------------
    def set_label(self, avg, averagetouch, avergeplateau):
        """
        Convert average avg into a symbolic state label:
            press      = strong positive 
            release    = strong negative 
            plateau    = slope near zero
            transition = intermediate, neither touch nor release
        """
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
        """
        Check if the last 3 window states indicate a return-to-zero (reset).
        Condition: first two are 'release' AND the third is any 'low activity' state.
        """
        return (
            states[0] == "release"
            and states[1] == "release"
            and (states[2] in ["release", "transition", "plateau"])
        )

    # ---------------------------------------------------------
    def integral(self, data_raw: np.ndarray, sensor_idx: int):
        """
        Main reconstruction pipeline:
        1. Compute thresholds.
        2. Detect first threshold crossing.
        3. Start integrating.
        4. Use sliding window to classify touch / release / plateau transitions.
        5. Reset when a full release sequence is detected.
        """
        signal_raw = data_raw[:, sensor_idx].astype(float)
        N = len(signal_raw)

        # Must have enough samples for threshold
        if N < self.Thr_samples + self.NW:
            raise ValueError("Not enough samples")

        # Compute thresholds
        self.compute_thresholds(signal_raw)

        # remove offset online should be removed from each samples
        signal = signal_raw - self.mu0

        # Output integral array
        integral = 0.0
        integral_out = np.zeros(N)

        # Sliding window buffer for estimating average 
        buffer = np.zeros(self.NW)
        counter = 0

        # Flags
        first_cross = False       # Have we crossed threshold yet?
        sign = +1                

        # Sliding-window state machine
        first_window_done = False # True once the first window after touch is processed
        states = ["", "", ""]     # Last 3 state labels
        idx_states = 0            # Index for filling the states array

        reset_integral = False    # Set to True when release is detected

        # ---------------------------------------------------------
        # MAIN LOOP
        # ---------------------------------------------------------
        for k in range(N):

            x = signal[k]

            # --------------------------------------
            # RESET BLOCK (called after full release)
            # --------------------------------------
            if reset_integral:
                # Reset integral output
                integral = 0
                integral_out[k] = 0

                # Reset sliding window state
                counter = 0
                idx_states = 0
                states = ["", "", ""]
                first_window_done = False

                # Return to pre-touch state
                first_cross = False
                sign = +1
                buffer[:] = 0

                reset_integral = False
                continue

            # --------------------------------------
            # THRESHOLD CROSS DETECTION
            # --------------------------------------
            if not first_cross:
                # Detect first cross
                if x > self.thr_upper or x < self.thr_lower:
                    first_cross = True

                    # Determine direction of movement
                    sign = -1 if x < self.thr_lower else +1

            # BEFORE first threshold crossing: do NOT integrate
            if not first_cross:
                integral_out[k] = 0.0
                integral = 0.0
                counter = 0
                continue

            # --------------------------------------
            # AFTER FIRST CROSS → TRUE INTEGRATION
            # --------------------------------------
            # Align sign so integration always increases for touch
            x = sign * x

            integral += x

            # Prevent negative drift
            if integral < 0:
                integral = 0

            integral_out[k] = integral

            # Store in sliding buffer
            buffer[counter] = integral
            counter += 1

            # --------------------------------------
            # PROCESS FULL SLIDING WINDOW
            # --------------------------------------
            if counter == self.NW:

                # Average estimation: diff(window_end - window_start) / window_length
                avg = (buffer[-1] - buffer[0]) / self.NW
                counter = 0

                # -------------------------
                # FIRST WINDOW AFTER TOUCH
                # -------------------------
                if not first_window_done:

                    # Determine amplitude scale for labeling
                    if self.averagetouch is None:
                        self.averagetouch = 10 ** np.floor(np.log10(avg))
                        self.avergeplateau = self.averagetouch / 10

                    first_window_done = True

                    # Label this first window
                    states[0] = self.set_label(avg, self.averagetouch, self.avergeplateau)

                    print(
                        f"FIRST WINDOW  avg={avg:.6f}, "
                        f"averagetouch={self.averagetouch:.6f}, avergeplateau={self.avergeplateau:.6f}, "
                        f"states={states}"
                    )
                    continue

                # -------------------------
                # NORMAL WINDOWS
                # -------------------------
                st = self.set_label(avg, self.averagetouch, self.avergeplateau)

                # Fill state buffer (3 latest states)
                if idx_states < 2:
                    idx_states += 1
                    states[idx_states] = st
                else:
                    # Shift left and append new state
                    states[0] = states[1]
                    states[1] = states[2]
                    states[2] = st

                # Check if we must reset (i.e., full release detected)
                reset_integral = self.check_states(states)

                print(
                    f"AVG={avg:.6f}, averagetouch={self.averagetouch:.6f}, "
                    f"avergeplateau={self.avergeplateau:.6f}, "
                    f"label={st}, states={states}, reset={reset_integral}"
                )

        # ---------------------------------------------------------
        # RETURN RESULTS
        # ---------------------------------------------------------
        return {
            "integral": integral_out,
            "thr_upper": self.thr_upper,
            "thr_lower": self.thr_lower,
            "averagetouch": self.averagetouch,
            "avergeplateau": self.avergeplateau,
        }
