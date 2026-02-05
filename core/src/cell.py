import math
import numpy as np

k = 1.380649e-23  # J/K
q = 1.602176634e-19  # C
EG_EV = 1.12 # silicon bandgap

class Cell:

    def __init__(
        self,
        isc_ref,
        voc_ref,
        diode_ideality,
        r_s=0,
        r_sh=np.inf,
        # Maxeon Gen 7 (typical) defaults (STC)
        voc_temp_coeff=-0.00175,     # V/°C  (≈ -0.236%/°C of ~0.741 V)
        isc_temp_coeff=0.00374,      # A/°C  (≈ +0.058%/°C of ~6.44 A)
        irradiance=1000.0,
        temperature_c=25.0,
        vmpp=0.647,                  # V (cell Vmpp)
        impp=6.114,                  # A (cell Impp)
        autofit=False
    ):
        self.isc_ref = float(isc_ref)       # A (STC)
        self.voc_ref = float(voc_ref)       # V (STC)
        self.diode_ideality = float(diode_ideality)  # n
        self.r_s = float(r_s)               # ohm (initial guess; can be fit later)
        self.r_sh = float(r_sh)             # ohm (initial guess; can be fit later)
        self.voc_temp_coeff = float(voc_temp_coeff)  # V/°C (beta_Voc)
        self.isc_temp_coeff = float(isc_temp_coeff)  # A/°C (alpha_Isc)
        self.irradiance = float(irradiance)         # W/m^2
        self.temperature_c = float(temperature_c)   # °C
        self.vmpp = vmpp
        self.impp = impp
        self._calibrated = False

        # cached values

        self.temperature_k = None
        self.thermal_voltage = None   # n*k*T/q (often called nVt)
        self.Iph = None               # photocurrent at current G,T
        self.I0_ref = None            # saturation current at STC (computed once)
        self.I0 = None                # saturation current at current T
        self.voc_lin = None           # linear Voc(T) estimate for heuristics

        self._update_cache(init_reference=True)

        if autofit:
            self.calibrate_at_stc()
            self._calibrated = True
        
    def set_conditions(self, irradiance, temperature_c):
        self.irradiance = irradiance
        self.temperature_c = temperature_c
        self._update_cache()

    def _i0_from_temperature(self, T_k):
        """
        Scale I0 from STC to current temperature using diffusion-current model:
        I0(T) = I0_ref * (T/Tref)^3 * exp( -Eg/k * (1/T - 1/Tref) )
        """
        Tref = 25.0 + 273.15
        Eg_J = EG_EV * q                      # convert eV → Joules
        return self.I0_ref * (T_k / Tref)**3 * math.exp(-(Eg_J / k) * (1.0/T_k - 1.0/Tref))

    def _update_cache(self, init_reference=False):
        self.temperature_k = self.temperature_c + 273.15
        self.thermal_voltage = self.diode_ideality * k * self.temperature_k / q

        # Photocurrent scales with G and alpha_Isc (temperature):
        Iph_ref_25 = self.isc_ref
        Iph_T = Iph_ref_25 + self.isc_temp_coeff * (self.temperature_c - 25.0)
        self.Iph = Iph_T * (self.irradiance / 1000.0)

        # A simple linear Voc(T) estimate (optional, just for heuristics/guesses):
        self.voc_lin = max(0.0, self.voc_ref + self.voc_temp_coeff * (self.temperature_c - 25.0))

        if init_reference:
            # Compute I0_ref at STC one time, using your existing STC equation:
            TrefK = 25.0 + 273.15
            Vt_ref = (k * TrefK) / q
            nVt_ref = self.diode_ideality * Vt_ref
            denom_ref = math.exp(self.voc_ref / max(nVt_ref, 1e-30)) - 1.0
            self.I0_ref = (self.isc_ref - self.voc_ref / max(self.r_sh, 1e-12)) / max(denom_ref, 1e-30)

        # Runtime: scale I0 from I0_ref using current T (no mixing with linearized Voc):
        self.I0 = self._i0_from_temperature(self.temperature_k)



    def calibrate_at_stc(self, **fit_kwargs):
        """
        One-time calibration of Rs, Rsh using datasheet (Isc,Voc,Vmpp,Impp) at STC.
        Keeps Rs/Rsh fixed afterward. Restores original G,T when done.
        """
        # stash current operating conditions
        G0, T0 = self.irradiance, self.temperature_c

        # set STC for calibration
        self.set_conditions(irradiance=1000.0, temperature_c=25.0)

        # run your fitter (scipy version you implemented)
        rs, rsh, info = self.fit_rs_rsh_scipy((1e-4, 0.05), (500.0, 5000.0), start_from="grid", **fit_kwargs)

        self._calibrated = True

        # restore previous conditions and caches
        self.set_conditions(G0, T0)
        return rs, rsh, info



    def solve_i_at_v(self, v, i0=None, iph=None, max_iter=80, tol=1e-10, i_init=None):
        # Cache attributes and functions locally to reduce attribute lookups
        r_sh = self.r_sh
        if np.isinf(r_sh) or r_sh <= 0:
            inv_r_sh = 0.0
        else:
            inv_r_sh = 1.0 / r_sh

        if i0 is None:
            i0 = self.I0
        if iph is None:
            iph = self.Iph

        rs = self.r_s

        # initial linear approximation
        denom_lin = 1.0 + rs * inv_r_sh
        if i_init is None:
            i = (iph - v * inv_r_sh) / denom_lin if denom_lin > 0 else iph
        else:
            i = float(i_init)

        # small helpers cached locally
        clip = np.clip
        exp = math.exp
        isfinite = np.isfinite
        abs_f = abs

        def clamp_I(x):
            # wide bounds to avoid boxing Newton too hard
            lo = -10.0 * abs_f(iph) - 1e-6
            hi =  2.0  * abs_f(iph) + 1e-6
            return min(max(x, lo), hi)

        i = clamp_I(i)

        nVt = max(self.thermal_voltage, 1e-30)

        for _ in range(max_iter):
            z = (v + i * rs) / nVt
            z = float(clip(z, -100.0, 100.0))
            e = exp(z)

            F = i - iph + i0 * (e - 1.0) + (v + i * rs) * inv_r_sh
            dF = 1.0 + i0 * e * (rs / nVt) + rs * inv_r_sh
            if abs_f(dF) < 1e-30:
                dF = 1e-30 if dF >= 0 else -1e-30
            # calculate step from F and dF
            step = -F / dF
            i_new = i + step

            if not isfinite(i_new):
                i_new = i + (-F / (abs_f(dF) + 1e-30)) * 0.5

            # backtracking
            # if the new step residual is worse than the old one, we halve the step size
            backtracked = 0
            while backtracked < 5:
                z_try = (v + i_new * rs) / nVt
                z_try = float(clip(z_try, -100.0, 100.0))
                e_try = exp(z_try)
                F_try = i_new - iph + i0 * (e_try - 1.0) + (v + i_new * rs) * inv_r_sh
                if abs_f(F_try) <= abs_f(F):
                    break
                i_new = i + 0.5 * (i_new - i)
                backtracked += 1

            i_new = clamp_I(i_new)

            # step-size + residual convergence
            if abs_f(i_new - i) <= tol * max(1.0, abs_f(i)):
                z_fin = (v + i_new * rs) / nVt
                z_fin = float(clip(z_fin, -100.0, 100.0))
                e_fin = exp(z_fin)
                F_fin = i_new - iph + i0 * (e_fin - 1.0) + (v + i_new * rs) * inv_r_sh
                if abs_f(F_fin) <= 1e-12:
                    i = i_new
                    break

            i = i_new

        return float(i)

    def _i0_iph_from_isc_voc(self):
        # STC-consistent: use 25°C thermal voltage with STC Rs/Rsh and STC Isc/Voc
        TrefK = 25.0 + 273.15
        Vt_ref = (k * TrefK) / q
        nVt_ref = self.diode_ideality * Vt_ref

        exp_voc = math.exp(self.voc_ref / max(nVt_ref, 1e-30))
        exp_isc = math.exp((self.isc_ref * self.r_s) / max(nVt_ref, 1e-30))

        denom = exp_voc - exp_isc
        if denom <= 0:
            return None, None

        inv_rsh = 0.0 if (np.isinf(self.r_sh) or self.r_sh <= 0) else 1.0 / self.r_sh
        i0 = (self.isc_ref * (1.0 + self.r_s * inv_rsh)) / denom
        iph = self.isc_ref + i0 * (exp_isc - 1.0) + self.isc_ref * self.r_s * inv_rsh
        return i0, iph

    # we are trying to fit rs and rsh based on known values
    def fit_rs_rsh_scipy(
        self,
        rs_bounds=(1e-4, 0.05),
        rsh_bounds=(500.0, 5000.0),
        start_from="grid",
        loss="huber",
        f_scale=0.2,
        max_nfev=600,
        fit_n=True
    ):
        from scipy.optimize import least_squares
        import numpy as np

        # for seeding
        # user's settings
        if isinstance(start_from, (tuple, list)) and len(start_from) == 2:
            rs0, rsh0 = map(float, start_from)
        # use whatever is in there right now
        elif start_from == "current":
            rs0, rsh0 = float(self.r_s), float(self.r_sh)
        # uses a grid to find the best starting point, reuses residual that we use when fitting rs and rsh with scipy
        elif start_from == "grid":
            # tiny grid seed: try a few candidates and pick the lowest L2 residual norm
            rs_cands = np.geomspace(max(1e-5, rs_bounds[0]), rs_bounds[1], 4)
            rsh_cands = np.geomspace(max(50.0, rsh_bounds[0]), rsh_bounds[1], 4)
            best = None
            # temporarily stash operating point
            G0, T0 = self.irradiance, self.temperature_c
            for rs_c in rs_cands:
                for rsh_c in rsh_cands:
                    rvec = self._bundle_residuals(rs_c, rsh_c, n=self.diode_ideality, weights=None)
                    if not np.all(np.isfinite(rvec)):
                        continue
                    val = float(np.linalg.norm(rvec))
                    if (best is None) or (val < best[0]):
                        best = (val, rs_c, rsh_c)
            # restore state (bundle_residuals already guards, but this is explicit)
            self.set_conditions(G0, T0)
            rs0, rsh0 = (best[1], best[2]) if best is not None else (0.01, 1500.0)
        else:
            # simple midpoints if grid omitted or fails
            rs0, rsh0 = 0.01, 1500.0

        # ---- Optimize in log-space for positivity & conditioning ----
        if fit_n:
            x0 = np.array([np.log(rs0), np.log(rsh0), self.diode_ideality], float)
            lb = np.array([np.log(max(1e-5, rs_bounds[0])), np.log(max(50.0, rsh_bounds[0])), 1.00])
            ub = np.array([np.log(rs_bounds[1]),                np.log(rsh_bounds[1]),       1.70])

            def resid(x):
                log_rs, log_rsh, n = map(float, x)
                return self._bundle_residuals(np.exp(log_rs), np.exp(log_rsh), n=n)
        else:
            x0 = np.array([np.log(rs0), np.log(rsh0)], float)
            lb = np.array([np.log(max(1e-5, rs_bounds[0])), np.log(max(50.0, rsh_bounds[0]))])
            ub = np.array([np.log(rs_bounds[1]),                np.log(rsh_bounds[1])])

            def resid(x):
                log_rs, log_rsh = map(float, x)
                return self._bundle_residuals(np.exp(log_rs), np.exp(log_rsh), n=None)

        # keep x0 strictly inside bounds
        x0 = np.minimum(np.maximum(x0, lb + 1e-12), ub - 1e-12)

        res = least_squares(
            resid, x0, bounds=(lb, ub),
            method="trf", loss=loss, f_scale=f_scale, max_nfev=max_nfev,
            jac="2-point"  # good default for smooth residuals
        )

        if fit_n:
            rs, rsh, n = float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(res.x[2])
            self.diode_ideality = n
        else:
            rs, rsh = float(np.exp(res.x[0])), float(np.exp(res.x[1]))

        self.r_s, self.r_sh = rs, rsh
        self._update_cache(init_reference=False)

        info = {
            "success": bool(res.success),
            "message": res.message,
            "nfev": int(res.nfev),
            "cost": float(res.cost),
            "rs": rs, "rsh": rsh, "n": float(self.diode_ideality)
        }
        return rs, rsh, info

    # ---------- Utilities for residuals (normalized & stable) ----------


    def _bundle_residuals(self, rs, rsh, n=None, weights=None):
        """
        Bundle residuals that *anchor the physics*:
        - r_isc = ( I(0) - Isc_ref ) / Isc_ref
        - r_voc = I(Voc_ref) / Isc_ref
        - r_mpp = ( Vmpp·I(Vmpp) - Vmpp·Impp ) / (Vmpp·Impp)
        - r_tc_voc = ( β_model - β_spec ) / |β_spec|
        - r_tc_isc = ( α_model - α_spec ) / |α_spec|
        - r_irr = (I_0V_800 / I_0V_400) - 2
        - reg_rs adds penalty for extreme values
        - reg_rsh adds penalty for extreme values
        Using these residuals we input to our scip py optimizer to find a good Rs and Rsh
        """
        W = weights or {
            "isc": 1.0, "voc": 1.0, "mpp": 6.0,
            "tc_voc": 1.2, "tc_isc": 1.0,
            "irr_lin": 0.5,
            "reg_rs": 1.0, "reg_rsh": 1.0
        }

        G0, T0 = self.irradiance, self.temperature_c

        rs_old, rsh_old, n_old = self.r_s, self.r_sh, self.diode_ideality
        try:
            self.r_s, self.r_sh = float(rs), float(rsh)
            if n is not None:
                self.diode_ideality = float(n)
            self._update_cache(init_reference=False)

            # --- STC (25C, 1000 W/m^2)
            self.set_conditions(1000.0, 25.0)
            i0_stc, iph_stc = self._i0_iph_from_isc_voc()
            if i0_stc is None or not np.isfinite(i0_stc) or not np.isfinite(iph_stc) or i0_stc < 0:
                return np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 10.0, 10.0], float)

            I0V   = self.solve_i_at_v(0.0,       i0=i0_stc, iph=iph_stc)
            Ivoc  = self.solve_i_at_v(self.voc_ref, i0=i0_stc, iph=iph_stc)
            Ivmpp = self.solve_i_at_v(self.vmpp, i0=i0_stc, iph=iph_stc)

            Isc_ref = max(1e-12, abs(self.isc_ref))
            Pmpp_ref = max(1e-12, abs(self.vmpp * self.impp))
            r_isc = (I0V - self.isc_ref) / Isc_ref
            r_voc = Ivoc / Isc_ref
            r_mpp = (self.vmpp * Ivmpp - self.vmpp * self.impp) / Pmpp_ref

            # --- Temperature coefficient constraints (finite difference around 25°C)
            # basically we find the condition at 15 and 40 find the Voc so we can then compare voc temp coeff
            dT = 15.0
            self.set_conditions(1000.0, 25.0 + dT)
            i0_hi, iph_hi = self._i0_iph_from_isc_voc()
            Voc_hi = self._solve_voc(i0=i0_hi, iph=iph_hi)

            self.set_conditions(1000.0, 25.0 - dT)
            i0_lo, iph_lo = self._i0_iph_from_isc_voc()
            Voc_lo = self._solve_voc(i0=i0_lo, iph=iph_lo)

            beta_voc_model = (Voc_hi - Voc_lo) / (2 * dT)          # V/°C
            beta_voc_spec  = self.voc_temp_coeff                   # V/°C
            r_tc_voc = (beta_voc_model - beta_voc_spec) / max(1e-6, abs(beta_voc_spec))

            # Isc(T) slope
            # same thing here but with Isc
            self.set_conditions(1000.0, 25.0 + dT); I_0V_hi = self.solve_i_at_v(0.0)
            self.set_conditions(1000.0, 25.0 - dT); I_0V_lo = self.solve_i_at_v(0.0)
            alpha_isc_model = (I_0V_hi - I_0V_lo) / (2 * dT)       # A/°C
            alpha_isc_spec  = self.isc_temp_coeff                   # A/°C
            r_tc_isc = (alpha_isc_model - alpha_isc_spec) / max(1e-6, abs(alpha_isc_spec))

            # --- Irradiance linearity (near STC)
            # set irradiance to 800 and 400 see if the increase is 2x
            self.set_conditions(800.0, 25.0); I_0V_800 = self.solve_i_at_v(0.0)
            self.set_conditions(400.0, 25.0); I_0V_400 = self.solve_i_at_v(0.0)
            # Expect ~2× ratio; turn into a scalar residual
            r_irr = (I_0V_800 / max(1e-9, I_0V_400)) - 2.0

            # --- Tiny L2 regularization to avoid extremes that "match" but are non-physical
            reg_rs  = 1e-3 * (rs  / 0.05)**2
            reg_rsh = 1e-6 * (rsh / 1000.0)**2

            return np.array([
                W["isc"]*r_isc, W["voc"]*r_voc, W["mpp"]*r_mpp,
                W["tc_voc"]*r_tc_voc, W["tc_isc"]*r_tc_isc,
                W["irr_lin"]*r_irr,
                W["reg_rs"]*reg_rs, W["reg_rsh"]*reg_rsh
            ], dtype=float)

        finally:
            self.r_s, self.r_sh = rs_old, rsh_old
            self.diode_ideality = n_old
            self.set_conditions(G0, T0)
            self._update_cache(init_reference=False)


    def _solve_voc(self, i0=None, iph=None):
        """
        Numerically find Voc: the voltage where I(V)=~0.
        Uses a coarse sweep + 1D refine that is robust even before good fitting.
        """
        # Bracket around linear estimate; allow extra headroom
        vmax_guess = max(self.voc_lin if self.voc_lin is not None else self.voc_ref, self.voc_ref)
        v_lo = 0.0
        v_hi = max(vmax_guess, 0.9 * self.voc_ref) + 0.05  # small pad

        # coarse scan to get close
        i_guess = iph if iph is not None else self.Iph
        best_v, best_absI = 0.0, 1e9
        for v in np.linspace(v_lo, v_hi, 80):
            i_guess = self.solve_i_at_v(v, i0=i0, iph=iph, i_init=i_guess)
            ai = abs(i_guess)
            if ai < best_absI:
                best_absI, best_v = ai, v

        # quick golden-section refine on |I(v)|
        phi = (1 + 5**0.5) / 2
        a, b = max(v_lo, best_v - 0.08), min(v_hi, best_v + 0.08)
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi
        f = lambda v, ig: (abs(self.solve_i_at_v(v, i0=i0, iph=iph, i_init=ig)), self.solve_i_at_v(v, i0=i0, iph=iph, i_init=ig))
        f1, i_guess = f(x1, i_guess)
        f2, i_guess = f(x2, i_guess)
        for _ in range(30):
            if (b - a) <= 1e-6 or min(f1, f2) <= 1e-9:
                break
            if f1 > f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (b - a) / phi
                f2, i_guess = f(x2, i_guess)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - (b - a) / phi
                f1, i_guess = f(x1, i_guess)
        return 0.5 * (a + b)

    # ------IV CURVE GENERATION------

    def get_iv_curve(self, num_points, vmin=0.0, vmax=None):

        if vmax is None:
            vmax = self.voc_lin if self.voc_lin is not None else self.voc_ref

        voltages = np.linspace(vmin, vmax, num_points)
        points = []

        i_guess = self.Iph  # start near Isc
        for v in voltages:
            i = self.solve_i_at_v(v, i_init=i_guess)
            points.append((float(v), float(i)))
            i_guess = i  # warm start for next voltage initial guess

        return points
    
    def v_at_i(self, i_target: float, tol: float = 1e-7, max_iter: int = 80) -> float:
        """
        Find the voltage at which the cell current equals i_target (A).
        Uses a robust bisection search over the monotonic I–V curve.
        Works for forward-bias region (0..Isc).
        """
        # Bracket around Voc range
        v_lo = -0.05
        v_hi_guess = self.voc_lin if self.voc_lin is not None else self.voc_ref
        v_hi = max(0.1, float(v_hi_guess)) * 1.10  # pad above Voc

        i_lo = self.solve_i_at_v(v_lo)   # ~Isc
        i_hi = self.solve_i_at_v(v_hi)   # ~0

        # If target outside bounds, clamp
        if i_target >= i_lo:
            return v_lo
        if i_target <= i_hi:
            return v_hi

        # Bisection
        # keep getting the middle of v and solve for i
        for _ in range(max_iter):
            vm = 0.5 * (v_lo + v_hi)
            im = self.solve_i_at_v(vm)
            if abs(im - i_target) < tol:
                return vm
            if im > i_target:
                v_lo = vm
            else:
                v_hi = vm
        return 0.5 * (v_lo + v_hi)
