import math
import numpy as np

k = 1.380649e-23  # J/K
q = 1.602176634e-19  # C
EG_EV = 1.12 # silicon bandgap

class Cell:

    def __init__(self, isc_ref, voc_ref, diode_ideality, r_s=0, r_sh=0, voc_temp_coeff=-0.00174, isc_temp_coeff=0.0029, irradiance=1000.0, temperature_c=25.0, vmpp=0.621, impp=5.84, autofit=False):
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
        rs, rsh, info = self.fit_rs_rsh_scipy(start_from="grid", **fit_kwargs)

        self._calibrated = True

        # restore previous conditions and caches
        self.set_conditions(G0, T0)
        return rs, rsh, info



    def solve_i_at_v(self, v, i0=None, iph=None, max_iter=80, tol=1e-10, i_init=None):
        
        if np.isinf(self.r_sh) or self.r_sh<=0:
            inv_r_sh = 0.0
        else:
            inv_r_sh = 1/self.r_sh

        if i0 is None:
            i0 = self.I0
        if iph is None:
            iph = self.Iph

        rs = self.r_s
        denom_lin = 1.0 + rs * inv_r_sh

        if i_init is None:
            denom_lin = 1.0 + rs * inv_r_sh
            i = (iph - v * inv_r_sh) / denom_lin if denom_lin > 0 else iph
        else:
            i = float(i_init)


        def clamp_I(x):
            # wide bounds to avoid boxing Newton too hard
            # below zero can happen slightly under reverse bias; allow some slack
            lo = -10.0 * abs(iph) - 1e-6
            hi =  2.0  * abs(iph) + 1e-6
            return min(max(x, lo), hi)

        i = clamp_I(i)

        nVt = max(self.thermal_voltage, 1e-30)

        for _ in range(max_iter):
            z = (v + i*rs) / nVt
            z = float(np.clip(z, -100.0, 100.0))
            e = math.exp(z)

            F  = i - iph + i0*(e - 1.0) + (v + i*rs) * inv_r_sh
            dF = 1.0 + i0*e*(rs/nVt) + rs * inv_r_sh
            if abs(dF) < 1e-30:
                dF = 1e-30 if dF >= 0 else -1e-30

            step  = -F / dF
            i_new = i + step

            if not np.isfinite(i_new):
                i_new = i + (-F / (abs(dF) + 1e-30)) * 0.5

            # backtracking
            backtracked = 0
            while backtracked < 5:
                z_try = (v + i_new*rs) / nVt
                z_try = float(np.clip(z_try, -100.0, 100.0))
                e_try = math.exp(z_try)
                F_try = i_new - iph + i0*(e_try - 1.0) + (v + i_new*rs) * inv_r_sh
                if abs(F_try) <= abs(F):
                    break
                i_new = i + 0.5*(i_new - i)
                backtracked += 1

            i_new = clamp_I(i_new)

            # step-size + residual convergence
            if abs(i_new - i) <= tol * max(1.0, abs(i)):
                z_fin = (v + i_new*rs) / nVt
                z_fin = float(np.clip(z_fin, -100.0, 100.0))
                e_fin = math.exp(z_fin)
                F_fin = i_new - iph + i0*(e_fin - 1.0) + (v + i_new*rs) * inv_r_sh
                if abs(F_fin) <= 1e-12:
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

    
    def _mpp_error(self):
        i0, iph = self._i0_iph_from_isc_voc()
        if i0 is None:
            return np.inf, None, None


        i_at_vmpp = self.solve_i_at_v(self.vmpp, i0, iph)
        real_power = self.vmpp * self.impp
        simulated_power = self.vmpp * i_at_vmpp

        error = abs(real_power-simulated_power)

        return error, i0, iph


    def fit_rs_rsh_scipy(self,
                     rs_bounds=(1e-6, 0.2),
                     rsh_bounds=(10.0, 20000.0),
                     start_from="grid",        # "grid", "current", or (rs0,rsh0)
                     use_slopes=False,
                     weights=(1.0, 1.0, 3.0, 0.25, 0.25),
                     loss="soft_l1",
                     f_scale=0.2,
                     max_nfev=200,
                     grid_kwargs=None):
        """
        Robust SciPy least-squares fit for (Rs, Rsh), with normalized residuals.
        - start_from: "grid" runs a quick grid refine first; "current" uses current Rs/Rsh; or pass a tuple (rs0,rsh0).
        - use_slopes: include endpoint slope residuals to better shape the knee (optional).
        - weights: (w_isc, w_voc, w_mpp[, w_slope_left, w_slope_right]) — MPP usually deserves more weight.
        - loss: "linear" (plain LS), "soft_l1", or "huber".
        - f_scale: scale at which residuals start getting down-weighted (since residuals are normalized, 0.1–0.5 works well).
        """
        try:
            from scipy.optimize import least_squares
        except Exception as e:
            raise RuntimeError("SciPy not available: " + str(e))

        # choose starting point
        if isinstance(start_from, (tuple, list)) and len(start_from) == 2:
            rs0, rsh0 = map(float, start_from)
        elif start_from == "current":
            rs0, rsh0 = float(self.r_s), float(self.r_sh)
        else:
            # small, fast grid to get a decent seed
            gk = grid_kwargs or {
                "rs_range": (0.0, 0.08),
                "rsh_range": (20.0, 6000.0),
                "coarse_steps": (15, 15),
                "refine_factor": 0.3,
                "refine_steps": (17, 17),
                "weights": (1.0, 1.0, 3.0),
            }
            # quick inline grid-refine using the normalized residuals
            def grid_search(rs_lo, rs_hi, rsh_lo, rsh_hi, n_rs, n_rsh):
                best = (float("inf"), None, None)
                rs_vals  = np.linspace(rs_lo,  rs_hi,  int(max(2, n_rs)))
                rsh_vals = np.linspace(rsh_lo, rsh_hi, int(max(2, n_rsh)))
                for rs in rs_vals:
                    for rsh in rsh_vals:
                        r = self._fit_residuals_norm(rs, rsh, use_slopes=False, w=gk["weights"])
                        err = float(np.linalg.norm(r))
                        if err < best[0]:
                            best = (err, float(rs), float(rsh))
                return best

            (rs_lo, rs_hi), (rsh_lo, rsh_hi) = gk["rs_range"], gk["rsh_range"]
            n_rs, n_rsh = gk["coarse_steps"]
            err1, rs1, rsh1 = grid_search(rs_lo, rs_hi, rsh_lo, rsh_hi, n_rs, n_rsh)

            rs_span  = max(1e-6, gk["refine_factor"] * max(1e-6, rs1))
            rsh_span = max(1.0,  gk["refine_factor"] * max(1.0,  rsh1))
            err2, rs2, rsh2 = grid_search(max(0.0, rs1 - rs_span), rs1 + rs_span,
                                        max(1.0,  rsh1 - rsh_span), rsh1 + rsh_span,
                                        *gk["refine_steps"])
            rs0, rsh0 = rs2, rsh2

        # residual callback for SciPy (normalized)
        def resid(x):
            rs, rsh = float(x[0]), float(x[1])
            return self._fit_residuals_norm(rs, rsh, use_slopes=use_slopes, w=weights)

        lb = np.array([rs_bounds[0],  rsh_bounds[0]], dtype=float)
        ub = np.array([rs_bounds[1],  rsh_bounds[1]], dtype=float)

        res = least_squares(
            resid,
            x0=np.array([rs0, rsh0], dtype=float),
            bounds=(lb, ub),
            method="trf",
            loss=loss,         # "linear", "soft_l1", or "huber"
            f_scale=f_scale,   # assumes residuals are normalized
            max_nfev=max_nfev
        )

        best_rs, best_rsh = float(res.x[0]), float(res.x[1])
        # apply and refresh
        self.r_s, self.r_sh = best_rs, best_rsh
        self._update_cache(init_reference=False)

        info = {
            "success": bool(res.success),
            "message": res.message,
            "nfev": int(res.nfev),
            "cost": float(res.cost),                   # 0.5 * sum(residuals^2 after loss transform)
            "final_residuals": res.fun.tolist(),       # raw residuals (pre-loss)
            "x0": [rs0, rsh0],
            "bounds": [rs_bounds, rsh_bounds],
            "weights": list(weights),
            "loss": loss,
            "f_scale": f_scale
        }
        return best_rs, best_rsh, info




    # ---------- Utilities for residuals (normalized & stable) ----------

    def _predict_io_given_rs_rsh(self, rs, rsh):
        """Temporarily set (rs,rsh), refresh cache, and estimate (i0, iph)."""
        rs_old, rsh_old = self.r_s, self.r_sh
        try:
            self.r_s, self.r_sh = float(rs), float(rsh)
            self._update_cache(init_reference=False)
            i0, iph = self._i0_iph_from_isc_voc()
            return i0, iph
        finally:
            self.r_s, self.r_sh = rs_old, rsh_old
            self._update_cache(init_reference=False)

    def _endpoint_slopes(self, i0, iph, dv=1e-4):
        """
        Approximate endpoint slopes:
        near V=0:      dI/dV ≈ (I(dv)-I(0)) / dv   ~ -1/Rsh (if Rs small)
        near V≈Voc:    dI/dV ≈ (I(Voc)-I(Voc-dv)) / dv  (Rs influences)
        """
        I0 = self.solve_i_at_v(0.0, i0=i0, iph=iph)
        I_dv = self.solve_i_at_v(dv,  i0=i0, iph=iph, i_init=I0)
        slope_left = (I_dv - I0) / dv

        I_voc = self.solve_i_at_v(self.voc_ref, i0=i0, iph=iph)
        I_voc_m = self.solve_i_at_v(max(0.0, self.voc_ref - dv), i0=i0, iph=iph, i_init=I_voc)
        slope_right = (I_voc - I_voc_m) / dv
        return slope_left, slope_right

    def _fit_residuals_norm(self, rs, rsh, use_slopes=False, w=(1.0, 1.0, 3.0, 0.3, 0.3)):
        """
        Normalized residual vector for least-squares:
        r1 = (I(0) - Isc)/Isc
        r2 = I(Voc)/Isc
        r3 = (P(Vmpp) - Vmpp*Impp) / (Vmpp*Impp)
        r4 = (slope_left - target_left)/|target_left|         [optional]
        r5 = (slope_right - target_right)/max(1e-6, |target_right|)  [optional]
        Weights w = (w1,w2,w3[,w4,w5]).
        """
        # estimate i0, iph under these Rs/Rsh
        i0, iph = self._predict_io_given_rs_rsh(rs, rsh)
        if i0 is None or not np.isfinite(i0) or i0 < 0 or not np.isfinite(iph):
            # invalid combo -> big penalty
            return np.array([1e6, 1e6, 1e6], dtype=float)

        # base residuals
        Isc_ref = max(1e-12, abs(self.isc_ref))
        Pmpp_ref = max(1e-12, abs(self.vmpp * self.impp))

        I0V   = self.solve_i_at_v(0.0, i0=i0, iph=iph)
        Ivoc  = self.solve_i_at_v(self.voc_ref, i0=i0, iph=iph)
        Ivmpp = self.solve_i_at_v(self.vmpp,   i0=i0, iph=iph)

        r1 = (I0V - self.isc_ref) / Isc_ref
        r2 = Ivoc / Isc_ref
        r3 = (self.vmpp * Ivmpp - self.vmpp * self.impp) / Pmpp_ref

        res = [w[0]*r1, w[1]*r2, w[2]*r3]

        # optional slope constraints (help pin Rs/Rsh shape)
        if use_slopes:
            # crude targets: near 0V, slope ~ -1/Rsh ; near Voc, slope magnitude grows with Rs
            # Use current (rs,rsh) inferred targets for normalization to avoid over-constraining units.
            # Here we “nudge” toward theoretical expectations:
            slope_left, slope_right = self._endpoint_slopes(i0, iph, dv=1e-4)

            # heuristic targets:
            target_left  = -1.0 / max(1.0, rsh)          # amps/volt
            target_right = -1.0 / max(1e-6, rs + 1e-3)   # larger magnitude for larger Rs (very rough)

            # normalized slope residuals
            r4 = (slope_left  - target_left)  / max(1e-6, abs(target_left))
            r5 = (slope_right - target_right) / max(1e-6, abs(target_right))

            res.extend([w[3]*r4, w[4]*r5])

        bad = (i0 is None) or (not np.isfinite(i0)) or (i0 < 0) or (not np.isfinite(iph))
        if bad:
            base = [1e6, 1e6, 1e6]
            if use_slopes:
                base += [1e6, 1e6]
            return np.array(base, dtype=float)

        return np.array(res, dtype=float)


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