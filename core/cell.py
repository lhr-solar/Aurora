import math
import numpy as np

k = 1.380649e-23  # J/K
q = 1.602176634e-19  # C

class Cell:

    def __init__(self, isc_ref, voc_ref, diode_ideality, r_s, r_sh, voc_temp_coeff=-0.00174, isc_temp_coeff=0.0029, irradiance=1000.0, temperature_c=25.0, vmpp=0.621, impp=5.84):
        self.isc_ref = float(isc_ref)       # A (STC)
        self.voc_ref = float(voc_ref)       # V (STC)
        self.diode_ideality = float(diode_ideality)  # n
        self.r_s = float(r_s)               # ohm (initial guess; can be fit later)
        self.r_sh = float(r_sh)             # ohm (initial guess; can be fit later)
        self.voc_temp_coeff = float(voc_temp_coeff)  # V/°C (beta_Voc)
        self.isc_temp_coeff = float(isc_temp_coeff)  # A/°C (alpha_Isc)
        self.irradiance = float(irradiance)         # W/m^2
        self.temperature_c = float(temperature_c)   # °C


        # cached values

        self.temperature_k = None
        self.thermal_voltage = None   # n*k*T/q (often called nVt)
        self.Iph = None               # photocurrent at current G,T
        self.I0_ref = None            # saturation current at STC (computed once)
        self.I0 = None                # saturation current at current T (kept = I0_ref here)
        self.voc_lin = None           # linear Voc(T) estimate for heuristics

        self._update_cache(init_reference=True)
        
    def set_conditions(self, irradiance, temperature_c):
        self.irradiance = irradiance
        self.temperature_c = temperature_c
        self._update_cache()

    def _update_cache(self, init_reference):
        self.temperature_k = self.temperature_c + 273.15
        self.thermal_voltage = self.diode_ideality*k*self.temperature_k / q
        
        # Photocurrent: linearized in T and proportional to G
        # STC reference: 25°C, 1000 W/m^2
        Iph_ref_25 = self.isc_ref  # good approximation at STC
        Iph_T = Iph_ref_25 + self.isc_temp_coeff * (self.temperature_c - 25.0)
        self.Iph = Iph_T * (self.irradiance / 1000.0)

        # Linear Voc estimate vs temperature (useful for guesses)
        self.voc_lin = max(0.0, self.voc_ref + self.voc_temp_coeff * (self.temperature_c - 25.0))

        if init_reference:
            TrefK = 25.0 + 273.15
            Vt_ref = (k * TrefK) / q
            nVt_ref = self.diode_ideality * Vt_ref
            denom_ref = math.exp(self.voc_ref / max(nVt_ref, 1e-30)) - 1.0
            # Uses current r_sh as the STC shunt value
            self.I0_ref = (self.isc_ref - self.voc_ref / max(self.r_sh, 1e-12)) / max(denom_ref, 1e-30)

        nVt = self.thermal_voltage
        denom = math.exp(self.voc_lin / max(nVt, 1e-30)) - 1.0
        self.I0 = (self.Iph - self.voc_lin / max(self.r_sh, 1e-12)) / max(denom, 1e-30)


    def fit_rs_rsh(self):
        return None

    # solved using Newton-Ralphson
    def singlediode_current(self):
        return None

    def solve_i_at_v(self, v, i0, max_iter=80, tol=1e-10):
        
        if np.isinf(self.r_sh):
            inv_r_sh = 0.0
        else:
            inv_r_sh = 1/self.r_sh

        i = self.Iph

        for _ in range(max_iter):

            arg = v+i*self.r_s
            arg1 = np.exp(arg / self.thermal_voltage)
            # in case the number is too big
            arg1 = np.clip(arg1, -50, 50)

            F = i - self.Iph + i0 * (arg1 - 1) + (arg/self.r_sh)
            dF = 1 + (i0 * arg1 * self.r_s / self.thermal_voltage) + self.r_s / self.r_sh

            if abs(dF) < 1e-30:
                denom = 1e-30 if dF >= 0 else -1e-30
            else:
                denom = dF

            step = -F / denom

            i += step

            if i<tol:
                break
        
        return i

    def _i0_iph_from_isc_voc(self):

        i0_denom = math.exp(self.voc_ref/self.thermal_voltage) - math.exp(self.isc_ref*self.r_s/self.thermal_voltage)
        if i0_denom <= 0:
            return None, None
        # if denom is <=0 not possible

        i0_num = self.isc*(1+(self.r_s/self.r_sh))
        i0 = i0_num / i0_denom

        iph = self.isc_ref + i0*(math.exp((self.isc_ref*self.r_s / self.thermal_voltage) - 1)) + self.isc_ref*self.r_s/self.r_sh

        return i0, iph
    
    def _mpp_error(self):
        i0, iph = self._i0_iph_from_isc_voc()
        if i0 is None:
            return np.inf, None, None
        # need to finish