import math

boltzmann_const = 1.380649e-23
charge_electron = 1.602176634e-19
EG0 = 1.17  # eV (assume silicon)
ALPHA = 4.73e-4
BETA = 636.0


class Bypass_Diode:
    """Wrapper for a reverse-wired bypass diode.

    Numerically stable diode model:
      - Varshni bandgap for temperature dependence
      - Is(T) ~ T^3 * exp(-Eg/(kT)) scaling
      - v_at_i(I) = - (n*Vt*log1p(I/Is) + I*Rs) for I>0, else -inf
    """

    def __init__(self, irradiance=None, temperature=None, ideality_factor=1.2, series_resistance=0.005, V_cells: list | None = None):
        self.irradiance = irradiance
        self.temperature = temperature
        self.ideality_factor = float(ideality_factor)
        self.series_resistance = float(series_resistance)
        self.V_cells = list(V_cells) if V_cells is not None else []

        # runtime cached values
        self.thermal_voltage = None
        self.temp_kelvin = None
        # reverse_saturation_current mirrors older name but stores Is (A)
        self.reverse_saturation_current = None
        self.diode_voltage_curve_slope = None
        self.V_string = None
        self.bypass_voltage = None
        self.sub_voltage = None

        # If temperature provided at construction, initialize caches
        if temperature is not None:
            try:
                self.set_temperature(float(temperature))
            except Exception:
                # ignore failures at init; callers may call set_temperature later
                pass

    @staticmethod
    def _Eg_eV(T_kelvin: float) -> float:
        """Varshni model for bandgap in eV.

        T in Kelvin.
        """
        return EG0 - (ALPHA * T_kelvin * T_kelvin) / (T_kelvin + BETA)

    def set_temperature(self, temp_celcius: float):
        self.temperature = float(temp_celcius)
        self.temp_kelvin = self.temperature + 273.15
        # thermal voltage Vt = k*T/q
        self.thermal_voltage = boltzmann_const * self.temp_kelvin / charge_electron

        # compute reverse saturation current Is (store in reverse_saturation_current)
        # Use a common approximation: Is(T) = Is_ref * (T/Tref)^3 * exp( -q/k * (Eg(T)/T - Eg(Tref)/Tref) )
        # We don't have an Is_ref passed here in the old API; fall back to a small default
        # if not present on the instance (some callers might set it externally).
        Is_ref = getattr(self, 'is_ref', 1e-9)
        # choose a reference temperature (25°C) in Kelvin
        Tref = 25.0 + 273.15
        EgT = self._Eg_eV(self.temp_kelvin)
        EgTref = self._Eg_eV(Tref)
        # exponent uses q/k with Eg in eV -> convert Eg (eV) to Joules by multiplying q
        expo = -(charge_electron / boltzmann_const) * (EgT / self.temp_kelvin - EgTref / Tref)
        # guard numerics
        try:
            self.reverse_saturation_current = float(Is_ref * (self.temp_kelvin / Tref) ** 3 * math.exp(expo))
        except OverflowError:
            # if exponent is huge negative, Is ~ 0; cap at tiny positive value
            self.reverse_saturation_current = max(1e-50, float(Is_ref))

    def v_at_i(self, terminal_current) -> float:
        """Return diode voltage at given string current (terminal_current).

        Keeps the original method name but interprets the argument as current.
        For non-positive current returns -inf so substring logic chooses cell voltages.
        """
        I = float(terminal_current)
        # If temperature hasn't been set, attempt to compute with defaults
        if self.reverse_saturation_current is None:
            # conservative default small saturation current
            self.reverse_saturation_current = getattr(self, 'is_ref', 1e-9)
            if self.thermal_voltage is None:
                # approximate at 25°C
                self.temp_kelvin = 25.0 + 273.15
                self.thermal_voltage = boltzmann_const * self.temp_kelvin / charge_electron

        Is = max(self.reverse_saturation_current, 1e-300)
        n = max(self.ideality_factor, 1e-6)
        Rs = float(self.series_resistance)

        if I <= 0.0:
            return float('-inf')

        # Use log1p for numerical stability when I/Is is small
        return -(n * self.thermal_voltage * math.log1p(I / Is) + I * Rs)

    def dv_dI(self, terminal_current) -> float | None:
        """Analytic derivative dV/dI of the diode V(I) model.

        Returns None for non-conducting (I<=0).
        """
        I = float(terminal_current)
        if I <= 0.0:
            return None
        Is = max(self.reverse_saturation_current, 1e-300)
        n = max(self.ideality_factor, 1e-6)
        Rs = float(self.series_resistance)
        # derivative: d/dI [ - (n*Vt*ln(1 + I/Is) + I*Rs) ] = - (n*Vt/(I+Is) + Rs)
        return -(n * self.thermal_voltage / (I + Is) + Rs)

    def activation_condition(self, terminal_current):
        """Return True if bypass diode voltage dominates substring voltage.

        The method computes `self.V_string = sum(self.V_cells)` and compares with
        `self.v_at_i(terminal_current)`.
        """
        try:
            self.V_string = sum(self.V_cells)
        except Exception:
            # fallback if V_cells is not iterable
            self.V_string = float(getattr(self, 'V_string', 0.0))
        self.bypass_voltage = self.v_at_i(terminal_current)
        return self.V_string <= self.bypass_voltage

    def Clamp(self):
        # set sub_voltage to the larger of the
        # string voltage or the bypass diode voltage.
        try:
            self.V_string = sum(self.V_cells)
        except Exception:
            self.V_string = float(getattr(self, 'V_string', 0.0))
        # ensure bypass_voltage computed
        if self.bypass_voltage is None:
            self.bypass_voltage = self.v_at_i(0.0)
        self.sub_voltage = max(self.V_string, self.bypass_voltage)