"""Standard atmosphere and atmospheric absorption of sound.

ISA (1976 standard atmosphere's layers to 32 km: the troposphere, the tropopause's isothermal layer from 11 km, the
stratosphere warming 1 K/km from 20 km) for density, temperature and speed of sound; ISO 9613-1 for the absorption
coefficient of pure tones (dB/m), which also holds for fractional octave bands when evaluated at the band's exact
mid-frequency.
"""
from dataclasses import dataclass

import numpy as np

T0_ISA = 288.15        # K
P0_ISA = 101325.0      # Pa
LAPSE = 0.0065         # K/m
R_AIR = 287.053        # J/(kg K)
GAMMA = 1.4
G0 = 9.80665
# the layers above the troposphere: base altitude (m), lapse rate (K/m)
LAYERS = ((0.0, LAPSE), (11000.0, 0.0), (20000.0, -0.001), (32000.0, None))


def isa(h):
    """ISA temperature (K) and pressure (Pa) at geopotential altitude h (m), to 32 km"""
    T, p = T0_ISA, P0_ISA
    for (h0, a), (h1, _) in zip(LAYERS, LAYERS[1:]):
        dh = min(h, h1) - h0
        if a:
            p *= ((T - a * dh) / T) ** (G0 / (R_AIR * a))
        else:
            p *= np.exp(-G0 * dh / (R_AIR * T))
        T -= a * dh
        if h <= h1:
            break
    return T, p


@dataclass(frozen=True)
class Atmosphere:
    """Air at the rotor: ISA at altitude_m with an optional temperature offset."""
    altitude_m: float = 0.0
    dT: float = 0.0              # K, offset from ISA
    humidity: float = 0.7        # relative humidity 0..1 (for absorption)

    @property
    def T(self):
        return isa(self.altitude_m)[0] + self.dT

    @property
    def p(self):
        return isa(self.altitude_m)[1]

    @property
    def rho(self):
        return self.p / (R_AIR * self.T)

    @property
    def c(self):
        return float(np.sqrt(GAMMA * R_AIR * self.T))

    @property
    def nu(self):
        """Kinematic viscosity (Sutherland)."""
        mu = 1.458e-6 * self.T ** 1.5 / (self.T + 110.4)
        return mu / self.rho

    def absorption(self, f):
        """Pure-tone absorption coefficient in dB/m at frequency f (Hz)."""
        return iso9613_alpha(f, self.T, self.humidity * 100.0, self.p)


def iso9613_alpha(f, T=293.15, rh_percent=70.0, p=P0_ISA):
    """ISO 9613-1 atmospheric absorption, dB per metre."""
    f = np.asarray(f, dtype=float)
    pr = 101.325e3
    T0 = 293.15
    T01 = 273.16
    pa = p / pr
    C = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    psat = 10.0 ** C
    h = rh_percent * psat / pa           # molar concentration of water vapour, %
    frO = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = pa * (T / T0) ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0)))
    return 8.686 * f ** 2 * (1.84e-11 / pa * (T / T0) ** 0.5
                             + (T / T0) ** -2.5 * (0.01275 * np.exp(-2239.1 / T) / (frO + f ** 2 / frO)
                                                   + 0.1068 * np.exp(-3352.0 / T) / (frN + f ** 2 / frN)))
