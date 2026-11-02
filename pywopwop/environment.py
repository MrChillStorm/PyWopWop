"""The listener's surroundings, for loops: the ground's reflection and the diffuse outdoor field.

A listener outside stands on the ground: every pulse arrives twice, directly and reflected, the reflection a few
milliseconds later and filtered by the ground's impedance (Delany-Bazley, at the grazing angle). Terrain, trees and
buildings add a weak, quickly decaying diffuse field. Both are linear and time-invariant over a loop, so they are
applied as circular filters: the loop stays seamless, and layers filtered alike still add up to the mix.
"""
from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft

ENVIRONMENTS = ("free", "ground", "open", "urban")


@dataclass
class Environment:
    kind: str = "open"              # free (anechoic), ground (reflection only), open (and a light diffuse field), urban
    listener_height: float = 1.7
    flow_resistivity: float = 2.0e5     # Pa s/m^2: grass; 2e7 for asphalt
    seed: int = 7

    @property
    def reverb(self):
        """(level of the diffuse field relative to the direct sound's energy, dB; T60, s)"""
        return {"free": None, "ground": None, "open": (-20.0, 0.6), "urban": (-12.0, 1.1)}[self.kind]


def ground_gain(f, height_above, distance, env, rho=1.2, c=340.0):
    """Direct plus ground-reflected path (image source) as a gain on the direct sound, for a source height_above the
    listener (m) at distance (m)."""
    if env.kind == "free" or height_above <= 0:
        return np.ones_like(f, dtype=complex)
    h = env.listener_height
    horiz = np.sqrt(max(distance ** 2 - height_above ** 2, 0.0))
    r1 = distance
    r2 = np.hypot(horiz, height_above + 2.0 * h)
    sin_g = (height_above + 2.0 * h) / r2
    X = np.maximum(rho * np.maximum(f, 1.0) / env.flow_resistivity, 1e-4)
    Z = 1.0 + 0.0571 * X ** -0.754 - 1j * 0.087 * X ** -0.732
    R = (Z * sin_g - 1.0) / (Z * sin_g + 1.0)
    return 1.0 + R * (r1 / r2) * np.exp(-2j * np.pi * f * (r2 - r1) / c)


def diffuse_ir(fs, level_db, t60, seed, lowpass=4000.0):
    """A diffuse outdoor tail: Gaussian noise under an exponential decay, starting 15 ms after the direct sound,
    darker than it, with `level_db` of the direct sound's energy."""
    rng = np.random.default_rng(seed)
    n = int(1.5 * t60 * fs)
    t = np.arange(n) / fs
    ir = rng.normal(size=n) * np.exp(-6.9078 * t / t60) * (t > 0.015)
    X = np.fft.rfft(ir)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    ir = np.fft.irfft(X / np.sqrt(1.0 + (f / lowpass) ** 2), n)
    return ir * np.sqrt(10.0 ** (level_db / 10.0) / np.sum(ir ** 2))


def apply(x, fs, height_above, distance, env):
    """The listener's surroundings on a loop (circular)."""
    if env is None or env.kind == "free":
        return x
    L = len(x)
    X = sfft.rfft(x, workers=-1)
    f = np.fft.rfftfreq(L, 1.0 / fs)
    G = ground_gain(f, height_above, distance, env)
    rv = env.reverb
    if rv is not None:
        ir = diffuse_ir(fs, rv[0], rv[1], env.seed)
        if len(ir) > L:
            ir = np.add.reduceat(np.pad(ir, (0, (-len(ir)) % L)).reshape(-1, L), [0], axis=0)[0]
        G = G + sfft.rfft(ir, L, workers=-1)
    return sfft.irfft(X * G, L, workers=-1)
