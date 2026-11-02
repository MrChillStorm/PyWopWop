"""Elastic blades: rotating-beam flap modes and a torsion mode, for any rotor.

Flap: a Galerkin solution of the rotating Euler-Bernoulli beam (bending stiffness EI, centrifugal tension), hinged
at the flap hinge for articulated rotors or clamped at the hub for hingeless, bearingless and teetering ones (and
propellers). The mass per length comes from the Lock number unless given; EI is uniform and set so that the mode
the data name matches its published frequency (the first for clamped blades, the first elastic one for hinged).
So a hingeless rotor's coning is the bent shape of its first mode, not a straight line.

Torsion: one mode, frequency from the data, shape between a fixed-free torsion bar (sin(pi x / 2)) and a rigid
pitch rotation about the control system's flexibility.

The periodic response to the blade loads is found by harmonic balance (aeroelastic.py).
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


@dataclass
class BladeModes:
    x: np.ndarray               # r/R where shapes are tabulated (the aerodynamic stations)
    flap_shapes: np.ndarray     # (nmodes, Nr), tip value 1
    flap_slopes: np.ndarray     # (nmodes, Nr), d(shape)/dr (per metre)
    flap_freq: np.ndarray       # rad/s
    flap_mass: np.ndarray       # generalised masses (kg)
    flap_ratio: np.ndarray      # per rev
    torsion_shape: np.ndarray   # (Nr,)
    torsion_freq: float         # rad/s
    torsion_inertia: float      # generalised inertia (kg m^2)
    torsion_ratio: float
    m: float                    # blade mass per length, kg/m
    EI: float
    hinged: bool
    root: float                 # r/R of the hinge or clamp
    zeta_flap: float
    zeta_torsion: float


def _basis(xi, hinged, n):
    """Polynomial basis on xi = (r - root)/(R - root) satisfying the root conditions; values, d/dxi, d2/dxi2."""
    powers = ([1] + list(range(3, n + 2))) if hinged else list(range(2, n + 2))
    xi = np.asarray(xi)[None, :]
    p = np.array(powers, float)[:, None]
    phi = xi ** p
    d1 = p * xi ** np.maximum(p - 1, 0)
    d2 = p * (p - 1) * xi ** np.maximum(p - 2, 0)
    return phi, d1, d2


def _flap_eigen(R, root, m, EI, omega, hinged, nbasis=8):
    L = R * (1.0 - root)
    xi = 0.5 * (_GL_X + 1.0)
    w = 0.5 * _GL_W
    r = R * root + L * xi
    phi, d1, d2 = _basis(xi, hinged, nbasis)
    d1 = d1 / L
    d2 = d2 / L ** 2
    # centrifugal tension at r: Omega^2 int_r^R m s ds
    T = m * omega ** 2 * (R ** 2 - r ** 2) / 2.0
    K = (EI * (d2 * w) @ d2.T + (T * d1 * w) @ d1.T) * L
    Mm = (m * phi * w) @ phi.T * L
    vals, vecs = eigh(K, Mm)
    return np.sqrt(np.maximum(vals, 0.0)), vecs, nbasis


def blade_modes(rotor, x, rho, n_flap=2):
    """Flap and torsion modes of a rotor's blade, tabulated at the stations x (r/R)."""
    R = rotor.radius
    omega = rotor.omega
    c75 = float(rotor.chord_at(0.75))
    m = rotor.blade_mass_per_length or 3.0 * rho * rotor.airfoil.lift_slope * c75 * R / rotor.lock_number
    hinged = rotor.hub == "articulated"
    if hinged:
        nu1 = max(rotor.flap_frequency, 1.0)
        e = 2.0 * (nu1 ** 2 - 1.0) / (3.0 + 2.0 * (nu1 ** 2 - 1.0))
        root, target_mode, target = e, 1, rotor.flap2_frequency
    else:
        root = rotor.clamp_radius
        target_mode = 0
        target = rotor.flap_frequency if rotor.hub != "teetering" else rotor.coning_frequency
    # EI from the target frequency (per rev), by bisection in log EI
    lo, hi = -2.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f, _, _ = _flap_eigen(R, root, m, 10.0 ** mid, omega, hinged)
        if f[target_mode] / omega > target:
            hi = mid
        else:
            lo = mid
    EI = 10.0 ** (0.5 * (lo + hi))
    freqs, vecs, nb = _flap_eigen(R, root, m, EI, omega, hinged)
    xi = np.clip((np.asarray(x) - root) / (1.0 - root), 0.0, None)
    phi, d1, _ = _basis(xi, hinged, nb)
    shapes, slopes, masses = [], [], []
    xq = 0.5 * (_GL_X + 1.0)
    pq, _, _ = _basis(xq, hinged, nb)
    tip, _, _ = _basis(np.array([1.0]), hinged, nb)
    for k in range(n_flap):
        v = vecs[:, k]
        scale = float(tip[:, 0] @ v)
        v = v / scale
        s = v @ phi
        s = np.where(np.asarray(x) >= root, s, 0.0)
        shapes.append(s)
        slopes.append(np.where(np.asarray(x) >= root, (v @ d1) / (R * (1.0 - root)), 0.0))
        masses.append(m * R * (1.0 - root) * np.sum(0.5 * _GL_W * (v @ pq) ** 2))
    # torsion
    kappa = rotor.torsion_rigid_fraction
    xs = np.asarray(x)
    ts = (1.0 - kappa) * np.sin(0.5 * np.pi * xs) + kappa
    xg = 0.5 * (_GL_X + 1.0)
    i_theta = m * (rotor.torsion_gyration * c75) ** 2
    tg = (1.0 - kappa) * np.sin(0.5 * np.pi * xg) + kappa
    I_t = i_theta * R * np.sum(0.5 * _GL_W * tg ** 2)
    return BladeModes(x=xs, flap_shapes=np.array(shapes), flap_slopes=np.array(slopes), flap_freq=freqs[:n_flap],
                      flap_mass=np.array(masses), flap_ratio=freqs[:n_flap] / omega, torsion_shape=ts,
                      torsion_freq=rotor.torsion_frequency * omega, torsion_inertia=I_t,
                      torsion_ratio=rotor.torsion_frequency, m=m, EI=EI, hinged=hinged, root=root,
                      zeta_flap=rotor.structural_damping, zeta_torsion=rotor.torsion_damping)
