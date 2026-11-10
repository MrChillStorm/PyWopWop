"""Blade loads: a trimmed blade-element rotor model with momentum inflow, elastic blades and a prescribed
tip-vortex wake.

Rotor frame ("canonical"): origin at the hub, z along the tip-path-plane normal (thrust), x downstream in the plane
(the in-plane component of the air's velocity relative to the rotor), y = z x x; the rotor turns anticlockwise about
z (clockwise rotors are computed mirrored, see operating.py). Azimuth psi is measured from x (downstream) in the
direction of rotation, so the advancing blade is at psi = 90 deg.

- Hover and low speed: blade-element momentum theory per annulus with Prandtl tip loss.
- Forward flight: Glauert momentum for the mean inflow, Drees' linear distribution over the disc.
- Trim: collective and cyclic (relative to the tip-path plane) for the required thrust with no first-harmonic
  flapping, so the tip-path plane is where the aircraft's balance of forces puts it.
- Elastic blades (structure.py): flap bending (with the coning) and torsion, their periodic response by harmonic
  balance coupled into the trim. Deflections enter the angle of attack (flap velocity, slope, elastic twist) and
  the wake (release height, blade height: the blade-vortex miss distance).
- Blade-vortex interaction: Beddoes' prescribed tip-vortex wake (the vortex sinks with the local Beddoes inflow,
  doubled downstream of the disc), Vatistas cores growing by Squire's law, Biot-Savart induced velocity sampled
  over the chord with thin-airfoil lift weights (a lifting chord rather than a lifting line). Only the impulsive part
  is kept (the difference from the same wake with a wide core), the smooth part being the inflow model's; it builds
  up on the blade through Wagner's indicial response, scaled for compressibility.
"""
from dataclasses import dataclass, field
import warnings

import numpy as np
from numba import njit, prange

from .model import Rotor
from .structure import blade_modes


@dataclass
class AeroOptions:
    n_radial: int = 40
    n_azimuth: int = 720            # per revolution (a multiple of the blade count is enforced)
    bvi: bool = True
    # a ground run's trim at a fixed collective (operating._at_collective) turns this on: its rotors that set signed_hover_inflow give
    # their annuli loaded downward the mirrored momentum balance (_hover_inflow). Nowhere else: in flight the option moved the
    # MV-22B's hover and airplane-mode trims off their tests
    signed_hover_inflow: bool = False
    elastic: bool = True            # use the rotor's elastic blade model when it has one
    wake_dpsi_deg: float = 2.0      # wake discretisation
    wake_revs_max: float = 4.0
    # tip vortex core and strength: effective values for a lifting line, calibrated on HART II's own rotor so its baseline
    # case's 6-40 BPF noise on the microphones' plane comes out at the level measured (mean error about zero, its load's
    # swings within 1.5 times the measured): with the measured core (about 0.12 chord) and the whole of the peak
    # circulation a lifting line is 11 dB too loud there, as the real vortex is displaced and spread by the approaching
    # blade
    core_radius_chords: float = 0.8
    core_growth_a1: float = 1.0e-4         # Squire / Bhagwat-Leishman turbulent viscosity coefficient
    smooth_core_chords: float = 1.5        # the wide core whose field is left to the inflow model
    near_wake_deg: float = 30.0            # the blade's own vortex is left to the tip loss for this long
    rollup: float = 0.7                    # fraction of the peak bound circulation in the tip vortex
    chord_points: int = 4                  # lifting-chord sampling of the vortex-induced velocity
    unsteady: bool = True                  # Wagner response to the vortex-induced angle of attack
    wake_iterations: int = 3
    harmonics: int = 24                    # elastic response harmonics kept
    # blade-wake interaction: turbulence around the older tip vortices, met by the following blades
    wake_turbulence: float = 0.2           # rms turbulent velocity / the vortex's velocity at its turbulent radius
    wake_turbulent_radius_chords: float = 0.5  # turbulent radius after one revolution, chords (grows as sqrt(age))
    wake_turbulent_core_chords: float = 0.4    # the envelope's radius at the vortex's birth


# A tiltrotor's wing under its discs in a hover: it blocks part of each rotor's downwash, which runs inboard along the
# wing, meets the other rotor's over the fuselage and rises back through the discs, a fountain. From the XV-15: the
# inflow over the wing falls to 80 % across 0.6 of its chord with ramps of 0.1 chord either side (Rutledge, Coffen and
# George 1991, which gives the spikes of pressure measured behind it); the turbulence it brings back, from hot wires on
# a 1/12 scale model (Coffen, George, Hardinge and Stevenson 1991; George 1996, NASA CR-203040): about 1.5 m/s rms on
# the inflow of 6 m/s, over the disc's outer third (0.67 < r/R < 1) from the wing's leading edge rearward, of a length
# scale the disc's height over the wing, and intermittent: it switches from one rotor to the other.
FOUNTAIN_DEFICIT = 0.2
FOUNTAIN_FLAT_CHORDS = 0.6
FOUNTAIN_RAMP_CHORDS = 0.1
FOUNTAIN_RMS = 0.25                 # of the hover induced velocity
FOUNTAIN_INTERMITTENCY = 0.5        # the share of the time a rotor has it
FOUNTAIN_R_INNER = 0.67
FOUNTAIN_AFT_DEG = 60.0             # how far behind the wing it reaches round the disc


@dataclass
class Fountain:
    """Where a tiltrotor's wing lies under its disc (rotor frame)."""
    psi_inboard: float              # rad: the azimuth pointing at the other rotor
    aft_sign: float                 # +1: the tail lies at higher azimuths than psi_inboard, -1 at lower
    mid_chord_ahead: float          # m: the wing's mid chord ahead of the hub
    leading_edge_ahead: float       # m: its leading edge ahead of the hub
    chord: float                    # m
    clearance: float                # m: the disc's height over the wing
    v_hover: float                  # m/s: the hover induced velocity
    strength: float = 1.0           # 1 in a hover, falling as forward speed sweeps the wake off the wing


def _over_the_wing(bm, f):
    """(Npsi, Nr): each blade element's distance (m) inboard of the hub along the wing and aft of the hub across it"""
    r = bm.x[None, :] * bm.R
    d = bm.psi[:, None] - f.psi_inboard
    return r * np.cos(d), f.aft_sign * r * np.sin(d)


def fountain_inflow(bm, lam0, f):
    """The change in inflow ratio (Npsi, Nr) over the wing: a deficit, in the rotor's mean inflow, with a quarter sine
    up each side"""
    along, aft = _over_the_wing(bm, f)
    half = 0.5 * FOUNTAIN_FLAT_CHORDS * f.chord
    ramp = FOUNTAIN_RAMP_CHORDS * f.chord
    s = np.clip((half + ramp - np.abs(aft + f.mid_chord_ahead)) / ramp, 0.0, 1.0)
    return -FOUNTAIN_DEFICIT * f.strength * lam0 * np.sin(0.5 * np.pi * s) * (along > 0.0)


def fountain_turbulence(bm, f):
    """The fountain's rms turbulent velocity (Npsi, Nr, m/s) as the blades meet it, its intermittency in the mean
    square"""
    along, aft = _over_the_wing(bm, f)
    fwd = np.degrees(np.arcsin(np.clip(f.leading_edge_ahead / bm.R, -1.0, 1.0)))
    ang = np.degrees(np.arctan2(aft, along))
    edge = 10.0                     # deg: softened, as the fountain wanders
    win = np.clip((ang + fwd + edge) / edge, 0.0, 1.0) * np.clip((FOUNTAIN_AFT_DEG + edge - ang) / edge, 0.0, 1.0)
    rad = np.clip((bm.x[None, :] - FOUNTAIN_R_INNER + 0.05) / 0.1, 0.0, 1.0)
    return FOUNTAIN_RMS * f.v_hover * f.strength * np.sqrt(FOUNTAIN_INTERMITTENCY) * win * rad


@dataclass
class AeroSolution:
    rotor: Rotor
    omega: float
    R: float
    x: np.ndarray                   # r/R of the stations (Nr)
    dx: np.ndarray
    psi: np.ndarray                 # azimuths (Npsi)
    chord: np.ndarray               # m (Nr)
    mu_x: float
    mu_z: float                     # upflow through the disc / tip speed (positive: air rising through the disc)
    lambda0: float                  # mean induced inflow ratio
    theta0: float                   # collective at 0.75 R, rad
    theta1c: float
    theta1s: float
    beta0: float                    # mean coning of the tip, rad
    z: np.ndarray                   # (Npsi, Nr) height of the lifting line above the hub plane, m
    zdot: np.ndarray                # (Npsi, Nr) its velocity, m/s
    theta_e: np.ndarray             # (Npsi, Nr) elastic twist, rad
    UT: np.ndarray                  # (Npsi, Nr) m/s, into the leading edge
    UP: np.ndarray                  # (Npsi, Nr) m/s, down through the blade
    alpha: np.ndarray               # (Npsi, Nr) effective angle of attack, rad
    mach: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    fz: np.ndarray                  # (Npsi, Nr) N/m on the blade, along the rotor axis
    ft: np.ndarray                  # (Npsi, Nr) N/m on the blade, in the direction of rotation (negative: drag)
    gamma_bound: np.ndarray         # (Npsi, Nr) m^2/s
    thrust: float
    torque: float
    rho: float
    c: float
    nu: float
    dlambda_bvi: np.ndarray = None  # (Npsi, Nr) impulsive inflow from the wake
    dlambda_fountain: np.ndarray = None     # (Npsi, Nr) a tiltrotor's inflow over its wing in a hover
    fz_base: np.ndarray = None      # the loads without the impulsive part (same trim): BVI = fz - fz_base
    ft_base: np.ndarray = None
    wake: dict = field(default=None, repr=False)
    modes: object = field(default=None, repr=False)
    ground_effect: float = 1.0      # induced inflow factor of the ground (1: out of ground effect)
    bwi_rms: np.ndarray = None      # (Npsi, Nr) rms turbulent velocity from the wake met by the blade, m/s
    bwi_scale: np.ndarray = None    # (Npsi, Nr) its length scale, m
    model: object = field(default=None, repr=False)     # the blade model, for load variants
    lam_i: object = field(default=None, repr=False)
    opts: object = field(default=None, repr=False)

    @property
    def power(self):
        return self.torque * self.omega

    @property
    def ct(self):
        return self.thrust / (self.rho * np.pi * self.R ** 2 * (self.omega * self.R) ** 2)

    @property
    def ct_sigma(self):
        return self.ct / self.rotor.solidity

    @property
    def r(self):
        return self.x * self.R

    @property
    def dr(self):
        return self.dx * self.R

    @property
    def advancing_tip_mach(self):
        return (1.0 + self.mu_x) * self.omega * self.R / self.c


def radial_stations(root, n):
    """Panel edges clustered towards the tip (half-cosine), their midpoints and widths, as r/R."""
    e = root + (1.0 - root) * np.sin(0.5 * np.pi * np.arange(n + 1) / n)
    return 0.5 * (e[1:] + e[:-1]), np.diff(e)


def glauert_inflow(ct, mu_x, mu_z):
    """Mean induced inflow ratio from Glauert's momentum theory; in near-axial descent the empirical vortex-ring
    curve (Leishman, Principles of Helicopter Aerodynamics, eq. 2.124) bridges the gap momentum theory leaves."""
    ct = max(ct, 1e-8)
    lh = np.sqrt(ct / 2.0)
    lam_i = lh
    ok = False
    for _ in range(100):
        lam = lam_i - mu_z
        s = np.sqrt(mu_x ** 2 + lam ** 2)
        f = lam_i - ct / (2.0 * s)
        df = 1.0 + ct * lam / (2.0 * s ** 3)
        step = f / df if abs(df) > 1e-6 else f
        lam_i = max(lam_i - np.clip(step, -0.5 * lh, 0.5 * lh), 1e-6)
        if abs(step) < 1e-10:
            ok = True
            break
    vrs = mu_x < 0.6 * lh and -2.0 * lh < -mu_z < 0.0
    if not ok or vrs:
        xd = -mu_z / lh                 # axial velocity / hover induced velocity, negative descending
        if -2.0 <= xd <= 0.0:
            k = 1.15 - 1.125 * xd - 1.372 * xd ** 2 - 1.718 * xd ** 3 - 0.655 * xd ** 4
            lam_axial = k * lh
            w = np.clip(mu_x / (0.6 * lh), 0.0, 1.0)
            lam_i = (1.0 - w) * lam_axial + w * lam_i if ok else lam_axial
    return lam_i


def ground_effect(R, height, mu_x, lam):
    """Cheeseman and Bennett's ground effect: the induced inflow's reduction factor for a hub `height` (m) above the
    ground, fading with forward speed as the wake is swept back: 1 - (R/4z)^2 / (1 + (mu/lambda)^2). Held above 0.5
    (the formula's validity ends near z = R/4)."""
    if height is None or height <= 0:
        return 1.0
    k = (R / (4.0 * height)) ** 2 / (1.0 + (mu_x / max(lam, 1e-4)) ** 2)
    return float(np.clip(1.0 - k, 0.5, 1.0))


def drees(lam_total, mu_x):
    """Drees' first-harmonic inflow coefficients kx (fore-aft) and ky (lateral)."""
    chi = np.arctan2(mu_x, max(lam_total, 1e-6))
    s = np.sin(chi)
    kx = 4.0 / 3.0 * (1.0 - np.cos(chi) - 1.8 * mu_x ** 2) / s if s > 1e-4 else 0.0
    return kx, -2.0 * mu_x, chi


class _BladeModel:
    """Vectorised blade-element evaluation on the (psi, r) grid, with the blade's elastic state."""

    def __init__(self, rotor, omega, V, alpha_tpp, rho, c, nu, opts):
        self.rotor = rotor
        self.opts = opts
        self.B = rotor.blades
        self.R = rotor.radius
        self.omega = omega
        self.rho, self.c, self.nu = rho, c, nu
        npsi = int(np.ceil(opts.n_azimuth / self.B) * self.B)
        self.psi = np.arange(npsi) * 2.0 * np.pi / npsi
        self.x, self.dx = radial_stations(rotor.root_cutout, opts.n_radial)
        self.r = self.x * self.R
        self.dr = self.dx * self.R
        self.chord = rotor.chord_at(self.x)
        self.twist = rotor.twist_at(self.x)
        self.mu_x = V * np.cos(alpha_tpp) / (omega * self.R)
        self.mu_z = V * np.sin(alpha_tpp) / (omega * self.R)
        self.cos, self.sin = np.cos(self.psi)[:, None], np.sin(self.psi)[:, None]
        self.area = np.pi * self.R ** 2
        self.vtip = omega * self.R
        c75 = rotor.chord_at(0.75)
        self.I_beta = rho * rotor.airfoil.lift_slope * c75 * self.R ** 4 / rotor.lock_number
        self.extra_lambda = np.zeros((npsi, len(self.x)))
        self.dalpha_unsteady = None
        self.w_ff = float(np.clip((self.mu_x - 0.01) / 0.04, 0.0, 1.0))
        self.lam_hover = None
        self.ground_height = None      # hub height above the ground (m), for ground effect
        self.k_ge = 1.0
        self.beta_p = np.radians(rotor.precone_deg)
        self.modes = blade_modes(rotor, self.x, rho) if (rotor.elastic and opts.elastic) else None
        if self.modes is not None:
            md = self.modes
            # with the rotor at this speed (the modes were set up at the nominal one)
            md.flap_freq = md.flap_ratio * omega
            md.torsion_freq = md.torsion_ratio * omega
            self.q = np.zeros((len(md.flap_freq), npsi), complex)       # flap modal amplitudes, spectra
            self.p = np.zeros(npsi, complex)                              # torsion
        self.set_rigid(self.beta_p)

    # ---------------------------------------------------------------- elastic state
    def z_base(self):
        """Undeflected blade: the precone line (clamped blades) or the hub plane (hinged)."""
        if self.modes is not None and self.modes.hinged:
            return np.zeros_like(self.r), np.zeros_like(self.r)
        return self.r * np.sin(self.beta_p), np.full_like(self.r, np.sin(self.beta_p))

    def set_rigid(self, beta0):
        n = len(self.psi)
        self.beta0 = beta0
        self.z = np.broadcast_to(self.r * np.sin(beta0), (n, len(self.r))).copy()
        self.slope = np.full_like(self.z, np.sin(beta0))
        self.zdot = np.zeros_like(self.z)
        self.zddot = np.zeros_like(self.z)
        self.theta_e = np.zeros_like(self.z)

    def modal_response(self, ld):
        """Flap and torsion modal spectra in equilibrium with the loads (harmonic balance), the aerodynamic flap
        damping and torsional stiffness moved to the left-hand side to precondition the iteration."""
        md = self.modes
        n = len(self.psi)
        harm = np.fft.fftfreq(n, 1.0 / n)
        keep = np.abs(harm) <= self.opts.harmonics
        W = self.omega
        a = self.rotor.airfoil.lift_slope
        fz = ld['fz']
        if not md.hinged:
            # centrifugal force on the precone line, normal to it, drives the deflection back towards the plane
            fz = fz - md.m * W ** 2 * self.r[None, :] * np.sin(self.beta_p)
        nf = len(md.flap_freq)
        iw = 1j * harm * W
        w2 = (harm * W) ** 2
        phi = md.flap_shapes
        tsh = md.torsion_shape
        dr = self.dr
        # structure: flap modes and the torsion mode, coupled by the section CG's offset d aft of the elastic axis
        # (kinetic energy m (w' - d theta')^2 / 2): flap gets + m d theta'', torsion + m d w''
        d = (self.rotor.cg_axis - self.rotor.elastic_axis) * self.chord
        C = np.array([np.sum(md.m * d * phi[k] * tsh * dr) for k in range(nf)])
        # and the centrifugal force through the offset CG of a sloped blade: a nose-up moment m d Omega^2 r w'
        # (potential energy -int m d Omega^2 r w' theta dr, so the reciprocal term acts on the flap modes)
        cf = md.m * d * W ** 2 * self.r
        Kc = np.array([np.sum(cf * md.flap_slopes[k] * tsh * dr) for k in range(nf)])
        A = np.zeros((n, nf + 1, nf + 1), complex)
        for k in range(nf):
            A[:, k, k] = md.flap_mass[k] * (md.flap_freq[k] ** 2 - w2 + 2j * md.zeta_flap * md.flap_freq[k] * harm * W)
            A[:, k, nf] = w2 * C[k] - Kc[k]
            A[:, nf, k] = w2 * C[k] - Kc[k]
        wt = md.torsion_freq
        A[:, nf, nf] = md.torsion_inertia * (wt ** 2 - w2 + 2j * md.zeta_torsion * wt * harm * W)
        # the azimuth-mean linearised quasi-steady aerodynamics, moved to the left-hand side so the iteration is a
        # Newton method with an approximate Jacobian: lift from elastic twist (B), flap damping (Cf), the elastic
        # axis' moment from twist (G) and from flap velocity (E); da = theta - wdot / U
        beta = np.sqrt(np.maximum(1.0 - np.minimum(ld['mach'], 0.9) ** 2, 0.19))
        Um = np.mean(np.sqrt(ld['UT'] ** 2 + ld['UP'] ** 2), axis=0)
        qa = np.mean(0.5 * self.rho * (ld['UT'] ** 2 + ld['UP'] ** 2) * a / beta, axis=0) * self.chord
        ua = 0.5 * self.rho * Um * np.mean(a / beta, axis=0) * self.chord
        arm = (0.25 - self.rotor.elastic_axis) * self.chord
        L = np.zeros((nf + 1, nf + 1))
        D = np.zeros((nf + 1, nf + 1))
        for k in range(nf):
            for j in range(nf):
                D[k, j] = np.sum(ua * phi[k] * phi[j] * dr)
            L[k, nf] = -np.sum(qa * phi[k] * tsh * dr)
            D[nf, k] = -np.sum(ua * arm * tsh * phi[k] * dr)
        L[nf, nf] = max(np.sum(qa * arm * tsh ** 2 * dr), 0.0)
        Alin = L[None, :, :] + iw[:, None, None] * D[None, :, :]
        A = A + Alin
        x_old = np.concatenate([self.q, self.p[None, :]], axis=0).T          # (n, nf+1)
        rhs = np.zeros((n, nf + 1), complex)
        for k in range(nf):
            rhs[:, k] = np.fft.fft(np.sum(fz * phi[k][None, :] * dr[None, :], axis=1)) / n
        # the propeller moment: the section's mass off the plane of rotation pulls its pitch towards it. On the elastic twist
        # it is in the rotating torsion frequency; on the pitch the controls and the built-in twist set, it is a load
        prop = -md.m * (self.rotor.torsion_gyration * self.chord[None, :]) ** 2 * W ** 2 \
            * np.sin(ld['pitch']) * np.cos(ld['pitch'])
        rhs[:, nf] = np.fft.fft(np.sum((ld['mea'] + prop) * tsh[None, :] * dr[None, :], axis=1)) / n
        _, slope0 = self.z_base()
        rhs[0, nf] += np.sum(cf * slope0 * tsh * dr)          # the precone line's slope, steady
        rhs = rhs + np.einsum('nij,nj->ni', Alin, x_old)
        sol = np.linalg.solve(A[keep], rhs[keep][..., None])[..., 0]
        q_new = np.zeros_like(self.q)
        p_new = np.zeros_like(self.p)
        q_new[:, keep] = sol[:, :nf].T
        p_new[keep] = sol[:, nf]
        return q_new, p_new

    def tip_first_harmonic(self, q):
        """Cosine and sine first-harmonic tip deflection (m) of modal spectra q: the flapping relative to the
        tip-path plane, which the trim sets to zero."""
        t1 = np.sum(q[:, 1])
        return 2.0 * t1.real, -2.0 * t1.imag

    def update_elastic(self, ld, relax=0.6):
        """Relax the elastic state towards equilibrium with the loads; returns the change in tip deflection (m) and
        elastic twist (rad)."""
        n = len(self.psi)
        q_new, p_new = self.modal_response(ld)
        q = (1.0 - relax) * self.q + relax * q_new
        p = (1.0 - relax) * self.p + relax * p_new
        dtip = np.max(np.abs(np.fft.ifft(np.sum(q - self.q, axis=0)) * n))
        dth = np.max(np.abs(np.fft.ifft(p - self.p) * n))
        self.q, self.p = q, p
        self._rebuild()
        return dtip, dth

    def _rebuild(self):
        md = self.modes
        n = len(self.psi)
        harm = np.fft.fftfreq(n, 1.0 / n)
        qt = np.real(np.fft.ifft(self.q, axis=1) * n)                       # (modes, Npsi)
        qd = np.real(np.fft.ifft(1j * harm * self.q, axis=1) * n) * self.omega
        qdd = np.real(np.fft.ifft(-harm ** 2 * self.q, axis=1) * n) * self.omega ** 2
        zb, sb = self.z_base()
        self.z = zb[None, :] + qt.T @ md.flap_shapes
        self.slope = sb[None, :] + qt.T @ md.flap_slopes
        self.zdot = qd.T @ md.flap_shapes
        self.zddot = qdd.T @ md.flap_shapes
        pt = np.real(np.fft.ifft(self.p) * n)
        self.theta_e = pt[:, None] * md.torsion_shape[None, :]
        self.beta0 = float(np.arcsin(np.clip(np.mean(self.z[:, -1]) / self.r[-1], -1.0, 1.0)))

    # ---------------------------------------------------------------- loads
    def loads(self, theta0, th1c, th1s, lam_i, single=False):
        """Section loads for the controls and induced inflow lam_i (Npsi, Nr) or (1, Nr); single: at psi = 0 only
        (axisymmetric hover inflow iteration)."""
        x = self.x[None, :]
        vt = self.vtip
        if single:
            cos, sin = np.ones((1, 1)), np.zeros((1, 1))
            slope, zdot, the = self.slope[:1], self.zdot[:1], self.theta_e[:1]
        else:
            cos, sin = self.cos, self.sin
            slope, zdot, the = self.slope, self.zdot, self.theta_e
        ut = vt * (x + self.mu_x * sin)
        up_base = vt * (lam_i - self.mu_z + self.mu_x * cos * slope) + zdot
        theta = theta0 + self.twist[None, :] + th1c * cos + th1s * sin + the
        phi_base = np.arctan2(up_base, ut)
        up = up_base
        phi = phi_base
        alpha = theta - phi
        if not single and np.any(self.extra_lambda):
            # the vortex's impulsive upwash perturbs the angle of attack (through the indicial response); the forces
            # stay resolved about the base inflow, so the perturbation doesn't tilt the lift into spurious power
            if self.dalpha_unsteady is None:
                alpha = alpha - (np.arctan2(up_base + vt * self.extra_lambda, ut) - phi_base)
            else:
                alpha = alpha + self.dalpha_unsteady
        U = np.sqrt(ut ** 2 + up ** 2)
        mach = U / self.c
        af = self.rotor.airfoil
        cl, cd = af.coefficients(alpha, mach)
        # Prandtl tip loss with the local inflow angle (smooth to zero at the tip) on the lift in forward flight;
        # in the hover the annulus momentum balance carries it (w_ff blends the two)
        phi_tl = np.maximum(np.abs(np.mean(up_base) / vt) / x, 0.02)
        F = 2.0 / np.pi * np.arccos(np.exp(-self.B * (1.0 - x) / (2.0 * x * phi_tl)))
        cl = cl * F ** self.w_ff
        q = 0.5 * self.rho * U ** 2 * self.chord[None, :]
        L = q * cl
        D = q * cd
        Ue = np.maximum(U, 1e-6)
        fz = (L * ut - D * up) / Ue
        ft = -(L * up + D * ut) / Ue
        gamma = 0.5 * U * self.chord[None, :] * cl
        # pitching moment about the elastic axis, nose up positive (lift acts at the quarter chord); q holds one chord
        mea = q * self.chord[None, :] * (af.moment(alpha, mach) + cl * (self.rotor.elastic_axis - 0.25))
        return dict(UT=ut, UP=up, alpha=alpha, mach=mach, cl=cl, cd=cd, fz=fz, ft=ft, gamma=gamma, phi=phi, mea=mea,
                    pitch=theta - the)

    def integrate(self, ld):
        T = self.B * np.mean(np.sum(ld['fz'] * self.dr, axis=1))
        Q = -self.B * np.mean(np.sum(ld['ft'] * self.r * self.dr, axis=1))
        shape = self.modes.flap_shapes[0] * self.R if self.modes is not None else self.r
        Mb = np.sum(ld['fz'] * shape * self.dr, axis=1)       # first flap mode's generalised force (moment)
        m1c = 2.0 * np.mean(Mb * self.cos[:, 0])
        m1s = 2.0 * np.mean(Mb * self.sin[:, 0])
        return T, Q, np.mean(np.sum(ld['fz'] * self.r * self.dr, axis=1)), m1c, m1s

    def coning(self, mean_moment):
        """Coning of a rigid blade with an equivalent hinge spring."""
        rot = self.rotor
        if rot.hub == "teetering":
            return self.beta_p
        nu2 = rot.flap_frequency ** 2
        return (mean_moment / (self.I_beta * self.omega ** 2) + (nu2 - 1.0) * self.beta_p) / nu2


def _hover_inflow(bm, theta0, lam_mean_ff):
    """Induced inflow per annulus (BEMT with tip loss). An annulus loaded downward (a twisted blade's tip at flat pitch,
    its root still lifting) has no induced flow, unless the rotor has signed_hover_inflow and the trim is a ground run's
    (AeroOptions.signed_hover_inflow): then the momentum balance mirrored, the annulus hovering upside down and driving
    its air up through the disc, lambda = mu_z/2 - sqrt(mu_z^2/4 + q) (the hover's solution with the thrust and the
    flow reversed)"""
    x = bm.x
    B = bm.B
    signed = bool(getattr(bm.opts, "signed_hover_inflow", False) and getattr(bm.rotor, "signed_hover_inflow", False))
    lam = bm.lam_hover if bm.lam_hover is not None else np.full_like(x, max(lam_mean_ff, 0.01))
    for _ in range(80):
        ld = bm.loads(theta0, 0.0, 0.0, lam[None, :], single=True)
        fz = ld['fz'][0]
        phi = np.maximum(np.abs(lam - bm.mu_z) / x, 0.01)
        F = 2.0 / np.pi * np.arccos(np.exp(-B * (1.0 - x) / (2.0 * x * phi)))
        F = np.maximum(F, 0.05)
        qn = B * (np.abs(fz) if signed else np.maximum(fz, 0.0)) / (4.0 * np.pi * bm.rho * x * bm.R * F * bm.vtip ** 2)
        root = np.sqrt(bm.mu_z ** 2 + 4.0 * qn)
        new = 0.5 * (bm.mu_z + (np.where(fz < 0.0, -root, root) if signed else root)) * bm.k_ge
        if np.max(np.abs(new - lam)) < 1e-7:
            lam = new
            break
        lam = 0.5 * lam + 0.5 * new
    bm.lam_hover = lam
    return lam


def _induced(bm, theta0, lam0):
    kx, ky, _ = drees(lam0 - bm.mu_z, bm.mu_x)
    lam_ff = lam0 * (1.0 + kx * bm.x[None, :] * bm.cos + ky * bm.x[None, :] * bm.sin)
    if bm.w_ff >= 1.0:
        return lam_ff
    lam_h = _hover_inflow(bm, theta0, lam0)[None, :]
    return (1.0 - bm.w_ff) * lam_h + bm.w_ff * lam_ff


def trim(bm, thrust, controls0=None, cyclic=True, tol=1e-5):
    """Collective (and cyclic) for the thrust with zero first-harmonic flapping, the elastic response converged
    with it. Returns controls, mean induced inflow, induced inflow field, loads, (thrust, torque)."""
    ct = thrust / (bm.rho * bm.area * bm.vtip ** 2)
    lam0 = glauert_inflow(ct, bm.mu_x, bm.mu_z)
    bm.k_ge = ground_effect(bm.R, bm.ground_height, bm.mu_x, lam0 - bm.mu_z)
    lam0 = lam0 * bm.k_ge
    sigma = bm.rotor.solidity
    a = bm.rotor.airfoil.lift_slope
    if controls0 is None:
        # the inflow's part as the inflow angle at 0.75 R: 1.5 lambda for a helicopter's, a propeller's too (1.5 lambda
        # at a propeller's 0.5-0.6 starts the Newton iteration in the stall, where it finds a second, stalled solution)
        th0 = (6.0 * ct / (sigma * a) + 1.125 * np.arctan((lam0 - bm.mu_z) / 0.75)
               + np.radians(bm.rotor.airfoil.alpha0_deg))
        u = np.array([th0, 0.0, 0.0])
    else:
        u = np.array(controls0, float)
    m_ref = max(thrust, 1.0) * bm.R / bm.B
    t_ref = max(thrust, 1.0)
    n = 3 if cyclic else 1

    tip_ref = 0.01 * bm.R

    def evaluate(u):
        lam_i = _induced(bm, u[0], lam0)
        ld = bm.loads(u[0], u[1], u[2], lam_i)
        T, Q, Mb, m1c, m1s = bm.integrate(ld)
        if bm.modes is not None:
            # flapping relative to the tip-path plane, from the elastic response to these loads
            tc, ts = bm.tip_first_harmonic(bm.modal_response(ld)[0])
            m1c, m1s = tc / tip_ref * m_ref, ts / tip_ref * m_ref
        return ld, lam_i, (T, Q, Mb, m1c, m1s)

    def state():
        return (bm.q.copy(), bm.p.copy()) if bm.modes is not None else bm.beta0

    def restore(s):
        if bm.modes is not None:
            bm.q, bm.p = s[0].copy(), s[1].copy()
            bm._rebuild()
        else:
            bm.set_rigid(s)

    def residual(ev):
        T, _, _, c, s = ev[2]
        return np.array([(T - thrust) / t_ref, c / m_ref, s / m_ref])[:n]

    J = None
    elastic_change = 1.0
    radius = 0.05                   # rad: the largest step, shrunk if the iteration runs away
    best = None                     # the smallest residual so far: (its size, the controls, the elastic state)
    for it in range(120):
        ev = evaluate(u)
        ld, lam_i, (T, Q, Mb, m1c, m1s) = ev
        res = residual(ev)
        size = float(np.max(np.abs(res)))
        if size < tol and elastic_change < 2e-5:
            break
        if it >= 6:                 # by then the elastic state has left its start
            if best is None or size < best[0]:
                best = (size, u.copy(), state())
            elif size > max(3.0 * best[0], 0.05) and radius > 0.005:
                # running away, a stale Jacobian and the elastic state chasing each other: back to the best point, in
                # smaller steps from a fresh Jacobian
                u = best[1].copy()
                restore(best[2])
                radius *= 0.5
                J = None
                ev = evaluate(u)
                ld, lam_i, (T, Q, Mb, m1c, m1s) = ev
                res = residual(ev)
        if J is None or it % 3 == 0 or radius < 0.05:
            # with the elastic state frozen, as res was evaluated
            J = np.zeros((n, n))
            h = 1e-4
            for k in range(n):
                du = u.copy()
                du[k] += h
                _, _, (T2, _, _, c2, s2) = evaluate(du)
                J[:, k] = (np.array([(T2 - thrust) / t_ref, c2 / m_ref, s2 / m_ref])[:n] - res) / h
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        u[:n] += np.clip(step, -radius, radius)
        if bm.modes is not None:
            dtip, dth = bm.update_elastic(ld)
            elastic_change = dtip / bm.R + dth
        else:
            beta0 = bm.coning(Mb)
            elastic_change = abs(beta0 - bm.beta0)
            bm.set_rigid(beta0)
    ev = evaluate(u)
    if best is not None and float(np.max(np.abs(residual(ev)))) > best[0]:
        u = best[1].copy()
        restore(best[2])
        ev = evaluate(u)
    ld, lam_i, (T, Q, Mb, m1c, m1s) = ev
    # against the thrust asked for, or near none (a run-up at flat pitch) the rotor's own scale: 0.001 in C_T / sigma
    miss = abs(T - thrust) - 0.001 * sigma * bm.rho * bm.area * bm.vtip ** 2
    if miss > 0.1 * t_ref:
        raise ValueError(f"the trim failed: {T:.0f} N of the {thrust:.0f} N asked for, at advance ratio {bm.mu_x:.3f}")
    if miss > 0.02 * t_ref:
        warnings.warn(f"the trim fell short: {T:.0f} N of the {thrust:.0f} N asked for, at advance ratio {bm.mu_x:.3f}")
    return u, lam0, lam_i, ld, (T, Q)


# ---------------------------------------------------------------- prescribed wake and Biot-Savart

def beddoes_wake_z(psi_rel, zeta, mu_x, mu_z, lam0, r_v, z0):
    """Height (z/R) of the tip vortex released at azimuth psi_rel after age zeta (rad), integrated from Beddoes'
    descent velocities: lam0 (1 + E x - E |y|^3) under the disc, 2 lam0 (1 - E |y|^3) downstream of it.
    z0: release height (z/R) per release azimuth."""
    chi = np.arctan2(mu_x, max(lam0 - mu_z, 1e-6))
    E = chi / 2.0
    x = r_v * np.cos(psi_rel)[:, None] + mu_x * zeta[None, :]
    y = r_v * np.sin(psi_rel)[:, None] * np.ones_like(zeta)[None, :]
    inside = x <= np.sqrt(np.maximum(1.0 - y ** 2, 0.0))
    ay3 = np.abs(y) ** 3
    lam = np.where(inside, lam0 * (1.0 + E * x - E * ay3), 2.0 * lam0 * (1.0 - E * ay3))
    dz = mu_z - lam
    z = np.zeros_like(x)
    dzeta = np.diff(zeta)[None, :]
    z[:, 1:] = np.cumsum(0.5 * (dz[:, 1:] + dz[:, :-1]) * dzeta, axis=1)
    return np.asarray(z0)[:, None] + z, x, y


@njit(parallel=True, fastmath=True, cache=True)
def _bvi_kernel(xs, chord, zb, cospsi, sinpsi, wx, wy, wz, gam, rc, rc_s, nb, npsi, step, n_near, R, xq, wq, cull):
    """Vertical induced velocity, impulsive part, at blade 0's stations for every azimuth from all blades' tip
    vortices, averaged over the chord with the weights wq at chord fractions xq.

    wx, wy, wz (npsi_rel, nz): vortex points (m) by release azimuth index and age index; gam (npsi_rel,);
    rc, rc_s (nz,): true and wide core radii by age. Blade b's vortex at blade-0 azimuth j has release index
    (j + b * npsi / nb - i * step) mod npsi at age index i. zb (npsi, nr): blade height. Segments farther from the
    blade than cull plus their wide core and length are skipped: there the two fields are the same."""
    nr = xs.shape[0]
    nq = xq.shape[0]
    nz = wx.shape[1]
    out = np.zeros((npsi, nr))
    inv4pi = 1.0 / (4.0 * np.pi)
    r0 = xs[0] * R
    r1 = xs[nr - 1] * R
    for j in prange(npsi):
        cj, sj = cospsi[j], sinpsi[j]
        zr, zt = zb[j, 0], zb[j, nr - 1]
        for b in range(nb):
            i0 = n_near if b == 0 else 0
            for i in range(i0, nz - 1):
                ka = (j + b * (npsi // nb) - i * step) % npsi
                kb = (j + b * (npsi // nb) - (i + 1) * step) % npsi
                g = 0.5 * (gam[ka] + gam[kb])
                if g == 0.0:
                    continue
                ax, ay, az = wx[ka, i], wy[ka, i], wz[ka, i]
                bx, by, bz = wx[kb, i + 1], wy[kb, i + 1], wz[kb, i + 1]
                r0x, r0y, r0z = bx - ax, by - ay, bz - az
                l2 = r0x * r0x + r0y * r0y + r0z * r0z
                rcs = 0.5 * (rc_s[i] + rc_s[i + 1])
                # cull: distance from the segment's midpoint to the blade's span line
                mx, my, mz = 0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (az + bz)
                s = mx * cj + my * sj
                if s < r0:
                    s = r0
                elif s > r1:
                    s = r1
                qz = zr + (s - r0) * (zt - zr) / (r1 - r0)
                dx_, dy_, dz_ = mx - s * cj, my - s * sj, mz - qz
                if dx_ * dx_ + dy_ * dy_ + dz_ * dz_ > (0.5 * np.sqrt(l2) + 4.0 * rcs + cull) ** 2:
                    continue
                c4 = 0.5 * (rc[i] + rc[i + 1])
                c4 = c4 * c4 * c4 * c4
                s4 = rcs * rcs * rcs * rcs
                for p in range(nr):
                    acc = 0.0
                    for q in range(nq):
                        off = (0.25 - xq[q]) * chord[p]          # along the direction of motion
                        px = xs[p] * R * cj - off * sj
                        py = xs[p] * R * sj + off * cj
                        pz = zb[j, p]
                        r1x, r1y, r1z = px - ax, py - ay, pz - az
                        r2x, r2y, r2z = px - bx, py - by, pz - bz
                        cx = r1y * r2z - r1z * r2y
                        cy = r1z * r2x - r1x * r2z
                        cz = r1x * r2y - r1y * r2x
                        c2 = cx * cx + cy * cy + cz * cz
                        n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z) + 1e-12
                        n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z) + 1e-12
                        dot = (r0x * (r1x / n1 - r2x / n2) + r0y * (r1y / n1 - r2y / n2)
                               + r0z * (r1z / n1 - r2z / n2))
                        h2 = c2 / l2
                        h4 = h2 * h2
                        base = g * inv4pi * cz * dot / l2
                        acc += wq[q] * (base / np.sqrt(c4 + h4) - base / np.sqrt(s4 + h4))
                    out[j, p] += acc
    return out


def wagner_filter(dalpha, UT, chord, mach, omega, revs=3):
    """Wagner indicial response (1 - 0.165 e^-0.0455s - 0.335 e^-0.3s, s in semichords, exponents scaled by
    beta^2) to the vortex-induced angle-of-attack history, periodic in azimuth. dalpha, UT, mach: (Npsi, Nr)."""
    npsi = dalpha.shape[0]
    dt = 2.0 * np.pi / npsi / omega
    beta2 = np.maximum(1.0 - np.minimum(mach, 0.95) ** 2, 0.1)
    ds = 2.0 * np.abs(UT) * dt / chord[None, :]
    q = beta2 * ds
    return _wagner_steps(dalpha, np.exp(-0.0455 * q), np.exp(-0.3 * q), np.exp(-0.0455 * q / 2.0), np.exp(-0.3 * q / 2.0), revs)


@njit(fastmath=True, cache=True)
def _wagner_steps(dalpha, e0, e1, h0, h1, revs):
    """wagner_filter's two lag states, stepped round the azimuth to their periodic state; e: a step's decay, h: half of it"""
    npsi, nr = dalpha.shape
    out = np.zeros_like(dalpha)
    for i in range(nr):
        x0 = 0.0
        x1 = 0.0
        prev = dalpha[npsi - 1, i]
        for rev in range(revs):
            for j in range(npsi):
                a = dalpha[j, i]
                d = a - prev
                x0 = x0 * e0[j, i] + 0.165 * d * h0[j, i]
                x1 = x1 * e1[j, i] + 0.335 * d * h1[j, i]
                prev = a
                out[j, i] = a - x0 - x1
    return out


def solve_rotor(rotor, omega, V, alpha_tpp, thrust, rho, c, nu, opts=None, ground_height=None, fountain=None):
    """Trimmed loads of one rotor. V: air speed relative to the rotor (m/s); alpha_tpp: angle of the air's velocity
    above the tip-path plane (positive: air rising through the disc, as in descent); thrust (N) along the TPP normal;
    ground_height: the hub's height above the ground (m) for ground effect, None out of it; fountain: a Fountain, a
    tiltrotor's wing under the disc."""
    opts = opts or AeroOptions()
    bm = _BladeModel(rotor, omega, V, alpha_tpp, rho, c, nu, opts)
    bm.ground_height = ground_height
    cyclic = rotor.cyclic
    u, lam0, lam_i, ld, (T, Q) = trim(bm, thrust, cyclic=cyclic)

    def impose(dl):
        """an impulsive change in the inflow, through the blade's unsteady response"""
        bm.extra_lambda = dl
        if opts.unsteady:
            up_base = bm.vtip * (lam_i - bm.mu_z + bm.mu_x * bm.cos * bm.slope) + bm.zdot
            d_raw = -(np.arctan2(up_base + bm.vtip * dl, ld['UT']) - np.arctan2(up_base, ld['UT']))
            bm.dalpha_unsteady = wagner_filter(d_raw, ld['UT'], bm.chord, ld['mach'], omega)
    dlam_f = None
    if fountain is not None and fountain.strength > 0.01 and thrust > 0:
        dlam_f = fountain_inflow(bm, lam0, fountain)
        impose(dlam_f)
        u, lam0, lam_i, ld, (T, Q) = trim(bm, thrust, controls0=u, cyclic=cyclic)
    dlam = None
    wake = None
    if opts.bvi and bm.mu_x > 0.025 and thrust > 0:
        for it in range(opts.wake_iterations):
            dlam, wake = _bvi_inflow(bm, ld, lam0, opts)
            # ramp the impulsive part in over low advance ratios, where the wake model is least certain
            dlam = dlam * float(np.clip((bm.mu_x - 0.025) / 0.03, 0.0, 1.0))
            impose(dlam if dlam_f is None else dlam + dlam_f)
            u, lam0, lam_i, ld, (T, Q) = trim(bm, thrust, controls0=u, cyclic=cyclic)
    bwi_rms, bwi_scale = (_wake_turbulence(bm, ld, lam0, opts) if thrust > 0 and opts.wake_turbulence > 0
                          else (None, None))
    if dlam_f is not None:
        uf = fountain_turbulence(bm, fountain)
        if bwi_rms is None:
            bwi_rms, bwi_scale = uf, np.full_like(uf, fountain.clearance)
        else:
            u2 = bwi_rms ** 2 + uf ** 2
            bwi_scale = np.where(u2 > 0, (bwi_rms ** 2 * bwi_scale + uf ** 2 * fountain.clearance)
                                 / np.maximum(u2, 1e-30), bwi_scale)
            bwi_rms = np.sqrt(u2)
    base = None
    if dlam is not None:
        keep = bm.extra_lambda, bm.dalpha_unsteady
        if dlam_f is None:
            bm.extra_lambda = np.zeros_like(dlam)
            bm.dalpha_unsteady = None
        else:
            impose(dlam_f)
        base = bm.loads(u[0], u[1], u[2], lam_i)
        bm.extra_lambda, bm.dalpha_unsteady = keep
    return AeroSolution(rotor=rotor, omega=omega, R=bm.R, x=bm.x, dx=bm.dx, psi=bm.psi, chord=bm.chord,
                        mu_x=bm.mu_x, mu_z=bm.mu_z, lambda0=lam0, theta0=u[0], theta1c=u[1], theta1s=u[2],
                        beta0=bm.beta0, z=bm.z, zdot=bm.zdot, theta_e=bm.theta_e, UT=ld['UT'], UP=ld['UP'],
                        alpha=ld['alpha'], mach=ld['mach'], cl=ld['cl'], cd=ld['cd'], fz=ld['fz'], ft=ld['ft'],
                        gamma_bound=ld['gamma'], thrust=T, torque=Q, rho=rho, c=c, nu=nu, dlambda_bvi=dlam,
                        dlambda_fountain=dlam_f,
                        wake=wake, modes=bm.modes, fz_base=None if base is None else base['fz'],
                        ft_base=None if base is None else base['ft'], model=bm, lam_i=lam_i, opts=opts,
                        bwi_rms=bwi_rms, bwi_scale=bwi_scale, ground_effect=bm.k_ge)


def _bvi_inflow(bm, ld, lam0, opts, dz=None, gscale=None, dxy=None):
    """Impulsive inflow from the prescribed tip-vortex wake; dz(zeta) and dxy(zeta) (Npsi, nz arrays, in R) and
    gscale (Npsi) perturb the vortices' positions and strengths (wander and unsteadiness)."""
    rot = bm.rotor
    npsi = len(bm.psi)
    dpsi = 2.0 * np.pi / npsi
    step = max(1, int(round(np.radians(opts.wake_dpsi_deg) / dpsi)))
    zeta_max = np.clip(2.5 / max(bm.mu_x, 1e-3), 2.0 * np.pi, opts.wake_revs_max * 2.0 * np.pi)
    nz = int(zeta_max / (step * dpsi)) + 1
    zeta = np.arange(nz) * step * dpsi
    r_v = rot.tip_loss
    R = bm.R
    # released from the deflected tip
    z0 = np.array([np.interp(r_v, bm.x, bm.z[j]) for j in range(npsi)]) / R
    zt, xt, yt = beddoes_wake_z(bm.psi, zeta, bm.mu_x, bm.mu_z, lam0, r_v, z0)
    sel = bm.x >= 0.5
    gam_v = opts.rollup * np.maximum(np.max(ld['gamma'][:, sel], axis=1), 0.0)
    if dz is not None:
        zt = zt + dz(zeta)
    if dxy is not None:
        ddx, ddy = dxy(zeta)
        xt, yt = xt + ddx, yt + ddy
    if gscale is not None:
        gam_v = gam_v * gscale
    c_tip = float(rot.chord_at(1.0))
    rc0 = opts.core_radius_chords * c_tip
    gmean = max(np.mean(gam_v), 1e-6)
    delta = 1.0 + opts.core_growth_a1 * gmean / bm.nu
    rc = np.sqrt(rc0 ** 2 + 4.0 * 1.25643 * delta * bm.nu * zeta / bm.omega)
    rc_s = np.sqrt(rc ** 2 + (opts.smooth_core_chords * c_tip) ** 2)
    n_near = int(np.ceil(np.radians(opts.near_wake_deg) / (step * dpsi)))
    # thin-airfoil lift weighting of a non-uniform upwash over the chord: (1 - cos theta), x/c = (1 - cos theta)/2
    nq = max(1, opts.chord_points)
    th = (np.arange(nq) + 0.5) * np.pi / nq
    xq = (1.0 - np.cos(th)) / 2.0 if nq > 1 else np.array([0.75])
    wq = (1.0 - np.cos(th)) / nq if nq > 1 else np.array([1.0])
    w = _bvi_kernel(bm.x, np.ascontiguousarray(bm.chord, float), np.ascontiguousarray(bm.z), np.cos(bm.psi),
                    np.sin(bm.psi), np.ascontiguousarray(xt * R), np.ascontiguousarray(yt * R),
                    np.ascontiguousarray(zt * R), gam_v, rc, rc_s, bm.B, npsi, step, n_near, R, xq, wq,
                    float(np.max(bm.chord)))
    dlam = -w / bm.vtip              # inflow ratio is positive downward
    wake = dict(x=xt, y=yt, z=zt, zeta=zeta, gamma=gam_v, rc=rc)
    return dlam, wake


@njit(parallel=True, fastmath=True, cache=True)
def _turbulence_kernel(xs, zb, cospsi, sinpsi, wx, wy, wz, gam, rt, nb, npsi, step, n_near, R, ktu):
    """Mean-square turbulent velocity and its length scale at blade 0's stations, from the turbulent envelopes of
    every blade's tip vortex: (k Gamma / 2 pi r_t)^2 exp(-h^2 / 2 r_t^2) per unit filament length over
    sqrt(2 pi) r_t (so a crossing contributes its peak whatever the discretisation)."""
    nr = xs.shape[0]
    nz = wx.shape[1]
    u2 = np.zeros((npsi, nr))
    lw = np.zeros((npsi, nr))
    for j in prange(npsi):
        cj, sj = cospsi[j], sinpsi[j]
        for b in range(nb):
            i0 = n_near if b == 0 else 0
            for i in range(i0, nz - 1):
                ka = (j + b * (npsi // nb) - i * step) % npsi
                kb = (j + b * (npsi // nb) - (i + 1) * step) % npsi
                g = 0.5 * (gam[ka] + gam[kb])
                if g <= 0.0:
                    continue
                ax, ay, az = wx[ka, i], wy[ka, i], wz[ka, i]
                bx, by, bz = wx[kb, i + 1], wy[kb, i + 1], wz[kb, i + 1]
                sx, sy, sz = bx - ax, by - ay, bz - az
                l2 = sx * sx + sy * sy + sz * sz
                ln = np.sqrt(l2)
                r = 0.5 * (rt[i] + rt[i + 1])
                mx, my, mz = 0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (az + bz)
                s = mx * cj + my * sj
                if s < xs[0] * R - 4.0 * r - ln or s > R + 4.0 * r + ln:
                    continue
                amp = (ktu * g / (2.0 * np.pi * r)) ** 2 * ln / (np.sqrt(2.0 * np.pi) * r)
                for p in range(nr):
                    px = xs[p] * R * cj
                    py = xs[p] * R * sj
                    pz = zb[j, p]
                    # distance from the point to the segment
                    t = ((px - ax) * sx + (py - ay) * sy + (pz - az) * sz) / l2
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    dx = px - ax - t * sx
                    dy = py - ay - t * sy
                    dz = pz - az - t * sz
                    h2 = dx * dx + dy * dy + dz * dz
                    if h2 > 16.0 * r * r:
                        continue
                    e = amp * np.exp(-0.5 * h2 / (r * r))
                    u2[j, p] += e
                    lw[j, p] += e * r
    return u2, lw


def landgrebe_wake(psi_rel, zeta, ct, sigma, twist_deg, blades, z0):
    """Landgrebe's generalised hover tip-vortex geometry (z/R, x/R, y/R): contraction to 0.78 R, a slow descent
    until the next blade passes and a faster one after it. twist_deg: linear twist, tip minus root (negative)."""
    A = 0.78
    lam = 0.145 + 27.0 * ct
    k1 = -0.25 * (ct / sigma + 0.001 * twist_deg)
    k2 = -(1.41 + 0.0141 * twist_deg) * np.sqrt(ct / 2.0)
    zb = 2.0 * np.pi / blades
    r = A + (1.0 - A) * np.exp(-lam * zeta)
    z = np.where(zeta <= zb, k1 * zeta, k1 * zb + k2 * (zeta - zb))
    x = r[None, :] * np.cos(psi_rel)[:, None]
    y = r[None, :] * np.sin(psi_rel)[:, None]
    return np.asarray(z0)[:, None] + z[None, :], x, y


def _wake_turbulence(bm, ld, lam0, opts):
    """Blade-wake interaction: the rms turbulent velocity (m/s) and length scale (m) each blade element meets in
    the turbulent envelopes of the tip vortices ahead of it (Beddoes wake, as for the blade-vortex interaction)."""
    rot = bm.rotor
    npsi = len(bm.psi)
    dpsi = 2.0 * np.pi / npsi
    step = max(1, int(round(np.radians(opts.wake_dpsi_deg) / dpsi)))
    zeta_max = np.clip(2.5 / max(bm.mu_x, 1e-3), 2.0 * np.pi, opts.wake_revs_max * 2.0 * np.pi)
    nz = int(zeta_max / (step * dpsi)) + 1
    zeta = np.arange(nz) * step * dpsi
    r_v = rot.tip_loss
    R = bm.R
    z0 = np.array([np.interp(r_v, bm.x, bm.z[j]) for j in range(npsi)]) / R
    if bm.mu_x < 0.03:
        T = bm.B * np.mean(np.sum(ld['fz'] * bm.dr, axis=1))
        ct = max(T / (bm.rho * bm.area * bm.vtip ** 2), 1e-5)
        tw = float(np.degrees(rot.twist_at(1.0) - rot.twist_at(rot.root_cutout)))
        zt, xt, yt = landgrebe_wake(bm.psi, zeta, ct, rot.solidity, tw, bm.B, z0)
    else:
        zt, xt, yt = beddoes_wake_z(bm.psi, zeta, bm.mu_x, bm.mu_z, lam0, r_v, z0)
    sel = bm.x >= 0.5
    gam_v = opts.rollup * np.maximum(np.max(ld['gamma'][:, sel], axis=1), 0.0)
    c_tip = float(rot.chord_at(1.0))
    rc0 = opts.wake_turbulent_core_chords * c_tip
    rt = np.sqrt(rc0 ** 2 + (opts.wake_turbulent_radius_chords * c_tip) ** 2 * zeta / (2.0 * np.pi))
    n_near = int(np.ceil(np.radians(opts.near_wake_deg) / (step * dpsi)))
    u2, lw = _turbulence_kernel(bm.x, np.ascontiguousarray(bm.z), np.cos(bm.psi), np.sin(bm.psi),
                                np.ascontiguousarray(xt * R), np.ascontiguousarray(yt * R),
                                np.ascontiguousarray(zt * R), gam_v, rt, bm.B, npsi, step, n_near, R,
                                opts.wake_turbulence)
    scale = np.where(u2 > 0, lw / np.maximum(u2, 1e-30), rc0)
    return np.sqrt(u2), np.maximum(scale, 0.1 * c_tip)


# ---------------------------------------------------------------- unsteadiness: one revolution at a time

def _smooth_random(rng, n, harmonics):
    """A unit-rms random periodic function on n points with Fourier content up to `harmonics` per revolution."""
    k = np.arange(1, harmonics + 1)
    a = rng.normal(size=len(k)) / np.sqrt(k)
    p = rng.uniform(0, 2 * np.pi, len(k))
    psi = np.arange(n) * 2 * np.pi / n
    g = np.sum(a[:, None] * np.cos(k[:, None] * psi[None, :] + p[:, None]), axis=0)
    return g / np.sqrt(np.mean(g ** 2))


def gust_field(bm, rng, rms, scale, n_modes=256):
    """Vertical velocity (m/s, up) of a frozen isotropic turbulent field (von Karman energy spectrum, integral
    scale `scale`) met by blade 0's elements over one revolution, the air passing the rotor with its forward speed.
    Scales below a chord are left to the broadband noise model."""
    c = float(np.max(bm.chord))
    k = np.geomspace(0.05 / scale, 2 * np.pi / c, n_modes)
    E = (k * scale) ** 4 / (1.0 + (k * scale) ** 2) ** (17.0 / 6.0)
    dk = np.gradient(k)
    amp = np.sqrt(E * dk)
    th = rng.uniform(0, 2 * np.pi, n_modes)
    ph = rng.uniform(0, 2 * np.pi, n_modes)
    tau = bm.psi / bm.omega
    X = bm.r[None, :] * bm.cos - bm.mu_x * bm.vtip * tau[:, None]
    Y = bm.r[None, :] * bm.sin
    return _gust_modes(X, Y, amp, k, np.cos(th), np.sin(th), ph) * rms / np.sqrt(0.5 * np.sum(amp ** 2))


@njit(parallel=True, fastmath=True, cache=True)
def _gust_modes(X, Y, amp, k, cos_th, sin_th, ph):
    w = np.zeros_like(X)
    for i in prange(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            s = 0.0
            for q in range(amp.shape[0]):
                s += amp[q] * np.cos(k[q] * (cos_th[q] * x + sin_th[q] * y) + ph[q])
            w[i, j] = s
    return w


def load_variants(aero, n=8, gust_rms=1.0, gust_scale=20.0, wander_chords=0.25, strength_rms=0.1, seed=1):
    """n revolutions of blade loads, each with its own turbulent gusts and tip-vortex wander and strength, at the
    trim of the solution: the pass-to-pass unsteadiness of a real rotor. Returns a list of dicts with ft, fz (with
    the blade-vortex interaction) and ft_base, fz_base (without), shaped like aero.fz."""
    bm = aero.model
    opts = aero.opts or AeroOptions()
    rng = np.random.default_rng(seed)
    keep = bm.extra_lambda, bm.dalpha_unsteady
    up_base = bm.vtip * (aero.lam_i - bm.mu_z + bm.mu_x * bm.cos * bm.slope) + bm.zdot
    UT, mach = aero.UT, aero.mach
    ramp = float(np.clip((bm.mu_x - 0.025) / 0.03, 0.0, 1.0))
    c_tip = float(bm.rotor.chord_at(1.0))
    npsi = len(bm.psi)
    out = []
    for v in range(n):
        dl_g = -gust_field(bm, rng, gust_rms, gust_scale) / bm.vtip
        if aero.dlambda_fountain is not None:
            # the fountain comes and goes: spikes of a strength of their own in each revolution
            dl_g = dl_g + aero.dlambda_fountain * max(1.0 + 0.5 * rng.normal(), 0.0)
        dl_b = 0.0
        if aero.dlambda_bvi is not None:
            g_z, g_x, g_y = (_smooth_random(rng, npsi, 8) for _ in range(3))
            g_s = _smooth_random(rng, npsi, 6)
            sig = wander_chords * c_tip / bm.R

            def grow(zeta):
                return np.minimum(1.0, zeta / (0.5 * np.pi))[None, :]

            def dz(zeta, g_z=g_z):
                return sig * g_z[:, None] * grow(zeta)

            def dxy(zeta, g_x=g_x, g_y=g_y):
                # in-plane wander, larger than the vertical (the wake is carried by the free stream and gusts)
                return 2.0 * sig * g_x[:, None] * grow(zeta), 2.0 * sig * g_y[:, None] * grow(zeta)
            dl_b, _ = _bvi_inflow(bm, dict(gamma=aero.gamma_bound), aero.lambda0, opts, dz=dz, dxy=dxy,
                                  gscale=np.maximum(1.0 + strength_rms * g_s, 0.2))
            dl_b = dl_b * ramp
        res = {}
        for kind, dl in (("", dl_g + dl_b), ("_base", dl_g)):
            d_raw = -(np.arctan2(up_base + bm.vtip * dl, UT) - np.arctan2(up_base, UT))
            bm.extra_lambda = dl
            bm.dalpha_unsteady = wagner_filter(d_raw, UT, bm.chord, mach, bm.omega) if opts.unsteady else None
            ld = bm.loads(aero.theta0, aero.theta1c, aero.theta1s, aero.lam_i)
            res["ft" + kind], res["fz" + kind] = ld["ft"], ld["fz"]
        out.append(res)
    bm.extra_lambda, bm.dalpha_unsteady = keep
    return out
