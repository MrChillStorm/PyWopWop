"""A tiltrotor's wing behind its proprotors in airplane mode, as a source of their harmonics.

Each blade's vortex system (its bound vortex, a horseshoe's, and the tip and root vortices it trails, convected with
the flow through the disc) sweeps an upwash over the wing's leading edge once a revolution. Where the tip circle,
seen along the axis, crosses the wing (on the MV-22B at its root), every tip vortex runs into the leading edge a few
milliseconds after it is shed, a sharp pulse whose harmonics fall slowly. The wing answers each strip's upwash as a
two-dimensional section a convected gust (Sears' function: at these reduced frequencies the lift gathers at the
leading edge and falls as one over the square root of the frequency), and its unsteady lift radiates as a dipole per
strip. Its harmonics at an observer are added to the blade's loading harmonics (synth.rotor_sound), so they reach
the walls and the listener as the rotor's sound does.

The vortex is the rotor's prescribed wake's (Beddoes': released at the deflected tip, convected with the free stream
and twice the mean induced inflow, its strength the rollup of the peak bound circulation), but with its measured
core, not the rotor's effective one: that one makes up for a lifting line's response to a vortex passing close,
which Sears' function here already holds. Where the wing is bolted to the fuselage its loads also reach the cabin
through the structure (structure_borne), which passes their high harmonics as the walls don't. Not modelled: the vortex cut and displaced by the wing (it passes through),
the gust's spanwise variation on the section's response (strip theory), the wing's scattering as a half plane, and
the wing in helicopter mode, under the discs' wake (the download's fountain)."""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

CORE_RADIUS_CHORDS = 0.12       # HART II's measured tip vortex core, in tip chords
MAX_NACELLE_DEG = 30.0          # the wing is behind the discs (airplane mode and the late conversion)
STRUCTURE_REACH = 1.0           # m outboard of the fuselage's side: the wing's loads there go into the cabin's frames


@dataclass
class WingLoads:
    """One blade's vortex system's load on the wing, per strip and rotor harmonic, in the rotor's frame."""
    pos: np.ndarray             # (Ns, 3) each strip's dipole, m
    normal: np.ndarray          # (Ns, 3) the wing's normal (its lift's direction)
    force: np.ndarray           # (Ns, M) complex amplitude of the strip's lift, N: L(t) = Re(sum F_m e^{i m Omega t})
    f_rot: float                # revolutions per second
    c: float                    # speed of sound, m/s
    rho: float = 1.225
    y: np.ndarray = None        # (Ns,) each strip's spanwise station, aircraft frame
    attached: np.ndarray = None  # (Ns,) the strips whose load goes into the cabin's structure


def wing_strips(wing, dy_fine=0.02, dy=0.1, fine_to=2.5):
    """The wing's spanwise strips on both sides, from its sections [|y|, leading edge x, chord, chord line z]
    (aircraft frame, linear between them): dy_fine wide out to |y| fine_to (where a tip circle may cross a wing
    root), dy outboard. Returns the strips' leading edges (aircraft frame), chords and widths."""
    s = np.asarray(sorted(wing), float)
    y0, y1 = s[0, 0], s[-1, 0]
    edges = np.unique(np.concatenate([np.arange(y0, min(fine_to, y1), dy_fine), np.arange(min(fine_to, y1), y1, dy),
                                      [y1]]))
    yc = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    xle, chord, z = (np.interp(yc, s[:, 0], s[:, k]) for k in (1, 2, 3))
    le = np.concatenate([np.c_[xle, yc, z], np.c_[xle, -yc, z]])
    return le, np.concatenate([chord, chord]), np.concatenate([width, width])


@njit(parallel=True, fastmath=True, cache=True)
def _upwash_kernel(P, N, n_t, r_tip, z_tip, r_root, z_root, r_b, z_b, zeta, Uc, omega, gam, rc_tip, rc_root, rc_b):
    """The normal velocity at the points P (Ns, 3) along N over one revolution (n_t steps) from one blade's vortex
    system: at blade azimuth psi the bound line r_b (heights z_b) along the blade, the tip vortex released at
    psi - zeta, age zeta, from (r_tip, z_tip) convected by Uc zeta / omega, and the root vortex the same way, all of
    strength gam[release step]. Vatistas (n = 2) cores rc_tip, rc_root (by age) and rc_b."""
    ns = P.shape[0]
    nz = zeta.shape[0]
    ng = gam.shape[0]
    nb = r_b.shape[0]
    out = np.zeros((ns, n_t))
    for i in prange(ns):
        px, py, pz = P[i, 0], P[i, 1], P[i, 2]
        nx, ny, nz_ = N[i, 0], N[i, 1], N[i, 2]
        for k in range(n_t):
            psi = 2.0 * np.pi * k / n_t
            w = 0.0
            g0 = gam[(k * ng) // n_t]
            # the bound vortex, root to tip (positive circulation points outboard)
            cs, sn = np.cos(psi), np.sin(psi)
            for j in range(nb - 1):
                ax, ay, az = r_b[j] * cs, r_b[j] * sn, z_b[j]
                bx, by, bz = r_b[j + 1] * cs, r_b[j + 1] * sn, z_b[j + 1]
                w += _segment(px, py, pz, ax, ay, az, bx, by, bz, g0, rc_b, nx, ny, nz_)
            # the trailed vortices: the tip's leaves the tip into the wake, the root's comes out of it into the root
            for side in range(2):
                r = r_tip if side == 0 else r_root
                z0 = z_tip if side == 0 else z_root
                sgn = 1.0 if side == 0 else -1.0
                for j in range(nz - 1):
                    a1, a2 = zeta[j], zeta[j + 1]
                    t1, t2 = a1 / omega, a2 / omega
                    c1, s1 = np.cos(psi - a1), np.sin(psi - a1)
                    c2, s2 = np.cos(psi - a2), np.sin(psi - a2)
                    ax, ay, az = r * c1 + Uc[0] * t1, r * s1 + Uc[1] * t1, z0 + Uc[2] * t1
                    bx, by, bz = r * c2 + Uc[0] * t2, r * s2 + Uc[1] * t2, z0 + Uc[2] * t2
                    kr = int(((psi - a1) / (2.0 * np.pi) % 1.0) * ng) % ng
                    rc = rc_tip[j] if side == 0 else rc_root[j]
                    w += _segment(px, py, pz, ax, ay, az, bx, by, bz, sgn * gam[kr], rc, nx, ny, nz_)
            out[i, k] = w
    return out


@njit(fastmath=True, cache=True, inline='always')
def _segment(px, py, pz, ax, ay, az, bx, by, bz, g, rc, nx, ny, nz):
    """A straight vortex segment's (a to b, circulation g) velocity at p along n, with a Vatistas (n = 2) core."""
    r1x, r1y, r1z = px - ax, py - ay, pz - az
    r2x, r2y, r2z = px - bx, py - by, pz - bz
    r0x, r0y, r0z = bx - ax, by - ay, bz - az
    cx = r1y * r2z - r1z * r2y
    cy = r1z * r2x - r1x * r2z
    cz = r1x * r2y - r1y * r2x
    c2 = cx * cx + cy * cy + cz * cz
    l2 = r0x * r0x + r0y * r0y + r0z * r0z
    n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
    if c2 < 1e-20 or l2 < 1e-20 or n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    h2 = c2 / l2
    core = h2 / np.sqrt(rc ** 4 + h2 * h2)
    dot = r0x * (r1x / n1 - r2x / n2) + r0y * (r1y / n1 - r2y / n2) + r0z * (r1z / n1 - r2z / n2)
    k = g / (4.0 * np.pi) * dot / c2 * core
    return k * (cx * nx + cy * ny + cz * nz)


def sears(sigma):
    """Sears' function for a gust met at the leading edge (its convection to the midchord taken out): the section's
    lift over the quasi-steady pi rho c U w; at high reduced frequency 1 / sqrt(2 pi sigma), a quarter period late."""
    s = np.asarray(sigma, float)
    a = 2.0 * np.pi * s
    return np.exp(-0.25j * np.pi * a / (1.0 + a)) / np.sqrt(1.0 + a)


def viscous_wake_upwash(aero, P, normal, Uc, M):
    """One blade's viscous wake over the points P (rotor frame) as upwash along their normals, harmonics m = 0..M-1
    (w(t) = Re(sum a_m e^{i m Omega t})): the sheet its trailing edge leaves, convected as its vortices are, carries
    the air the blade drags along, a Gaussian deficit across it, decaying with the distance s behind the trailing
    edge (Lakshminarayana and Davino: at the centre 2.42 sqrt(Cd) / (s/c + 0.3) of the section's speed, half width
    0.68 c sqrt(Cd (s/c + 0.15))). It crosses each point once a revolution, a pulse whose transform is closed form."""
    R = aero.R
    omega = aero.omega
    rot = aero.rotor
    zmean = np.mean(aero.z, axis=0)
    cd = np.mean(aero.cd, axis=0)
    T = 2.0 * np.pi / omega
    w_m = np.arange(M) * omega
    out = np.zeros((len(P), M), complex)
    if Uc[2] >= 0.0:
        return out
    for i in range(len(P)):
        p, n = P[i], normal[i]
        zb = float(np.interp(np.hypot(p[0], p[1]) / R, aero.x, zmean))
        if p[2] >= zb:
            continue
        age = (p[2] - zb) / Uc[2]
        q = p[:2] - Uc[:2] * age                     # where in the disc the wake passing the point was shed
        r = float(np.hypot(q[0], q[1]))
        if r < rot.root_cutout * R or r > R:
            continue
        c = float(np.interp(r / R, aero.x, aero.chord))
        Cd = max(float(np.interp(r / R, aero.x, cd)), 0.0)
        phi = float(np.arctan2(q[1], q[0]))
        V = float(np.hypot(omega * r, Uc[2]))
        s = V * age
        u_c = V * 2.42 * np.sqrt(Cd) / (s / c + 0.3)
        half = c * 0.68 * np.sqrt(Cd * (s / c + 0.15))
        e = omega * r * np.array([-np.sin(phi), np.cos(phi), 0.0]) - Uc     # the blade's way through the air
        proj = float(e @ n) / float(np.linalg.norm(e))
        t_k = ((phi + 0.75 * c / r) / omega + age) % T        # the trailing edge passed phi, age before
        Th = half / (omega * r)                              # the pulse's half width, s
        out[i] = (2.0 / T) * proj * u_c * Th * np.sqrt(np.pi / np.log(2.0)) \
            * np.exp(-(w_m * Th) ** 2 / (4.0 * np.log(2.0)) - 1j * w_m * t_k)
    return out


def wing_loads(point, wing, pitch, n_t=2048, revs=2.0, f_max=6000.0, attach_y=None):
    """The wing's unsteady lift from one blade of a solved rotor (operating.RotorPoint) whose disc it is behind.
    wing: the aircraft's sections; pitch: the airframe's (rad), its frame to the heading frame; attach_y: the
    fuselage's half width, where the wing is bolted to it (its loads within STRUCTURE_REACH of it go into the cabin's
    structure)."""
    from .synth import _pitch_matrix
    aero = point.aero
    rot = point.rotor
    R = aero.R
    omega = aero.omega
    rho = aero.rho
    U = np.asarray(point.U, float)
    Uw = float(np.linalg.norm(U))
    # the vortex: the rotor's prescribed wake's, its measured core
    r_v = rot.tip_loss * R
    zmean = np.mean(aero.z, axis=0)
    z_tip = float(np.interp(rot.tip_loss, aero.x, zmean))
    r_root = rot.root_cutout * R
    z_root = float(np.interp(rot.root_cutout, aero.x, zmean))
    opts = aero.opts
    rollup = getattr(opts, "rollup", 0.7)
    sel = aero.x >= 0.5
    gam = rollup * np.maximum(np.max(aero.gamma_bound[:, sel], axis=1), 0.0)
    Uc = U - np.array([0.0, 0.0, 2.0 * aero.lambda0 * omega * R])
    c_tip = float(rot.chord_at(1.0))
    rc0 = CORE_RADIUS_CHORDS * c_tip
    a1 = getattr(opts, "core_growth_a1", 1e-4)
    delta = 1.0 + a1 * max(float(np.mean(gam)), 1e-6) / aero.nu
    fine = 2.0 * np.pi / n_t
    zeta = np.unique(np.concatenate([np.arange(0.0, 1.2, fine), np.arange(1.2, revs * 2.0 * np.pi, 8.0 * fine),
                                     [revs * 2.0 * np.pi]]))
    rc = np.sqrt(rc0 ** 2 + 4.0 * 1.25643 * delta * aero.nu * zeta / omega)
    rc_root = np.sqrt((CORE_RADIUS_CHORDS * float(rot.chord_at(rot.root_cutout))) ** 2
                      + 4.0 * 1.25643 * delta * aero.nu * zeta / omega)
    r_b = np.linspace(r_root, r_v, 17)
    z_b = np.interp(r_b / R, aero.x, zmean)
    # the strips in the rotor's frame
    le, chord, width = wing_strips(wing)
    Pm = _pitch_matrix(pitch)
    O = point.to_rotor(np.zeros(3))
    J = np.array([point.to_rotor(e) - O for e in np.eye(3)]).T        # rotor = O + J @ heading
    to_rot = lambda X: (J @ (Pm @ np.asarray(X, float).T)).T + O
    rot_dir = lambda v: (J @ (Pm @ np.asarray(v, float).T)).T
    P = to_rot(le)
    normal = rot_dir(np.tile([0.0, 0.0, 1.0], (len(le), 1)))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    aft = rot_dir(np.tile([-1.0, 0.0, 0.0], (len(le), 1)))
    w = _upwash_kernel(np.ascontiguousarray(P), np.ascontiguousarray(normal), n_t, r_v, z_tip, r_root, z_root,
                       r_b, z_b, zeta, Uc, omega, np.ascontiguousarray(gam), rc, rc_root, rc0)
    f_rot = omega / (2.0 * np.pi)
    M = min(n_t // 2, int(f_max / f_rot) + 1)
    a = 2.0 * np.fft.rfft(w, axis=1)[:, :M] / n_t                     # w(t) = Re(sum a_m e^{i m Omega t})
    a = a + viscous_wake_upwash(aero, P, normal, Uc, M)
    a[:, 0] = 0.0
    sigma = np.arange(M)[None, :] * omega * chord[:, None] / (2.0 * Uw)
    F = np.pi * rho * chord[:, None] * Uw * a * sears(sigma) * width[:, None]
    # the lift gathers at the leading edge at these frequencies: a tenth of the chord behind it
    pos = P + 0.1 * chord[:, None] * aft
    attached = (np.abs(le[:, 1]) <= attach_y + STRUCTURE_REACH) if attach_y is not None else np.zeros(len(le), bool)
    return WingLoads(pos=pos, normal=normal, force=F, f_rot=f_rot, c=aero.c, rho=rho, y=le[:, 1], attached=attached)


def structure_borne(loads, cabin, area, f):
    """The cabin's mean-square pressure (Pa^2) at the harmonics' frequencies f (shaft orders 0..) from the wing's loads
    where it is bolted to the fuselage, by statistical energy analysis: each load feeds the structure the power
    |F|^2 Re(Y) / 2 (Y the cabin's wing_mobility, or the skin's point mobility pi f_c / 4 m c^2 from its surface
    mass and coincidence), the loads
    within half a bending wavelength of each other together, farther apart by power; the structure stores it against
    its loss factor and radiates it into the cabin with a stiffened panel's edge-mode efficiency (Maidanik: lambda_c
    P / pi^2 A sqrt(f / f_c) between its frames and stringers), built up by the interior's absorption. Below those
    panels' first mode they have no resonances to store it: the structure moves as a spring, F omega / K, (f / f_11)^3
    of the power the modes would hold. The trim's mounts are taken to pass the frames' motion."""
    f = np.asarray(f, float)
    out = np.zeros(len(f))
    sel = loads.attached if loads.attached is not None else np.zeros(0, bool)
    if not np.any(sel) or not cabin.coincidence_hz:
        return out
    c0, rho = loads.c, loads.rho
    m, fc = cabin.surface_mass, cabin.coincidence_hz
    Y = cabin.wing_mobility or np.pi * fc / (4.0 * m * c0 ** 2)
    a, b = cabin.frame_pitch, cabin.stringer_pitch
    edge = (c0 / fc) * 2.0 * (a + b) / (np.pi ** 2 * a * b)
    f11 = 0.5 * np.pi * c0 ** 2 / (2.0 * np.pi * fc) * (1.0 / a ** 2 + 1.0 / b ** 2)
    y = loads.y[sel]
    F = loads.force[sel]
    side = np.sign(y)
    M = min(F.shape[1], len(f))
    for k in range(1, M):
        fk = f[k]
        half = 0.5 * c0 / np.sqrt(fk * fc)                  # half the skin's bending wavelength
        chunk = side * (np.floor(np.abs(y) / half) + 1.0)
        P = 0.0
        for g in np.unique(chunk):
            P += abs(np.sum(F[chunk == g, k])) ** 2
        sigma = min(edge * np.sqrt(fk / fc), 1.0) if fk < fc else 1.0
        w = 2.0 * np.pi * fk
        out[k] = 2.0 * (rho * c0) ** 2 * sigma * Y * P * float(cabin.room(np.array([fk]))[0]) / (
            w * cabin.structure_loss_factor * m * area) * min(1.0, fk / f11) ** 3
    return out


def covered(patches, wing):
    """Which wall patches (aircraft frame) the wing sits on: within its planform and above its lower surface (the
    sections' half thickness, 5th column), under the wing box and so out of the outside air."""
    s = np.asarray(sorted(wing), float)
    if s.shape[1] < 5:
        return np.zeros(len(patches), bool)
    y = np.abs(patches[:, 1])
    xle, chord, zc, half = (np.interp(y, s[:, 0], s[:, k]) for k in (1, 2, 3, 4))
    return (y <= s[-1, 0]) & (patches[:, 0] <= xle) & (patches[:, 0] >= xle - chord) & (patches[:, 2] >= zc - half)


def harmonics(loads, observer, n, radiating=False):
    """The wing's pressure at an observer (rotor frame, m) as single-blade harmonics in f1a's convention, p(t) =
    sum_m Re(2 H_m e^{i m Omega t}) / n, m = 0..n/2: each strip a point force on the air, minus its lift. radiating:
    the dipole's sound alone, for the airframe's own walls: the 1/r^2 term is the pressure hugging the wing, no sound,
    and what it does to the cabin goes through the structure (structure_borne)."""
    d = np.asarray(observer, float)[None, :] - loads.pos
    r = np.maximum(np.linalg.norm(d, axis=1), 1e-3)
    cosn = np.sum(d * loads.normal, axis=1) / r
    M = loads.force.shape[1]
    k = 2.0 * np.pi * loads.f_rot * np.arange(M) / loads.c
    near = 0.0 if radiating else 1.0 / r[:, None] ** 2
    G = cosn[:, None] * (1j * k[None, :] / r[:, None] + near) * np.exp(-1j * k[None, :] * r[:, None])
    P = -np.sum(loads.force * G, axis=0) / (4.0 * np.pi)
    H = np.zeros(n // 2 + 1, complex)
    m = min(M, len(H))
    H[:m] = P[:m] * n / 2.0
    return H
