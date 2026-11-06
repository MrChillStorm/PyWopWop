"""Tonal rotor noise: Farassat's Formulation 1A of the Ffowcs Williams-Hawkings equation.

Thickness noise from the blade surface (panels), loading noise from the blade loads (compact in span, spread over
the chord at thin-airfoil load points, or compact at the quarter chord). Source-time-dominant: every source element
is followed in emission time, its reception time found in closed form for an observer in uniform motion (a
quadratic), and its contribution interpolated onto the observer's time grid. Kinematics are analytic: rotation,
the hub's translation through still air, and the blade's elastic heave from the aeroelastic solution.

Frames are the rotor's canonical frame (aero.py), fixed to the undisturbed air: the hub starts at the origin and
moves with -U (U: air velocity relative to the rotor).
"""
from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
from numba import get_num_threads, njit, prange

from .aero import radial_stations


@dataclass
class SourceModel:
    """Everything the F1A kernel needs for one rotor, on an acoustic azimuth grid of n_psi points per revolution."""
    omega: float
    blades: int
    n_psi: int
    U: np.ndarray               # air velocity relative to the rotor (canonical), m/s
    rho: float
    c: float
    # loading points (blade frame at psi = 0, before elastic heave)
    lp_pos: np.ndarray          # (El, 3)
    lp_station: np.ndarray      # (El,) aero station index
    lp_weight: np.ndarray       # (El,) chordwise share times span width (m)
    # thickness panels
    tp_pos: np.ndarray          # (Et, 3) centroids
    tp_area: np.ndarray         # (Et, 3) area vectors, outward
    tp_station: np.ndarray      # (Et,)
    # tables over (station, psi)
    f_tab: np.ndarray           # (Nr, n_psi, 2): (ft, fz) on the blade, N/m
    df_tab: np.ndarray          # (Nr, n_psi, 2): d/dpsi
    z_tab: np.ndarray           # (Nr, n_psi): elastic heave above the static position, m
    dz_tab: np.ndarray          # (Nr, n_psi): d/dpsi
    ddz_tab: np.ndarray         # (Nr, n_psi): d2/dpsi2

    @property
    def period(self):
        return 2.0 * np.pi / self.omega


def _fourier_resample(a, n, axis, taper=0.3, derivatives=2):
    """Periodic band-limited resampling along axis to n points, with value and that many derivatives per radian; the
    top `taper` fraction of the source harmonics is rolled off (no Gibbs ringing from a coarse grid)."""
    a = np.moveaxis(np.asarray(a, float), axis, -1)
    m = a.shape[-1]
    A = sfft.rfft(a, axis=-1)
    k = np.arange(A.shape[-1])
    kmax = m // 2
    w = np.ones_like(k, dtype=float)
    k0 = (1.0 - taper) * kmax
    sel = k > k0
    w[sel] = 0.5 * (1.0 + np.cos(np.pi * np.clip((k[sel] - k0) / (kmax - k0), 0.0, 1.0)))
    A = A * w
    B = np.zeros(a.shape[:-1] + (n // 2 + 1,), complex)
    B[..., :A.shape[-1]] = A
    scale = n / m
    kk = np.arange(B.shape[-1])
    out = [sfft.irfft(B * (1j * kk) ** d, n, axis=-1, workers=-1) * scale for d in range(derivatives + 1)]
    return [np.moveaxis(x, -1, axis) for x in out]


def chordwise_points(n):
    """Load points and shares for a flat-plate loading (sqrt((1-x)/x)) in n chordwise bins; n = 1: quarter chord."""
    if n <= 1:
        return np.array([0.25]), np.array([1.0])
    edges = 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))
    F = lambda x: np.sqrt(x * (1.0 - x)) + np.arcsin(np.sqrt(x))          # antiderivative of sqrt((1-x)/x)
    share = np.diff(F(edges)) / (np.pi / 2.0)
    # load centroid of each bin, by quadrature
    xc = np.empty(n)
    for i in range(n):
        x = np.linspace(edges[i], edges[i + 1], 401)[1:-1]
        wgt = np.sqrt((1.0 - x) / x)
        xc[i] = np.sum(x * wgt) / np.sum(wgt)
    return xc, share


def blade_surface(rotor, x_edges, theta_geo, n_chord=24):
    """Closed blade surface as quad panels: (centroids, area vectors, station index). x_edges: r/R panel edges of
    the aero stations (each strip belongs to one station); theta_geo: the geometric pitch per station (rad)."""
    R = rotor.radius
    af = rotor.airfoil
    xc = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_chord + 1) / n_chord))      # x/c, LE to TE
    ht = af.half_thickness(xc)
    ht[-1] = 0.0                     # closed trailing edge
    ht[0] = 0.0
    cb = af.camber(xc)
    nst = len(x_edges) - 1

    def section(r_over_R, th):
        """Upper and lower surface points (n_chord+1, 3) of the section at r/R with pitch th."""
        c = float(rotor.chord_at(r_over_R))
        r = r_over_R * R
        sweep = 0.0
        if r_over_R > rotor.tip_sweep_start:
            sweep = (r_over_R - rotor.tip_sweep_start) * R * np.tan(np.radians(rotor.tip_sweep_deg))
        cv = np.array([0.0, np.cos(th), np.sin(th)])        # chord direction, towards the leading edge
        nv = np.array([0.0, -np.sin(th), np.cos(th)])       # upper normal
        base = np.array([r, -sweep, 0.0])
        up = base + ((0.25 - xc) * c)[:, None] * cv + ((cb + ht) * c)[:, None] * nv
        lo = base + ((0.25 - xc) * c)[:, None] * cv + ((cb - ht) * c)[:, None] * nv
        return up, lo

    cents, areas, stn = [], [], []

    def quad(p1, p2, p3, p4, s, outward_hint):
        A = 0.5 * np.cross(p3 - p1, p4 - p2)
        if np.dot(A, outward_hint) < 0:
            A = -A
        cents.append(0.25 * (p1 + p2 + p3 + p4))
        areas.append(A)
        stn.append(s)

    th_edges = np.interp(x_edges, 0.5 * (x_edges[1:] + x_edges[:-1]), theta_geo)
    secs = [section(x_edges[i], th_edges[i]) for i in range(nst + 1)]
    for s in range(nst):
        (ua, la), (ub, lb) = secs[s], secs[s + 1]
        th = theta_geo[s]
        nv = np.array([0.0, -np.sin(th), np.cos(th)])
        for k in range(n_chord):
            quad(ua[k], ua[k + 1], ub[k + 1], ub[k], s, nv)
            quad(la[k], la[k + 1], lb[k + 1], lb[k], s, -nv)
    for s, (u, l), out in ((0, secs[0], np.array([-1.0, 0, 0])), (nst - 1, secs[-1], np.array([1.0, 0, 0]))):
        for k in range(n_chord - 1):
            quad(u[k], u[k + 1], l[k + 1], l[k], s, out)
    return np.array(cents), np.array(areas), np.array(stn, dtype=np.int64)


def build_sources(point, n_psi=None, chord_points=4, surface_chord=24, fs_target=96000.0, thickness=True,
                  loads="full"):
    """The F1A source model of a solved rotor (operating.RotorPoint). loads: 'full', or 'base' (without the
    blade-vortex interaction's impulsive part, where the solution has it)."""
    aero = point.aero
    rot = point.rotor
    B = rot.blades
    omega = aero.omega
    if n_psi is None:
        need = 2.0 * np.pi / omega * fs_target
        n_psi = B
        while n_psi < need:
            n_psi *= 2
    na = len(aero.psi)
    # tables, resampled to the acoustic grid
    if isinstance(loads, tuple):
        f = np.stack(loads, axis=-1)                       # explicit (ft, fz), e.g. a load variant
    elif loads == "base" and aero.fz_base is not None:
        f = np.stack([aero.ft_base, aero.fz_base], axis=-1)
    else:
        f = np.stack([aero.ft, aero.fz], axis=-1)          # (Npsi, Nr, 2)
    fv, fd = _fourier_resample(f, n_psi, axis=0, derivatives=1)
    zstat = np.mean(aero.z, axis=0)
    tables = aero.__dict__.setdefault("_heave_tables", {})    # the same for every load on this solution
    if n_psi not in tables:
        tables[n_psi] = [np.ascontiguousarray(x.T) for x in _fourier_resample(aero.z - zstat[None, :], n_psi, axis=0)]
    z_tab, dz_tab, ddz_tab = tables[n_psi]
    f_tab = np.ascontiguousarray(np.transpose(fv, (1, 0, 2)))
    df_tab = np.ascontiguousarray(np.transpose(fd, (1, 0, 2)))
    R = aero.R
    x = aero.x
    chord = aero.chord
    sweep = np.where(x > rot.tip_sweep_start, (x - rot.tip_sweep_start) * R * np.tan(np.radians(rot.tip_sweep_deg)),
                     0.0)
    xc, share = chordwise_points(chord_points)
    pos, st, wt = [], [], []
    for i in range(len(x)):
        for j in range(len(xc)):
            pos.append([x[i] * R, (0.25 - xc[j]) * chord[i] - sweep[i], zstat[i]])
            st.append(i)
            wt.append(share[j] * aero.dx[i] * R)
    if thickness:
        _, dx = radial_stations(rot.root_cutout, len(x))
        edges = np.concatenate([[rot.root_cutout], rot.root_cutout + np.cumsum(dx)])
        theta_geo = aero.theta0 + rot.twist_at(x) + np.mean(aero.theta_e, axis=0)
        tc, ta, ts = blade_surface(rot, edges, theta_geo, surface_chord)
        tc = tc.copy()
        tc[:, 2] += zstat[ts]
    else:
        tc, ta, ts = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    return SourceModel(omega=omega, blades=B, n_psi=n_psi, U=np.asarray(point.U, float), rho=aero.rho, c=aero.c,
                       lp_pos=np.array(pos), lp_station=np.array(st, np.int64), lp_weight=np.array(wt),
                       tp_pos=tc, tp_area=ta, tp_station=ts, f_tab=f_tab, df_tab=df_tab, z_tab=z_tab,
                       dz_tab=dz_tab, ddz_tab=ddz_tab)


def morph_sequence(src, point, variants, revs, rng, concentration=0.4):
    """Give a source model blade loads spanning `revs` revolutions, each a random mix (Dirichlet) of the load
    variants, cross-faded from one revolution to the next: blades meet a different revolution on every pass."""
    aero = point.aero
    n = src.n_psi
    tabs = []
    for v in variants:
        fv, fd = _fourier_resample(np.stack([v["ft"], v["fz"]], axis=-1), n, axis=0, derivatives=1)
        tabs.append((np.transpose(fv, (1, 0, 2)), np.transpose(fd, (1, 0, 2))))
    W = rng.dirichlet(np.full(len(variants), concentration), size=revs)
    u = np.arange(revs * n) / n
    k0 = np.floor(u).astype(int)
    f = u - k0
    sm = 0.5 - 0.5 * np.cos(np.pi * f)
    dsm = 0.5 * np.pi * np.sin(np.pi * f) / (2.0 * np.pi)          # d(sm)/dpsi
    nr = src.f_tab.shape[0]
    F = np.zeros((nr, revs * n, 2))
    dF = np.zeros((nr, revs * n, 2))
    idx = np.arange(revs * n) % n
    for vi, (fv, fd) in enumerate(tabs):
        w = (1.0 - sm) * W[k0 % revs, vi] + sm * W[(k0 + 1) % revs, vi]
        dw = dsm * (W[(k0 + 1) % revs, vi] - W[k0 % revs, vi])
        F += w[None, :, None] * fv[:, idx, :]
        dF += w[None, :, None] * fd[:, idx, :] + dw[None, :, None] * fv[:, idx, :]
    src.f_tab = np.ascontiguousarray(F)
    src.df_tab = np.ascontiguousarray(dF)
    return src


# ---------------------------------------------------------------- kernels

@njit(fastmath=True, cache=True, inline='always')
def _kinematics(px, py, pz, cs, sn, Om, z, dz, ddz, Ux, Uy, Uz, tau):
    """Position, velocity and acceleration of a blade point (blade frame px, py, pz + elastic heave z) at azimuth
    (cos, sin) in the air-fixed frame."""
    zz = pz + z
    rx = cs * px - sn * py
    ry = sn * px + cs * py
    yx = rx - Ux * tau
    yy = ry - Uy * tau
    yz = zz - Uz * tau
    vx = -Ux - Om * ry
    vy = -Uy + Om * rx
    vz = -Uz + Om * dz
    ax = -Om * Om * rx
    ay = -Om * Om * ry
    az = Om * Om * ddz
    return yx, yy, yz, vx, vy, vz, ax, ay, az


@njit(fastmath=True, cache=True, inline='always')
def _retarded(yx, yy, yz, ox, oy, oz, vox, voy, voz, tau, c):
    """Reception time and the vector source -> observer for emission at tau from y, observer at o + vo t."""
    dx = ox + vox * tau - yx
    dy = oy + voy * tau - yy
    dz = oz + voz * tau - yz
    dv = dx * vox + dy * voy + dz * voz
    a2 = c * c - (vox * vox + voy * voy + voz * voz)
    d2 = dx * dx + dy * dy + dz * dz
    s = (dv + np.sqrt(dv * dv + a2 * d2)) / a2
    rx = dx + vox * s
    ry = dy + voy * s
    rz = dz + voz * s
    return tau + s, rx, ry, rz, c * s


@njit(fastmath=True, cache=True)
def _accumulate(out, t_a, f_a, t_b, f_b, t0, dt, nout, periodic):
    i0 = int(np.ceil((t_a - t0) / dt))
    i1 = int(np.ceil((t_b - t0) / dt)) - 1
    if i1 < i0:
        return
    inv = 1.0 / (t_b - t_a)
    for i in range(i0, i1 + 1):
        w = (t0 + i * dt - t_a) * inv
        val = f_a + w * (f_b - f_a)
        if periodic:
            out[i % nout] += val
        elif 0 <= i < nout:
            out[i] += val


@njit(parallel=True, fastmath=True, cache=True)
def f1a_kernel(src_pos, src_area, src_station, src_weight, is_loading, f_tab, df_tab, z_tab, dz_tab, ddz_tab,
               cos_tab, sin_tab, Om, n_psi, blades, Ux, Uy, Uz, obs, vobs, c, rho, n_start, n_steps, dtau, mod,
               t0, dt, nout, periodic, nth, stride, near_only=False, both=False):
    """Accumulate thickness (is_loading 0) or loading (1) noise of all elements onto the observer grid.
    near_only: only the terms falling as 1/r^2, the moving blade's incompressible (hydrodynamic) pressure; both: the
    whole and those terms apart, in one pass over the geometry.

    Blade b (b < blades) at step n has azimuth index (n stride + b n_psi / blades) mod n_psi and emission time
    n stride dtau. mod (blades, n_steps + 1): load scale per blade and step (loading only). periodic: wrap the
    output with period nout dt (one revolution, blade 0 only). Returns (n_threads, 1 or, with both, 2, nout) partial
    sums."""
    ne = src_pos.shape[0]
    n_tab = f_tab.shape[1]
    part = np.zeros((nth, 2 if both else 1, nout))
    ox, oy, oz = obs[0], obs[1], obs[2]
    vox, voy, voz = vobs[0], vobs[1], vobs[2]
    inv4pi = 1.0 / (4.0 * np.pi)
    for th in prange(nth):
        acc = part[th, 0]
        acc_near = part[th, part.shape[1] - 1]
        for e in range(th, ne, nth):
            px, py, pz = src_pos[e, 0], src_pos[e, 1], src_pos[e, 2]
            s = src_station[e]
            for b in range(blades):
                t_prev = 0.0
                f_prev = 0.0
                g_prev = 0.0
                for k in range(n_steps + 1):
                    n = (n_start + k) * stride
                    tau = n * dtau
                    jj = n + b * (n_psi // blades)
                    j = jj % n_psi
                    jt = jj % n_tab              # the loads may span several (different) revolutions
                    cs = cos_tab[j]
                    sn = sin_tab[j]
                    yx, yy, yz, vx, vy, vz, ax, ay, az = _kinematics(px, py, pz, cs, sn, Om, z_tab[s, j],
                                                                     dz_tab[s, j], ddz_tab[s, j], Ux, Uy, Uz, tau)
                    t, rx, ry, rz, r = _retarded(yx, yy, yz, ox, oy, oz, vox, voy, voz, tau, c)
                    ir = 1.0 / r
                    hx, hy, hz = rx * ir, ry * ir, rz * ir
                    Mr = (vx * hx + vy * hy + vz * hz) / c
                    dMr = (ax * hx + ay * hy + az * hz) / c
                    M2 = (vx * vx + vy * vy + vz * vz) / (c * c)
                    om = 1.0 / (1.0 - Mr)
                    if is_loading == 0:
                        Ax0, Ay0, Az0 = src_area[e, 0], src_area[e, 1], src_area[e, 2]
                        Ax = cs * Ax0 - sn * Ay0
                        Ay = sn * Ax0 + cs * Ay0
                        Az = Az0
                        vn = vx * Ax + vy * Ay + vz * Az
                        dvn = ax * Ax + ay * Ay + az * Az + Om * (-vx * Ay + vy * Ax)
                        g = rho * vn * c * (Mr - M2) * ir * ir * om * om * om
                        f = rho * (dvn * ir * om * om + vn * (r * dMr + c * (Mr - M2)) * ir * ir * om * om * om)
                    else:
                        w = src_weight[e] * mod[b, k]
                        # force on the fluid: minus the blade's (0, ft, fz), rotated; its rate adds the rotation
                        ft = f_tab[s, jt, 0]
                        fz = f_tab[s, jt, 1]
                        dft = df_tab[s, jt, 0] * Om
                        dfz = df_tab[s, jt, 1] * Om
                        Fx = w * (sn * ft)
                        Fy = w * (-cs * ft)
                        Fz = w * (-fz)
                        dFx = w * (sn * dft + Om * cs * ft)
                        dFy = w * (-cs * dft + Om * sn * ft)
                        dFz = w * (-dfz)
                        Fr = Fx * hx + Fy * hy + Fz * hz
                        dFr = dFx * hx + dFy * hy + dFz * hz
                        FM = (Fx * vx + Fy * vy + Fz * vz) / c
                        g = (Fr - FM) * ir * ir * om * om + Fr * (Mr - M2) * ir * ir * om * om * om
                        f = (dFr * ir * om * om / c + (Fr - FM) * ir * ir * om * om
                             + Fr * (r * dMr + c * (Mr - M2)) * ir * ir * om * om * om / c)
                    if near_only:
                        f = g
                    f *= inv4pi
                    g *= inv4pi
                    if k > 0:
                        _accumulate(acc, t_prev, f_prev, t, f, t0, dt, nout, periodic)
                        if both:
                            _accumulate(acc_near, t_prev, g_prev, t, g, t0, dt, nout, periodic)
                    t_prev = t
                    f_prev = f
                    g_prev = g
    return part


@njit(parallel=True, fastmath=True, cache=True)
def f1a_loads_kernel(src_pos, src_station, src_weight, f_tabs, df_tabs, z_tab, dz_tab, ddz_tab, cos_tab, sin_tab, Om, n_psi,
                     Ux, Uy, Uz, obs, vobs, c, n_steps, dtau, t0, dt, nout, nth):
    """f1a_kernel's loading noise of one blade over one revolution (periodic), for many sets of loads on the same blade
    at once. The motion and the geometry to the observer don't depend on the loads, and the pressure is linear in them:
    each step's four coefficients are worked out once, and each set costs four products. f_tabs, df_tabs:
    (sets, stations, 2, n_psi). Returns (n_threads, nout, sets) partial sums."""
    ne = src_pos.shape[0]
    nset = f_tabs.shape[0]
    part = np.zeros((nth, nout, nset))
    ox, oy, oz = obs[0], obs[1], obs[2]
    vox, voy, voz = vobs[0], vobs[1], vobs[2]
    inv4pi = 1.0 / (4.0 * np.pi)
    for th in prange(nth):
        acc = part[th]
        fa = np.zeros(nset)
        fb = np.zeros(nset)
        for e in range(th, ne, nth):
            px, py, pz = src_pos[e, 0], src_pos[e, 1], src_pos[e, 2]
            s = src_station[e]
            w = src_weight[e] * inv4pi
            t_prev = 0.0
            for k in range(n_steps + 1):
                tau = k * dtau
                j = k % n_psi
                cs = cos_tab[j]
                sn = sin_tab[j]
                yx, yy, yz, vx, vy, vz, ax, ay, az = _kinematics(px, py, pz, cs, sn, Om, z_tab[s, j], dz_tab[s, j],
                                                                 ddz_tab[s, j], Ux, Uy, Uz, tau)
                t, rx, ry, rz, r = _retarded(yx, yy, yz, ox, oy, oz, vox, voy, voz, tau, c)
                ir = 1.0 / r
                hx, hy, hz = rx * ir, ry * ir, rz * ir
                Mr = (vx * hx + vy * hy + vz * hz) / c
                dMr = (ax * hx + ay * hy + az * hz) / c
                M2 = (vx * vx + vy * vy + vz * vz) / (c * c)
                om = 1.0 / (1.0 - Mr)
                g1 = ir * om * om / c
                g2 = ir * ir * om * om
                g3 = (r * dMr + c * (Mr - M2)) * ir * ir * om * om * om / c
                # the force's terms, as f1a_kernel's, by the load each is a multiple of
                along = sn * hx - cs * hy
                c_ft = w * (g1 * Om * (cs * hx + sn * hy) + g2 * (along - (sn * vx - cs * vy) / c) + g3 * along)
                c_fz = -w * (g2 * (hz - vz / c) + g3 * hz)
                c_dft = w * g1 * along * Om
                c_dfz = -w * g1 * hz * Om
                for q in range(nset):
                    fb[q] = (c_ft * f_tabs[q, s, 0, j] + c_fz * f_tabs[q, s, 1, j]
                             + c_dft * df_tabs[q, s, 0, j] + c_dfz * df_tabs[q, s, 1, j])
                if k > 0:
                    i0 = int(np.ceil((t_prev - t0) / dt))
                    i1 = int(np.ceil((t - t0) / dt)) - 1
                    if i1 >= i0:
                        inv = 1.0 / (t - t_prev)
                        for i in range(i0, i1 + 1):
                            u = (t0 + i * dt - t_prev) * inv
                            ii = i % nout
                            for q in range(nset):
                                acc[ii, q] += fa[q] + u * (fb[q] - fa[q])
                t_prev = t
                fa, fb = fb, fa
    return part


def _azimuths(n_psi):
    psi = 2.0 * np.pi * np.arange(n_psi) / n_psi
    return np.cos(psi), np.sin(psi)


@dataclass
class LoadSets:
    """Many sets of blade loads on one source model, for loading_harmonics."""
    src: SourceModel
    f: np.ndarray                       # (sets, stations, 2, n_psi)
    df: np.ndarray


def load_sets(src, loads):
    """loads: (ft, fz) pairs on the source model's blade, resampled to its acoustic grid (as build_sources does)."""
    nr = src.f_tab.shape[0]
    f = np.empty((len(loads), nr, 2, src.n_psi))
    df = np.empty_like(f)
    for i, pair in enumerate(loads):
        f[i], df[i] = _fourier_resample(np.transpose(np.stack(pair), (2, 0, 1)), src.n_psi, axis=-1, derivatives=1)
    return LoadSets(src, f, df)


def loading_harmonics(sets, observer, n_out=None):
    """single_blade_harmonics' loading harmonics of every set: (sets, n_out / 2 + 1)."""
    src = sets.src
    n_out = n_out or src.n_psi
    Ux, Uy, Uz = src.U
    cos_tab, sin_tab = _azimuths(src.n_psi)
    nth = get_num_threads()
    part = f1a_loads_kernel(src.lp_pos, src.lp_station, src.lp_weight, sets.f, sets.df, src.z_tab, src.dz_tab, src.ddz_tab,
                            cos_tab, sin_tab, src.omega, src.n_psi, Ux, Uy, Uz, np.asarray(observer, float), -src.U,
                            src.c, src.n_psi, src.period / src.n_psi, 0.0, src.period / n_out, n_out, nth)
    return sfft.rfft(part.sum(axis=0).T, axis=-1)


def _run(src, obs, vobs, n_start, n_steps, t0, dt, nout, periodic, blades, mod=None, parts=("thickness", "loading"),
         thickness_stride=1, near_only=False, both=False):
    dtau = src.period / src.n_psi
    cos_tab, sin_tab = _azimuths(src.n_psi)
    Ux, Uy, Uz = src.U
    if mod is None:
        mod = np.ones((blades, n_steps + 1))
    obs = np.asarray(obs, float)
    vobs = np.asarray(vobs, float)
    nth = get_num_threads()
    out = {}
    if "thickness" in parts and len(src.tp_pos):
        st = thickness_stride
        out["thickness"] = f1a_kernel(src.tp_pos, src.tp_area, src.tp_station, np.zeros(len(src.tp_pos)), 0,
                                      src.f_tab, src.df_tab, src.z_tab, src.dz_tab, src.ddz_tab, cos_tab, sin_tab,
                                      src.omega, src.n_psi, blades, Ux, Uy, Uz, obs, vobs, src.c, src.rho,
                                      n_start // st, n_steps // st + (0 if periodic else 1), dtau,
                                      np.ones((blades, n_steps // st + 2)),
                                      t0, dt, nout, periodic, nth, st, near_only, both).sum(axis=0)
    else:
        out["thickness"] = np.zeros((2 if both else 1, nout))
    if "loading" in parts:
        out["loading"] = f1a_kernel(src.lp_pos, np.zeros((len(src.lp_pos), 3)), src.lp_station, src.lp_weight, 1,
                                    src.f_tab, src.df_tab, src.z_tab, src.dz_tab, src.ddz_tab, cos_tab, sin_tab,
                                    src.omega, src.n_psi, blades, Ux, Uy, Uz, obs, vobs, src.c, src.rho, n_start, n_steps,
                                    dtau, mod,
                                    t0, dt, nout, periodic, nth, 1, near_only, both).sum(axis=0)
    else:
        out["loading"] = np.zeros((2 if both else 1, nout))
    return out if both else {k: v[0] for k, v in out.items()}


def periodic_signature(src, observer, n_out=None, thickness_stride=None, near_only=False, both=False):
    """One blade's thickness and loading pressure over one revolution at an observer riding with the rotor
    (position relative to the hub, canonical frame). Returns (t, p_thickness, p_loading), t from 0 to the period; with
    both, each pressure has two rows, the whole and its near-field terms. thickness_stride: emission steps of the
    thickness panels, in acoustic azimuth steps (default: about 8192 per revolution; thickness noise is low-frequency
    unless the advancing tip is near Mach 0.9)."""
    n_out = n_out or src.n_psi
    T = src.period
    obs = np.asarray(observer, float)
    if thickness_stride is None:
        thickness_stride = max(1, src.n_psi // 8192) if thickness_mach(src) < 0.85 else 1
    out = _run(src, obs, -src.U, 0, src.n_psi, 0.0, T / n_out, n_out, True, 1,
               thickness_stride=thickness_stride, near_only=near_only, both=both)
    return np.arange(n_out) * T / n_out, out["thickness"], out["loading"]


def thickness_mach(src):
    """Advancing tip Mach number of the source model."""
    r_tip = np.max(np.linalg.norm(src.lp_pos[:, :2], axis=1))
    return (src.omega * r_tip + np.linalg.norm(src.U[:2])) / src.c


def single_blade_harmonics(src, observer, n_out=None, near_only=False):
    """Complex harmonics of one blade's thickness and loading noise, per rotor harmonic m = 0..n_out/2:
    p(t) = sum_m Re(2 H_m e^{i m Omega t}) / n_out (rfft convention). near_only: their 1/r^2 terms alone."""
    t, pt, pl = periodic_signature(src, observer, n_out, near_only=near_only)
    return np.fft.rfft(pt), np.fft.rfft(pl), len(t)


def harmonics_with_near(src, observer, n_out=None):
    """single_blade_harmonics and, from the same pass, its near_only terms: H_thickness, H_loading, n, and those two
    of the near field"""
    t, pt, pl = periodic_signature(src, observer, n_out, both=True)
    return np.fft.rfft(pt[0]), np.fft.rfft(pl[0]), len(t), np.fft.rfft(pt[1]), np.fft.rfft(pl[1])


def time_history(src, observer, t_start, duration, fs, observer_velocity=(0.0, 0.0, 0.0), mod=None,
                 parts=("thickness", "loading"), thickness_stride=1, mod_fn=None):
    """All blades' thickness and loading pressure at an observer in uniform motion (air-fixed canonical frame,
    position at t = 0), sampled at fs from t_start for duration (s). Emission covers what reaches the observer.
    mod: optional (blades, steps) load scale per blade (unsteadiness); steps cover the emission span; or
    mod_fn(n_start, n_steps, dtau) returning it."""
    obs = np.asarray(observer, float)
    vo = np.asarray(observer_velocity, float)
    T = src.period
    dtau = T / src.n_psi
    # emission span: what arrives in [t_start, t_start + duration], with margin
    span = np.linalg.norm(obs) + np.linalg.norm(src.U) * (abs(t_start) + duration) + 2.0 * np.linalg.norm(src.lp_pos[-1])
    span += np.linalg.norm(vo) * (abs(t_start) + duration)
    tmax = span / src.c + 1.0
    n_start = int(np.floor((t_start - tmax) / dtau))
    n_steps = int(np.ceil((duration + tmax) / dtau)) + 1
    nout = int(round(duration * fs))
    if mod_fn is not None:
        mod = mod_fn(n_start, n_steps, dtau)
    if mod is not None and mod.shape[1] < n_steps + 1:
        raise ValueError("mod must cover the emission steps")
    out = _run(src, obs, vo, n_start, n_steps, t_start, 1.0 / fs, nout, False, src.blades, mod, parts,
               thickness_stride)
    return t_start + np.arange(nout) / fs, out["thickness"], out["loading"], (n_start, n_steps)
