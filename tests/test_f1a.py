"""Formulation 1A against closed-form and independent solutions."""
import numpy as np
from scipy.special import jv

from pywopwop import f1a

C0, RHO = 340.0, 1.225


def _point_source(B=2, Re=1.0, omega=100.0, n_psi=4096, f_tab=None, z_tab=None, U=(0.0, 0.0, 0.0), area=None):
    """A hand-built source model: one element per blade at radius Re (a load point, or a thickness panel)."""
    nr = 1
    if f_tab is None:
        f_tab = np.zeros((nr, n_psi, 2))
    if z_tab is None:
        z_tab = np.zeros((nr, n_psi))
    k = np.fft.rfftfreq(n_psi, 1.0 / n_psi)
    d = lambda a, p: np.fft.irfft(np.fft.rfft(a, axis=1) * (1j * k)[None, :, None][..., :a.ndim - 1].reshape(
        (1, -1) + (1,) * (a.ndim - 2)) ** p, n_psi, axis=1)
    df = np.stack([np.fft.irfft(np.fft.rfft(f_tab[..., i], axis=1) * 1j * k, n_psi, axis=1) for i in range(2)], -1)
    dz = np.fft.irfft(np.fft.rfft(z_tab, axis=1) * 1j * k, n_psi, axis=1)
    ddz = np.fft.irfft(np.fft.rfft(z_tab, axis=1) * -(k ** 2), n_psi, axis=1)
    tp = np.zeros((0, 3)) if area is None else np.array([[Re, 0.0, 0.0]])
    ta = np.zeros((0, 3)) if area is None else np.array([area])
    return f1a.SourceModel(omega=omega, blades=B, n_psi=n_psi, U=np.asarray(U, float), rho=RHO, c=C0,
                           lp_pos=np.array([[Re, 0.0, 0.0]]), lp_station=np.zeros(1, np.int64),
                           lp_weight=np.ones(1), tp_pos=tp, tp_area=ta,
                           tp_station=np.zeros(len(tp), np.int64), f_tab=f_tab, df_tab=df, z_tab=z_tab,
                           dz_tab=dz, ddz_tab=ddz)


def test_gutin_hover_harmonics():
    """Far-field loading harmonics of a hovering rotor with concentrated thrust and torque (Gutin 1936)."""
    B, Re, omega, n_psi = 2, 1.0, 120.0, 4096
    T, Q = 400.0, 60.0
    f_tab = np.zeros((1, n_psi, 2))
    f_tab[0, :, 0] = -Q / (B * Re)        # drag opposes rotation
    f_tab[0, :, 1] = T / B
    src = _point_source(B, Re, omega, n_psi, f_tab=f_tab)
    r = 200.0
    for theta_deg in (30.0, 60.0, 80.0, 100.0, 150.0):
        th = np.radians(theta_deg)
        obs = r * np.array([np.sin(th), 0.0, np.cos(th)])
        _, H_l, n = f1a.single_blade_harmonics(src, obs, n_psi)
        for m in (1, 2, 3):
            got = B * 2.0 * np.abs(H_l[m * B]) / n / np.sqrt(2.0)
            k = m * B * omega / C0
            ref = (m * B * omega / (2 * np.sqrt(2) * np.pi * C0 * r)
                   * abs(-T * np.cos(th) + Q * C0 / (omega * Re ** 2)) * abs(jv(m * B, k * Re * np.sin(th))))
            assert abs(got - ref) / ref < 0.03, (theta_deg, m, got, ref)


def _reference_f1(B, Re, omega, n_psi, f_tab, z_tab, area, U, obs, n_out):
    """Formulation 1 by brute force: the retarded integrands accumulated in observer time with numerical
    kinematics (finite differences of the position), then differentiated spectrally in observer time."""
    T = 2 * np.pi / omega
    tau = np.arange(n_psi + 1) * T / n_psi
    psi = omega * tau
    j = np.arange(n_psi + 1) % n_psi
    z = z_tab[0, j]
    pos = lambda tt, zz: np.stack([Re * np.cos(omega * tt) - U[0] * tt, Re * np.sin(omega * tt) - U[1] * tt,
                                   zz - U[2] * tt], -1)
    y = pos(tau, z)
    zs = np.fft.irfft(np.fft.rfft(z_tab[0]) * np.exp(1j * np.fft.rfftfreq(n_psi, 1 / n_psi) * 2 * np.pi * 1e-4),
                      n_psi)
    h = T / n_psi
    # velocity by central differences of the position (z from the periodic table)
    zp = np.roll(z_tab[0], -1)[j]
    zm = np.roll(z_tab[0], 1)[j]
    v = (pos(tau + h, zp) - pos(tau - h, zm)) / (2 * h)
    vo = -np.asarray(U)
    d = obs[None, :] + vo[None, :] * tau[:, None] - y
    a2 = C0 ** 2 - vo @ vo
    dv = d @ vo
    s = (dv + np.sqrt(dv ** 2 + a2 * np.sum(d * d, 1))) / a2
    rv = d + vo[None, :] * s[:, None]
    r = C0 * s
    rh = rv / r[:, None]
    Mr = np.sum(v * rh, 1) / C0
    t = tau + s
    cs, sn = np.cos(psi), np.sin(psi)
    A = np.stack([cs * area[0] - sn * area[1], sn * area[0] + cs * area[1], np.full_like(cs, area[2])], -1)
    gT = RHO * np.sum(v * A, 1) / (r * (1 - Mr))
    ft, fz = f_tab[0, j, 0], f_tab[0, j, 1]
    F = np.stack([sn * ft, -cs * ft, -fz], -1)
    Fr = np.sum(F * rh, 1)
    gL1 = Fr / (C0 * r * (1 - Mr))
    gL2 = Fr / (r ** 2 * (1 - Mr))
    tg = (np.arange(n_out) * T / n_out)
    out = []
    for g in (gT, gL1, gL2):
        tt = np.concatenate([t - T, t, t + T])
        gg = np.concatenate([g, g, g])
        tw = t[0] + (tg - t[0]) % T
        out.append(np.interp(tw, tt, gg))
    kk = np.fft.rfftfreq(n_out, T / n_out) * 2 * np.pi
    dt = lambda x: np.fft.irfft(np.fft.rfft(x) * 1j * kk, n_out)
    pT = dt(out[0]) / (4 * np.pi)
    pL = (dt(out[1]) + out[2]) / (4 * np.pi)
    return tg, pT, pL, t[0]


def _sources_for_f1(n_psi, U):
    psi = np.arange(n_psi) * 2 * np.pi / n_psi
    f_tab = np.zeros((1, n_psi, 2))
    f_tab[0, :, 0] = -20.0 + 6.0 * np.cos(psi + 0.3)
    f_tab[0, :, 1] = 300.0 + 80.0 * np.sin(psi) + 40.0 * np.cos(3 * psi) + 25.0 * np.exp(-((psi - 1.2) / 0.05) ** 2)
    z_tab = np.zeros((1, n_psi))
    z_tab[0] = 0.04 * np.cos(2 * psi) + 0.02 * np.sin(psi)
    area = np.array([0.0, 0.004, 0.01])
    return f_tab, z_tab, area


def test_formulation_1_vs_1a_hover_with_unsteady_loads_and_heave():
    """F1A (analytic time derivatives) equals Formulation 1 (numerical observer-time derivative) for a panel and a
    load point with unsteady loads and elastic heave."""
    B, Re, omega, n_psi = 1, 1.5, 80.0, 8192
    f_tab, z_tab, area = _sources_for_f1(n_psi, (0, 0, 0))
    src = _point_source(B, Re, omega, n_psi, f_tab=f_tab, z_tab=z_tab, area=area)
    obs = np.array([-20.0, 25.0, -10.0])
    t, pt, pl = f1a.periodic_signature(src, obs, 4096)
    _, rT, rL, _ = _reference_f1(B, Re, omega, n_psi, f_tab, z_tab, area, (0.0, 0.0, 0.0), obs, 4096)
    for got, ref in ((pt, rT), (pl, rL)):
        assert np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)) < 1e-3


def test_formulation_1_vs_1a_forward_flight_fixed_observer():
    """The same with the rotor flying past a fixed observer (aperiodic, Doppler-shifted, all of it)."""
    B, Re, omega, n_psi = 1, 1.5, 80.0, 4096
    U = np.array([60.0, 0.0, -4.0])
    f_tab, z_tab, area = _sources_for_f1(n_psi, U)
    src = _point_source(B, Re, omega, n_psi, f_tab=f_tab, z_tab=z_tab, U=U, area=area)
    obs = np.array([-30.0, 20.0, -15.0])
    fs = 40000.0
    t, pt, pl, (n0, ns) = f1a.time_history(src, obs, 0.25, 0.5, fs)
    # reference: emission times over the same span, numerical kinematics, derivative at the fixed observer
    T = 2 * np.pi / omega
    dtau = T / n_psi
    n = np.arange(n0, n0 + ns + 1)
    tau = n * dtau
    j = n % n_psi
    pos = lambda tt, jj: np.stack([Re * np.cos(omega * tt) - U[0] * tt, Re * np.sin(omega * tt) - U[1] * tt,
                                   z_tab[0, jj % n_psi] - U[2] * tt], -1)
    y = pos(tau, j)
    v = (pos(tau + dtau, j + 1) - pos(tau - dtau, j - 1)) / (2 * dtau)
    d = obs[None, :] - y
    r = np.linalg.norm(d, axis=1)
    rh = d / r[:, None]
    Mr = np.sum(v * rh, 1) / C0
    te = tau + r / C0
    psi = omega * tau
    cs, sn = np.cos(psi), np.sin(psi)
    A = np.stack([cs * area[0] - sn * area[1], sn * area[0] + cs * area[1], np.full_like(cs, area[2])], -1)
    ft, fz = f_tab[0, j, 0], f_tab[0, j, 1]
    F = np.stack([sn * ft, -cs * ft, -fz], -1)
    Fr = np.sum(F * rh, 1)
    gT = np.interp(t, te, RHO * np.sum(v * A, 1) / (r * (1 - Mr)))
    gL1 = np.interp(t, te, Fr / (C0 * r * (1 - Mr)))
    gL2 = np.interp(t, te, Fr / (r ** 2 * (1 - Mr)))
    rT = np.gradient(gT, t) / (4 * np.pi)
    rL = (np.gradient(gL1, t) + gL2) / (4 * np.pi)
    inner = slice(200, -200)
    for got, ref in ((pt, rT), (pl, rL)):
        g, rf = got[inner], ref[inner]
        assert np.sqrt(np.mean((g - rf) ** 2)) / np.sqrt(np.mean(rf ** 2)) < 5e-3


def test_reception_time_closes_the_retarded_equation():
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = rng.normal(size=3) * 5
        o = rng.normal(size=3) * 50
        vo = rng.normal(size=3) * 60
        tau = rng.uniform(-1, 1)
        t, rx, ry, rz, r = f1a._retarded(*y, *o, *vo, tau, C0)
        x_obs = o + vo * t
        assert abs(np.linalg.norm(x_obs - y) - C0 * (t - tau)) < 1e-9 * C0
        assert abs(np.linalg.norm([rx, ry, rz]) - r) < 1e-9


def test_near_field_terms_fade_with_distance():
    """the 1/r^2 terms (the blade's hydrodynamic pressure) lead close to the blade and vanish far from it"""
    from pywopwop import f1a
    from pywopwop.aero import AeroOptions
    from pywopwop.model import load_aircraft, FlightCondition
    from pywopwop.operating import operate
    st = operate(load_aircraft("mv22b"), FlightCondition(airspeed_kt=220.0, nacelle_deg=0.0),
                 AeroOptions(n_radial=16, n_azimuth=180))
    p = st.points[0]
    src = f1a.build_sources(p, chord_points=1, surface_chord=12, fs_target=24000.0)
    R = p.rotor.radius
    share = []
    for d, m in ((0.3, slice(1, 6)), (300.0, slice(1, 6))):     # the far case tens of wavelengths out
        o = (R + d, 0.0, 0.0)
        Ht, Hl, n = f1a.single_blade_harmonics(src, o)
        Htn, Hln, _ = f1a.single_blade_harmonics(src, o, near_only=True)
        share.append(np.abs(Htn + Hln)[m].sum() / np.abs(Ht + Hl)[m].sum())
    assert share[0] > 0.3 and share[1] < 0.05
    both = f1a.harmonics_with_near(src, o)                      # the whole and the near field from one pass
    for a, b in zip(both, (Ht, Hl, n, Htn, Hln)):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9 * np.max(np.abs(Hl)))


def test_load_sets_match_one_at_a_time():
    """Many sets of loads at once give what each gives alone: the geometry is shared and the pressure linear in the loads"""
    from pywopwop.model import load_aircraft, FlightCondition
    from pywopwop.operating import operate
    from pywopwop.aero import AeroOptions
    p = operate(load_aircraft("uh60a"), FlightCondition(airspeed_kt=80.0), AeroOptions(n_radial=16, n_azimuth=180)).points[0]
    src = f1a.build_sources(p, chord_points=2, thickness=False, fs_target=24000.0)
    a = p.aero
    loads = [(a.ft, a.fz), (1.5 * a.ft, 0.5 * a.fz), (a.fz, a.ft)]
    o = p.to_rotor(np.array([60.0, -40.0, -30.0]))
    H = f1a.loading_harmonics(f1a.load_sets(src, loads), o)
    for h, (ft, fz) in zip(H, loads):
        one = f1a.build_sources(p, n_psi=src.n_psi, chord_points=2, thickness=False, loads=(ft, fz))
        ref = f1a.single_blade_harmonics(one, o)[1]
        assert np.max(np.abs(h - ref)) < 1e-9 * np.max(np.abs(ref))
