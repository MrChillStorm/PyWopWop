"""Checks of the models against published and closed-form results, and solver timings."""
import time

import numpy as np
from numba import njit, prange
from scipy.special import jv

from . import bpm, f1a, synth
from .atmosphere import iso9613_alpha

# NASA RP-1218 table D2 (fig. 45(a)): pressure side, suction side, laminar, total at every 1/3-octave centre
_D2 = np.array([[20.654, 28.704, -17.142, 29.336], [24.461, 31.965, -13.285, 32.676], [28.291, 35.244, -9.018, 36.042],
                [31.437, 37.937, -5.161, 38.815], [34.309, 40.400, -1.304, 41.356], [37.023, 42.736, 2.690, 43.768],
                [39.577, 44.949, 6.820, 46.057], [41.761, 46.859, 10.677, 48.034], [43.845, 48.706, 14.671, 49.954],
                [45.839, 50.503, 18.801, 51.849], [47.581, 52.106, 22.658, 53.568], [49.233, 53.664, 26.515, 55.255],
                [50.987, 55.368, 30.782, 57.106], [52.533, 56.907, 37.725, 58.817], [54.074, 57.750, 47.262, 60.167],
                [55.570, 57.500, 48.959, 60.496], [56.044, 56.082, 41.796, 59.455], [55.399, 54.541, 32.428, 58.208],
                [53.840, 52.942, 28.433, 56.553], [52.190, 51.253, 24.304, 54.821], [50.638, 49.614, 20.447, 53.192],
                [49.044, 47.890, 16.590, 51.523], [47.202, 45.851, 12.323, 49.591], [45.436, 43.863, 8.466, 47.731],
                [43.549, 41.710, 4.609, 45.737], [41.440, 39.279, 0.614, 43.503], [39.065, 36.522, -3.515, 40.987]])

# ISO 9613-2 table 2, 20 C and 70 % relative humidity, dB/km at the octave bands' exact mid-frequencies
_ISO = {63: 0.1, 125: 0.3, 250: 1.1, 500: 2.8, 1000: 5.0, 2000: 9.0, 4000: 22.9, 8000: 76.6}


def _point_rotor(B, Re, omega, n_psi, T, Q):
    f_tab = np.zeros((1, n_psi, 2))
    f_tab[0, :, 0] = -Q / (B * Re)
    f_tab[0, :, 1] = T / B
    z = np.zeros((1, n_psi))
    return f1a.SourceModel(omega=omega, blades=B, n_psi=n_psi, U=np.zeros(3), rho=1.225, c=340.0,
                           lp_pos=np.array([[Re, 0.0, 0.0]]), lp_station=np.zeros(1, np.int64),
                           lp_weight=np.ones(1), tp_pos=np.zeros((0, 3)), tp_area=np.zeros((0, 3)),
                           tp_station=np.zeros(0, np.int64), f_tab=f_tab, df_tab=np.zeros_like(f_tab), z_tab=z,
                           dz_tab=z, ddz_tab=z)


def check_bpm():
    f = bpm.FRCEN_BPM
    a = dict(alpha=1.516, c=0.3048, U=71.3, f=f, L=0.4572, r=1.22, theta=90.0, phi=90.0, trip=0, nu=1.4529e-5,
             c0=340.46)
    p, s, _, _ = bpm.tblte(**a)
    lam = bpm.lblvs(**a)
    tot = 10 * np.log10(10 ** (p / 10) + 10 ** (s / 10) + 10 ** (lam / 10)
                        + 10 ** (bpm.tblte(**a)[2] / 10))
    err = np.max(np.abs(np.column_stack([p, s, lam, tot]) - _D2))
    return "BPM self-noise vs NASA RP-1218 table D2", f"max {err:.3f} dB", err < 0.02


def check_gutin():
    B, Re, om, n = 2, 1.0, 120.0, 4096
    T, Q = 400.0, 60.0
    src = _point_rotor(B, Re, om, n, T, Q)
    worst = 0.0
    for th_deg in (30.0, 60.0, 80.0, 100.0, 150.0):
        th = np.radians(th_deg)
        obs = 200.0 * np.array([np.sin(th), 0.0, np.cos(th)])
        _, H, nn = f1a.single_blade_harmonics(src, obs, n)
        for m in (1, 2, 3):
            got = B * 2 * abs(H[m * B]) / nn / np.sqrt(2)
            ref = (m * B * om / (2 * np.sqrt(2) * np.pi * 340.0 * 200.0) * abs(-T * np.cos(th) + Q * 340.0 / om)
                   * abs(jv(m * B, m * B * om / 340.0 * Re * np.sin(th))))
            worst = max(worst, abs(20 * np.log10(got / ref)))
    return "F1A loading vs Gutin's hover harmonics (far field)", f"max {worst:.2f} dB", worst < 0.3


def check_absorption():
    worst = 0.0
    for f, ref in _ISO.items():
        fm = 1000.0 * 10.0 ** (0.3 * np.round(np.log2(f / 1000.0)))        # exact mid-frequency
        got = iso9613_alpha(fm, 293.15, 70.0) * 1000.0
        worst = max(worst, abs(round(got, 1) - ref))
    return ("ISO 9613-1 absorption vs ISO 9613-2 table 2 (dB/km, as printed)", f"max {worst:.1f} dB/km difference",
            worst < 0.05)


def check_hover_momentum():
    from .model import load_aircraft, FlightCondition
    from .operating import operate
    ac = load_aircraft("uh60a")
    st = operate(ac, FlightCondition())
    a = st.points[0].aero
    ideal = a.thrust ** 1.5 / np.sqrt(2 * a.rho * np.pi * a.R ** 2)
    fm = ideal / a.power
    return "Hover figure of merit (UH-60A main rotor, momentum theory)", f"FM {fm:.3f}", 0.65 < fm < 0.82


def check_modes():
    from .model import load_aircraft
    from .structure import blade_modes
    ac = load_aircraft("bo105")
    md = blade_modes(ac.main, np.linspace(0.2, 1.0, 40), 1.225)
    f2 = md.flap_ratio[1]
    return ("Bo 105 second flap mode from the first (published 2.78/rev)", f"{f2:.3f}/rev",
            abs(f2 - 2.78) / 2.78 < 0.03)


def check_loop():
    n = 4096
    t = np.arange(n) / n
    p = np.exp(-((t - 0.3) / 0.002) ** 2) - 0.5 * np.exp(-((t - 0.31) / 0.003) ** 2)
    H = np.fft.rfft(p)
    s = synth.RotorSound("r", 2, 11.37, H * 0, H, n)
    fs = 48000.0
    sig = synth.synthesize([s], fs, 3.0, synth.Variability.none(), pitch_headroom=1.25)
    x = sig["total"]
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    above = 20 * np.log10(np.max(X[f > 0.5 * fs / 1.25]) / np.max(X) + 1e-300)
    jump = abs(x[0] - x[-1]) / np.max(np.abs(np.diff(x)))
    return ("Loop: alias-free above Nyquist/headroom, seamless", f"{above:.0f} dB above the limit, junction "
            f"{jump:.2f} of the largest step", above < -100 and jump <= 1.0)


def check_t700():
    from . import engine
    e = engine.load_engine("t700")
    # GE's nonlinear simulation (NASA TM-105805 table 1): Ng %, compressor delivery psi, power turbine inlet R
    ge = ((87.5, 132.0, 1472.0), (91.4, 161.7, 1578.0), (94.5, 188.9, 1675.0), (96.7, 212.3, 1778.0),
          (98.5, 231.5, 1896.0), (99.5, 240.5, 1954.0))
    dp = dt = 0.0
    for ng, ps3, t45 in ge:
        st = engine.engine_state(e, ng)
        dp = max(dp, abs(st.pr / (ps3 / 14.696) - 1.0))
        dt = max(dt, abs(st.T45 - t45 / 1.8))
    return ("T700 cycle vs GE's simulation (NASA TM-105805), 87.5-99.5 % Ng",
            f"delivery pressure within {100 * dp:.1f} %, TGT within {dt:.0f} K", dp < 0.02 and dt < 80.0)


def run(out=print):
    checks = [check_bpm, check_gutin, check_absorption, check_hover_momentum, check_modes, check_loop, check_t700]
    ok = True
    for c in checks:
        name, result, passed = c()
        ok &= passed
        out(f"{'PASS' if passed else 'FAIL'}  {name}: {result}")
    out("(the F1A thickness and loading terms are also checked against Formulation 1 in tests/test_f1a.py)")
    return ok


# ---------------------------------------------------------------- benchmark

@njit(parallel=True, fastmath=True, cache=True)
def _retarded_time_f1a(radii, f_t, f_z, omega, B, obs, c, n_out, period):
    """The classical retarded-time algorithm (as the original WOPWOP): for every observer time, blade and element,
    Newton iteration for the emission time, then the compact loading integrand. Hover, steady loads."""
    out = np.zeros(n_out)
    d0 = np.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2)
    for i in prange(n_out):
        t = i * period / n_out
        acc = 0.0
        for b in range(B):
            ph = 2.0 * np.pi * b / B
            for e in range(radii.shape[0]):
                Re = radii[e]
                tau = t - d0 / c
                for it in range(20):
                    ps = omega * tau + ph
                    yx, yy = Re * np.cos(ps), Re * np.sin(ps)
                    dx, dy, dz = obs[0] - yx, obs[1] - yy, obs[2]
                    r = np.sqrt(dx * dx + dy * dy + dz * dz)
                    g = t - tau - r / c
                    dg = -1.0 + (dx * (-omega * yy) + dy * (omega * yx)) / (r * c)
                    step = g / dg
                    tau -= step
                    if abs(step) < 1e-12:
                        break
                ps = omega * tau + ph
                cs, sn = np.cos(ps), np.sin(ps)
                yx, yy = Re * cs, Re * sn
                dx, dy, dz = obs[0] - yx, obs[1] - yy, obs[2]
                r = np.sqrt(dx * dx + dy * dy + dz * dz)
                hx, hy, hz = dx / r, dy / r, dz / r
                vx, vy = -omega * yy, omega * yx
                ax, ay = -omega * omega * yx, -omega * omega * yy
                Mr = (vx * hx + vy * hy) / c
                dMr = (ax * hx + ay * hy) / c
                M2 = (vx * vx + vy * vy) / (c * c)
                ft, fz = f_t[e], f_z[e]
                Fx, Fy, Fz = sn * ft, -cs * ft, -fz
                dFx, dFy = omega * cs * ft, omega * sn * ft
                Fr = Fx * hx + Fy * hy + Fz * hz
                dFr = dFx * hx + dFy * hy
                FM = (Fx * vx + Fy * vy) / c
                om = 1.0 / (1.0 - Mr)
                acc += (dFr / (c * r) * om * om + (Fr - FM) / (r * r) * om * om
                        + Fr * (r * dMr + c * (Mr - M2)) / (c * r * r) * om * om * om) / (4.0 * np.pi)
        out[i] = acc
    return out


def benchmark(out=print):
    from .model import load_aircraft, FlightCondition, STANDARD_OBSERVERS
    from .operating import operate
    from . import broadband
    # 1. retarded-time vs source-time-dominant on the same compact rotor (40 stations, 4 blades, hover)
    B, R, om, n, ne = 4, 8.0, 27.0, 32768, 40
    x = np.linspace(0.2, 1.0, ne)
    ft = -40.0 * x
    fz = 900.0 * x
    src = _point_rotor(B, 1.0, om, n, 0.0, 0.0)
    src.lp_pos = np.column_stack([x * R, np.zeros(ne), np.zeros(ne)])
    src.lp_station = np.arange(ne)
    src.lp_weight = np.ones(ne)
    src.f_tab = np.repeat(np.stack([ft, fz], -1)[:, None, :], n, axis=1)
    src.df_tab = np.zeros_like(src.f_tab)
    src.z_tab = src.dz_tab = src.ddz_tab = np.zeros((ne, n))
    obs = np.array([60.0, 20.0, -50.0])
    period = 2 * np.pi / om
    _retarded_time_f1a(x * R, ft, fz, om, B, obs, 340.0, 64, period)
    f1a.periodic_signature(src, obs, 64)
    t0 = time.perf_counter()
    ref = _retarded_time_f1a(x * R, ft, fz, om, B, obs, 340.0, n, period)
    t_ret = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, _, pl = f1a.periodic_signature(src, obs, n)
    tot = sum(np.roll(pl, k * n // B) for k in range(B))
    t_src = time.perf_counter() - t0
    err = np.sqrt(np.mean((tot - ref) ** 2)) / np.sqrt(np.mean(ref ** 2))
    out(f"compact loading, {ne} stations x {B} blades, one observer, {n} samples per revolution:")
    out(f"  retarded-time algorithm (Newton per observer sample, every blade): {1e3 * t_ret:8.1f} ms")
    out(f"  source-time-dominant, closed form, blade symmetry:                {1e3 * t_src:8.1f} ms "
        f"({t_ret / t_src:.0f}x faster), difference {100 * err:.4f} %")
    # 2. a full UH-60A condition
    ac = load_aircraft("uh60a")
    cond = FlightCondition(airspeed_kt=65, climb_fpm=-800)
    operate(ac, cond)
    t0 = time.perf_counter()
    st = operate(ac, cond)
    t_trim = time.perf_counter() - t0
    pt = st.points[0]
    s = f1a.build_sources(pt)
    o = pt.to_rotor(STANDARD_OBSERVERS["front_right"].offset())
    t0 = time.perf_counter()
    f1a.periodic_signature(s, o)
    t_obs = time.perf_counter() - t0
    evals = len(s.lp_pos) * s.n_psi + len(s.tp_pos) * s.n_psi // max(1, s.n_psi // 8192)
    t0 = time.perf_counter()
    broadband.spectral_history(pt, o)
    t_bb = time.perf_counter() - t0
    t0 = time.perf_counter()
    synth.render(ac, cond, STANDARD_OBSERVERS["front_right"], state=st, duration=8.0)
    t_render = time.perf_counter() - t0
    out(f"UH-60A, 65 kt, 800 ft/min descent (elastic blades, prescribed wake, BVI):")
    out(f"  trim, elastic response and wake:  {t_trim:6.2f} s")
    out(f"  F1A at one observer:              {t_obs:6.3f} s  ({len(s.lp_pos)} load points and {len(s.tp_pos)} "
        f"panels, {evals / t_obs / 1e6:.0f} M source evaluations per second)")
    out(f"  broadband spectral history:       {t_bb:6.3f} s")
    out(f"  8 s loop, main and tail rotor:    {t_render:6.2f} s")
