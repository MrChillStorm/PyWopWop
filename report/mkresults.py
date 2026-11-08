#!/usr/bin/env python3
"""Run the report's test programme into report/results/*.json (the tables, numbers and figures are made from these by
mktables.py and figures.py).

    python report/mkresults.py [stage ...]          all stages, or the ones named

Stages: verify (the solvers against closed forms and NASA's code), rotors (hover and level-flight power against tests), stand (the XV-15
rotor's noise on NASA's test stand), hart2 (the
HART II rotor's loads and noise), bvi (slap's level against the descent, on HART II and the UH-60A), texture (loops against recordings),
fountain (the XV-15 over ground boards), outside (the MV-22 against the published table), engines (cycles, levels), tones (the
engines' and gearbox's tones in UH-60 recordings), cabin (the walls), wind, mv22 (the cockpit against its recording), mv22_terms (the
same, each physical term taken out), ask21 (the gliders' recordings), slip (the glider slipping), props (the X-57's propellers), tyres
(tyres rolling, against two landings' recordings: tools/tyre_recordings.py), bench (timings: run on a quiet machine).

texture, tones, mv22, ask21 and tyres read recordings (refs/recordings, kept out of the repository) and the FlightGear aircrafts' sounds; the
results they leave are what the report needs, so it can be rebuilt without them."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter
from scipy.signal import bilinear, butter, hilbert, lfilter, sosfiltfilt, welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import refdata as D                                         # noqa: E402
from pywopwop import bpm, engine as en, f1a, operating, synth, validation      # noqa: E402
from pywopwop.aero import AeroOptions, load_variants, solve_rotor              # noqa: E402
from pywopwop.atmosphere import Atmosphere, iso9613_alpha                      # noqa: E402
from pywopwop.environment import Environment                # noqa: E402
from pywopwop.model import (Airfoil, Cabin, FlightCondition, KT, Observer, Rotor, STANDARD_OBSERVERS,  # noqa: E402
                            load_aircraft)
from pywopwop.operating import RotorPoint, operate          # noqa: E402
from pywopwop.structure import blade_modes                  # noqa: E402
from pywopwop.synth import Quality, Variability, a_weighted_ms, render         # noqa: E402

OUT = os.path.join(HERE, "results")
REC = os.path.join(ROOT, "refs", "recordings")
CACHE = os.path.join(ROOT, "refs", "report_cache")          # git-ignored, as recordings are
FS = 48000
A_RHO, A_C, A_NU = 1.225, 340.3, 1.46e-5


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


TAG = os.environ.get("PWW_RESULTS_TAG", "")


def dump(name, obj):
    with open(os.path.join(OUT, TAG + name + ".json"), "w") as f:
        json.dump(plain(obj), f, indent=0)


def load(name):
    return json.load(open(os.path.join(OUT, name + ".json")))


def read_wav(path):
    fs, x = wavfile.read(path)
    x = x.astype(float) / (32768.0 if x.dtype == np.int16 else (2.0 ** 31 if x.dtype == np.int32 else 1.0))
    return fs, (x.mean(axis=1) if x.ndim > 1 else x)


def dba(y, fs=FS):
    return 10.0 * np.log10(a_weighted_ms(y, fs) / 4e-10)


def thirds(x, fs=FS, nper=8192, lo=20.0, hi=16000.0):
    """third-octave levels (dB, arbitrary reference) and their centres"""
    fc = 1000.0 * 2.0 ** (np.arange(-17, 13) / 3.0)
    fc = fc[(fc >= lo) & (fc <= hi)]
    f, P = welch(x, fs, nperseg=nper)
    return fc, np.array([10 * np.log10(P[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].sum() * (f[1] - f[0]) + 1e-30) for c in fc])


def lufs(x, fs=FS):
    """ITU-R BS.1770-4: K-weighting, 400 ms blocks, the absolute and relative gates"""
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    y = lfilter(b2, a2, lfilter(b1, a1, x))
    n, h = int(0.4 * fs), int(0.1 * fs)
    z = np.array([np.mean(y[i:i + n] ** 2) for i in range(0, len(y) - n, h)])
    z = z[-0.691 + 10 * np.log10(z + 1e-30) > -70.0]
    z = z[-0.691 + 10 * np.log10(z) > -0.691 + 10 * np.log10(z.mean()) - 10.0]
    return -0.691 + 10 * np.log10(z.mean())


# ---------------------------------------------------------------- verify

def _f1_errors():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_f1a as T
    B, Re, omega, n_psi = 1, 1.5, 80.0, 8192
    f_tab, z_tab, area = T._sources_for_f1(n_psi, (0, 0, 0))
    src = T._point_source(B, Re, omega, n_psi, f_tab=f_tab, z_tab=z_tab, area=area)
    obs = np.array([-20.0, 25.0, -10.0])
    t, pt, pl = f1a.periodic_signature(src, obs, 4096)
    _, rT, rL, _ = T._reference_f1(B, Re, omega, n_psi, f_tab, z_tab, area, (0.0, 0.0, 0.0), obs, 4096)
    rel = lambda g, r: float(np.sqrt(np.mean((g - r) ** 2)) / np.sqrt(np.mean(r ** 2)))
    hover = dict(thickness=rel(pt, rT), loading=rel(pl, rL))
    # the rotor flying past a fixed listener, against the retarded integrands differentiated numerically
    n_psi = 4096
    U = np.array([60.0, 0.0, -4.0])
    f_tab, z_tab, area = T._sources_for_f1(n_psi, U)
    src = T._point_source(B, Re, omega, n_psi, f_tab=f_tab, z_tab=z_tab, U=U, area=area)
    obs = np.array([-30.0, 20.0, -15.0])
    t, pt, pl, (n0, ns) = f1a.time_history(src, obs, 0.25, 0.5, 40000.0)
    Tp = 2 * np.pi / omega
    dtau = Tp / n_psi
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
    Mr = np.sum(v * rh, 1) / T.C0
    te = tau + r / T.C0
    psi = omega * tau
    cs, sn = np.cos(psi), np.sin(psi)
    A = np.stack([cs * area[0] - sn * area[1], sn * area[0] + cs * area[1], np.full_like(cs, area[2])], -1)
    ft, fz = f_tab[0, j, 0], f_tab[0, j, 1]
    Fr = np.sum(np.stack([sn * ft, -cs * ft, -fz], -1) * rh, 1)
    gT = np.interp(t, te, T.RHO * np.sum(v * A, 1) / (r * (1 - Mr)))
    g1 = np.interp(t, te, Fr / (T.C0 * r * (1 - Mr)))
    g2 = np.interp(t, te, Fr / (r ** 2 * (1 - Mr)))
    rT = np.gradient(gT, t) / (4 * np.pi)
    rL = (np.gradient(g1, t) + g2) / (4 * np.pi)
    s = slice(200, -200)
    flyby = dict(thickness=rel(pt[s], rT[s]), loading=rel(pl[s], rL[s]))
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        yy, o, vo, ta = rng.normal(size=3) * 5, rng.normal(size=3) * 50, rng.normal(size=3) * 60, rng.uniform(-1, 1)
        tt, *_r = f1a._retarded(*yy, *o, *vo, ta, T.C0)
        worst = max(worst, abs(np.linalg.norm(o + vo * tt - yy) - T.C0 * (tt - ta)) / T.C0)
    return hover, flyby, worst


def verify():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_bpm as TB
    f = bpm.FRCEN_BPM
    a = dict(alpha=1.516, c=0.3048, U=71.3, f=f, L=0.4572, r=1.22, theta=90.0, phi=90.0, trip=0, nu=1.4529e-5, c0=340.46)
    p, s, sep, _ = bpm.tblte(**a)
    lam = bpm.lblvs(**a)
    tot = 10 * np.log10(10 ** (p / 10) + 10 ** (s / 10) + 10 ** (lam / 10) + 10 ** (sep / 10))
    got = np.column_stack([p, s, lam, tot])
    d2 = float(np.max(np.abs(got - validation._D2)))
    nseg = 10
    args = dict(alpha=np.full(nseg, 5.4), c=0.1524, U=71.3, f=TB.F, L=0.0305, r=1.22, theta=90.0, phi=90.0, trip=bpm.TRIP_HEAVY,
                nu=TB.VISC, c0=TB.C0)
    p3, s3, a3, _ = bpm.tblte(**args)
    p3, s3, a3 = (TB._sum_db(*x) for x in (p3, s3, a3))
    tip = bpm.tip(alpha_tip=7.7, c=0.1524, U=71.3, f=TB.F, r=1.22, theta=90.0, phi=90.0, c0=TB.C0, round_tip=True)
    g3 = np.column_stack([p3, s3, a3, tip, TB._sum_db(p3, s3, a3, tip)])
    g3[:, 2] = np.maximum(g3[:, 2], -90.0)
    e3 = np.abs(g3 - TB.D3)
    # Gutin
    from scipy.special import jv
    B, Re, om, n, Tq, Q = 2, 1.0, 120.0, 4096, 400.0, 60.0
    src = validation._point_rotor(B, Re, om, n, Tq, Q)
    gut = []
    for th_deg in (30.0, 60.0, 80.0, 100.0, 150.0):
        th = np.radians(th_deg)
        obs = 200.0 * np.array([np.sin(th), 0.0, np.cos(th)])
        _, H, nn = f1a.single_blade_harmonics(src, obs, n)
        for m in (1, 2, 3):
            g = B * 2 * abs(H[m * B]) / nn / np.sqrt(2)
            ref = (m * B * om / (2 * np.sqrt(2) * np.pi * 340.0 * 200.0) * abs(-Tq * np.cos(th) + Q * 340.0 / om)
                   * abs(jv(m * B, m * B * om / 340.0 * Re * np.sin(th))))
            gut.append(dict(theta=th_deg, m=m, got=20 * np.log10(g / 2e-5), ref=20 * np.log10(ref / 2e-5)))
    hover, flyby, closure = _f1_errors()
    iso = []
    for fq, ref in validation._ISO.items():
        fm = 1000.0 * 10.0 ** (0.3 * np.round(np.log2(fq / 1000.0)))
        iso.append(dict(f=fq, got=round(float(iso9613_alpha(fm, 293.15, 70.0) * 1000.0), 1), ref=ref))
    # a loop
    nn = 4096
    t = np.arange(nn) / nn
    pp = np.exp(-((t - 0.3) / 0.002) ** 2) - 0.5 * np.exp(-((t - 0.31) / 0.003) ** 2)
    H = np.fft.rfft(pp)
    sig = synth.synthesize([synth.RotorSound("r", 2, 11.37, H * 0, H, nn)], 48000.0, 3.0, Variability.none(), pitch_headroom=1.25)
    x = sig["total"]
    X = np.abs(np.fft.rfft(x))
    ff = np.fft.rfftfreq(len(x), 1 / 48000.0)
    above = 20 * np.log10(np.max(X[ff > 0.5 * 48000.0 / 1.25]) / np.max(X) + 1e-300)
    jump = abs(x[0] - x[-1]) / np.max(np.abs(np.diff(x)))
    # blades
    ac = load_aircraft("bo105")
    md = blade_modes(ac.main, np.linspace(0.2, 1.0, 40), 1.225)
    # cycle and hover
    import pywopwop
    pkg = os.path.dirname(os.path.dirname(pywopwop.__file__))        # the tests of the code under test
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"], capture_output=True, text=True,
                         cwd=pkg).stdout.strip().splitlines()[-1]
    dump("verify", dict(bpm_f=f, bpm_d2_table=validation._D2, bpm_d2_port=got, bpm_d2_err=d2, bpm_d3_err=float(np.max(e3[:, [0, 1, 2, 4]])),
                        bpm_d3_tip_err=float(np.max(e3[:, 3])), gutin=gut, f1_hover=hover, f1_flyby=flyby, closure=closure,
                        iso=iso, loop_above_db=above, loop_junction=jump, bo105_flap=md.flap_ratio, pytest=out))


# ---------------------------------------------------------------- rotors against tests

def _power(rot, omega, V, alpha, T, rho=A_RHO, c=A_C, nu=A_NU, **kw):
    return solve_rotor(rot, omega, V, alpha, T, rho, c, nu, AeroOptions(**kw))


def _torsion_moment(a, x0):
    """the blade's steady torsion moment at r/R = x0 (N m): the aerodynamic moment about the elastic axis, the centrifugal force on the
    section's offset centre of mass through the blade's slope, and the propeller moment, outboard of it"""
    bm = a.model
    r = a.rotor
    ld = bm.loads(a.theta0, a.theta1c, a.theta1s, a.lam_i)
    sel = bm.x >= x0
    d = (r.cg_axis - r.elastic_axis) * bm.chord
    cf = bm.modes.m * d * a.omega ** 2 * bm.r * np.mean(bm.slope, axis=0)
    th = a.theta0 + r.twist_at(bm.x)
    prop = -bm.modes.m * (r.torsion_gyration * bm.chord) ** 2 * a.omega ** 2 * np.sin(th) * np.cos(th)
    q = 0.5 * a.rho * np.mean(ld["UT"] ** 2 + ld["UP"] ** 2, axis=0)
    c = bm.chord
    aero = q * c * c * (np.mean(r.airfoil.moment(ld["alpha"], ld["mach"]), axis=0) + np.mean(ld["cl"], axis=0) * (r.elastic_axis - 0.25))
    return float(np.sum(((aero + cf + prop) * bm.dr)[sel]))


def rotors():
    # XV-15: the full-scale rotor in the hover
    ac = load_aircraft("xv15")
    r = ac.main
    h = D.XV15_HOVER
    vt = h["mtip"] * A_C
    om = vt / r.radius
    A = np.pi * r.radius ** 2
    xv = []
    for cts in np.arange(0.04, 0.181, 0.01):
        T = cts * h["sigma"] * A_RHO * A * vt ** 2
        a = _power(r, om, 0.0, 0.0, T, bvi=False)
        xv.append(dict(ct_s=cts, cp_s=a.power / (A_RHO * A * vt ** 3 * h["sigma"]), fm=T ** 1.5 / np.sqrt(2 * A_RHO * A) / a.power))
    # Bo 105: the full-scale rotor on NASA's test stand, at the tunnel's speed the rotor drew
    ac = load_aircraft("bo105")
    r = ac.main
    b = D.BO105_HOVER
    om = b["rpm"] * 2 * np.pi / 60
    vt = om * r.radius
    A = np.pi * r.radius ** 2
    bo = []
    for th, cts, cps, kt in b["points"]:
        if cts < 0.01:
            continue
        T = cts * b["sigma"] * b["rho"] * A * vt ** 2
        a = _power(r, om, kt * KT, np.radians(-b["shaft_deg"]) if kt > 0 else 0.0, T, rho=b["rho"], bvi=False)
        bo.append(dict(ct_s=cts, cp_s_test=cps, theta_test=th, cp_s=a.power / (b["rho"] * A * vt ** 3 * b["sigma"]),
                       theta=float(np.degrees(a.theta0) + 0.4),               # 0.75 R to 0.7 R: the -8 deg twist
                       twist_tip=float(np.degrees(np.mean(a.theta_e[:, -1]))), torsion_034=_torsion_moment(a, 0.34) * 8.8507))
    fw = D.BO105_FORWARD
    T = fw["ct_s"] * fw["sigma"] * fw["rho"] * A * vt ** 2
    bof = []
    for mu, cps in fw["points"]:
        a = _power(r, om, mu * vt, np.radians(-fw["shaft_deg"]), T, rho=fw["rho"], n_azimuth=360)
        bof.append(dict(mu=mu, cp_s_test=cps, cp_s=a.power / (fw["rho"] * A * vt ** 3 * fw["sigma"])))
    # UH-60A: both engines' power, in the hover and level flight, against both rotors' (the rest: transmission, accessories)
    ac = load_aircraft("uh60a")
    m = ac.main
    om = m.omega
    vt = om * m.radius
    A = np.pi * m.radius ** 2
    uh = []
    for ct in np.arange(0.0055, 0.0101, 0.0005):
        W = ct * A_RHO * A * vt ** 2
        st = operate(ac, FlightCondition(mass_kg=W / 9.80665), AeroOptions(bvi=False))
        P = sum(p.aero.power for p in st.points)
        uh.append(dict(ct=ct, cp=P / (A_RHO * A * vt ** 3), cp_main=st.points[0].aero.power / (A_RHO * A * vt ** 3),
                       cp_test=D.uh60_hover_cp(ct)))
    lv = {}
    for cte4 in (65, 80):
        W = cte4 * 1e-4 * A_RHO * A * vt ** 2
        rows = []
        for mu, d in D.UH60_LEVEL.items():
            st = operate(ac, FlightCondition(airspeed_kt=mu * vt / KT, mass_kg=W / 9.80665), AeroOptions(n_azimuth=360))
            P = sum(p.aero.power for p in st.points)
            rows.append(dict(mu=mu, cp=1e5 * P / (A_RHO * A * vt ** 3), cp_test=d[cte4]))
        lv[cte4] = rows
    dump("rotors", dict(xv15=xv, xv15_test=dict(ct_s=D.XV15_HOVER["ct_s"], cp_s=D.XV15_HOVER["cp_s"]), bo105=bo, bo105_forward=bof, uh60_hover=uh, uh60_level=lv))


def stand():
    """the full-scale XV-15 rotor at the microphones of NASA's outdoor test stand (CR-177436 figs. 9.1, 9.2): overall level against thrust"""
    from pywopwop import broadband
    from pywopwop.synth import rotor_sound
    ex = json.load(open(os.path.join(HERE, "data", "xv15_stand_noise.json")))
    ac = load_aircraft("xv15")
    r = ac.main
    vt = 0.69 * A_C
    om = vt / r.radius
    near = np.array([r.radius + 0.61, 0.0, -2.44])
    far = 76.2 * np.array([np.cos(np.radians(15.0)), 0.0, -np.sin(np.radians(15.0))])
    rows = []
    for lb in (200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 10500):
        T = lb * 4.448222
        a = solve_rotor(r, om, 0.0, 0.0, T, A_RHO, A_C, A_NU, AeroOptions())
        pt = RotorPoint(r, om, 0.0, 0.0, T, np.eye(3), np.zeros(3), np.zeros(3), aero=a)
        fz = np.mean(a.fz, axis=0)
        row = dict(lb=lb, ct_s=a.ct_sigma, tip_lift=float(r.blades * np.sum((fz * a.dr)[a.x > 0.8]) / 4.448222))     # the outer fifth's, lb
        for tag, o in (("near", near), ("far", far)):
            s = rotor_sound(pt, o, "normal", broadband.BroadbandOptions(n_azimuth=128), True, {}, Variability(variants=0))
            k = np.arange(1, 160) * r.blades
            lv = lambda H: 10 * np.log10(np.sum((r.blades * 2 * np.abs(H[k]) / s.n) ** 2 / 2) / 4e-10)
            tonal, bb = lv(s.H_thickness + s.H_loading), s.history.oaspl
            row[tag] = dict(tonal=tonal, thickness=lv(s.H_thickness), loading=lv(s.H_loading), broadband=bb, total=10 * np.log10(10 ** (tonal / 10) + 10 ** (bb / 10)))
        rows.append(row)
        print(f"stand {lb} lb: near {row['near']['total']:.1f} far {row['far']['total']:.1f}", flush=True)
    dump("stand", dict(rows=rows, test=ex))


# ---------------------------------------------------------------- HART II

def hart2_rotor():
    h = D.HART2
    return Rotor(name="hart2", radius=h["radius"], blades=h["blades"], rpm=h["rpm"], chord=h["chord"], root_cutout=h["root_cutout"], twist=h["twist"], hub="hingeless", precone_deg=h["precone_deg"],
                 flap_frequency=h["flap_frequency"], lock_number=5.0, torsion_frequency=h["torsion_frequency"], elastic_axis=0.25,
                 cg_axis=0.25, clamp_radius=0.1, torsion_damping=0.2,
                 airfoil=Airfoil(name="NACA 23012", table="naca23012_c81", shape="naca23012", cl_max=1.5))


def hart2_solve(alpha_deg, core=None, rollup=None):
    h = D.HART2
    o = AeroOptions()
    o.core_radius_chords = core if core is not None else o.core_radius_chords
    o.rollup = rollup if rollup is not None else o.rollup
    r = hart2_rotor()
    om = h["rpm"] * 2 * np.pi / 60
    return solve_rotor(r, om, h["speed_ms"], np.radians(alpha_deg), h["thrust_n"], A_RHO, 340.0, A_NU, o)


def hart2_loads(a):
    i = int(np.argmin(np.abs(a.x - 0.87)))
    q = 0.5 * a.rho * a.c ** 2 * a.chord[i]
    return a.fz[:, i] / q, (a.fz_base[:, i] / q if a.fz_base is not None else a.fz[:, i] / q)


def bvi_part(y):
    """what lies above 10/rev"""
    F = np.fft.rfft(y)
    F[10:] = 0
    return y - np.fft.irfft(F, len(y))


def load_stats(psi_deg, y):
    hp = bvi_part(np.asarray(y, float))
    d = np.asarray(psi_deg)
    adv, ret = (d >= 20) & (d <= 90), (d >= 270) & (d <= 350)
    return dict(mean=float(np.mean(y)), adv=float(np.ptp(hp[adv])), ret=float(np.ptp(hp[ret])), rms=float(np.std(hp)))


def hart2_map(a, alpha_deg, ys, mxs, harmonics=(6, 41)):
    h = D.HART2
    al = np.radians(alpha_deg)
    V = h["speed_ms"]
    U = V * np.array([np.cos(al), 0.0, np.sin(al)])
    pt = RotorPoint(a.rotor, a.omega, V, al, h["thrust_n"], np.eye(3), np.zeros(3), U, aero=a)
    src = f1a.build_sources(pt, chord_points=4, surface_chord=24, fs_target=96000.0)
    srcb = f1a.build_sources(pt, n_psi=src.n_psi, chord_points=4, thickness=False, loads="base") if a.fz_base is not None else None
    ex, ez, ey = np.array([np.cos(al), 0, np.sin(al)]), np.array([-np.sin(al), 0, np.cos(al)]), np.array([0.0, 1.0, 0.0])
    k = np.arange(*harmonics) * h["blades"]
    L = np.zeros((len(mxs), len(ys)))
    Lb = np.zeros_like(L)
    for i, m in enumerate(mxs):
        for j, y in enumerate(ys):
            o = m * ex + y * ey - h["mic_below_hub_m"] * ez          # the map's "-x" axis runs downstream (refdata.py)
            Ht, Hl, n = f1a.single_blade_harmonics(src, o)
            L[i, j] = 10 * np.log10(np.sum((h["blades"] * 2 * np.abs(Ht[k] + Hl[k]) / n) ** 2 / 2) / 4e-10)
            if srcb is not None:
                _, Hb, _ = f1a.single_blade_harmonics(srcb, o, n)
                Lb[i, j] = 10 * np.log10(np.sum((h["blades"] * 2 * np.abs(Ht[k] + Hb[k]) / n) ** 2 / 2) / 4e-10)
    return L, Lb


def hart2():
    ex = json.load(open(os.path.join(HERE, "data", "hart2_bl_noise.json")))
    ld = json.load(open(os.path.join(HERE, "data", "hart2_bl_loads.json")))
    ys, mxs, G = np.array(ex["y"]), np.array(ex["minus_x"]), np.array(ex["level"], float)
    # the shipped constants, the effective core's calibration (a sweep at the test's shaft angle), the shaft angle less an open jet's
    # usual correction (the paper gives none), and the measured core with all the circulation rolled up
    cases = [(5.3, None, None), (4.5, None, None), (5.3, 0.12, 1.0)] + [(5.3, c, 0.7) for c in (0.4, 0.6, 0.7, 0.9, 1.0, 1.2)]
    out = dict(y=ys, minus_x=mxs, measured=G, loads_psi=ld["psi"], loads_measured=ld["m2cn"], measured_stats=load_stats(ld["psi"], ld["m2cn"]), cases=[])
    for alpha, core, roll in cases:
        a = hart2_solve(alpha, core, roll)
        y, yb = hart2_loads(a)
        psi = np.degrees(a.psi)
        L, Lb = hart2_map(a, alpha, ys, mxs)
        d = L - G
        pos, neg = ys > 0, ys < 0
        i, j = np.unravel_index(np.argmax(np.where(pos[None, :], L, -1)), L.shape)
        i2, j2 = np.unravel_index(np.argmax(np.where(neg[None, :], L, -1)), L.shape)
        c = dict(alpha=alpha, core=core if core is not None else AeroOptions().core_radius_chords,
                 rollup=roll if roll is not None else AeroOptions().rollup, power_kw=a.power / 1e3, theta75=float(np.degrees(a.theta0)),
                 mu=a.mu_x, mu_z=a.mu_z, lambda0=a.lambda0, tip_twist=np.degrees(a.theta_e[::2, -1]), tip_flap=(a.z[::2, -1] - a.r[-1] * np.sin(np.radians(2.5))), stats=load_stats(psi, y), psi=psi[::2], m2cn=y[::2], base=yb[::2], L=L, Lb=Lb, max_adv=L[:, pos].max(),
                 adv_at=[mxs[i], ys[j]], max_ret=L[:, neg].max(), ret_at=[mxs[i2], ys[j2]], no_bvi_max=Lb.max(), mean_err=float(np.mean(d)),
                 rms_err=float(np.sqrt(np.mean((d - d.mean()) ** 2))), rms_abs=float(np.sqrt(np.mean(d ** 2))))
        out["cases"].append(c)
        print(f"hart2 alpha {alpha} core {c['core']}: adv {c['max_adv']:.1f} at {c['adv_at']} ret {c['max_ret']:.1f} at {c['ret_at']} no BVI {c['no_bvi_max']:.1f} mean {c['mean_err']:+.1f} rms {c['rms_err']:.1f} loads {c['stats']}", flush=True)
        dump("hart2", out)


# ---------------------------------------------------------------- slap against the descent

def _bvi_alpha(alpha):
    ys = np.arange(-2.7, 2.71, 0.9)
    mxs = np.arange(-3.0, 3.01, 1.0)
    a = hart2_solve(alpha)
    L, Lb = hart2_map(a, alpha, ys, mxs)
    return dict(alpha=alpha, mu_z=a.mu_z, lambda0=a.lambda0, peak=L.max(), mean=10 * np.log10(np.mean(10 ** (L / 10))), peak_no_bvi=Lb.max(),
                rms=load_stats(np.degrees(a.psi), hart2_loads(a)[0])["rms"])


def _bvi_descent(a):
    name, kt, fpm = a
    ac = load_aircraft(name)
    q = Quality.get("normal")
    fc = FlightCondition(airspeed_kt=kt, climb_fpm=fpm)
    st = operate(ac, fc, AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
    r = render(ac, fc, STANDARD_OBSERVERS["front_right"], duration=4.0, quality=q, state=st, variability=Variability(seed=1))
    lv = r.levels()
    p = st.points[0].aero
    return dict(aircraft=name, kt=kt, fpm=fpm, mu=p.mu_x, lam=p.lambda0 - p.mu_z, total=lv["total"]["dba"], slap=lv.get("layer/slap", dict(dba=-99))["dba"],
                rotor=lv["layer/rotor"]["dba"], oaspl=lv["total"]["oaspl"], tpp=float(np.degrees(p.mu_z / max(p.mu_x, 1e-6))))


def bvi():
    alpha = [_bvi_alpha(a) for a in (0.0, 1.5, 3.0, 4.0, 5.3, 6.5, 8.0, 10.0)]
    jobs = [("uh60a", 65.0, c) for c in np.arange(-1600.0, 801.0, 400.0)] + [("uh1h", 70.0, c) for c in np.arange(-1600.0, 801.0, 400.0)]
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(_bvi_descent, jobs))
    dump("bvi", dict(alpha=alpha, descent=rows))


# ---------------------------------------------------------------- texture

def features(x, fs, bpf):
    """what told the first loops from recordings: how alike consecutive blade passes are, harmonics against noise from 100 Hz to 3 kHz,
    crest factor"""
    from scipy import signal
    x = x / np.max(np.abs(x))
    x = x[:int(min(len(x), 12 * fs))]
    y = sosfiltfilt(butter(4, [40, min(8000, fs / 2.2)], "bandpass", fs=fs, output="sos"), x)
    n = int(round(fs / bpf))
    m = len(y) // n - 1
    seg = [y[i * n:(i + 1) * n] for i in range(m)]
    best = []
    for i in range(m - 1):
        a = seg[i]
        b = y[max((i + 1) * n - 60, 0):(i + 2) * n + 60]
        c = signal.correlate(b, a, "valid") / np.sqrt(np.sum(a * a) * np.convolve(b * b, np.ones(n), "valid") + 1e-30)
        best.append(c.max())
    f, P = welch(y, fs, nperseg=int(fs * 2))
    k = (f > 100) & (f < 3000)
    h = np.zeros_like(f, bool)
    for mm in range(1, int(3000 / bpf) + 1):
        h |= np.abs(f - mm * bpf) < 1.0
    return dict(corr=float(np.median(best)), tones_db=float(10 * np.log10(np.sum(P[k & h]) / np.sum(P[k & ~h]))),
                crest_db=float(20 * np.log10(np.max(np.abs(y)) / np.sqrt(np.mean(y ** 2)))))


def _texture_model(a):
    name, kt, fpm, obs, tag = a
    ac = load_aircraft(name)
    q = Quality.get("normal")
    fc = FlightCondition(airspeed_kt=kt, climb_fpm=fpm)
    st = operate(ac, fc, AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
    r = render(ac, fc, STANDARD_OBSERVERS[obs], duration=15.0, quality=q, state=st, variability=Variability(seed=1))
    per = render(ac, fc, STANDARD_OBSERVERS[obs], duration=4.0, quality=q, state=st, variability=Variability.none(), with_broadband=False)
    bpf = ac.main.blades * st.points[0].aero.omega / (2 * np.pi)
    lv = r.levels()
    n = int(0.5 * FS)
    return dict(aircraft=name, tag=tag, kt=kt, fpm=fpm, obs=obs, varied=features(r.total, FS, bpf), steady=features(per.total, FS, bpf),
                dba=lv["total"]["dba"], slap_dba=lv.get("layer/slap", dict(dba=-99))["dba"], wave=r.total[:n:4], fs=FS / 4)


def texture():
    rec = []
    for path, bpf, tag in (("UH-60/UH-60A/Sounds/uh60_idle_outside_loop.wav", 17.14, "UH-60, outside, flat pitch"),
                           ("UH-60/UH-60A/Sounds/uh60_idle_outside_vortexinteraction_loop.wav", 17.14, "UH-60, outside, blade slap"),
                           ("UH-60/UH-60A/Sounds/blade_vortex.wav", 17.14, "UH-60, blade slap"),
                           ("Bo105/MBB_Bo_105/Sounds/rotor.wav", 4 * 424 / 60 * 0.959, "Bo 105, rotor"),
                           ("Bo105/MBB_Bo_105/Sounds/blade_vortex.wav", 4 * 424 / 60 * 0.959, "Bo 105, blade slap")):
        p = os.path.join(os.path.dirname(ROOT), path)
        if not os.path.exists(p):
            continue
        fs, x = read_wav(p)
        rec.append(dict(tag=tag, **features(x, fs, bpf)))
    jobs = [("uh60a", 0.0, 0.0, "side", "hover"), ("uh60a", 120.0, 0.0, "side", "cruise"), ("uh60a", 65.0, -800.0, "front_right", "approach"),
            ("bo105", 120.0, 0.0, "side", "cruise"), ("bo105", 65.0, -800.0, "front_right", "approach"),
            ("uh60a", 120.0, 0.0, "front_right", "cruise_ahead")]       # the approach's listener, for like with like
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(_texture_model, jobs))
    dump("texture", dict(recordings=rec, model=rows))


# ---------------------------------------------------------------- the XV-15 over ground boards

_FOUNTAIN = operating._fountain


def _xv15(a):
    az, el, on = a
    operating._fountain = _FOUNTAIN if on else (lambda *x: None)
    ac = load_aircraft("xv15")
    q = Quality.get("normal")
    fc = FlightCondition(airspeed_kt=0.0)
    st = operate(ac, fc, AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
    r = render(ac, fc, Observer("m", az, el, D.XV15_BOARD["radii"] * ac.main.radius), duration=4.0, quality=q, state=st,
               variability=Variability(seed=1), environment="free", f_highpass=10.0)
    y = 2.0 * r.total                                     # a ground board doubles the free field
    bpf = ac.main.blades * ac.main.rpm / 60.0
    T = FS / bpf
    k = int(len(y) / T) - 1
    idx = (np.arange(k)[:, None] * T + np.arange(int(T))[None, :]).astype(int)
    m = y[idx].mean(axis=0)
    hi = sosfiltfilt(butter(4, 150.0, "hp", fs=FS, output="sos"), y)
    lv = r.levels()["total"]
    return dict(az=az, el=el, on=on, mean_hi=m.max(), mean_lo=m.min(), hi_p=np.percentile(hi, 99.9), hi_n=np.percentile(hi, 0.1), whole_hi=y.max(),
                whole_lo=y.min(), dba=lv["dba"], oaspl=lv["oaspl"], wave=y[int(FS):int(FS) + int(0.25 * FS):4])


def fountain():
    jobs = [(az, el, on) for az in (180, 0) for el in (-12.6, -23.0, -45.7) for on in (True, False)]
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(_xv15, jobs))
    dump("fountain", dict(rows=rows, fs=FS / 4))


# ---------------------------------------------------------------- the MV-22 outside

def a_filter():
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    num = [(2 * np.pi * f4) ** 2 * 10 ** (2.0 / 20), 0, 0, 0, 0]
    den = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    den = np.polymul(np.polymul(den, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return bilinear(num, den, FS)


def lmax(y, tau):
    b, a = a_filter()
    e = lfilter(b, a, np.r_[y, y]) ** 2
    k = np.exp(-1.0 / (tau * FS))
    return 10 * np.log10(lfilter([1 - k], [1, -k], e)[len(y):].max() / 4e-10)


def _mv22_out(a):
    az, slant, el_deg, on, low = a
    operating._fountain = _FOUNTAIN if on else (lambda *x: None)
    ac = load_aircraft("mv22b")
    q = Quality.get("normal")
    if low:
        H = 30.0
        D_ = float(np.sqrt(max(slant ** 2 - (H - 1.7) ** 2, 1.0)))
    else:
        D_ = slant * np.cos(np.radians(el_deg))
        H = 1.7 + slant * np.sin(np.radians(el_deg))
    fc = FlightCondition(airspeed_kt=0.0, ground_height_m=H if H < 3 * ac.main.radius else None)
    st = operate(ac, fc, AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
    el = -np.degrees(np.arctan2(H - 1.7, D_))
    o = Observer(f"a{az}", az, el, float(np.hypot(D_, H - 1.7)))
    r = render(ac, fc, o, duration=15.0, quality=q, state=st, variability=Variability(seed=1), environment=Environment("open", flow_resistivity=2e7))
    lv = r.levels()
    return dict(az=az, slant=slant, el=el, on=on, low=low, H=H, leq=lv["total"]["dba"], slow=lmax(r.total, 1.0), fast=lmax(r.total, 0.125),
                rotor=lv["layer/rotor"]["dba"], swish=lv["layer/swish"]["dba"])


def _mv22_below(wing):
    ac = load_aircraft("mv22b")
    if not wing:
        ac.wing = None
    q = Quality.get("normal")
    fc = FlightCondition(airspeed_kt=220.0, nacelle_deg=0.0)
    r = render(ac, fc, Observer("below", 0.0, -89.0, 100.0), duration=6.0, quality=q, variability=Variability(seed=1),
               environment=Environment("open", flow_resistivity=2e7))
    return dict(wing=wing, dba=r.levels()["total"]["dba"], fast=lmax(r.total, 0.125))


def outside():
    jobs = [(az, s, 30.0, True, False) for s in (50.0, 100.0, 200.0, 500.0) for az in (180, 90)]
    jobs += [(180, s, 30.0, False, False) for s in (50.0, 100.0, 200.0, 500.0)]
    jobs += [(180, s, 0.0, True, True) for s in (50.0, 100.0, 200.0, 500.0)]
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(_mv22_out, jobs))
        below = list(ex.map(_mv22_below, (True, False)))
    dump("outside", dict(hover=rows, below=below))


# ---------------------------------------------------------------- engines

def engines():
    e = en.load_engine("t700")
    cyc = []
    for ng, ps3, t45 in D.T700_GE:
        s = en.engine_state(e, ng)
        cyc.append(dict(ng=ng, pr=s.pr, pr_test=ps3 / 14.696, t45=s.T45, t45_test=t45 / 1.8, W=s.W, m_rel=s.m_rel))
    a = en.load_engine("ae1107c")
    s = en.engine_state(a, a.design_ng)
    ae = dict(ng=a.design_ng, pr=s.pr, W=s.W, t4_c=s.T4 - 273.15, power_shp=s.power_w / 745.7)
    op = []
    for ng in (20, 30, 45, 65, 80, 87.5, 92, 96, 100, 104):
        st = en.engine_state(e, ng)
        op.append(dict(ng=ng, pr=st.pr, W=st.W, t4=st.T4, t45=st.T45, m_rel=st.m_rel, fuel=st.fuel_pph, power_shp=st.power_w / 745.7))
    # levels, one engine, on the ground 50 m to the right
    inst = next(i for i in en.load_installations("uh60a") if i.name == "right")      # the near engine, its exhaust turned to the listener
    o = Observer("side", 90.0, 0.0, 50.0)
    lv = []
    for ng, tag in ((67.5, "idle"), (87.5, "FLY"), (100.0, "full power")):
        r = en.render_engine(inst, en.EngineCondition(ng), o, duration=5.0)
        L = r.levels()
        lv.append(dict(tag=tag, ng=ng, total=L["total"]["dba"], **{k.split("/")[1]: v["dba"] for k, v in L.items() if k.startswith("layer/")}))
    apu = en.load_apu("uh60a")
    ra = en.render_apu(apu, 100.0, o, duration=5.0).levels()["total"]["dba"]
    # MV-22B
    mv = []
    instm = en.load_installations("mv22b")[0]
    for ng in (65.0, 90.5, 104.0):
        L = en.render_engine(instm, en.EngineCondition(ng), o, duration=5.0, hub_height=5.2).levels()
        mv.append(dict(ng=ng, total=L["total"]["dba"]))
    # the whine's frequencies by blade count
    tr = en.load_transmission("uh60a")
    ac = load_aircraft("uh60a")
    wh = [dict(ng=ng, f1=e.compressor["rotor_blades"] * e.ng_rpm * ng / 100 / 60, f2=e.compressor["stage2"]["rotor_blades"] * e.ng_rpm * ng / 100 / 60)
          for ng in (67.5, 87.0, 94.0)]
    fa = a.compressor["rotor_blades"] * a.ng_rpm / 100 / 60
    # the gearbox's growth from flat pitch to the hover: its amplitude goes with the torque
    hover = operate(ac, FlightCondition(), AeroOptions(bvi=False)).points[0].aero.torque
    flat = operate(ac, FlightCondition(collective_deg=ac.main.flat_pitch_deg, ground_height_m=0.0), AeroOptions(bvi=False)).points[0].aero.torque
    dump("engines", dict(t700=cyc, op=op, ae1107c=ae, ae1107c_test=D.AE1107C, levels=lv, apu_dba=ra, mv22=mv, whine=wh, gear_hz=tr.tones[0][0], gear_rise_model=20 * np.log10(hover / flat), ng_rpm=e.ng_rpm,
                         ae_rpm=a.ng_rpm, ae_blades=a.compressor["rotor_blades"], ae_f_per_pct=fa))


def _line(f, P, lo, hi, side=250.0):
    """the strongest line in a band, its level, and how far it stands over the median floor around it"""
    k = (f >= lo) & (f <= hi)
    i = np.argmax(np.where(k, P, 0.0))
    fl = np.median(P[(f > f[i] - side) & (f < f[i] + side)])
    return float(f[i]), float(10 * np.log10(P[i] + 1e-30)), float(10 * np.log10(P[i] / fl))


def tones():
    fs, x = read_wav(os.path.join(REC, "uh60_start_short", "audio.wav"))
    f, P = welch(x[44 * fs:56 * fs], fs, nperseg=65536)
    a, _, pa = _line(f, P, 9900, 10300)
    b, _, pb = _line(f, P, 10900, 11300)
    out = dict(idle=dict(f1=a, f2=b, ratio=b / a, over1=pa, over2=pb))
    fs, x = read_wav(os.path.join(REC, "uh60_start_long", "mono.wav"))
    seg = lambda t0, t1: welch(x[int(t0 * fs):int(t1 * fs)], fs, nperseg=65536)
    # the whine's wander: the two engines' lines, 8 s at a time, either side of the moment an earlier check took for the doors
    track = []
    for t in range(372, 460, 8):
        f, P = seg(t, t + 8)
        k = (f > 9900) & (f < 10400)
        L = 10 * np.log10(P[k] + 1e-30)
        from scipy.signal import find_peaks
        pk, _ = find_peaks(L, prominence=5, distance=10)
        top = sorted(sorted(pk, key=lambda i: -L[i])[:2])
        track.append(dict(t=t, lines=[(float(f[k][i]), float(L[i])) for i in top], band=float(10 * np.log10(P[k].sum() * (f[1] - f[0])))))
    f, Pa = seg(400, 417.5)
    f, Pb = seg(425, 445)
    band = lambda P, lo, hi: float(10 * np.log10(P[(f > lo) & (f < hi)].sum() * (f[1] - f[0])))

    def narrow(P, fc=10060.0, w=15.0):
        sel = (f > fc - w) & (f < fc + w)
        base = np.median(P[(f > fc - 250) & (f < fc + 250)])
        return float(10 * np.log10(max(np.sum(P[sel] - base) * (f[1] - f[0]), 1e-20)))
    out["doors"] = dict(track=track, narrow_before=narrow(Pa), narrow_after=narrow(Pb), band_before=band(Pa, 9900, 10400), band_after=band(Pb, 9900, 10400),
                        low_before=band(Pa, 50, 400), low_after=band(Pb, 50, 400))
    # FLY on the ground and the hover
    f, P = seg(505, 570)
    out["fly"] = _line(f, P, 12700, 13100)
    gf = _line(f, P, 975, 1010, side=60.0)
    f, P = seg(620, 700)
    out["hover"] = _line(f, P, 13900, 14200)
    gh = _line(f, P, 975, 1010, side=60.0)
    out["gear"] = dict(fly=gf, hover=gh)
    dump("tones", out)


# ---------------------------------------------------------------- cabin and wind

def cabin():
    f = 1000.0 * 2.0 ** (np.arange(-18, 13) / 3.0)
    out = dict(f=f)
    for name in ("uh60a", "mv22b"):
        c = load_aircraft(name).cabin
        d = dict(opaque=c.opaque_tl(f), window=Cabin.panel_tl(f, c.window_mass, c.window_coincidence_hz, c.window_loss_factor),
                 tau=-10 * np.log10(c.tau(f)), alpha=c.alpha(f), room=10 * np.log10(c.room(f)), inside=10 * np.log10(c.inside(f)))
        if c.openings:
            d["inside_open"] = 10 * np.log10(c.inside(f, True))
            d["door_db"] = float(10 * np.log10(c.inside(np.array([10000.0]), True)[0] / c.inside(np.array([10000.0]))[0]))
        if c.trim_mass:
            m1, m2, g = c.surface_mass, c.trim_mass, c.trim_gap
            d["f0"] = float(np.sqrt(1.2 * 340.0 ** 2 * (m1 + m2) / (g * m1 * m2)) / (2 * np.pi))
        out[name] = d
    dump("cabin", out)


def wind():
    from pywopwop import wind as w
    ac = load_aircraft("mv22b")
    atm0 = Atmosphere(0.0)
    fq = 1000.0 * 2.0 ** (np.arange(-12, 13) / 3.0)
    rows = []
    for kt in (60.0, 120.0, 240.0, 350.0):
        U = kt * KT
        q = 0.5 * atm0.rho * U ** 2
        delta, tau, re_tau, rt = w.boundary_layer(U, 2.1, atm0)
        p2 = tau ** 2 * (6.5 + 1.86 * np.log(max(re_tau, 333.0) / 333.0))
        rows.append(dict(kt=kt, p_rms_q=float(np.sqrt(p2) / q), delta=delta, re_tau=re_tau))
    alt = []
    for h in (3000.0, 6000.0, 9000.0, 12000.0, 16000.0, 21000.0):
        atm = Atmosphere(h)
        V = 220.0
        veq, amp = w.equivalent(V, atm)
        a = w.wall_pressure_psd(fq, V * KT, 2.1, atm)
        b = w.wall_pressure_psd(fq, veq * KT, 2.1, atm0) * amp ** 2
        d = np.abs(10 * np.log10(a / b))
        alt.append(dict(h=h, veq=veq, amp=amp, err=float(d[(fq >= 125) & (fq <= 2000)].max()), err_all=float(d.max())))
    t = D.TU144
    atm = Atmosphere(t["alt_m"])
    U = t["mach"] * atm.c
    fq = np.array(t["f"], float)
    _, tau, re_tau, _ = w.boundary_layer(U, t["x_m"], atm)
    p2 = tau ** 2 * (6.5 + 1.86 * np.log(re_tau / 333.0))
    level = 10 * np.log10(w.wall_pressure_psd(fq, U, t["x_m"], atm) / 4e-10)
    dump("wind", dict(rows=rows, altitude=alt, tu144=dict(f=fq, model=level, test=t["level"], p_rms_q=float(np.sqrt(p2) / (0.5 * atm.rho * U ** 2)))))


def octaves(x, fc=D.UH60_OCTAVES):
    f, P = welch(x, FS, nperseg=2 ** 15)
    return np.array([10 * np.log10(P[(f >= c / 2 ** 0.5) & (f < c * 2 ** 0.5)].sum() * (f[1] - f[0]) / 4e-10 + 1e-30) for c in fc])


def _cockpit_point(kt):
    from pywopwop.wind import wind_loop
    ac = load_aircraft("uh60a")
    o = Observer("cockpit", position=tuple(ac.cockpit))
    tr = en.load_transmission("uh60a")
    q = Quality.get("normal")
    ao = AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth)
    p_hover = sum(p.aero.power for p in operate(ac, FlightCondition(), ao).points)
    fc = FlightCondition(airspeed_kt=kt)
    st = operate(ac, fc, ao)
    torque = 75.0 * sum(p.aero.power for p in st.points) / p_hover
    parts = dict(rotors=render(ac, fc, o, duration=6.0, quality=q, state=st, cabin=ac.cabin, variability=Variability(seed=1)).total,
                 drive=en.render_gearbox(tr, o, duration=6.0, torque_pct=torque).total)
    eng = [en.render_engine(i, en.EngineCondition(94.0), o, duration=6.0, cabin=ac.cabin).total for i in en.load_installations("uh60a")]
    n = min(len(v) for v in list(parts.values()) + eng)
    parts["engines"] = sum(e[:n] for e in eng)
    if kt > 0 and (ac.windscreen_m or ac.fuselage):
        parts["wind"] = wind_loop(ac, kt, 0.0, duration=6.0)
    n = min(len(v) for v in parts.values())
    parts = {k: v[:n] for k, v in parts.items()}
    total = sum(parts.values())
    return dict(kt=kt, torque=torque, dba=dba(total), parts={k: dba(v) for k, v in parts.items()}, octaves=octaves(total),
                part_octaves={k: octaves(v) for k, v in parts.items()})


def cockpit():
    """the UH-60A at the pilot's seat: everything, against Mace's octave bands at 120 kt and NASA's levels over the speed range"""
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(_cockpit_point, sorted(D.UH60_NASA_DBA)))
    dump("cockpit", dict(rows=rows, fc=D.UH60_OCTAVES, pilot=D.UH60_PILOT, copilot=D.UH60_COPILOT, nasa=D.UH60_NASA_DBA, army=D.ARMY_COCKPIT_DBA))


def _flight(a):
    """one flight of the FAA's: the largest A-weighted level along the path, the rotors and both engines at each angle under it, for their microphone"""
    kt, fpm, H, test_dba, test_oaspl = a
    ac = load_aircraft("uh60a")
    q = Quality.get("normal")
    fc = FlightCondition(airspeed_kt=kt, climb_fpm=fpm)
    ao = AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth)
    st = operate(ac, fc, ao)
    p_hover = sum(p.aero.power for p in operate(ac, FlightCondition(), ao).points)
    insts = en.load_installations("uh60a")
    env = Environment("ground", listener_height=1.2)
    rows = []
    for az, el in ((0, -25), (0, -35), (0, -50), (0, -70), (0, -89), (180, -70), (180, -50), (180, -35)):
        d = H / np.sin(np.radians(-el))
        r = render(ac, fc, Observer("p", az, el, d), duration=4.0, quality=q, state=st, variability=Variability(seed=1), environment=env)
        lv = r.levels()
        o = Observer("side", az, 0.0, d * np.cos(np.radians(el)))
        e = [en.render_engine(i, en.EngineCondition(94.0), o, duration=3.0, hub_height=H, environment=env).levels()["total"] for i in insts]
        add = lambda x: 10 * np.log10(sum(10 ** (v / 10) for v in x))
        rows.append(dict(az=az, el=el, rotors=lv["total"]["dba"], slap=lv.get("layer/slap", dict(dba=-99))["dba"], engines=add([v["dba"] for v in e]),
                         dba=add([lv["total"]["dba"]] + [v["dba"] for v in e]), oaspl=add([lv["total"]["oaspl"]] + [v["oaspl"] for v in e])))
    b = max(rows, key=lambda x: x["dba"])
    return dict(kt=kt, fpm=fpm, height=H, test_dba=test_dba, test_oaspl=test_oaspl, dba=b["dba"], oaspl=max(x["oaspl"] for x in rows), at=[b["az"], b["el"]], rotors=b["rotors"], engines=b["engines"],
                slap=b["slap"], torque=75.0 * sum(p.aero.power for p in st.points) / p_hover, rows=rows)


def flight():
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(_flight, D.UH60_FAA))
    dump("flight", dict(rows=rows, torque={str(k): v for k, v in D.UH60_FAA_TORQUE.items()}))


def _site(a):
    """the 150 kt flyover at 152 m from a microphone beside the path: the largest level along it"""
    y, side, h, test = a
    ac = load_aircraft("uh60a")
    q = Quality.get("normal")
    fc = FlightCondition(airspeed_kt=150.0)
    st = operate(ac, fc, AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
    insts = en.load_installations("uh60a")
    env = Environment("ground", listener_height=h)
    H = 152.4
    lat = y if side == "right" else -y
    add = lambda x: 10 * np.log10(sum(10 ** (v / 10) for v in x))
    best = -99.0
    for x in (-300.0, -150.0, -75.0, 0.0, 75.0, 150.0, 300.0):
        hor = float(np.hypot(x, lat))
        d = float(np.hypot(hor, H))
        az = float(np.degrees(np.arctan2(lat, x)))
        el = max(-float(np.degrees(np.arcsin(H / d))), -89.0)
        r = render(ac, fc, Observer("p", az, el, d), duration=4.0, quality=q, state=st, variability=Variability(seed=1), environment=env)
        e = [en.render_engine(i, en.EngineCondition(94.0), Observer("side", az, 0.0, max(hor, 1.0)), duration=3.0, hub_height=H, environment=env).levels()["total"]["dba"] for i in insts]
        best = max(best, add([r.levels()["total"]["dba"]] + e))
    return dict(y=y, side=side, mic=h, test=test, dba=best)


def sites():
    with ProcessPoolExecutor(3) as ex:
        dump("sites", dict(rows=list(ex.map(_site, D.UH60_FAA_SITES)), gap=D.UH60_FAA_GAP))


def core():
    """GE's combustion noise and the jet's against what NASA measured, and what the method's doubt is worth in a T700"""
    f = np.array(sorted(D.YF102_CORE), float)
    shape = 10.0 * np.interp(np.log10(f / 400.0), *en._S_COMB)
    th = np.array(sorted(D.YF102_DI), float)
    t = np.radians(en._TH)
    Dm = 10.0 ** np.array(en._D_COMB)
    mean = np.trapezoid(Dm * np.sin(t), t) / 2.0
    di = 10.0 * np.interp(th, en._TH, en._D_COMB) - 10.0 * np.log10(mean)
    inst = en.load_installations("uh60a")[0]
    o = Observer("side", 90.0, 0.0, 50.0)
    parts = []
    for ng in (87.5, 94.0, 99.5):
        L = en.render_engine(inst, en.EngineCondition(ng), o, duration=4.0).levels()
        parts.append(dict(ng=ng, **{k: v["dba"] for k, v in L.items() if isinstance(v, dict) and "dba" in v}))
    atm = en.Atmosphere(0.0)
    area = np.pi * 0.0508 ** 2
    jets = []
    for v in (150.0, 200.0, 250.0, 300.0):
        b = en.jet_noise(v, atm.rho, area, atm).bands
        k = b.sum() * 4.0 * np.pi / (atm.rho * atm.c) / (atm.rho * area * v ** 8 / atm.c ** 5)
        jets.append(dict(v=v, k=k, st=float(en.BANDS[int(np.argmax(b))] * 0.1016 / v)))
    dump("core", dict(f=f, measured=[D.YF102_CORE[int(x)] for x in f], shape=shape, th=th, di_test=[D.YF102_DI[int(x)] for x in th], di=di,
                      power={str(k): v for k, v in D.YF102_POWER.items()}, parts=parts, jets=jets, modern=D.GE_MODERN, k_test=D.NACA1329_K))


def prue():
    """the ASK 21 at the Prue-2's height over the microphones, clean and with its wheels; Fink's wheels on the test's Cessna; the wheels in the cockpit"""
    from pywopwop.glider import BANDS as GB, GliderCondition, fink_wheel, load_glider, spectra, wheel_noise
    g = load_glider("ask21")
    atm = Atmosphere(0.0)
    pos = np.array([0.0, 0.0, -152.4])
    A = 10 ** (-iso9613_alpha(GB, 288.15, 50.0, 101325.0) * 152.4 / 10)
    db = lambda m: 10 * np.log10(m + 1e-30) + 93.98
    rows = []
    for kt, test in D.PRUE2.items():
        c = GliderCondition(airspeed_kt=kt)
        sp, _ = spectra(g, c, pos, inside=False)
        w = wheel_noise(g, c.V, pos, atm)
        L = db((sp["airframe"] - w) * A)
        f = np.array(sorted(test))
        M = kt * 0.514444 / 340.3
        rows.append(dict(kt=kt, f=GB, model=L, total=db(sp["airframe"] * A), wheels=db(w * A), test_f=f, test=[test[k] for k in f], at_test=np.interp(np.log(f), np.log(GB), L),
                         oaspl=10 * np.log10((10 ** (L[GB >= 300] / 10)).sum()) + 3.0, test_oaspl=D.PRUE2_OASPL[kt],
                         law=10 * np.log10(M ** 6 * g.wing_area / (g.span ** 2 / g.wing_area) ** 4 / 152.4 ** 2) + 178.4))
    cessna = []
    for kt, test in D.CESSNA150.items():
        p = sum(n * fink_wheel(d, kt * 0.514444, pos, atm) for d, n in D.CESSNA150_TYRES) * A
        f = np.array(sorted(test))
        cessna.append(dict(kt=kt, f=f, test=[test[k] for k in f], wheels=np.interp(np.log(f), np.log(GB), db(p))))
    head = to_ask21_head(g)
    inside = []
    for kt in (60.0, 100.0):
        c = GliderCondition(airspeed_kt=kt)
        sp, _ = spectra(g, c, head, inside=True)
        w = wheel_noise(g, c.V, head, atm) * sp_transmission(g)
        band = (GB >= 100) & (GB <= 400)
        inside.append(dict(kt=kt, under=float(np.max(db(w[band]) - db(sp["wind"][band])))))
    dump("prue", dict(rows=rows, cessna=cessna, inside=inside))


def to_ask21_head(g):
    from pywopwop.glider import to_body
    return to_body(*g.cockpit["front"])


def sp_transmission(g):
    from pywopwop.glider import BANDS as GB, _mass_law
    return _mass_law(GB, g.fuselage["canopy_surface_mass"])


# ---------------------------------------------------------------- the MV-22B's cockpit against its recording

CAM_T0, CAM_T1 = 20.0, 80.0


def camera(ac):
    return (ac.cockpit[0] - 1.0, 0.0, ac.cockpit[2] + 0.2)


def mv22_rotors(ac, altitude=3048.0, seed=11):
    o = Observer("camera", position=camera(ac))
    fc = FlightCondition(airspeed_kt=240.0, nacelle_deg=0.0, altitude_m=altitude, rpm_percent=99.64)
    return render(ac, fc, o, duration=15.0, quality=Quality.get("normal"), cabin=ac.cabin, variability=Variability(seed=seed))


def mv22_mix():
    from pywopwop.wind import wind_loop
    ac = load_aircraft("mv22b")
    N = int(FS * (CAM_T1 - CAM_T0))
    o = Observer("camera", position=camera(ac))
    parts = dict(proprotors=np.resize(mv22_rotors(ac).total, N))
    eng = np.zeros(N)
    for i, inst in enumerate(en.load_installations("mv22b", 0.0)):
        eng += np.resize(en.render_engine(inst, en.EngineCondition(90.5, np_pct=83.7), o, duration=15.0, seed=21 + i, cabin=ac.cabin).total, N)
    parts["engines"] = eng
    parts["wind"] = np.resize(wind_loop(ac, 240.0, 3048.0, duration=15.0, seed=31, listener=camera(ac)), N)
    tr = en.load_transmission("mv22b")
    sig, _ = en._render_sources({"gearbox": (None, en.gearbox_source(tr, 83.7, 46.0))}, o, None, FS, 15.167, 41, "free", None, 1.3, 25.0, Atmosphere(0.0))
    parts["drive"] = np.resize(sig["layer/gearbox"], N)
    return parts


def fit_bpf(y, lo=16.45, hi=16.80):
    f, P = welch(y, FS, nperseg=2 ** 18)
    best = None
    for b in np.arange(lo, hi, 0.001):
        s = sum(P[np.abs(f - h * b) < 0.2].sum() for h in range(6, 26))
        if best is None or s > best[1]:
            best = (float(b), s)
    return best[0]


def over_floor(y, bpf, hs):
    f, P = welch(y, FS, nperseg=2 ** 18)
    out = []
    for h in hs:
        on = np.abs(f - h * bpf) < 0.4
        off = (np.abs(f - h * bpf) > 2.0) & (np.abs(f - h * bpf) < 6.0)
        out.append(10 * np.log10(P[on].sum() / (np.median(P[off]) * on.sum())))
    return np.array(out)


def _passes(y, b, n=128):
    T = FS / b
    k = int(len(y) / T) - 1
    return y[(np.arange(k)[:, None] * T + (np.arange(n)[None, :] + 0.5) * T / n).astype(int)]


BANDS = ((50, 100), (100, 200), (200, 300), (300, 400), (400, 510), (510, 770), (770, 1270), (1270, 2000))
HIGH = ((500, 1000), (1000, 2000), (2000, 5000), (5000, 12000))


def pulse_stats(x, fixed=None):
    """each critical band's share in the blade lines and how hard it pulses with the blades, 10 s at a time at that stretch's own
    blade-passing frequency (or fixed); and the broadband's pulsing over 500 Hz"""
    segs = [x[i:i + 10 * FS] for i in range(0, len(x) - 10 * FS + 1, 10 * FS)]
    bpfs = [fixed or fit_bpf(s) for s in segs]
    rows = []
    for lo, hi in BANDS + HIGH:
        sh, pu = [], []
        for s, b in zip(segs, bpfs):
            f, P = welch(s, FS, nperseg=2 ** 18)
            k = (f >= lo) & (f < hi)
            hh = np.round(f / b)
            sh.append(P[k & (np.abs(f - hh * b) < 0.5)].sum() / P[k].sum())
            e = np.abs(hilbert(sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=FS, output="sos"), s)))
            E = _passes(e, b).mean(axis=0)
            c = np.fft.rfft(E)
            c[5:] = 0
            w = np.fft.irfft(c, len(E))
            pu.append(20 * np.log10(w.max() / w.min()))
        rows.append(dict(lo=lo, hi=hi, share=100 * float(np.mean(sh)), pulse=float(np.mean(pu))))
    return rows, bpfs


def prominence(x, f0):
    f, P = welch(x, FS, nperseg=8192)
    L = 10 * np.log10(P + 1e-30)
    return float((L - median_filter(L, 41))[np.abs(f - f0) < 60].max())


def _recording():
    fs, x = read_wav(os.path.join(REC, "mv22b_cockpit", "mono.wav"))
    return fs, x


def mv22():
    parts = mv22_mix()
    total = sum(parts.values())
    fs, x = _recording()
    rec = x[int(CAM_T0 * fs):int(CAM_T1 * fs)]
    gain = 10 ** ((lufs(rec) - lufs(total)) / 20.0)
    mod = total * gain
    fc, Lr = thirds(rec, lo=25.0, hi=12500.0)
    _, Lm = thirds(mod, lo=25.0, hi=12500.0)
    H = np.arange(1, 31)
    br = fit_bpf(rec)
    bm = fit_bpf(mod)
    sr, bpfs = pulse_stats(rec)
    sm, _ = pulse_stats(mod)

    def floor_thirds(y, bpf):
        f, P = welch(y, FS, nperseg=2 ** 18)
        out = []
        for c in fc[(fc >= 39) & (fc <= 500)]:
            k = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
            h = np.round(f / bpf)
            off = k & (np.abs(f - h * bpf) > 2.0)
            out.append(10 * np.log10(np.median(P[off]) * k.sum() + 1e-30))
        return np.array(out)
    bands = ((25, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 12000))
    oct_ = lambda y: [float(10 * np.log10(welch(y, FS, nperseg=2 ** 15)[1][(welch(y, FS, nperseg=2 ** 15)[0] >= a) & (welch(y, FS, nperseg=2 ** 15)[0] < b)].sum())) for a, b in bands]
    # the cabin window beside the nacelle
    win = {}
    for t0, t1, tag in ((160, 200, "cruise"), (215, 300, "power_off")):
        f, P = welch(x[t0 * fs:t1 * fs], fs, nperseg=16384)
        win[tag] = _line(f, P, 6500, 7300, side=400.0)
    e = np.abs(hilbert(sosfiltfilt(butter(4, [6900, 7250], "bandpass", fs=fs, output="sos"), x[160 * fs:200 * fs])))[::16]
    fe, Pe = welch(e - e.mean(), fs / 16, nperseg=2 ** 13)
    Le = 10 * np.log10(Pe + 1e-30)
    chop = float(Le[np.abs(fe - 16.6) < 0.5].max() - np.median(Le[(fe > 3) & (fe < 60)]))
    dump("mv22", dict(fc=fc, recording=Lr, model=Lm, floor_fc=fc[(fc >= 39) & (fc <= 500)], floor_rec=floor_thirds(rec, br), floor_mod=floor_thirds(mod, bm), harmonics=H,
                      over_rec=over_floor(rec, br, H), over_mod=over_floor(mod, bm, H), bpf_rec=br, bpf_mod=bm, bpf_segments=bpfs, stats_rec=sr, stats_mod=sm,
                      parts={k: dba(v) for k, v in parts.items()}, all_dba=dba(total), bands=bands, oct_rec=oct_(rec), oct_mod=oct_(mod),
                      tones={str(f0): dict(rec=prominence(rec, f0), mod=prominence(mod, f0)) for f0 in (7080.0, 2030.0, 8170.0)}, window=win, chop=chop))
    os.makedirs(CACHE, exist_ok=True)               # holds the recording's audio: never in the repository
    np.savez(os.path.join(CACHE, "mv22_mix.npz"), model=mod.astype(np.float32), recording=rec.astype(np.float32))


def chop():
    """the whine at the cabin window, chopped by the proprotor: its line's envelope at the blade rate, over the envelope's floor"""
    fs, x = _recording()
    y = x[160 * fs:200 * fs]
    f, P = welch(y, fs, nperseg=2 ** 16)
    f0 = _line(f, P, 6900, 7250, side=400.0)[0]
    out = {}
    for tag, lo, hi in (("line", f0 - 12, f0 + 12), ("beside", f0 + 60, f0 + 300)):
        e = np.abs(hilbert(sosfiltfilt(butter(2, [lo, hi], "bandpass", fs=fs, output="sos"), y)))[::16]
        fe, Pe = welch(e - e.mean(), fs / 16, nperseg=2 ** 13)
        L = 10 * np.log10(Pe + 1e-30)
        out[tag] = float(L[np.abs(fe - 16.6) < 0.5].max() - np.median(L[(fe > 3) & (fe < 60)]))
    dump("chop", dict(f0=f0, **out))


def _terms(name):
    ac = load_aircraft("mv22b")
    altitude = 3048.0
    if name == "no_wing":
        ac.wing = None
    elif name == "in_step":
        ac.rotors[1].index_deg = 0.0
    elif name == "sea_level":
        altitude = 0.0
    elif name == "single_wall":
        ac.cabin.trim_mass = None
    elif name == "no_structure":
        ac.cabin.coincidence_hz, keep = None, ac.cabin.coincidence_hz
    r = mv22_rotors(ac, altitude)
    x = r.total
    return name, x


def mv22_terms():
    """each physical term of the proprotors' path taken out of the mix in turn: how far their harmonics stand out of the rest then"""
    full = mv22_mix()
    rest = sum(v for k, v in full.items() if k != "proprotors")
    d = np.load(os.path.join(CACHE, "mv22_mix.npz"))
    rec = d["recording"].astype(float)
    g = float(np.sqrt(np.mean(d["model"].astype(float) ** 2) / np.mean(sum(full.values()) ** 2)))
    H = np.arange(1, 31)
    br = fit_bpf(rec)
    orr = over_floor(rec, br, H)
    rows = []
    for name in ("full", "no_wing", "in_step", "sea_level", "single_wall"):
        ac = load_aircraft("mv22b")
        if name == "full":
            x = full["proprotors"]
        else:
            n, y = _terms(name)
            x = np.resize(y, len(rest))
        t = (x + rest) * g
        bm = fit_bpf(t)
        o = over_floor(t, bm, H)
        lo, mid, hi = (H >= 2) & (H <= 5), (H >= 6) & (H <= 13), (H >= 14) & (H <= 30)
        s, _ = pulse_stats(t[:20 * FS])
        rows.append(dict(name=name, rotors_dba=dba(x), h2_5=float(o[lo].mean()), h6_13=float(o[mid].mean()), h14_30=float(o[hi].mean()),
                         pulse_100=s[1]["pulse"], share_100=s[1]["share"]))
        print(rows[-1], flush=True)
    lo, mid, hi = (H >= 2) & (H <= 5), (H >= 6) & (H <= 13), (H >= 14) & (H <= 30)
    dump("mv22_terms", dict(rows=rows, rec=dict(h2_5=float(orr[lo].mean()), h6_13=float(orr[mid].mean()), h14_30=float(orr[hi].mean()))))


# ---------------------------------------------------------------- the ASK 21

def median_bands(x, fs, fc, n=16384):
    w = np.hanning(n)
    segs = [x[j:j + n] * w for j in range(0, len(x) - n + 1, n // 2)]
    P = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], 0)
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.array([10 * np.log10(np.median(P[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))]) * ((f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))).sum() + 1e-20) for c in fc])


def roar():
    """the cockpit's low roar, measured: each band of the first recording split into a part fixed in time and a part growing with the airspeed, set to the
    model's 1-4 kHz, less what the model has there without it. Checked on a recording it doesn't use: the roll, another glider, a phone"""
    from dataclasses import replace
    from scipy.optimize import nnls
    from pywopwop.glider import BANDS as GB, GliderCondition, load_glider, spectra, to_body
    g = load_glider("ask21")
    g0 = replace(g, roar=None)
    ear = to_body(*g.cockpit["front"])
    use = (GB >= 30) & (GB <= 6400)
    fc = GB[use]
    KTS = np.array([40.0, 50.0, 60.0, 75.0, 90.0, 105.0, 120.0])

    def model(gl, v):
        sp, _ = spectra(gl, GliderCondition(airspeed_kt=v), ear, inside=True)
        return 10 * np.log10(sp["wind"] + sp["airframe"] + 1e-30)[use] + 93.98
    M = np.array([model(g0, v) for v in KTS])
    fs, x = read_wav(os.path.join(REC, "ask21_aerobatics", "mono.wav"))
    d = json.load(open(os.path.join(HERE, "data", "ask21_asi.json")))
    tt, kt = np.array(d["t"]), np.array(d["kt"])
    T = np.arange(0.0, len(x) / fs - 1.0, 0.5)
    R = np.array([median_bands(x[int(t * fs):int((t + 1.0) * fs)], fs, fc) for t in T])
    V = np.array([np.mean(kt[(tt >= t) & (tt < t + 1.0)]) if ((tt >= t) & (tt < t + 1.0)).any() else np.nan for t in T])
    s = np.isfinite(V) & (V > 30)
    ref = (fc >= 1000) & (fc <= 4000)
    lv = np.log10(KTS)
    mref = np.interp(np.log10(V[s]), lv, 10 * np.log10((10 ** (M[:, ref] / 10)).mean(1)))
    off = float(np.median(mref - 10 * np.log10((10 ** (R[s][:, ref] / 10)).mean(1))))
    low = np.where(fc <= 520)[0]
    fits = {}
    for n in (2, 3, 4, 5, 6):
        fl, fw, e = [], [], []
        for j in low:
            y = 10 ** (R[s, j] / 10)
            A = np.c_[np.ones(s.sum()), (V[s] / 100.0) ** n]
            c = nnls(A / y[:, None], np.ones(s.sum()))[0]
            fl.append(10 * np.log10(max(c[0], 1e-9)) + off)
            fw.append(10 * np.log10(max(c[1], 1e-9)) + off)
            e.append(float(np.std(10 * np.log10((A @ c) / y))))
        fits[n] = dict(floor=fl, flow=fw, rms=e)
    m100 = np.array([np.interp(2.0, lv, M[:, j]) for j in low])
    flow = np.array(fits[3]["flow"])
    res = 10 * np.log10(np.maximum(10 ** (flow / 10) - 10 ** (m100 / 10), 1e-3))
    stall = (V > 30) & (V < 42)
    fast = (V > 108) & (V < 118)
    # the roll: another glider and microphone, relative to its own 1-4 kHz, without and with the roar
    fs3, z = read_wav(os.path.join(REC, "roll_short", "mono.wav"))
    win = np.arange(0.25, 18.8, 0.5)
    Vr = np.interp(win, *np.array(D.ROLL_ASI).T) / 1.852
    gap = np.zeros(len(win), bool)
    for a_, b_ in D.ROLL_GAPS:
        gap |= (win > a_) & (win < b_)
    Rr = np.array([median_bands(z[int((t - 0.25) * fs3):int((t + 0.75) * fs3)], fs3, fc) for t in win])
    k = (fc >= 120) & (fc <= 330)
    roll = []
    for lo, hi in ((60, 75), (75, 90), (90, 110)):
        m = (Vr >= lo) & (Vr < hi) & ~gap
        r = Rr[m].mean(0)
        v = float(Vr[m].mean())
        a, b = model(g0, v), model(g, v)
        rr = r - r[ref].mean()
        roll.append(dict(kt=v, n=int(m.sum()), rec=rr, without=a - a[ref].mean(), with_roar=b - a[ref].mean(), rms_without=float(np.sqrt(np.mean((rr - a + a[ref].mean())[k] ** 2))),
                         rms_with=float(np.sqrt(np.mean((rr - b + a[ref].mean())[k] ** 2)))))
    dump("roar", dict(fc=fc[low], offset=off, floor=fits[3]["floor"], flow=flow, model=m100, residual=res, rms={str(n): float(np.mean(f["rms"])) for n, f in fits.items()},
                      stall=R[stall][:, low].mean(0), fast=R[fast][:, low].mean(0), corr=[float(np.corrcoef(np.log10(V[s]), R[s, j])[0, 1]) for j in low], roll=roll, preset=dict(hz=g.roar["hz"], db=g.roar["db"]) if g.roar else None))


def ask21():
    from pywopwop.glider import BANDS as GB, GliderCondition, load_glider, spectra, stall_speed, to_body
    g = load_glider("ask21")
    ear = to_body(*g.cockpit["front"])
    fc = GB[(GB >= 100) & (GB <= 12500)]
    use = (GB >= 100) & (GB <= 12500)
    KTS = np.array([50.0, 60.0, 75.0, 90.0, 105.0, 120.0])
    M = []
    for v in KTS:
        sp, st = spectra(g, GliderCondition(airspeed_kt=v), ear, inside=True)
        M.append(10 * np.log10(sp["wind"] + sp["airframe"] + 1e-30)[use] + 93.98)
    M = np.array(M)
    fs, x = read_wav(os.path.join(REC, "ask21_aerobatics", "mono.wav"))
    d = json.load(open(os.path.join(HERE, "data", "ask21_asi.json")))
    tt, kt = np.array(d["t"]), np.array(d["kt"])
    T = np.arange(44.0, 140.0, 0.5)
    R = np.array([median_bands(x[int(t * fs):int((t + 1.0) * fs)], fs, fc) for t in T])
    V = np.array([np.mean(kt[(tt >= t) & (tt < t + 1.0)]) for t in T])
    bins = [(46, 56), (56, 66), (70, 80), (86, 96), (108, 118)]
    ref = (fc >= 1000) & (fc <= 4000)
    shapes = []
    lv = np.log10(KTS)
    for a, b in bins:
        s = (V >= a) & (V < b)
        r = R[s].mean(0)
        v = float(V[s].mean())
        mo = np.array([np.interp(np.log10(v), lv, M[:, j]) for j in range(len(fc))])
        shapes.append(dict(kt=v, n=int(s.sum()), rec=r - r[ref].mean(), mod=mo - mo[ref].mean()))
    # the sound power's growth with speed, band by band from 400 Hz to 8 kHz, without the bands the whistle crosses (1.25-3.2 kHz)
    k = (fc >= 400) & (fc <= 8000) & ~((fc >= 1250) & (fc <= 3200))
    ok = V > 45
    summed = lambda L: 10 * np.log10(np.sum(10 ** (L[:, k] / 10), axis=1))
    pr = float(np.polyfit(10 * np.log10(V[ok]), summed(R)[ok], 1)[0])
    pm = float(np.polyfit(10 * np.log10(KTS), summed(M), 1)[0])
    hi = np.where((fc >= 3200) & (fc <= 8000))[0]
    hr = float(np.median([np.polyfit(np.log10(V[ok]), R[ok, j], 1)[0] for j in hi]))
    hm = float(np.median([np.polyfit(np.log10(KTS), M[:, j], 1)[0] for j in hi]))
    # the second glider: shape only, the stretches free of the music's cymbals
    fs2, y = read_wav(os.path.join(REC, "ask21_4g", "mono.wav"))
    rows, keep = [], 0
    for t in np.arange(0, len(y) / fs2 - 1.0, 1.0):
        s = y[int(t * fs2):int((t + 1) * fs2)]
        f, P = welch(s, fs2, nperseg=8192)
        L = 10 * np.log10(P + 1e-30)
        pr_ = L - median_filter(L, 41)
        if max(pr_[np.abs(f - c) < 30].max() for c in (7860.0, 12280.0, 13500.0, 13980.0)) > 6.0:
            continue
        rows.append(median_bands(s, fs2, fc))
        keep += 1
    r2 = np.mean(rows, axis=0)
    mo = np.array([np.interp(np.log10(80.0), lv, M[:, j]) for j in range(len(fc))])
    # the roll
    KG = np.arange(60.0, 101.0, 10.0)
    NS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    Gd = np.full((len(KG), len(NS)), np.nan)
    hiB = (GB >= 1600) & (GB <= 6300)
    for i, v in enumerate(KG):
        for j, n in enumerate(NS):
            try:
                sp, _ = spectra(g, GliderCondition(airspeed_kt=v, load_factor=n), ear, inside=True)
                Gd[i, j] = 10 * np.log10(np.sum((sp["wind"] + sp["airframe"])[hiB])) + 93.98
            except Exception:
                pass
    from scipy.interpolate import RegularGridInterpolator
    from scipy.signal import stft
    fs3, z = read_wav(os.path.join(REC, "roll_short", "mono.wav"))
    f, ts, Z = stft(z, fs3, nperseg=2048, noverlap=1024)
    P = np.abs(Z) ** 2
    win = np.arange(0.25, 18.8, 0.5)
    b = GB[hiB]
    rec = []
    for c in win:
        med = np.median(P[:, np.abs(ts - c) <= 0.25], axis=1)
        rec.append(10 * np.log10(sum(np.median(med[(f >= q * 2 ** (-1 / 6)) & (f < q * 2 ** (1 / 6))]) * ((f >= q * 2 ** (-1 / 6)) & (f < q * 2 ** (1 / 6))).sum() for q in b)))
    rec = np.array(rec)
    Vr = np.interp(win, *np.array(D.ROLL_ASI).T) / 1.852
    nr = np.interp(win, *np.array(D.ROLL_G).T)
    gap = np.zeros(len(win), bool)
    for a_, b_ in D.ROLL_GAPS:
        gap |= (win > a_) & (win < b_)
    mod = RegularGridInterpolator((KG, NS), Gd)(np.c_[np.clip(Vr, KG[0], KG[-1]), np.clip(nr, 0, 5)])
    rec_c, mod_c = rec - rec[~gap].mean(), mod - mod[~gap].mean()
    X = np.c_[np.ones((~gap).sum()), 20 * np.log10(Vr[~gap]), nr[~gap]]
    cr = np.linalg.lstsq(X, rec[~gap], rcond=None)[0]
    cm = np.linalg.lstsq(X, mod[~gap], rcond=None)[0]
    dump("ask21", dict(stall_kt=stall_speed(g), fc=fc, kts=KTS, model=M, shapes=shapes, exponent_rec=pr, exponent_mod=pm, high_rec=hr, high_mod=hm, second=dict(rec=r2 - r2[ref].mean(), mod=mo - mo[ref].mean(), seconds=keep),
                       dba={str(v): float(10 * np.log10(np.sum(10 ** ((M[i] + synth.a_weighting_db(fc)) / 10)))) for i, v in enumerate(KTS)},
                       roll=dict(t=win, rec=rec_c, mod=mod_c, gap=gap, kt=Vr, g=nr, per_g_rec=float(cr[2]), per_g_mod=float(cm[2]), exp_rec=float(cr[1]), exp_mod=float(cm[1]),
                                 swing_rec=float(np.ptp(rec_c[~gap])), swing_mod=float(np.ptp(mod_c[~gap])))))


# ---------------------------------------------------------------- timings

def slip():
    """the glider slipping: where the crossflow separates at the seats, the cockpit's halves and the outside at the angles, the wing at the
    flight model's full slip, what Fink's strut amounts to as a cylinder's lift, and the layers' fades below the lowest loop"""
    from scipy.optimize import brentq
    from pywopwop.glider import (BANDS as GB, GliderCondition, crossflow_time, load_glider, low_speed_power, spectra, strip_states, strips,
                                 to_body)
    from pywopwop.model import Observer, STANDARD_OBSERVERS
    from pywopwop.synth import a_weighting_db
    g = load_glider("ask21")
    head, rear = to_body(*g.cockpit["front"]), to_body(*g.cockpit["rear"])
    place = lambda n: Observer(n, STANDARD_OBSERVERS[n].azimuth_deg, STANDARD_OBSERVERS[n].elevation_deg, 30.0).offset()
    side, below = place("side"), place("below")
    wa = 10.0 ** (a_weighting_db(GB) / 10.0)
    dbA = lambda p: float(10 * np.log10(np.sum(p * wa) / 4e-10))
    oa = lambda p: float(10 * np.log10(np.sum(p) / 4e-10))
    hiss = (GB >= 1500) & (GB <= 6800)
    FULL, FULL_KT = 46.0, 53.5                            # the flight model's full-rudder slip: 99 km/h
    rows = []
    for kt in (FULL_KT, 90.0):
        for b in (0.0, 10.0, 15.0, 17.3, 20.0, 25.0, 30.0, 35.0, 40.0, FULL, 50.0):
            c = GliderCondition(airspeed_kt=kt, sideslip_deg=b)
            i, st = spectra(g, c, head, True)
            r = spectra(g, c, rear, True)[0]
            o, u = spectra(g, c, side)[0]["airframe"], spectra(g, c, below)[0]["airframe"]
            rows.append(dict(kt=kt, beta=b, front=dbA(i["wind"]), windward=dbA(i["windward"]), leeward=dbA(i["leeward"]), rear=dbA(r["wind"]),
                             front_oa=oa(i["wind"]), share=st["canopy_separated"], hiss=float(10 * np.log10(np.sum(i["wind"][hiss]) / 4e-10)),
                             side=dbA(o), side_oa=oa(o), below=dbA(u), below_oa=oa(u)))
    c0, c1 = GliderCondition(airspeed_kt=FULL_KT), GliderCondition(airspeed_kt=FULL_KT, sideslip_deg=FULL)
    a, b = spectra(g, c0, head, True)[0], spectra(g, c1, head, True)[0]
    o0, o1 = spectra(g, c0, side)[0]["airframe"], spectra(g, c1, side)[0]["airframe"]
    u0, u1 = spectra(g, c0, below)[0]["airframe"], spectra(g, c1, below)[0]["airframe"]
    brakes = spectra(g, GliderCondition(airspeed_kt=FULL_KT, brakes=1.0), side)[0]["brakes"]
    S = strips(g)
    st = strip_states(g, c1)[0]
    w = [(s, x) for s, x in zip(S, st) if s.kind == "wing"]
    share = lambda sign: float(sum(s.span * x["sep"] for s, x in w if s.side == sign) / sum(s.span for s, x in w if s.side == sign))
    onset = {k: float(np.degrees(brentq(lambda t: float(crossflow_time(g, g.cockpit[k][0], np.degrees(t))) - 1.5, 0.01, 1.3)))
             for k in ("front", "rear")}
    db = lambda p: 10 * np.log10(np.maximum(p, 1e-30) / 4e-10)
    dump("slip", dict(rows=rows, fc=GB, full=FULL, full_kt=FULL_KT, onset=onset, fin_balance=strip_states(g, GliderCondition(airspeed_kt=60.0, rudder=1.0))[1]["sideslip_deg"],
                      straight_half=db(a["windward"]), windward=db(b["windward"]), leeward=db(b["leeward"]),
                      side_straight=db(o0), side_slip=db(o1), below_straight=db(u0), below_slip=db(u1), brakes=db(brakes), side_added=db(np.maximum(o1 - o0, 0.0)),
                      wing_windward=share(1), wing_leeward=share(-1), wing_stalled=any(x["stalled"] for s, x in w),
                      fin_stalled=any(x["stalled"] for s, x in zip(S, st) if s.kind == "fin"),
                      lift_rms=float(np.sqrt(16 * 3 * 2.753e-4 / (4 * np.pi * 0.2 ** 2 * 5))), power=low_speed_power(g, 40.0)))


def props():
    """the X-57's propellers: the high-lift prop against NASA's schedule and its tunnel, the cruise prop against NASA's CFD"""
    from dataclasses import replace
    from pywopwop import propeller as P, synth, broadband
    from pywopwop.aero import AeroOptions
    from pywopwop.atmosphere import Atmosphere
    from pywopwop.model import KT
    ac = P.load_prop_aircraft("x57")
    hl = next(r for r in ac.rotors if r.name == "hl_7")
    o = AeroOptions(n_radial=24, n_azimuth=180, bvi=False, elastic=False)
    atm = Atmosphere(762.0)
    sched, cruise = [], []
    for kt, rpm, nm, kw, lbf, hp in D.X57_SCHEDULE:
        om = rpm * 2 * np.pi / 60
        air = Atmosphere(hp * 0.3048)
        a, _ = P._solved_at_pitch(hl, om, kt * KT / np.sqrt(air.rho / 1.225), np.radians(-89.9), np.radians(ac.pitch_deg), air, o, max(lbf, 5) * 4.448)
        sched.append(dict(kt=kt, rpm=rpm, nm=nm, kw=kw, lbf=lbf, model_kw=a.torque * om / 1e3, model_lbf=a.thrust / 4.448, model_nm=a.torque))
    om = 2250 * 2 * np.pi / 60
    for v, beta, lbf, kw in D.X57_CRUISE:
        a, _ = P._solved_at_pitch(ac.rotors[0], om, v * 0.3048, np.radians(-88.0), np.radians(beta), atm, o, lbf * 4.448)
        cruise.append(dict(v=v, beta=beta, kw=kw, lbf=lbf, model_kw=a.torque * om / 1e3, model_lbf=a.thrust / 4.448))
    alone = replace(ac, rotors=[replace(hl, position=(0.0, 0.0, 0.0))], wing=[[0.0, -5.0, 0.5, 0.0, 0.05], [1.0, -5.0, 0.5, 0.0, 0.05]])
    alone.pitch_deg = ac.pitch_deg
    pt = P.operate_props(alone, P.PropCondition("tunnel", 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4800.0), AeroOptions(bvi=False, elastic=False)).points[0]
    q = synth.Quality.get("normal")
    bo = broadband.BroadbandOptions(n_azimuth=q.bb_azimuth, turbulence_ms=0.3, turbulence_scale=0.2, from_zero_lift=True)
    cache, bpf, h95 = {}, [], None
    B = hl.blades
    for th in D.X57_ANGLE:
        t = np.radians(th)
        r = 3.54 / np.sin(t)
        s = synth.rotor_sound(pt, r * np.array([np.cos(t), 0.0, -np.sin(t)]), q, bo, th == 95, cache, None)
        amp = lambda k: float(20 * np.log10(2 * abs(B * (s.H_thickness + s.H_loading)[k * B]) / s.n / np.sqrt(2) / 2e-5))
        bpf.append(amp(1))
        if th == 95:
            h95, harm = s.history, [amp(k) for k in (1, 2, 3)]
    f = np.array(h95.f)
    m = h95.msp.mean(0)
    per10 = 10 * np.log10(m / 4e-10 * 10.0 / (f * (2 ** (1 / 6) - 2 ** (-1 / 6))))
    m = json.load(open(os.path.join(ROOT, "out", "x57", "manifest.json")))
    level = lambda c, g, s, o: next(f["dba"] for f in m["files"] if (f["condition"], f["group"], f["source"], f["observer"]) == (c, g, s, o))
    under = [level("100kt_2250rpm_ct0.05", "cruise_left", "prop", "cockpit") - level("motor", "cruise_left", "motor", "cockpit"),
             level("60kt_4800rpm", "hl_b_left", "prop", "cockpit") - level("motor", "hl_b_left", "motor", "cockpit")]
    dump("props", dict(schedule=sched, cruise=cruise, under=under, sources=m["max_sources"]["cockpit"], loops=len(m["files"]), angle=D.X57_ANGLE, bpf=bpf, lh=D.X57_BPF_LH, rh=D.X57_BPF_RH, harmonics=harm,
                       thrust_lbf=pt.aero.thrust / 4.448, power_kw=pt.aero.torque * pt.omega / 1e3, f=f, per10=per10,
                       test_hz=D.X57_BROADBAND_HZ, test_db=D.X57_BROADBAND_DB, background_db=D.X57_BACKGROUND_DB,
                       model_db=[float(np.interp(np.log(x), np.log(f), per10)) for x in D.X57_BROADBAND_HZ],
                       pitch=ac.pitch_deg, upwash=[P.wing_upwash_deg(ac, r, 1.5) for r in ac.rotors if r.name.startswith("hl_")]))


def bench():
    lines = []
    validation.benchmark(out=lines.append)
    t0 = time.perf_counter()
    ac = load_aircraft("uh60a")
    fc = FlightCondition(airspeed_kt=65, climb_fpm=-800)
    st = operate(ac, fc)
    r = render(ac, fc, STANDARD_OBSERVERS["front_right"], state=st, duration=15.0)
    t = time.perf_counter() - t0
    load_avg = os.getloadavg()[0]
    dump("bench", dict(lines=lines, loop_15s=t, load=load_avg, machine=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                       cores=os.cpu_count()))


def tyres():
    """tyres rolling: the checks outside, the cockpit's transfers from an ASK 21's and a P2006T's landing, the X-57's tyres"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import tyre_recordings
    dump("tyres", tyre_recordings.results())


STAGES = dict(verify=verify, rotors=rotors, stand=stand, hart2=hart2, bvi=bvi, texture=texture, fountain=fountain, outside=outside, engines=engines, tones=tones, cabin=cabin,
              wind=wind, cockpit=cockpit, prue=prue, flight=flight, sites=sites, core=core, mv22=mv22, chop=chop, mv22_terms=mv22_terms, roar=roar, ask21=ask21, slip=slip, props=props, tyres=tyres, bench=bench)

if __name__ == "__main__":
    import pywopwop
    print("pywopwop from", pywopwop.__file__, flush=True)
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[1:] or STAGES:
        t = time.time()
        STAGES[name]()
        print(f"{name}: {time.time() - t:.0f} s", flush=True)
