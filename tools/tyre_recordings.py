#!/usr/bin/env python3
"""The tyre model (pywopwop/tyre.py) against what is measured: the checks outside, and the cockpit's structure-borne
transfers taken from two recorded landings, with their texture, and the X-57's three tyres by speed. Writes
report/results/tyres.json (report/mkresults.py runs it as its stage tyres).

    python tools/tyre_recordings.py

Reads refs/recordings (kept out of the repository): ask21b_landing/full.wav, an ASK 21B's front seat, a camera fixed to
the airframe, touching down on a paved runway at 595.34 s at 44 kt (the airspeed indicator in shot); p2006t_ymen/full.wav,
a Tecnam P2006T's cabin, a camera at the right seat, touching down on asphalt at 388.5 s (a 2-5 kHz chirp), the speed not
in shot (about 58 kt: a P2006T flight test touched down at 65, short landings come in slower). Both cameras' gain controls
hold their level, so only the instants either side of a touchdown, or one band against another, can be read."""
import json
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pywopwop import glider as G, tyre as T, wind as Wd           # noqa: E402
from pywopwop.atmosphere import Atmosphere                        # noqa: E402
from pywopwop.glider import BANDS as fc                           # noqa: E402
from pywopwop.propeller import load_prop_aircraft                 # noqa: E402

REC = os.path.join(ROOT, "refs", "recordings")
A = 10.0 ** (T.a_weight(fc) / 10.0)
KT = 0.514444
ALL = np.ones(len(fc), bool)
REF = (fc >= 2000) & (fc <= 5000)                # the camera's gain is set here, by the airflow
BAND = (fc >= 150) & (fc <= 1700)                # the roll
SEL = (fc >= 120) & (fc <= 2100)                 # the transfers' bands
MAIN_5 = dict(width_m=0.1257, diameter_m=0.3607)    # 5.00-5 (the ASK 21's main, the P2006T's nose)
MAIN_6 = dict(width_m=0.16, diameter_m=0.4445)      # 6.00-6 (the P2006T's mains)
# the X-57's airframe as its struts feel it (x57-mod3.xml: 3000 lb, Iyy 1804.3 slug ft^2, the centre of gravity at
# 155.101 in, the mains at 172.0 and the nose at 57.83): the two mains together, the nose pivoting about them
_M, _IYY = 3000.0 * 0.45359237, 1804.3 * 1.3558179
X57_MASS = dict(main=T.strut_mass(_M, _IYY, (172.0 - 155.101) * 0.0254) / 2,
                nose=T.strut_mass(_M, _IYY, (155.101 - 57.83) * 0.0254, (172.0 - 57.83) * 0.0254))


def dba(p2, sel=BAND):
    return float(10 * np.log10(np.sum((p2 * A)[sel]) / 4e-10))


def load(name):
    fs, x = wavfile.read(os.path.join(REC, name, "full.wav"))
    return fs, (x.astype(float).mean(1) if x.ndim > 1 else x.astype(float)) / 32768.0


def thirds(m, fs, t0, t1, nperseg=8192):
    f, P = welch(m[int(t0 * fs):int(t1 * fs)], fs, nperseg=nperseg)
    df = f[1] - f[0]
    return np.array([P[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].sum() * df for c in fc])


def power(tyres, kt):
    return sum(T.tyre_power(t["width_m"], t["diameter_m"], kt * KT) for t in tyres)


def transfer(p2, W):
    return 10 * np.log10(np.maximum(p2, 1e-20) / 4e-10) - 10 * np.log10(np.maximum(W, 1e-30) / 1e-12)


def texture(x, fs):
    """the roll band's kurtosis (Gaussian noise: 3)"""
    b = sosfiltfilt(butter(4, [150, 1700], "bandpass", fs=fs, output="sos"), x)
    return float(np.mean((b - b.mean()) ** 4) / np.var(b) ** 2)


def outside():
    r = T.REFERENCE
    W = T.tyre_power(r["width_m"], r["diameter_m"], r["kmh"] / 3.6, r["surface"])
    ref = dict(model=dba(W * T.RHO_C / (2 * np.pi), ALL), measured=float(10 * np.log10(np.sum(10 ** (np.asarray(r["dba"]) / 10)))))
    car = [dict(position=(x, y, 0.0), width_m=0.205, diameter_m=0.632) for x in (1.3, -1.3) for y in (0.75, -0.75)]
    car = dict(model=dba(T.tyre_noise(car, 80 / 3.6, (0.0, -7.5, 1.2), "smooth asphalt"), ALL), label=70.0)
    tow = [dict(position=(x, y, 0.0), **MAIN_5) for x, y in ((1.5, 0.0), (-0.5, -1.0), (-0.5, 1.0))]
    v = 55 * 0.44704
    tow = dict(model_smooth=dba(T.tyre_noise(tow, v, (0.0, -30.48, 1.2), "smooth asphalt"), ALL),
               model_asphalt=dba(T.tyre_noise(tow, v, (0.0, -30.48, 1.2), "asphalt"), ALL), measured=60.0)
    return dict(reference_1m=ref, car_80kmh_7m5=car, towed_aircraft_55mph_30m=tow)


def ask21():
    fs, m = load("ask21b_landing")
    g = G.load_glider("ask21")
    head = G.to_body(*g.cockpit["front"])
    air = lambda kt, n: (lambda s: s["wind"] + s["airframe"])(
        G.spectra(g, G.GliderCondition(airspeed_kt=kt, load_factor=n, turbulence_ms=0.5), head, True)[0])
    a47, a44 = air(47, 1.0), air(44, 0.4)
    # the gain in flight just before (the airflow), held through the roll's first 0.3 s
    flight = thirds(m, fs, 586.0, 595.3)
    k_flight = np.sum(a47[REF]) / np.sum(flight[REF])
    td = np.maximum(thirds(m, fs, 595.34, 595.64, 4096) * k_flight - a44, 0)
    # the gain from the airflow in the next 4 s
    r44 = thirds(m, fs, 596.0, 600.0)
    k44 = np.sum(a44[REF]) / np.sum(r44[REF])
    seg = np.maximum(r44 * k44 - a44, 0)
    t_td = transfer(td, power([MAIN_5], 45))
    t_44 = transfer(seg, power([MAIN_5], 44))
    # the 2-5 kHz across the touchdown: unchanged as the roll's bands jump, so it is airflow there and the gain held
    hi = lambda t0, t1: float(10 * np.log10(np.sum(thirds(m, fs, t0, t1, 4096)[REF])))
    ratio = [float(10 * np.log10(np.sum(thirds(m, fs, t0, t0 + 4)[BAND]) / np.sum(thirds(m, fs, t0, t0 + 4)[REF])))
             for t0 in (596, 600, 604, 608)]
    return dict(hz=fc[SEL].round().tolist(), touchdown=t_td[SEL].round(1).tolist(), at_44kt=t_44[SEL].round(1).tolist(),
                mean=((t_td + t_44) / 2)[SEL].round(1).tolist(), module=T.structure_db(fc[SEL], "ask21").round(1).tolist(),
                hi_before_after_db=[hi(594.9, 595.3), hi(595.34, 595.64)],
                roll_over_2_5khz_db_at_44_37_28_18kt=[round(r, 1) for r in ratio],
                kurtosis_44_37_28kt=[round(texture(m[int(t * fs):int((t + 4) * fs)], fs), 2) for t in (596, 600, 604)])


def wind(kt, ac=None):
    """the X-57's cockpit wind (its bank's: wind.py's windscreen 1.7 m from the nose, 3 mm acrylic), per third octave"""
    ac = ac or load_prop_aircraft("x57")
    atm = Atmosphere(0.0)
    f = np.linspace(1.0, 24000.0, 48000)
    U = kt * KT
    S = Wd.wall_pressure_psd(f, U, 1.7, atm) * Wd.transmission(f, U, Wd.windscreen(ac, 3.6), atm.c) * ac.cabin.room(f)
    df = f[1] - f[0]
    return np.array([S[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].sum() * df for c in fc])


# the X-57's tyres, from its flight model's gear (x57-mod3.xml): contacts in metres from the moment reference, x forward,
# y left, z up; the ground at rest 3 in under the contacts
X57_TYRES = [dict(name="nose", gear=0, position=(2.463, 0.0, -1.172), static_compression_ft=0.2439, **MAIN_5),
             dict(name="left_main", gear=1, position=(-0.437, 1.0, -1.172), static_compression_ft=0.2439, **MAIN_6),
             dict(name="right_main", gear=2, position=(-0.437, -1.0, -1.172), static_compression_ft=0.2439, **MAIN_6)]


def x57(speeds=(5, 10, 20, 30, 45, 60, 75, 90)):
    """the X-57's three tyres together by ground speed: the cockpit through each airframe's transfer, 30 m to the side,
    and the cockpit's wind at that airspeed"""
    from pywopwop.model import Observer, STANDARD_OBSERVERS
    ac = load_prop_aircraft("x57")
    so = STANDARD_OBSERVERS["side"]
    side = Observer("side", so.azimuth_deg, so.elevation_deg, 30.0).offset()
    out = dict(kt=list(speeds), cockpit=[], cockpit_ask21=[], airborne=[], side=[], wind=[])
    for kt in speeds:
        air, st = T.cockpit_noise(X57_TYRES, kt * KT, ac.cockpit, ac.cabin.inside, "asphalt", "p2006t")
        _, st21 = T.cockpit_noise(X57_TYRES, kt * KT, ac.cockpit, ac.cabin.inside, "asphalt", "ask21")
        out["cockpit"].append(dba(air + st, ALL))
        out["cockpit_ask21"].append(dba(air + st21, ALL))
        out["airborne"].append(dba(air, ALL))
        out["side"].append(dba(T.tyre_noise(X57_TYRES, kt * KT, side), ALL))
        out["wind"].append(dba(wind(kt, ac), ALL))
    out["low_speed_power"] = T.low_speed_power(MAIN_6["width_m"], MAIN_6["diameter_m"], speeds[0] * KT)
    return out


def p2006t():
    fs, m = load("p2006t_ymen")
    before = thirds(m, fs, 384.5, 388.3)
    k = np.sum(wind(65)[REF]) / np.sum(before[REF])
    before = before * k
    after = thirds(m, fs, 389.0, 392.0) * k
    later = thirds(m, fs, 391.0, 394.0) * k
    out = dict(hz=fc[SEL].round().tolist())
    for kt in (55, 58, 62):
        out[f"half_{kt}kt"] = transfer(after / 2, power([MAIN_6, MAIN_6], kt))[SEL].round(1).tolist()
    out["later_three_wheels_54kt"] = transfer(later / 2, power([MAIN_6, MAIN_6, MAIN_5], 54))[SEL].round(1).tolist()
    out["module"] = T.structure_db(fc[SEL], "p2006t").round(1).tolist()
    out["cabin_dba"] = dict(approach_all=dba(before, ALL), approach_band=dba(before), after_band=dba(after),
                            wind_model_65kt=dba(wind(65), ALL))
    # one band against the others either side of the touchdown: the gain control can't move this
    bal = lambda p: float(10 * np.log10(np.sum(p[BAND]) / (np.sum(p[(fc >= 25) & (fc <= 100)]) + np.sum(p[REF]))))
    out["band_balance_rise_db"] = bal(after) - bal(before)
    lo, hi = (fc >= 25) & (fc <= 100), REF
    out["rise_vs_low_db"] = float(10 * np.log10(np.sum(after[BAND]) / np.sum(after[lo]) * np.sum(before[lo]) / np.sum(before[BAND])))
    out["rise_vs_high_db"] = float(10 * np.log10(np.sum(after[BAND]) / np.sum(after[hi]) * np.sum(before[hi]) / np.sum(before[BAND])))
    # what closing the throttles did to the same measure, five seconds before
    out["throttles_db"] = bal(thirds(m, fs, 380.0, 383.0)) - bal(before)
    # an idling engine's line, as a check that the gain held
    f, Pb = welch(m[int(384.5 * fs):int(388.3 * fs)], fs, nperseg=16384)
    _, Pa = welch(m[int(389.0 * fs):int(392.0 * fs)], fs, nperseg=16384)
    j = np.argmin(abs(f - 556.6))
    out["line_557hz_change_db"] = float(10 * np.log10(Pa[j] / Pb[j]))
    out["kurtosis_by_second_380_399"] = [round(texture(m[int(t * fs):int((t + 1) * fs)], fs), 2) for t in range(380, 399)]
    # the band balance through the landing, by half second (dB, relative to the 3.8 s before the touchdown)
    t = np.arange(376.0, 399.0, 0.5)
    b = [bal(thirds(m, fs, x, x + 0.5, 4096)) for x in t]
    ref = bal(thirds(m, fs, 384.5, 388.3, 4096))
    out["balance_t"], out["balance_db"] = t.tolist(), [round(v - ref, 2) for v in b]
    out["touchdown_s"] = 388.5
    return out


def touchdown():
    """the squeals in both recordings, absolute at each touchdown, their sound energy through each airframe's transfer
    against what the wheel loses spinning up; the P2006T's sink rate from its slide; the one thump"""
    out = {}
    sq = (fc >= 500) & (fc <= 2000)              # the transfers' measured bands

    def squeal(name, m, fs, t0, t1, bg, k, airframe, tyre_type, wheels, kt):
        p2 = np.maximum(thirds(m, fs, t0, t1, 2048) - bg, 0) * k
        W = p2 / 4e-10 * 1e-12 / 10 ** (T.structure_db(fc, airframe) / 10)
        E = float(np.sum(W[sq]) * (t1 - t0))
        lost = wheels * 0.5 * T.wheel_inertia(tyre_type) * (kt * KT / T.TYRE_TYPES[tyre_type]["rolling_radius_m"]) ** 2
        # the squeal's own spectrum: the cockpit's less the transfer where it was measured (to 2 kHz), above that
        # falling as the cockpit's does; bands with nothing over the background interpolated
        band = (fc >= 500) & (fc <= 8000)
        L = 10 * np.log10(np.maximum(p2[band], 1e-20))
        f = fc[band]
        ok = L > -100
        L = np.interp(np.log(f), np.log(f[ok]), L[ok])
        Tb = T.structure_db(f, airframe)
        src = np.where(f <= 2000, L - Tb, L - L[f <= 2000][-1] + (L - Tb)[f <= 2000][-1])
        src = src - 10 * np.log10(np.sum(10 ** (src / 10)))
        out[name] = dict(seconds=t1 - t0, cockpit_dba=dba(p2, band), sound_j=E, lost_j=lost,
                         efficiency_db=float(10 * np.log10(E / lost)), source_db=src.round(1).tolist())
    fs, m = load("p2006t_ymen")
    before = thirds(m, fs, 384.5, 388.3)
    k = np.sum(wind(65)[REF]) / np.sum(before[REF])
    squeal("p2006t", m, fs, 388.512, 388.613, thirds(m, fs, 384.5, 388.3, 2048), k, "p2006t", "6.00-6", 2, 58)
    # the sink rate that makes a main wheel's slide last as long, on the X-57's strut (the same airframe); and how much
    # that depends on how fast the wing sheds the weight
    def sink(settle_s):
        durs = {fps: T.spin_up(58 * KT, "6.00-6", 8071.0, 76296.0, fps * 0.3048, 5671.0, settle_s=settle_s,
                               mass_kg=X57_MASS["main"])[0][-1] for fps in np.arange(0.2, 4.01, 0.02)}
        return round(float(min(durs, key=lambda x: abs(durs[x] - 0.101))), 2)
    out["p2006t"]["sink_fps"] = sink(T.SETTLE_S)
    out["p2006t"]["sink_fps_settling"] = {f"{s:g}": sink(s) for s in (0.1, 1.0, 3.0)}
    # the squeal's envelope, the line's band by 10 ms over the approach's, against the model's load and its power
    sos = butter(4, [900, 1500], "bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, m[int(384.0 * fs):int(389.0 * fs)]) ** 2
    bg = y[int(0.5 * fs):int(4.3 * fs)].mean()
    rec = np.array([y[int((t - 384.0) * fs):int((t + 0.01 - 384.0) * fs)].mean() - bg for t in 388.51 + 0.01 * np.arange(10)])
    tt, slip, P = T.spin_up(58 * KT, "6.00-6", 8071.0, 76296.0, out["p2006t"]["sink_fps"] * 0.3048, 5671.0,
                            mass_kg=X57_MASS["main"])
    bins = lambda e: np.array([e[(tt >= a) & (tt < a + 0.01)].mean() for a in 0.01 * np.arange(10)])
    rms = lambda e: float(np.sqrt(np.mean((10 * np.log10(bins(e) / bins(e).mean() / (rec / rec.mean()))) ** 2)))
    out["p2006t"]["envelope"] = dict(recorded=(rec / rec.mean()).round(2).tolist(), load_rms_db=rms(P / slip),
                                     power_rms_db=rms(P))
    # the mains' thump: what the 0.15 s from their touchdown hold over the approach's 0.15 s windows, band by band (the
    # sink rate known from the slide); and the later event, 1.8 s on, against the same
    lo = (fc >= 45) & (fc <= 700)
    win = np.array([thirds(m, fs, t, t + 0.15, 4096) for t in np.arange(385.0, 388.3, 0.15)]) * k
    mean, sd = win.mean(0), (10 * np.log10(np.maximum(win, 1e-30))).std(0)
    td = thirds(m, fs, 388.50, 388.65, 4096) * k
    own = mean * np.maximum(td / mean - 1.0, 0.0)
    ev = np.maximum(thirds(m, fs, 390.26, 390.41, 4096) - thirds(m, fs, 389.0, 390.1, 4096), 0) * k
    db = lambda x: (10 * np.log10(np.maximum(x, 1e-20) / 4e-10))[lo].round(1).tolist()
    out["thump"] = dict(hz=fc[lo].round().tolist(), mains=db(own), rise_db=(10 * np.log10(td / mean))[lo].round(1).tolist(),
                        scatter_db=sd[lo].round(1).tolist(), cabin=db(mean), later_event=db(ev),
                        force_n=8071.0 * out["p2006t"]["sink_fps"] * 0.3048)
    fs2, m2 = load("ask21b_landing")
    g = G.load_glider("ask21")
    head = G.to_body(*g.cockpit["front"])
    a47 = (lambda sp: sp["wind"] + sp["airframe"])(
        G.spectra(g, G.GliderCondition(airspeed_kt=47, load_factor=1.0, turbulence_ms=0.5), head, True)[0])
    k2 = np.sum(a47[REF]) / np.sum(thirds(m2, fs2, 586.0, 594.3)[REF])
    bg2 = thirds(m2, fs2, 592.5, 594.3, 2048)
    squeal("ask21_touch1", m2, fs2, 594.428, 594.503, bg2, k2, "ask21", "5.00-5", 1, 47)
    squeal("ask21_touch2", m2, fs2, 594.769, 594.94, bg2, k2, "ask21", "5.00-5", 1, 47)
    # a skip spins the wheel up in two goes: together they lose what one slide would
    E = out["ask21_touch1"]["sound_j"] + out["ask21_touch2"]["sound_j"]
    out["ask21"] = dict(sound_j=E, lost_j=out["ask21_touch1"]["lost_j"],
                        efficiency_db=float(10 * np.log10(E / out["ask21_touch1"]["lost_j"])))
    out["module_efficiency_db"] = T.SQUEAL_EFFICIENCY_DB
    out["tonal_share"] = [0.47, 0.62]
    # the X-57's: a main wheel and the nose wheel at 60 kt, by sink rate; exposures (dB(A) over the event)
    ac = load_prop_aircraft("x57")
    Tx = T.structure_db(fc, "p2006t")
    sel = lambda x: float(10 * np.log10(np.sum(x ** 2) / 48000.0 / 4e-10))
    from pywopwop.synth import a_weighted_ms
    sel_a = lambda x: float(10 * np.log10(a_weighted_ms(x, 48000.0) * len(x) / 48000.0 / 4e-10))
    struts = dict(main=("6.00-6", 8071.0, 76296.0, 5671.0), nose=("5.00-5", 2846.0, 26926.0, 2002.0))
    x57 = {}
    for name, (ty, c, k, N) in struts.items():
        m = X57_MASS[name]
        slides = [T.spin_up(60 * KT, ty, c, k, fps * 0.3048, N, mass_kg=m)[0][-1] for fps in (0.25, 4.0)]
        ck, E, _ = T.squeal_shot(ty, 60 * KT, c, k, 0.3048, N, transfer_db=Tx, mass_kg=m)
        sd, _, _ = T.squeal_shot(ty, 60 * KT, c, k, 0.3048, N, distance_m=30.0, mass_kg=m)
        x57[name] = dict(mass_kg=m, slide_s=slides, sound_mj=E * 1e3, cockpit_dba=sel_a(ck), side_dba=sel_a(sd),
                         thump_1fps_dba=sel_a(T.thump_shot(c, 0.3048)), thump_1fps_db=sel(T.thump_shot(c, 0.3048)))
    out["x57"] = x57
    return out


def results():
    res = dict(outside=outside(), ask21=ask21(), p2006t=p2006t(), x57=x57(), touchdown=touchdown())
    loop = T.noise_loop(T.tyre_power(0.16, 0.4445, 30.0) * T.RHO_C / (2 * np.pi), duration=4.0)
    res["loop_kurtosis"] = round(texture(loop, 48000.0), 2)
    return res


def main():
    res = results()
    o = res["outside"]
    print(f"outside: reference {o['reference_1m']['model']:.1f} dB(A) at 1 m (measured {o['reference_1m']['measured']:.1f}); "
          f"car {o['car_80kmh_7m5']['model']:.1f} (label about 70); towed aircraft {o['towed_aircraft_55mph_30m']['model_smooth']:.1f} "
          f"on smooth asphalt (measured 60)")
    for name in ("ask21", "p2006t", "x57", "touchdown"):
        print(f"\n{name}:")
        for key, v in res[name].items():
            print(f"  {key}: {v}")
    print(f"\nthe model's loop: kurtosis {res['loop_kurtosis']}")
    path = os.path.join(ROOT, "report", "results", "tyres.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
