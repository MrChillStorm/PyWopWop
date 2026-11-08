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


def results():
    res = dict(outside=outside(), ask21=ask21(), p2006t=p2006t(), x57=x57())
    loop = T.noise_loop(T.tyre_power(0.16, 0.4445, 30.0) * T.RHO_C / (2 * np.pi), duration=4.0)
    res["loop_kurtosis"] = round(texture(loop, 48000.0), 2)
    return res


def main():
    res = results()
    o = res["outside"]
    print(f"outside: reference {o['reference_1m']['model']:.1f} dB(A) at 1 m (measured {o['reference_1m']['measured']:.1f}); "
          f"car {o['car_80kmh_7m5']['model']:.1f} (label about 70); towed aircraft {o['towed_aircraft_55mph_30m']['model_smooth']:.1f} "
          f"on smooth asphalt (measured 60)")
    for name in ("ask21", "p2006t", "x57"):
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
