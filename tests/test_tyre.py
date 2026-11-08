import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pywopwop import tyre as T
from pywopwop.glider import BANDS

A = 10.0 ** (T.a_weight(BANDS) / 10.0)


def dba(p2):
    return 10 * np.log10(np.sum(p2 * A) / 4e-10)


def test_reference_reproduced():
    """the truck tyre it is scaled from comes back at its own speed, size and surface, 1 m away"""
    r = T.REFERENCE
    W = T.tyre_power(r["width_m"], r["diameter_m"], r["kmh"] / 3.6, r["surface"])
    p2 = W * T.RHO_C / (2 * np.pi)
    measured = 10 * np.log10(np.sum(10 ** (np.asarray(r["dba"]) / 10)))
    assert abs(dba(p2) - measured) < 1.0


def test_aircraft_measurement():
    """three 5.00-5 tyres towed at 55 mph on smooth asphalt, 30.48 m to the side: about 60 dB(A) (US 11198519)"""
    v = 55 * 0.44704
    tyres = [dict(position=(x, y, 0.0), width_m=0.1257, diameter_m=0.3607) for x, y in ((1.5, 0), (-0.5, -1), (-0.5, 1))]
    assert abs(dba(T.tyre_noise(tyres, v, (0.0, -30.48, 1.2), "smooth asphalt")) - 60.0) < 3.0


def test_speed_and_size():
    W = lambda v, w=0.16, d=0.4445: np.sum(T.tyre_power(w, d, v) * A)
    assert W(30.0) > W(15.0) * 2 ** 2      # faster than the speed squared, the air's part as the fourth power
    assert W(30.0) < W(15.0) * 2 ** 4.5
    assert W(20.0, w=0.32) == pytest.approx(2 * W(20.0))
    assert np.all(T.tyre_power(0.16, 0.4445, 0.0) == 0)
    # a smaller tyre's spectrum sits higher
    f = lambda d: np.sum(BANDS * T.tyre_power(0.16, d, 20.0)) / np.sum(T.tyre_power(0.16, d, 20.0))
    assert f(0.36) > f(0.9)


def test_structure_transfer():
    for name, st in T.STRUCTURES.items():
        assert T.structure_db(st["hz"], name) == pytest.approx(st["db"])
        assert T.structure_db(63.0, name) == pytest.approx(st["db"][0] - 12.0, abs=0.1)
        assert T.structure_db(4000.0, name) == pytest.approx(st["db"][-1] - 12.0, abs=0.1)
    # the glider's wheel, unsprung under the seat, puts more into its cockpit than the P2006T's sprung gear
    hi = np.asarray(T.STRUCTURES["ask21"]["hz"]) > 240
    assert np.all(np.asarray(T.STRUCTURES["ask21"]["db"])[hi] > np.asarray(T.STRUCTURES["p2006t"]["db"])[hi] + 3)
    head = (0.0, 0.0, 0.8)
    t = [dict(position=(0.0, 0.0, -1.0), width_m=0.1257, diameter_m=0.3607)]
    air, structure = T.cockpit_noise(t, 20.0, head, walls=lambda f: np.full(len(f), 1e-3))
    assert dba(structure) > dba(air) + 20


def test_noise_loop_levels_and_seam():
    p2 = T.tyre_power(0.16, 0.4445, 20.0) * T.RHO_C / (2 * np.pi)
    x = T.noise_loop(p2, duration=2.0, seed=3)
    assert 10 * np.log10(np.mean(x ** 2) / np.sum(p2[BANDS > 25])) == pytest.approx(0.0, abs=0.5)
    # seamless: the jump across the loop's ends is no bigger than one between neighbours
    assert abs(x[0] - x[-1]) < 6 * np.std(np.diff(x))


def test_bank_and_xml(tmp_path):
    from pywopwop.propeller import load_prop_aircraft
    ac = load_prop_aircraft("x57")
    tyres = [dict(name="nose", gear=0, position=(2.46, 0.0, -1.17), width_m=0.1257, diameter_m=0.3607,
                  static_compression_ft=0.244),
             dict(name="left_main", gear=1, position=(-0.44, 1.0, -1.17), width_m=0.16, diameter_m=0.4445,
                  static_compression_ft=0.244)]
    m = T.make_tyre_bank(ac, tmp_path, tyres, dict(cockpit=-30.0, exterior=-20.0), "p2006t", speeds=(5.0, 30.0, 60.0),
                         duration=1.0, progress=lambda s: None)
    assert len(m["files"]) == 2 * 3 * 2
    assert all((tmp_path / f["file"]).exists() for f in m["files"])
    cockpit = [f for f in m["files"] if f["view"] == "cockpit"]
    assert all(f["structure_dba"] > f["airborne_dba"] for f in cockpit)
    assert 2.0 < m["low_speed_power"]["nose"] < 4.0
    root = ET.fromstring((tmp_path / "pywopwop-tyres.xml").read_text())
    _comparisons_simgear_takes(root)
    entries = root.find("fx").findall("tyre")
    assert len(entries) == len(m["files"])
    for e in entries:
        c = ET.tostring(e.find("condition"), encoding="unicode")
        assert "wow" in c and "greater-than" in c
    assert json.loads((tmp_path / "manifest.json").read_text())["speeds_kt"] == [5.0, 30.0, 60.0]


MAIN = dict(damping=8071.0, spring=76296.0, static_load_n=5671.0)     # the X-57's main strut


def test_spin_up_loses_half_the_work():
    """whatever the load does, the slide loses what the wheel gains: I omega^2 / 2; a firmer touchdown slides shorter"""
    I, r = T.wheel_inertia("6.00-6"), T.TYRE_TYPES["6.00-6"]["rolling_radius_m"]
    V = 30.0
    durations = []
    for fps in (0.25, 1.0, 4.0):
        t, slip, P = T.spin_up(V, "6.00-6", MAIN["damping"], MAIN["spring"], fps * 0.3048, MAIN["static_load_n"])
        assert np.sum(P) * (t[1] - t[0]) == pytest.approx(0.5 * I * (V / r) ** 2, rel=0.01)
        assert slip[-1] < 0.01 * V
        durations.append(t[-1])
    assert durations[0] > durations[1] > durations[2]
    # the recorded P2006T touchdown: 58 kt, its slide 0.101 s, at 0.78 fps on these struts
    t, _, _ = T.spin_up(58 * 0.514444, "6.00-6", MAIN["damping"], MAIN["spring"], 0.78 * 0.3048, MAIN["static_load_n"])
    assert t[-1] == pytest.approx(0.101, abs=0.005)


def test_squeal_energy():
    """the squeal outside carries the efficiency's share of the spin-up's loss, the same at any sink rate"""
    fs, r = 48000.0, 30.0
    for fps in (0.5, 3.0):
        x, E, dur = T.squeal_shot("6.00-6", 30.0, MAIN["damping"], MAIN["spring"], fps * 0.3048, MAIN["static_load_n"],
                                  distance_m=r, fs=fs)
        lost = 0.5 * T.wheel_inertia("6.00-6") * (30.0 / T.TYRE_TYPES["6.00-6"]["rolling_radius_m"]) ** 2
        assert E == pytest.approx(10 ** (T.SQUEAL_EFFICIENCY_DB / 10) * lost, rel=0.01)
        heard = np.sum(x ** 2) / fs * 2 * np.pi * r ** 2 / T.RHO_C
        assert 10 * np.log10(heard / E) == pytest.approx(0.0, abs=1.5)


def test_thump_scales_with_force():
    a, b = T.thump_shot(8071.0, 0.3), T.thump_shot(8071.0, 0.6)
    assert 10 * np.log10(np.sum(b ** 2) / np.sum(a ** 2)) == pytest.approx(6.0, abs=1.5)


def _eval(e, P):
    """a SimGear expression or condition, as the sound system evaluates it"""
    t, k = e.tag, list(e)
    if t == "value":
        return float(e.text)
    if t == "property":
        return P[e.text]
    if t in ("expression", "condition"):
        return _eval(k[0], P)
    if t == "product":
        return float(np.prod([_eval(c, P) for c in k]))
    if t == "sum":
        return sum(_eval(c, P) for c in k)
    if t == "clip":
        x = _eval([c for c in k if c.tag not in ("clipMin", "clipMax")][0], P)
        return min(max(x, float(e.find("clipMin").text)), float(e.find("clipMax").text))
    if t == "and":
        return all(_eval(c, P) for c in k)
    if t == "not":
        return not _eval(k[0], P)
    if t in ("greater-than", "less-than"):
        # SimGear's conditions compare a property with a property or a value, nothing computed (condition.cxx)
        assert [c.tag for c in k][0] == "property" and k[1].tag in ("property", "value"), [c.tag for c in k]
        a, b = _eval(k[0], P), _eval(k[1], P)
        return a > b if t == "greater-than" else a < b
    raise KeyError(t)


def _comparisons_simgear_takes(root):
    """every comparison in every condition: a property on the left, a property or a value on the right"""
    for cond in root.iter("condition"):
        for c in cond.iter():
            if c.tag in ("greater-than", "less-than", "equals", "greater-than-equals", "less-than-equals"):
                kids = [k.tag for k in c]
                assert kids[0] == "property" and kids[1] in ("property", "value") and len(kids) == 2, kids


def test_touchdown_bank(tmp_path):
    """each wheel's files are one waveform under different envelopes; cross-faded by amplitude they keep the law's level
    anywhere on the grid; two wheels touching together are two tyres"""
    import scipy.io.wavfile as wavfile
    from pywopwop.propeller import load_prop_aircraft
    from pywopwop.synth import a_weighted_ms
    groups = [dict(name="main", tyre_type="6.00-6", wheels=[dict(gear=1, position=(-0.44, 1.0, -1.17)),
                                                            dict(gear=2, position=(-0.44, -1.0, -1.17))], **MAIN)]
    gain = dict(cockpit=-30.0, exterior=-20.0)
    m = T.make_touchdown_bank(load_prop_aircraft("x57"), tmp_path, groups, gain, "p2006t",
                              slips=(45.0, 75.0), sinks=(0.5, 2.0), progress=lambda s: None)
    assert len(m["files"]) == 2 * (2 * 2 * 2 + 2)
    files = {f["file"]: f for f in m["files"]}
    for kind, axis in (("squeal", "slip_kt"), ("thump", "sink_fps")):
        same = [f for f in m["files"] if f["kind"] == kind and f["view"] == "cockpit" and f["gear"] == 1]
        K = [f["law_dba"] - 20 * np.log10(f[axis]) for f in same]
        assert max(K) - min(K) < 0.01           # law_dba is kept to two decimals
    root = ET.fromstring((tmp_path / "pywopwop-touchdown.xml").read_text())
    _comparisons_simgear_takes(root)
    entries = root.find("fx").findall("touchdown")
    assert all(e.find("mode").text == "once" for e in entries)

    def heard(slip, sink, gear):
        P = {"sim/current-view/internal": True}
        for n in (1, 2):
            P.update({f"/fdm/jsbsim/gear/unit[{n}]/touchdown-sink-fps": sink, f"/gear/gear[{n}]/wow": True,
                      f"/fdm/jsbsim/gear/unit[{n}]/touchdown-slip-kt": slip})
        x = np.zeros(48000)
        for e in entries:
            name = e.find("name").text
            if f"_g{gear}_squeal" in name and _eval(e.find("condition"), P):
                fn = e.find("path").text.split("/")[-1]
                y = wavfile.read(tmp_path / fn)[1].astype(float) / 32768 / 10 ** (gain["cockpit"] / 20)
                x[:len(y)] += _eval(e.find("volume/expression"), P) * y
        return x
    K = np.mean([f["law_dba"] - 20 * np.log10(f["slip_kt"]) for f in files.values()
                 if f["kind"] == "squeal" and f["view"] == "cockpit" and f["gear"] == 1])
    for slip, sink in ((45.0, 0.5), (58.0, 1.0), (75.0, 2.0), (52.0, 1.6)):
        x = heard(slip, sink, 1)
        assert 10 * np.log10(a_weighted_ms(x, 48000.0) / 4e-10) == pytest.approx(K + 20 * np.log10(slip), abs=1.0)
    l, r = heard(58.0, 1.0, 1), heard(58.0, 1.0, 2)
    assert abs(np.corrcoef(l, r)[0, 1]) < 0.3
    # thumps at their estimate where they fit, through the knee past full scale, a firmer one still louder
    th = sorted((f for f in m["files"] if f["kind"] == "thump" and f["gear"] == 1), key=lambda f: f["sink_fps"])
    for f in th:
        assert f["limited_db"] < 0.05 if f["peak_over_db"] <= 0 else f["limited_db"] > 0
    assert all(b["played_dba"] > a["played_dba"] + 1.0 for a, b in zip(th, th[1:]))
