"""Turbine engines: the T700's cycle against GE's, the sources' scaling, the loops and the FlightGear section."""
from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pywopwop import engine as en
from pywopwop.model import Observer


def test_t700_cycle_matches_ge():
    e = en.load_engine("t700")
    # GE's nonlinear simulation (NASA TM-105805 table 1): Ng %, compressor delivery psi, power turbine inlet R
    for ng, ps3, t45 in ((87.5, 132.0, 1472.0), (91.4, 161.7, 1578.0), (94.5, 188.9, 1675.0), (96.7, 212.3, 1778.0),
                         (98.5, 231.5, 1896.0), (99.5, 240.5, 1954.0)):
        s = en.engine_state(e, ng)
        assert abs(s.pr / (ps3 / 14.696) - 1.0) < 0.02
        assert abs(s.T45 - t45 / 1.8) < 80.0
    d = en.engine_state(e, 99.5)
    assert 4.3 < d.W < 5.3                          # the T700's air flow, about 10 lb/s
    assert 1.3 < d.m_rel < 1.5                      # a transonic first stage


def test_cold_engine_and_idle():
    e = en.load_engine("t700")
    cold = en.engine_state(e, 24.0, lit=False)
    assert cold.fuel_pph == 0 and cold.T4 == cold.T3 and cold.power_w == 0
    idle = en.engine_state(e, e.idle_ng)
    assert 350.0 < idle.T45 - 273.15 < 650.0 and idle.power_w == 0.0


def test_compressor_tones_and_buzz_saw():
    e = en.load_engine("t700")
    B = e.compressor["rotor_blades"]
    lo = en.compressor_noise(e, en.engine_state(e, 65.0), 30.0)
    f_shaft = e.ng_rpm * 0.65 / 60.0
    B2 = e.compressor["stage2"]["rotor_blades"]
    assert abs(lo.tones[0][0] - B * f_shaft) < 1e-6           # the blade-passing frequencies
    assert abs(lo.tones[6][0] - B2 * f_shaft) < 1e-6
    assert len(lo.tones) == 12                                 # subsonic tips: no combination tones
    assert lo.tones[6][1] < 0.1 * lo.tones[0][1]               # the second stage heard through the first
    hi = en.compressor_noise(e, en.engine_state(e, 99.5), 30.0)
    shaft = e.ng_rpm * 0.995 / 60.0
    buzz = [f for f, _ in hi.tones[6:-6]]
    assert buzz and all(abs(f / shaft - round(f / shaft)) < 1e-9 and f < B * shaft for f in buzz)
    assert hi.total() > 5 * lo.total()


def test_core_and_turbine_follow_the_power():
    e = en.load_engine("t700")
    s1, s2 = en.engine_state(e, 80.0), en.engine_state(e, 99.5)
    c1, c2 = en.combustion_noise(e, s1, 120.0).total(), en.combustion_noise(e, s2, 120.0).total()
    assert 5.0 < 10 * np.log10(c2 / c1) < 20.0
    assert en.combustion_noise(e, s2, 120.0).total() > en.combustion_noise(e, s2, 30.0).total()   # aft
    assert en.turbine_noise(e, en.engine_state(e, 65.0), 120.0).total() == 0.0     # no power at idle
    assert en.combustion_noise(e, en.engine_state(e, 40.0, lit=False), 120.0).total() == 0.0
    st = [en.starter_noise(e, n).total() for n in (10.0, 30.0, 55.0)]
    assert max(st) / min(st) < 2.0                  # the choked supply keeps its loading
    open_st = en.starter_noise(replace(e, starter={**e.starter, "cowl_surface_mass": None}), 30.0)
    assert open_st.total() > 10 * en.starter_noise(e, 30.0).total()     # the cowling keeps its whine in


def test_engine_loop_is_seamless_and_layered():
    inst = en.load_installations("uh60a")[1]
    r = en.render_engine(inst, en.EngineCondition(40.0, starter=True), Observer("side", 90.0, 0.0, 50.0),
                         duration=2.0, seed=3)
    assert {"layer/compressor", "layer/core", "layer/starter"} <= set(r.signals)
    x = r.total
    step = np.abs(np.diff(np.concatenate([x, x[:1]])))
    assert step[-1] < 6 * np.median(step) + 1e-12   # the wrap is no louder a step than the rest
    assert np.allclose(sum(v for k, v in r.signals.items() if k.startswith("layer/")), x)


def test_ae1107c_cycle():
    e = en.load_engine("ae1107c")
    d = en.engine_state(e, e.design_ng)
    assert abs(d.T4 - 273.15 - 1209.0) < 15.0       # the published turbine inlet temperature
    assert 14.0 < d.W < 18.0 and abs(d.power_w / 745.7 - 6150.0) < 5.0
    temps = [en.engine_state(e, n).T45 for n in (78.6, 85.0, 90.5, 95.0, 100.0)]
    assert all(a < b for a, b in zip(temps, temps[1:]))     # hotter with the power, never past the rating's
    assert max(temps) < d.T45 + 10.0
    # the flight model's n1: 60 + 45 sqrt(P / 6150) %
    assert abs(en.engine_state(e, 90.0).power_w / 745.7 - 6150.0 * (30.0 / 45.0) ** 2) < 5.0


def test_tiltrotor_engines_turn_with_the_nacelles():
    heli, plane = en.load_installations("mv22b"), en.load_installations("mv22b", 0.0)
    for h, a in zip(heli, plane):
        assert np.allclose(h.inlet_axis, (0, 0, 1)) and np.allclose(a.inlet_axis, (1, 0, 0))
        assert np.allclose(a.exhaust_axis, (-1, 0, 0))
        pv = np.asarray(h.pivot)
        for p, q in ((h.inlet, a.inlet), (h.exhaust, a.exhaust)):
            assert abs(np.linalg.norm(np.subtract(p, pv)) - np.linalg.norm(np.subtract(q, pv))) < 1e-9
        assert a.inlet[0] > a.exhaust[0]                      # the inlet ahead in airplane mode
        assert h.n2_property.endswith("drive/nr-pct")
        assert h.inside_db == 0.0 and a.inside_db < -10.0     # the measured loss to the cockpit in airplane mode
    assert en.load_installations("uh60a", 0.0)[0].inlet == en.load_installations("uh60a")[0].inlet
    a = plane[0]
    cock, side = Observer("cockpit", position=(4.28, -0.57, -3.15)), Observer("side", 90.0, 0.0, 50.0)
    c = en.EngineCondition(92.0)
    for o in (cock, side):
        loud = en.render_engine(replace(a, inside_db=0.0), c, o, duration=1.0, seed=2).total
        quiet = en.render_engine(a, c, o, duration=1.0, seed=2).total
        gain = 20 * np.log10(np.std(quiet) / np.std(loud))
        assert abs(gain - (a.inside_db if o.position is not None else 0.0)) < 1e-9     # inside only


def test_flightgear_engine_section(tmp_path):
    m = dict(aircraft="X", exterior_distance_m=50.0, lit_ng=[65.0, 100.0], cold_ng=[10.0, 30.0],
             starter_ng=[10.0, 30.0],
             engines=[dict(name="left", properties="/engines/engine[0]", starter_property="/s0", idle_ng=65.0,
                           lightoff_ng=15.0)],
             files=[dict(file=f"f{i}.wav", engine="left", layer=l, state=s, ng=n, observer=v, fuel_pph=100.0)
                    for i, (l, s, n, v) in enumerate([(l, s, n, v) for v in ("cockpit", "side")
                                                      for l, s, g in (("compressor", "lit", (65.0, 100.0)),
                                                                      ("compressor", "cold", (10.0, 30.0)),
                                                                      ("core", "lit", (65.0, 100.0)),
                                                                      ("turbine", "lit", (100.0,)),
                                                                      ("starter", "starter", (10.0, 30.0)))
                                                      for n in g])])
    xml = en.flightgear_engine_xml(m)
    root = ET.fromstring(xml)
    sounds = root.find("fx").findall("engine")
    assert len(sounds) == len(m["files"])
    names = [s.find("name").text for s in sounds]
    assert len(set(names)) == len(names)
    turbine = next(s for s in sounds if "turbine" in s.find("name").text)
    assert "n2-pct" in ET.tostring(turbine.find("pitch")).decode()
    assert all(s.find("pitch") is None for s in sounds if "-core-" in s.find("name").text)   # the combustor's
    assert any("/s0" in ET.tostring(s.find("volume")).decode() for s in sounds)
    # a tiltrotor's airplane-mode bank: its own names, and the power turbine pitched by the rotors' speed
    m2 = dict(m, mode="airplane", engines=[dict(m["engines"][0], n2_property="/fdm/jsbsim/drive/nr-pct")])
    root2 = ET.fromstring(en.flightgear_engine_xml(m2))
    names2 = [s.find("name").text for s in root2.find("fx").findall("engine")]
    assert all(n.startswith("pww-airplane-") for n in names2) and not set(names2) & set(names)
    turbine2 = next(s for s in root2.find("fx").findall("engine") if "turbine" in s.find("name").text)
    assert "drive/nr-pct" in ET.tostring(turbine2.find("pitch")).decode()
    # the power turbine's level by its own speed (Smith and Bushell's tip Mach number cubed), off where it is 60 dB down,
    # and its pitch held only where it is off
    for t, prop in ((turbine, "/engines/engine[0]/n2-pct"), (turbine2, "/fdm/jsbsim/drive/nr-pct")):
        pw = t.find("volume/expression/product/pow")
        assert pw.find("div/property").text == prop and float(pw.findall("value")[0].text) == en.NP_EXPONENT
        gate = [g for g in t.find("condition/and").findall("greater-than") if g.findtext("property") == prop]
        assert len(gate) == 1 and abs(float(gate[0].findtext("value")) - 1.0) < 1e-9
        assert 20 * en.NP_EXPONENT * np.log10(en.NP_GATE_PCT / 100.0) == pytest.approx(-60.0)
        assert float(t.find("pitch").findtext("min")) == pytest.approx(en.NP_GATE_PCT * 0.01)
    assert not any(s.find("volume/expression/product/pow") is not None for s in sounds if "-turbine-" not in
                   s.find("name").text)
