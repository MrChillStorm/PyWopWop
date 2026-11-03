"""Turbine engines: the T700's cycle against GE's, the sources' scaling, the loops and the FlightGear section."""
from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np

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
