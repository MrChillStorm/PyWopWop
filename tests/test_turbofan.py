"""Turbofans: the AE 3007H's state against its cycle's reference row, the fan's discharge against Heidmann's formulas
worked by hand, the loops and the FlightGear section."""
import xml.etree.ElementTree as ET

import numpy as np

from pywopwop import engine as en
from pywopwop import turbofan as tf
from pywopwop.atmosphere import Atmosphere
from pywopwop.model import Observer


def test_reference_state_from_thrust_and_nozzle():
    e = tf.load_turbofan("ae3007h")
    r = e._ref
    # the Global Hawk flight model's cycle at the AE 3007A2's take-off row: 129 kg/s, fan pressure ratio 1.758
    assert abs(r.W / 129.0 - 1.0) < 0.10
    assert abs(r.fpr - 1.758) < 0.10          # with the duct's assumed 5 % loss; without it 1.60
    assert abs(r.opr - 20.2) < 1.0 and abs(r.T4 - 1380.0) < 20.0
    assert 1.25 < r.m_tip < 1.35              # 445 m/s at 8,700 rpm on 0.978 m


def test_thrust_lapse_keeps_the_fan_loaded_at_altitude():
    e = tf.load_turbofan("ae3007h")
    lo = tf.turbofan_state(e, 33362.0, 100.0)
    hi = tf.turbofan_state(e, 2945.0, 100.0, 0.6, Atmosphere(60000 * 0.3048))
    assert hi.W < 0.15 * lo.W                 # the thin air
    assert hi.m_rel > lo.m_rel                # the cold air: a higher tip Mach number at the same speed
    assert 1.3 < hi.fpr < 2.0


def test_discharge_levels_are_heidmanns():
    e = tf.load_turbofan("ae3007h")
    s = tf.turbofan_state(e, 20000.0, 75.0)
    assert s.m_rel < 1.05
    th = 130.0                                # the broadband's peak
    src = tf.fan_discharge(e, s, th)
    base = 10 * np.log10((s.dT_fan * 1.8) ** 2 * s.W / en.LB)
    md = 20 * np.log10(e._ref.m_rel)
    over = 20 * np.log10(max(s.m_rel, 1.0))
    # the overall broadband: the peak band's level plus the spectrum's sum over the bands
    bpf = 24 * 8700 * 0.75 / 60
    r = en.BANDS / bpf / 2
    shape = -np.where(r < 1, 2.445096095, 13.97197769) * np.log(r) ** 2
    want = base + 58.0 + md - over - 5 * np.log10(150 / 300) + 10 * np.log10(np.sum(10 ** (0.1 * shape)))
    got = 10 * np.log10(src.bands.sum() / en.P_REF ** 2)
    assert abs(got - want) < 0.2
    # the fundamental at the tones' peak (120 deg), cut off below a sonic tip: 8 dB down
    t = tf.fan_discharge(e, s, 120.0).tones[0]
    assert abs(t[0] - bpf) < 1e-6
    want_t = base + 60.0 + md - over - 10 * np.log10(150 / 300) - 8.0
    assert abs(10 * np.log10(t[1] / en.P_REF ** 2) - want_t) < 0.2
    # nothing forward of the nozzle to speak of: 41.6 dB down on the axis
    ahead = tf.fan_discharge(e, s, 0.0).bands.sum()
    assert abs(10 * np.log10(src.bands.sum() / ahead) - 41.6) < 0.2


def test_buzz_saw_only_with_a_supersonic_tip():
    e = tf.load_turbofan("ae3007h")
    slow = tf.fan_inlet(e, tf.turbofan_state(e, 6000.0, 55.0), 40.0)
    fast = tf.fan_inlet(e, tf.turbofan_state(e, 33362.0, 100.0), 40.0)
    f1 = 8700 / 60.0
    orders = lambda src: {round(f / (f1 * (0.55 if src is slow else 1.0))) for f, _ in src.tones}
    assert not any(o < 24 for o in orders(slow))          # the blade passing tones and the compressor's only
    assert sum(1 for o in orders(fast) if o < 24) >= 8     # the shaft's orders under the blade passing tone


def test_loops_are_seamless_and_altitude_quiets_them():
    jet = tf.load_jet("globalhawk")
    o = Observer("side", 90.0, 0.0, 50.0)
    lo = tf.render_jet(jet, 100.0, 0.0, 0.0, 33362.0, o, 16000.0, 3.0)
    hi = tf.render_jet(jet, 100.0, 60000.0, 0.6, 2945.0, o, 16000.0, 3.0)
    x = lo.signals["layer/fan"]
    assert abs(x[0] - x[-1]) <= 1.5 * np.max(np.abs(np.diff(x)))
    assert lo.levels()["layer/fan"]["oaspl"] - hi.levels()["layer/fan"]["oaspl"] > 10.0
    inside = Observer("inside", position=tuple(jet.cockpit))
    a, b = (tf.render_jet(jet, 100.0, alt, m, f, inside, 16000.0, 3.0).levels() for alt, m, f in ((0.0, 0.0, 33362.0), (60000.0, 0.6, 2945.0)))
    assert a["layer/fan"]["oaspl"] - b["layer/fan"]["oaspl"] > 30.0       # the thin air and the wall both
    assert abs(a["layer/structure"]["oaspl"] - b["layer/structure"]["oaspl"]) < 3.0     # the airframe's path is not the air's


def test_flightgear_section(tmp_path):
    m = tf.make_jet_bank("globalhawk", tmp_path, fs=8000.0, duration=1.0, progress=lambda *_: None)
    root = ET.parse(tmp_path / "pywopwop-jet.xml").getroot()
    names = [s.find("name").text for s in root.find("fx")]
    assert len(names) == len(set(names)) == len(m["files"])
    for s in root.find("fx"):
        assert (tmp_path / s.find("path").text.split("/")[-1]).exists()
        assert (s.find("pitch") is None) == ("-core-" in s.find("name").text)
        # a FlightGear condition takes properties and values only
        for c in s.find("condition").iter():
            assert c.tag in ("condition", "and", "not", "property", "value", "greater-than", "less-than")
