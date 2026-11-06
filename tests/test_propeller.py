"""Propeller aircraft: the X-57's discs, against NASA's tables and its tunnel, and the section's fades."""
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest

from pywopwop import propeller as P, synth
from pywopwop.aero import AeroOptions
from pywopwop.atmosphere import Atmosphere
from pywopwop.model import KT, Observer

QUICK = AeroOptions(n_radial=20, n_azimuth=60, bvi=False, elastic=False)


@pytest.fixture(scope="module")
def ac():
    return P.load_prop_aircraft("x57")


def test_the_fourteen_discs(ac):
    assert len(ac.rotors) == 14 and sum(len(v) for v in ac.groups.values()) == 14
    at = {r.name: r for r in ac.rotors}
    # disc 7 inboard on the right, 12 outboard, 6 and 1 their mirrors (y left in the aircraft frame)
    assert at["hl_7"].position[1] < 0 and at["hl_12"].position[1] < at["hl_7"].position[1]
    assert at["hl_6"].position[1] == pytest.approx(-at["hl_7"].position[1])
    assert at["hl_1"].position[1] == pytest.approx(-at["hl_12"].position[1])
    # neighbours counter-rotate; each contactor's group turns one way
    assert at["hl_7"].rotation != at["hl_8"].rotation and at["hl_1"].rotation != at["hl_2"].rotation
    for g, names in ac.groups.items():
        assert len({at[n].rotation for n in names}) == 1
    assert ac.groups["hl_a_left"] == ["hl_1", "hl_3", "hl_5"] and ac.groups["hl_b_right"] == ["hl_7", "hl_9", "hl_11"]


@pytest.mark.parametrize("kt, rpm, lbf, nm, hp", [(50, 4944, 52.6, 20.5, 2289), (58, 5020, 50.4, 20.5, 2289), (75, 4284, 22.2, 11.1, 2289),
                                                  (75, 4800, 38.2, 17.5, 0)])
def test_high_lift_prop_follows_nasas_tables(ac, kt, rpm, lbf, nm, hp):
    """NASA's final tables, Airspeed and Fixed modes, in equivalent knots, at the blade's effective pitch"""
    atm = Atmosphere(hp * 0.3048)
    rot = next(r for r in ac.rotors if r.name == "hl_7")
    V, om = kt * KT / np.sqrt(atm.rho / 1.225), rpm * 2 * np.pi / 60
    a, _ = P._solved_at_pitch(rot, om, V, np.radians(-89.9), np.radians(ac.pitch_deg), atm, QUICK, lbf * 4.448)
    assert a.torque == pytest.approx(nm, rel=0.12)
    assert a.thrust / 4.448 == pytest.approx(lbf, rel=0.17)


@pytest.mark.parametrize("v, beta, lbf, kw", [(150.0, 18.0, 96.8, 24.0), (150.0, 22.0, 194.6, 48.3), (112.5, 18.0, 194.2, 38.9)])
def test_cruise_prop_follows_nasas_cfd(ac, v, beta, lbf, kw):
    """2250 rpm, 2500 ft: speeds in ft/s, the blade pitch at 0.75 R"""
    atm = Atmosphere(762.0)
    om = 2250 * 2 * np.pi / 60
    a, _ = P._solved_at_pitch(ac.rotors[0], om, v * 0.3048, np.radians(-88.0), np.radians(beta), atm, QUICK, lbf * 4.448)
    assert a.thrust / 4.448 == pytest.approx(lbf, rel=0.15)
    assert a.torque * om / 1e3 == pytest.approx(kw, rel=0.15)


def test_blade_passing_tone_in_the_tunnel(ac):
    """4800 rpm, 75 KEAS: NASA's array has its two batches of blades at 79.4 and 71.7 dB in the plane, 3.54 m away"""
    rot = replace(next(r for r in ac.rotors if r.name == "hl_7"), position=(0.0, 0.0, 0.0))
    alone = replace(ac, rotors=[rot], wing=[[0.0, -5.0, 0.5, 0.0, 0.05], [1.0, -5.0, 0.5, 0.0, 0.05]])
    alone.pitch_deg = ac.pitch_deg
    pt = P.operate_props(alone, P.PropCondition("t", 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4800.0),
                         AeroOptions(n_radial=24, n_azimuth=120, bvi=False, elastic=False)).points[0]
    t = np.radians(95.0)
    s = synth.rotor_sound(pt, 3.55 * np.array([np.cos(t), 0.0, -np.sin(t)]), synth.Quality.get("draft"), with_broadband=False)
    B = rot.blades
    db = 20 * np.log10(2 * abs(B * (s.H_thickness + s.H_loading)[B]) / s.n / np.sqrt(2) / 2e-5)
    assert 71.7 < db < 79.4


def test_a_propeller_stays_between_its_pitch_stops(ac):
    """Standing still on its fine stop the cruise propeller makes more than a low thrust asked of it"""
    c = P.PropCondition("t", 0.0, 0.0, 2.0, 0.0, 2250.0)
    a = P.operate_props(ac, c, QUICK, discs=["cruise_left"], thrust=465.0).points[0].aero
    assert np.degrees(a.theta0) == pytest.approx(14.0, abs=0.1) and a.thrust > 465.0
    fast = P.operate_props(ac, P.PropCondition("t", 150.0, 0.0, 2.0, 0.0, 1500.0), QUICK, discs=["cruise_left"], thrust=2000.0).points[0].aero
    assert np.degrees(fast.theta0) == pytest.approx(40.0, abs=0.1)


def test_wing_upwash_at_the_discs(ac):
    rot = next(r for r in ac.rotors if r.name == "hl_7")
    assert 5.0 < P.wing_upwash_deg(ac, rot, 1.5) < 20.0
    assert P.wing_upwash_deg(ac, rot, 0.0) == 0.0


def test_motor_tones_are_seamless_and_at_their_level(ac):
    fs, L = 48000.0, 96000
    x = P.tone_loop([(40.0, 0.1)], 2250.0, L, fs, 1, wander=0.0)
    assert np.sqrt(np.mean(x ** 2)) == pytest.approx(0.1, rel=0.15)
    f = np.fft.rfftfreq(L, 1 / fs)
    assert f[np.argmax(np.abs(np.fft.rfft(x)))] == pytest.approx(1500.0, abs=2.0)      # 40 per revolution at 2250 rpm
    assert abs(x[0] - x[-1]) < 0.1 * np.ptp(x)
    assert ac.motors["cruise"]["level_db"] > ac.motors["high_lift"]["level_db"]
    assert [round(t[0], 1) for t in ac.hubs["high_lift"]["tones"]] == [45.6, 54.7]


def manifest(ac):
    files = []
    for g in ac.groups:
        cruise = g.startswith("cruise")
        for kt in (60.0, 100.0):
            for rpm in ((1500.0, 2250.0) if cruise else (3000.0, 4800.0)):
                for ct in ((0.0, 0.05) if cruise else (None,)):
                    name = f"{kt:g}kt_{rpm:g}rpm" + ("" if ct is None else f"_ct{ct:g}")
                    files += [dict(file=f"{name}_{g}_{v}_prop.wav", condition=name, group=g, source="prop", observer=v)
                              for v in ("cockpit", "side")]
        files += [dict(file=f"motor_{g}_{v}_motor.wav", condition="motor", group=g, source="motor", observer=v, rpm=2250.0)
                  for v in ("cockpit", "side")]
    return dict(aircraft="X-57", exterior_distance_m=30.0, cruise_speeds_kt=[60.0, 100.0], cruise_rpm=[1500.0, 2250.0],
                cruise_ct=[0.0, 0.05], high_lift_speeds_kt=[60.0, 100.0], high_lift_rpm=[3000.0, 4800.0], groups=ac.groups,
                positions=P._positions(ac), properties=P.PROPERTIES, files=files)


def test_section_fits_the_sources(ac):
    m = manifest(ac)
    fx = list(ET.fromstring(P.flightgear_prop_xml(m)).find("fx"))
    assert len(fx) == len(m["files"])
    names = {e.find("name").text: e for e in fx}
    e = names["pww-60kt_2250rpm_ct0.05-cruise_right-cockpit"]
    text = ET.tostring(e).decode()
    assert "cruise[1]/rpm" in text and "cruise[1]/ct" in text and e.find("pitch") is not None
    h = ET.tostring(names["pww-60kt_3000rpm-hl_b_right-side"]).decode()
    assert "hl[7]/rpm" in h                                # a group's first disc speaks for it
    src = P.prop_sources(m)
    assert 10 < src["cockpit"] < 64 and 10 < src["exterior"] < 64
