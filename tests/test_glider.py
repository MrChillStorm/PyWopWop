"""The glider model: lift, stall, the airbrakes' scaling, the loops' power and the FlightGear section."""
import re
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest

from pywopwop import glider as gl
from pywopwop.model import STANDARD_OBSERVERS


def test_geometry_and_lift():
    g = gl.load_glider("ask21")
    S = gl.strips(g)
    wing = [s for s in S if s.kind == "wing"]
    area = sum(s.chord * s.span for s in wing)
    root = 2 * g.fuselage_half_width * g.chord(0.0)            # the wing inside the fuselage is not in the strips
    assert abs(area + root - g.wing_area) / g.wing_area < 0.03
    ys = np.array([s.y for s in wing])
    cs = np.array([s.chord for s in wing])
    dy = np.array([s.span for s in wing])
    cl = gl.wing_cl(g, 0.8, 0.0, ys, cs)
    lift = np.sum(cl * cs * dy)                                 # both wings
    ys_f = np.linspace(0, g.span / 2, 2001)
    full = np.trapezoid(gl.wing_cl(g, 0.8, 0.0, ys_f, g.chord(ys_f)) * g.chord(ys_f), ys_f) * 2
    assert abs(full - 0.8 * g.wing_area) / (0.8 * g.wing_area) < 0.03      # Schrenk carries CL S
    assert lift < full


def test_stall_speed_matches_the_flight_manual():
    g = gl.load_glider("ask21")
    assert 32.0 < gl.stall_speed(g) < 38.0          # about 35 kt at 540 kg (65 km/h)


def test_airbrakes_grow_as_the_sixth_power_of_speed():
    g = gl.load_glider("ask21")
    o = STANDARD_OBSERVERS["side"].offset() * 0.3
    p = []
    for kt in (60.0, 120.0):
        sp, _ = gl.spectra(g, gl.GliderCondition(airspeed_kt=kt, brakes=1.0, turbulence_ms=0.0), o)
        p.append(np.sum(sp["brakes"]))
    slope = 10 * np.log10(p[1] / p[0]) / (10 * np.log10(2.0))
    assert 5.0 < slope < 7.0


def test_band_loop_has_the_band_powers():
    rng = np.random.default_rng(0)
    msp = np.zeros(len(gl.BANDS))
    msp[10] = 0.04
    msp[20] = 0.01
    x = gl._band_loop(msp, 48000 * 4, 48000.0, rng)
    assert abs(np.mean(x ** 2) / msp.sum() - 1.0) < 0.05


def test_flightgear_section_is_well_formed():
    kt = {"40kt": 40.0, "50kt": 50.0, "stall": 33.7}
    files = [dict(file=f"{c}_{v}_{s}.wav", condition=c, observer=v, source=s, airspeed_kt=kt[c])
             for c in ("40kt", "50kt", "stall") for v in ("cockpit", "side")
             for s in ("airframe", "windward", "leeward", "brakes")]
    m = dict(glider="ASK 21", stall_alpha_body_deg=10.0, speeds_kt=[40.0, 50.0], files=files)
    root = ET.fromstring(gl.flightgear_glider_xml(m))
    sounds = list(root.find("fx"))
    assert len(sounds) == 22                          # stall: airframe and the wind's halves only
    assert all(s.find("volume/expression") is not None for s in sounds)
    # the canopy's wind follows the airspeed alone, the stall's loop the lowest; the wing turns to the stall's loop
    # with the angle of attack
    vol = {s.find("name").text: ET.tostring(s.find("volume")).decode() for s in sounds}
    assert "alpha-deg" not in vol["pww-stall-cockpit-windward-right"]
    assert "airspeed-kt" in vol["pww-stall-cockpit-leeward-left"]
    assert "alpha-deg" not in vol["pww-40kt-cockpit-windward-right"]
    assert "alpha-deg" in vol["pww-stall-side-airframe"] and "alpha-deg" in vol["pww-40kt-side-airframe"]


def test_fuselage_flow_matches_a_prolate_spheroid():
    # exact potential flow about a 6:1 prolate spheroid: 1.0452 times the airspeed at the equator in axial flow; in
    # crossflow W the flow round the equator's section is 2 / (2 - beta0) = 1.9171 times W sin(angle), so on the
    # leeward line it converges at 1.9171 W / R
    import dataclasses
    g = gl.load_glider("ask21")
    L, R = 6.0, 0.5
    x = -L / 2 + L * (1 - np.cos(np.linspace(0, np.pi, 41))) / 2
    r = R * np.sqrt(np.maximum(1 - (2 * x / L) ** 2, 0))
    fu = dict(g.fuselage, x=list(x), top_z=list(r), bottom_z=list(-r), width=list(2 * r), section_exponent=2.0)
    F = gl.fuselage_flow(dataclasses.replace(g, name="spheroid-test", fuselage=fu))
    i = np.argmin(abs(F["x"]))
    assert abs(F["v"][0, i, 0] / 1.0452 - 1) < 0.005
    assert abs(F["div"][1, i] / (-1.9171 / R) - 1) < 0.03


def test_wing_separates_from_the_trailing_edge():
    g = gl.load_glider("ask21")
    root = [s for s in gl.strips(g) if s.kind == "wing"][0]
    Re = 2.0e6
    cruise = gl.section_state(root, Re, 0.5)
    high = gl.section_state(root, Re, 1.55)
    assert cruise["sep"] == 0.0 and not cruise["stalled"]
    assert 0.1 < high["sep"] < 0.8 and not high["stalled"]            # the FX S 02-196 stalls from the trailing edge
    M = 0.06
    g0, gw = 23.43 * M + 4.651, 27.094 * M + 3.31
    assert gl.bpm_angle(cruise, M) <= g0 - gw                          # attached: no BPM separation noise
    assert g0 - gw < gl.bpm_angle(high, M) < g0                        # separating: on BPM's bell, not deep stall


def test_canopy_layer_thickens_on_the_leeward_line():
    from pywopwop.atmosphere import Atmosphere
    g = gl.load_glider("ask21")
    atm = Atmosphere(0.0)
    V = 40.0 * 0.514444
    th = []
    for a in (0.0, 12.0):
        bl = gl.canopy_boundary_layer(g, V, a, 1.0, atm)
        assert bl["sep"] is None or bl["sep"] > g.fuselage["canopy_x"][1]
        th.append(np.interp(g.cockpit["rear"][0], bl["x"], bl["theta"]))
    assert th[1] > 1.5 * th[0]


def test_manoeuvres_roll_slip_and_trim():
    g = gl.load_glider("ask21")
    st = lambda **kw: gl.strip_states(g, gl.GliderCondition(**kw))[1]
    assert 10.0 < st(airspeed_kt=60, aileron=1.0)["roll_rate_dps"] < 60.0          # a steady roll, full aileron
    assert st(airspeed_kt=60, aileron=-1.0)["roll_rate_dps"] < -10.0
    assert 10.0 < st(airspeed_kt=50, rudder=1.0)["sideslip_deg"] < 25.0           # a full-rudder slip
    up40, dn100 = st(airspeed_kt=40)["elevator_deg"], st(airspeed_kt=100)["elevator_deg"]
    assert up40 < dn100                                                            # up elevator at low speed
    assert st(airspeed_kt=70, load_factor=3.5)["elevator_deg"] < st(airspeed_kt=70)["elevator_deg"] - 5.0


def test_accelerated_stall_is_louder_and_ailerons_add_sound():
    g = gl.load_glider("ask21")
    o = STANDARD_OBSERVERS["side"].offset() * 0.3
    p = lambda n: np.sum(gl.spectra(g, gl.GliderCondition(airspeed_kt=0.97 * gl.stall_speed(g, n), load_factor=n,
                                                          turbulence_ms=0.0), o)[0]["airframe"])
    assert 10 * np.log10(p(3.5) / p(1.0)) > 8.0
    sp, _ = gl.spectra(g, gl.GliderCondition(airspeed_kt=60, aileron=1.0, turbulence_ms=0.0), o)
    assert np.all(sp["ailerons"] >= 0) and np.sum(sp["ailerons"]) > 0.05 * np.sum(sp["airframe"])


def test_flightgear_section_has_accelerated_stalls_and_ailerons():
    files = [dict(file=f"{c}_side_{s}.wav", condition=c, observer="side", source=s, airspeed_kt=v)
             for c, v in (("40kt", 40.0), ("50kt", 50.0)) for s in ("airframe", "ailerons")]
    files += [dict(file=f"{c}_side_airframe.wav", condition=c, observer="side", source="airframe", airspeed_kt=v)
              for c, v in (("stall", 33.7), ("stall_2g", 47.6))]
    m = dict(glider="ASK 21", stall_alpha_body_deg=12.0, separation_alpha_body_deg=4.3, speeds_kt=[40.0, 50.0],
             files=files)
    root = ET.fromstring(gl.flightgear_glider_xml(m))
    vol = {s.find("name").text: ET.tostring(s.find("volume")).decode() for s in root.find("fx")}
    assert len(vol) == 6
    assert "aileron-pos-norm" in vol["pww-40kt-side-ailerons"]
    for c in ("stall", "stall_2g"):                   # turned to with the angle, cross-faded by airspeed
        assert "alpha-deg" in vol[f"pww-{c}-side-airframe"] and "airspeed-kt" in vol[f"pww-{c}-side-airframe"]


def test_inverted_flight():
    g = gl.load_glider("ask21")
    vs, vi = gl.stall_speed(g), gl.stall_speed(g, -1.0)
    assert vi > 1.15 * vs                             # the cambered wing flies worse upside down
    states, info = gl.strip_states(g, gl.GliderCondition(airspeed_kt=60, load_factor=-1.0))
    assert info["alpha_body"] < -8.0 and info["CL"] < 0
    front = gl.to_body(*g.cockpit["front"])
    up, _ = gl.spectra(g, gl.GliderCondition(airspeed_kt=60), front, True)
    inv, _ = gl.spectra(g, gl.GliderCondition(airspeed_kt=60, load_factor=-1.0), front, True)
    lo = gl.BANDS < 300
    # the windward canopy's thinner layer, without the measured roar that follows the airspeed alone
    up0, inv0 = (up["wind"] - gl.cockpit_roar(g, gl.GliderCondition(airspeed_kt=60).V, gl.Atmosphere(0.0)),
                 inv["wind"] - gl.cockpit_roar(g, gl.GliderCondition(airspeed_kt=60).V, gl.Atmosphere(0.0)))
    assert np.sum(inv0[lo]) < 0.8 * np.sum(up0[lo])
    files = [dict(file=f"{c}_side_{s}.wav", condition=c, observer="side", source=s, airspeed_kt=v, load_factor=n)
             for c, v, n in (("60kt", 60.0, 1.0), ("stall", 33.7, 1.0), ("inv_60kt", 60.0, -1.0),
                             ("stall_inv", 45.2, -1.0)) for s in ("airframe", "windward", "leeward")]
    m = dict(glider="ASK 21", stall_alpha_body_deg=12.0, separation_alpha_body_deg=4.3, speeds_kt=[60.0],
             inverted_speeds_kt=[60.0], inverted_stall_alpha_body_deg=-14.0, inverted_separation_alpha_body_deg=-11.0,
             files=files)
    vol = {s.find("name").text: ET.tostring(s.find("volume")).decode()
           for s in ET.fromstring(gl.flightgear_glider_xml(m)).find("fx")}
    assert all("pilot-g" in vol[f"pww-{c}-side-{s}"] for c in ("60kt", "inv_60kt", "stall_inv")
               for s in ("airframe", "windward-right", "leeward-left"))


def test_slip_turns_the_canopy_flow():
    from pywopwop.atmosphere import Atmosphere
    g = gl.load_glider("ask21")
    F = gl.fuselage_flow(g)
    c = gl.GliderCondition(airspeed_kt=60, rudder=1.0)
    _, info = gl.strip_states(g, c)
    bl = gl.canopy_layers(g, c.V, info["alpha_geo"], info["sideslip_deg"], info["CL"], Atmosphere(0.0))
    y = F["canopy_p"][:, 1]
    aft = F["canopy_p"][:, 0] > -0.4
    lee, wind = aft & (y < -0.15), aft & (y > 0.15)                   # the relative wind from the right
    assert np.mean(bl["theta"][lee]) > 2.0 * np.mean(bl["theta"][wind])
    straight = gl.canopy_layers(g, c.V, info["alpha_geo"], 0.0, info["CL"], Atmosphere(0.0))
    left = straight["theta"][y < 0].mean()
    assert abs(straight["theta"][y > 0].mean() / left - 1) < 0.02     # symmetric in straight flight


def slip_manifest():
    files = [dict(file=f"{n}_cockpit_{h}.wav", condition=n, observer="cockpit", source=h, airspeed_kt=60.0)
             for n in ("60kt", "slip25_60kt", "slip50_60kt") for h in ("windward", "leeward")]
    files += [dict(file=f"{n}_side_airframe.wav", condition=n, observer="side", source="airframe", airspeed_kt=60.0)
              for n in ("60kt", "slip25_60kt", "slip50_60kt")]
    return dict(glider="ASK 21", stall_alpha_body_deg=12.0, speeds_kt=[60.0], slip_speeds_kt=[60.0],
                slip_angles_deg=[25.0, 50.0], low_speed_power=dict(wind=3.0, airframe=6.0),
                cockpit_sound_position=dict(x=-2.5, y=0.33, z=0.9), files=files)


def test_flightgear_section_slips():
    """The angles cross-fade on the sideslip's size, the sign puts each half on its side, the outside slips too"""
    sounds = {s.find("name").text: s for s in ET.fromstring(gl.flightgear_glider_xml(slip_manifest())).find("fx")}
    text = {k: ET.tostring(s).decode() for k, s in sounds.items()}
    assert len(sounds) == 2 + 8 + 3
    for n in ("pww-60kt-cockpit-windward-right", "pww-slip25_60kt-cockpit-leeward-left", "pww-slip50_60kt-side-airframe"):
        assert "side-slip-deg" in text[n]
    # the wind from the right (positive): windward on the right; from the left, on the left
    for name, cmp, y in (("windward-right", "greater-than", -0.33), ("windward-left", "less-than", 0.33),
                         ("leeward-left", "greater-than", 0.33), ("leeward-right", "less-than", -0.33)):
        e = sounds[f"pww-slip50_60kt-cockpit-{name}"]
        assert e.find(f"condition/and/{cmp}") is not None
        assert float(e.find("position/y").text) == pytest.approx(y)     # FlightGear's +y is left
    straight = sounds["pww-60kt-cockpit-windward-right"]
    assert float(straight.find("position/y").text) < 0
    # a looped sound holds a source while its condition is true: only inside its own fades, and all of them
    m = slip_manifest()
    m["files"] += [dict(file=f"{c}_{v}_{s}.wav", condition=c, observer=v, source=s, airspeed_kt=k)
                   for c, k in (("50kt", 50.0), ("slip25_50kt", 50.0), ("slip50_50kt", 50.0)) for v in ("cockpit", "side")
                   for s in ("windward", "leeward", "airframe")]
    m["speeds_kt"] = m["slip_speeds_kt"] = [50.0, 60.0]
    fx = list(ET.fromstring(gl.flightgear_glider_xml(m)).find("fx"))

    def value(e, p):
        k = list(e)
        if e.tag == "value":
            return float(e.text)
        if e.tag == "property":
            return p[e.text.lstrip("/")]
        if e.tag == "clip":
            v = value(next(c for c in k if c.tag not in ("clipMin", "clipMax")), p)
            return min(max(v, float(e.find("clipMin").text)), float(e.find("clipMax").text))
        a = [value(c, p) for c in k]
        return dict(expression=a[0], product=float(np.prod(a)), sum=sum(a), sqrt=max(a[0], 0.0) ** 0.5, abs=abs(a[0]),
                    difference=a[0] - sum(a[1:]))[e.tag] if e.tag != "pow" else a[0] ** a[1]

    def true(e, p):
        if e.tag in ("and", "or"):
            return (all if e.tag == "and" else any)(true(c, p) for c in e)
        if e.tag == "not":
            return not true(e[0], p)
        if e.tag == "property":
            return bool(p[e.text.lstrip("/")])
        a, b = value(e[0], p), value(e[1], p)
        return a > b if e.tag == "greater-than" else a < b
    most = 0
    for inside in (True, False):
        for kt in (8.0, 49.0, 50.0, 55.0, 60.0, 75.0):
            for b in (-60.0, -30.0, -25.0, -10.0, 0.0, 10.0, 25.0, 40.0, 50.0, 60.0):
                p = {"sim/current-view/internal": inside, "velocities/airspeed-kt": kt, "orientation/side-slip-deg": b,
                     "orientation/alpha-deg": 3.0, "accelerations/pilot-g": 1.0, "surface-positions/speedbrake-pos-norm": 0.0,
                     "surface-positions/left-aileron-pos-norm": 0.0}
                on = [e for e in fx if true(e.find("condition")[0], p)]
                most = max(most, len(on))
                for e in fx:                              # nothing gated off would have sounded
                    name = e.find("name").text
                    side = name.rsplit("-", 1)[1]
                    elsewhere = ("cockpit" in name) != inside or ("slip" in name and side in ("left", "right") and b != 0.0
                                                                  and ("windward" in name) != (side == ("right" if b > 0 else "left")))
                    if e not in on and not elsewhere:
                        assert value(e.find("volume/expression"), p) < 1e-9, name
    assert most <= 12
    src = gl.glider_sources(m)
    assert 0 < src["cockpit"] <= 12 and 0 < src["exterior"] <= 12


def test_low_speed_fade():
    """Below the lowest loop a layer's power falls as the speed to its own exponent, not as the speed"""
    root = ET.fromstring(gl.flightgear_glider_xml(slip_manifest()))
    s = {e.find("name").text: e for e in root.find("fx")}
    assert s["pww-60kt-cockpit-windward-right"].find("volume/expression//pow/value").text == "1.5"
    assert s["pww-60kt-side-airframe"].find("volume/expression//pow/value").text == "3"
    g = gl.load_glider("ask21")
    p = gl.low_speed_power(g, 40.0)
    assert 2.5 < p["wind"] < 3.5 and 4.5 < p["airframe"] < 8.0 and p["slip_wind"] > p["wind"]


def test_slip_is_a_condition_of_its_own():
    g = gl.load_glider("ask21")
    S = gl.strips(g)
    root = S[0]
    balance = gl.strip_states(g, gl.GliderCondition(airspeed_kt=60, rudder=1.0))[1]["sideslip_deg"]
    st = lambda b, kt=60.0: gl.strip_states(g, gl.GliderCondition(airspeed_kt=kt, sideslip_deg=b))[0]
    info = gl.strip_states(g, gl.GliderCondition(airspeed_kt=60, sideslip_deg=46.0))[1]
    assert balance < 20.0 and info["sideslip_deg"] == 46.0
    # the wing at V cos(sideslip): its lift coefficient up by 1 / cos^2, the windward side further by the dihedral
    a, b = st(0.0)[0], st(46.0)[0]
    left = next(i for i, s in enumerate(S) if s.kind == "wing" and s.side < 0)
    assert root.side > 0 and b["cl"] > 1.5 * a["cl"] and st(46.0)[left]["alpha"] < b["alpha"] - 5.0
    assert any(x["stalled"] for s, x in zip(S, st(46.0)) if s.kind == "fin")
    assert not any(x["stalled"] for s, x in zip(S, st(46.0, 53.0)) if s.kind == "wing")   # the flight model's full slip


def test_crossflow_separation_follows_the_published_lines():
    g = gl.load_glider("ask21")
    x, right, left, top = g.cockpit["front"][0], np.pi / 2, -np.pi / 2, 0.0
    t = float(gl.crossflow_time(g, x, 46.0))
    assert 2.5 < t < 5.0
    sep = lambda p, b=46.0: bool(gl.crossflow_separated(g, x, p, 0.0, b))
    assert sep(left) and not sep(right) and not sep(top)
    # 151.2 t^-0.190 degrees from the windward line
    line = np.radians(151.2 * t ** -0.190)
    assert sep(right - line - 0.02) and not sep(right - line + 0.02)
    assert not sep(left, 15.0) and not sep(left, 0.0)   # the front seat's station starts past 20 degrees
    assert sep(left, 25.0)


def test_cockpit_halves():
    g = gl.load_glider("ask21")
    head = gl.to_body(*g.cockpit["front"])
    sp, _ = gl.spectra(g, gl.GliderCondition(airspeed_kt=75.0), head, True)
    assert np.allclose(sp["windward"] + sp["leeward"], sp["wind"])
    assert np.sum(sp["windward"]) == pytest.approx(np.sum(sp["leeward"]), rel=0.02)
    slip, state = gl.spectra(g, gl.GliderCondition(airspeed_kt=75.0, sideslip_deg=46.0), head, True)
    lo = (gl.BANDS > 40.0) & (gl.BANDS < 200.0)
    assert state["canopy_separated"] > 0.15
    assert np.sum(slip["leeward"][lo]) > 10.0 * np.sum(slip["windward"][lo])   # the separated side's rumble
    assert np.sum(slip["wind"][lo]) > 5.0 * np.sum(sp["wind"][lo])
    hi = gl.BANDS > 2500.0
    assert np.sum(slip["leeward"][hi]) < np.sum(slip["windward"][hi])          # and it has lost the hiss


def test_fuselage_across_the_flow():
    """Fink's strut sums to one over the third octaves and grows as M^6; the fuselage's lift dipole beams up and down"""
    atm = gl.Atmosphere(0.0)
    r, up = np.array([0.0, 0.0, -100.0]), np.array([0.0, 0.0, 1.0])
    p = gl.fink_strut(0.1, 2.0, 50.0, r, up, atm)
    M = 50.0 / atm.c
    assert p.sum() == pytest.approx((atm.rho * atm.c ** 2) ** 2 * 2.753e-4 * M ** 6 * 0.2 * 3.0 / (4.0 * np.pi * 1e4), rel=0.03)
    assert gl.fink_strut(0.1, 2.0, 100.0, r, up, atm).sum() / p.sum() == pytest.approx(2.0 ** 6, rel=0.03)
    g = gl.load_glider("ask21")
    V = 40.0
    n = lambda L, b=46.0: gl.crossflow_noise(g, V, 3.0, b, L, atm).sum()
    assert n(np.array([0.0, 0.0, -30.0])) > 5.0 * n(np.array([0.0, -30.0, 0.0]))
    assert n(np.array([0.0, 0.0, -30.0])) > 5.0 * n(np.array([0.0, 0.0, -30.0]), 25.0)
    assert n(np.array([0.0, 0.0, -30.0]), 0.0) == 0.0


def test_wakes_reach_the_leeward_strips():
    g = gl.load_glider("ask21")
    S = gl.strips(g)
    wing = [s for s in S if s.kind == "wing"]
    lee = next(s for s in wing if s.side < 0)
    wind = next(s for s in wing if s.side > 0)
    u, L = gl.wake_turbulence(g, lee, True, 3.0, 46.0)
    assert 0.03 < u < 0.3 and 0.1 < L < 1.0
    assert gl.wake_turbulence(g, wind, True, 3.0, 46.0)[0] == 0.0
    assert gl.wake_turbulence(g, wing[-1], True, 3.0, 46.0)[0] == 0.0          # the tip is out of it
    tail = [s for s in S if s.kind == "tailplane"]
    lee_t = next(s for s in tail if s.side < 0)
    assert gl.wake_turbulence(g, lee_t, True, 3.0, 46.0)[0] > 0.05             # in the stalled fin's
    assert gl.wake_turbulence(g, lee_t, False, 3.0, 46.0)[0] == 0.0
    assert gl.wake_turbulence(g, next(s for s in tail if s.side > 0), True, 3.0, 46.0)[0] == 0.0


def test_glider_bank_slips(tmp_path, monkeypatch):
    """A bank's slips: the halves in the cockpit, the airframe outside, the angle in the manifest and the section"""
    monkeypatch.setattr(gl, "low_speed_power", lambda *a: dict(wind=3.0, airframe=6.0, brakes=7.0, ailerons=7.0))
    g = gl.load_glider("ask21")
    m = gl.make_glider_bank(g, tmp_path, speeds=(60.0, 90.0), views=("cockpit", "side"), duration=0.5, fs=24000.0,
                            stall_load_factors=(1.0,), slip_angles=(25.0, 50.0), progress=lambda *_: None)
    have = {(f["condition"], f["observer"], f["source"]): f for f in m["files"]}
    assert ("slip50_60kt", "cockpit", "leeward") in have and ("slip50_60kt", "cockpit", "windward") in have
    assert ("slip25_90kt", "side", "airframe") in have and have[("slip25_90kt", "side", "airframe")]["sideslip_deg"] == 25.0
    assert ("slip50_60kt", "cockpit", "airframe") not in have and ("slip50_60kt", "cockpit", "mix") not in have
    assert ("60kt", "cockpit", "wind") not in have and ("60kt", "cockpit", "mix") in have
    assert m["slip_angles_deg"] == [25.0, 50.0] and m["cockpit_sound_position"]["y"] > 0.2
    assert 0 < m["max_sources"]["cockpit"] < 40 and 0 < m["max_sources"]["exterior"] < 40
    assert have[("slip50_60kt", "cockpit", "leeward")]["dba"] > have[("slip50_60kt", "cockpit", "windward")]["dba"] + 3.0
    x = (tmp_path / "pywopwop-glider.xml").read_text()
    assert ET.fromstring(x) is not None
    for f in re.findall(r"<path>Sounds/pywopwop/([^<]+)</path>", x):
        assert (tmp_path / f).exists()


def test_fixed_wheels():
    """Fink's wheel: its spectrum sums to one over the third octaves, it grows as M^6, and the ASK 21's wheels lead outside
    near their peak but not in the cockpit"""
    from pywopwop.atmosphere import Atmosphere
    from pywopwop.glider import BANDS, GliderCondition, fink_wheel, load_glider, spectra, to_body, wheel_noise
    atm = Atmosphere(0.0)
    r = np.array([0.0, 0.0, -100.0])
    p = fink_wheel(0.4, 50.0, r, atm)
    M = 50.0 / atm.c
    full = (atm.rho * atm.c ** 2) ** 2 * 4.349e-4 * M ** 6 * 0.16 * 1.5 / (4.0 * np.pi * 1e4)
    assert p.sum() == pytest.approx(full, rel=0.02)
    assert fink_wheel(0.4, 100.0, r, atm).sum() / p.sum() == pytest.approx(2.0 ** 6, rel=0.02)
    g = load_glider("ask21")
    c = GliderCondition(airspeed_kt=80.0)
    w = wheel_noise(g, c.V, r, atm)
    sp = spectra(g, c, r)[0]["airframe"]
    k = int(np.argmin(np.abs(BANDS - 630.0)))
    assert w[k] > 0.5 * sp[k]
    head = to_body(*g.cockpit["front"])
    inside = spectra(g, c, head, inside=True)[0]
    assert inside["airframe"][k] < 0.1 * inside["wind"][k]


def test_cockpit_roar():
    """The measured roar: low bands only, as the cube of the speed, in the cockpit and not outside"""
    g = gl.load_glider("ask21")
    atm = gl.Atmosphere(0.0)
    a, b = gl.cockpit_roar(g, 30.0, atm), gl.cockpit_roar(g, 60.0, atm)
    k = int(np.argmin(np.abs(gl.BANDS - 200.0)))
    assert b[k] / a[k] == pytest.approx(8.0, rel=1e-6)
    assert np.all(a[gl.BANDS > 400.0] == 0.0) and np.all(a[gl.BANDS < 45.0] == 0.0)
    head = gl.to_body(*g.cockpit["front"])
    c = gl.GliderCondition(airspeed_kt=60.0)
    assert gl.spectra(g, c, head, inside=True)[0]["wind"][k] > 2.0 * gl.spectra(replace(g, roar=None), c, head, inside=True)[0]["wind"][k]
    assert "wind" not in gl.spectra(g, c, np.array([0.0, 0.0, -100.0]))[0]
