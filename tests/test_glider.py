"""The glider model: lift, stall, the airbrakes' scaling, the loops' power and the FlightGear section."""
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
             for c in ("40kt", "50kt", "stall") for v in ("cockpit", "side") for s in ("airframe", "wind", "brakes")]
    m = dict(glider="ASK 21", stall_alpha_body_deg=10.0, speeds_kt=[40.0, 50.0], files=files)
    root = ET.fromstring(gl.flightgear_glider_xml(m))
    sounds = list(root.find("fx"))
    assert len(sounds) == 16                          # stall: airframe and wind only
    assert all(s.find("volume/expression") is not None for s in sounds)
    # the canopy's wind follows the airspeed alone, the stall's loop the lowest; the wing turns to the stall's loop
    # with the angle of attack
    vol = {s.find("name").text: ET.tostring(s.find("volume")).decode() for s in sounds}
    assert "alpha-deg" not in vol["pww-stall-cockpit-wind"] and "airspeed-kt" in vol["pww-stall-cockpit-wind"]
    assert "alpha-deg" not in vol["pww-40kt-cockpit-wind"]
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
                             ("stall_inv", 45.2, -1.0)) for s in ("airframe", "wind")]
    m = dict(glider="ASK 21", stall_alpha_body_deg=12.0, separation_alpha_body_deg=4.3, speeds_kt=[60.0],
             inverted_speeds_kt=[60.0], inverted_stall_alpha_body_deg=-14.0, inverted_separation_alpha_body_deg=-11.0,
             files=files)
    vol = {s.find("name").text: ET.tostring(s.find("volume")).decode()
           for s in ET.fromstring(gl.flightgear_glider_xml(m)).find("fx")}
    assert all("pilot-g" in vol[f"pww-{c}-side-{s}"] for c in ("60kt", "inv_60kt", "stall_inv") for s in ("airframe", "wind"))


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
    files = [dict(file=f"{n}_cockpit_wind.wav", condition=n, observer="cockpit", source="wind", airspeed_kt=60.0)
             for n in ("60kt", "slip_60kt")]
    m = dict(glider="ASK 21", stall_alpha_body_deg=12.0, speeds_kt=[60.0], slip_speeds_kt=[60.0],
             slip_sideslip_deg=17.0, files=files)
    vol = {s.find("name").text: ET.tostring(s.find("volume")).decode()
           for s in ET.fromstring(gl.flightgear_glider_xml(m)).find("fx")}
    assert all("side-slip-deg" in vol[f"pww-{n}-cockpit-wind"] for n in ("60kt", "slip_60kt"))


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
