"""The FlightGear sound section: its volumes evaluated as SimGear's xmlsound.cxx and SGExpression.cxx do."""
import xml.etree.ElementTree as ET

import numpy as np

from pywopwop.bank import BankOptions, HELICOPTER_CONDITIONS, flightgear_xml
from pywopwop.model import load_aircraft

MAXPROP = 5      # SimGear's xmlsound.hxx: a sound's <volume> entries beyond it are ignored


def _expr(e, props):
    """the SGExpression subset the section uses"""
    if e.tag == "property":
        return props[e.text]
    if e.tag == "value":
        return float(e.text)
    kids = [k for k in e if k.tag not in ("clipMin", "clipMax")]
    if e.tag == "sum":
        return sum(_expr(k, props) for k in kids)
    if e.tag == "difference":
        return _expr(kids[0], props) - sum(_expr(k, props) for k in kids[1:])
    if e.tag == "div":
        return _expr(kids[0], props) / _expr(kids[1], props)
    if e.tag == "pow":
        return _expr(kids[0], props) ** _expr(kids[1], props)
    if e.tag == "sqrt":
        return np.sqrt(_expr(kids[0], props))
    if e.tag == "product":
        return float(np.prod([_expr(k, props) for k in kids]))
    if e.tag == "clip":
        assert len(e) == 3
        return min(max(_expr(kids[0], props), float(e.findtext("clipMin"))), float(e.findtext("clipMax")))
    raise ValueError(e.tag)


def _volume(snd, props):
    """xmlsound.cxx SGXmlSound::volume() for expression entries (the section uses no others)"""
    vols = snd.findall("volume")
    assert 1 <= len(vols) <= MAXPROP
    v = 1.0
    for vol in vols:
        x = _expr(vol.find("expression")[0], props)
        if x >= 0:
            v *= x
    return min(v, 1.0)


def _section():
    ac = load_aircraft("uh60a")
    files = [dict(condition=c.name, observer=o, source=s) for c in HELICOPTER_CONDITIONS
             for o in ("side", "cockpit") for s in ("rotor", "slap", "swish", "tail")]
    rotors = {r.name: dict(nominal_rpm=r.rpm, blades=r.blades, role=r.role) for r in ac.rotors}
    xml = flightgear_xml(ac, HELICOPTER_CONDITIONS, dict(rotors=rotors, files=files), BankOptions())
    return ac, ET.fromstring(xml).find("fx")


def test_loops_add_up_everywhere():
    """outside and in the cockpit, the conditions' loops of a layer add up to 1 at any airspeed, vertical speed
    and height: no hole and no overlap"""
    ac, fx = _section()
    for view in ("side", "cockpit"):
        sounds = [s for s in fx if s.findtext("name").endswith(f"-{view}-rotor")]
        assert len(sounds) == len(HELICOPTER_CONDITIONS)
        for v in np.linspace(0.0, 160.0, 33):
            for fpm in np.linspace(-2500.0, 2500.0, 41):
                for agl in (2.0, 15.0, 30.0, 1000.0):
                    props = {"velocities/airspeed-kt": v, "velocities/vertical-speed-fps": fpm / 60.0,
                             "position/altitude-agl-ft": agl, "rotors/main/rpm": ac.main.rpm}
                    total = sum(_volume(s, props) for s in sounds)
                    assert abs(total - 1.0) < 1e-6, (view, v, fpm, agl, total)


def test_volumes_fade_and_follow_rpm():
    ac, fx = _section()
    snd = {s.findtext("name"): s for s in fx}
    hover = snd["pww-hover-side-rotor"]
    props = {"velocities/airspeed-kt": 0.0, "velocities/vertical-speed-fps": 0.0,
             "position/altitude-agl-ft": 1000.0, "rotors/main/rpm": ac.main.rpm}
    assert abs(_volume(hover, props) - 1.0) < 1e-6
    assert _volume(hover, {**props, "velocities/airspeed-kt": 60.0}) == 0.0              # the transition took over
    assert _volume(hover, {**props, "position/altitude-agl-ft": 5.0}) == 0.0             # in ground effect
    assert abs(_volume(hover, {**props, "rotors/main/rpm": 0.5 * ac.main.rpm}) - 0.5) < 1e-6
    slap = {**props, "velocities/airspeed-kt": 65.0, "velocities/vertical-speed-fps": -800.0 / 60.0}
    assert abs(_volume(snd["pww-bvi-side-slap"], slap) - 1.0) < 1e-6
    # a fast descent, outside the descents' airspeed windows (a glide slope at 120 kt): the cruise loop
    fast = {**props, "velocities/airspeed-kt": 120.0, "velocities/vertical-speed-fps": -650.0 / 60.0}
    assert abs(_volume(snd["pww-cruise-side-rotor"], fast) - 1.0) < 1e-6
