"""Loops, layers and the FlightGear snippet."""
import xml.etree.ElementTree as ET

import numpy as np

from pywopwop import synth
from pywopwop.bank import BankOptions, HELICOPTER_CONDITIONS, flightgear_xml
from pywopwop.model import FlightCondition, STANDARD_OBSERVERS, load_aircraft


def test_loop_length_is_fast_and_in_tune():
    for f_rot in (4.3, 7.07, 27.6, 90.0):
        L, n = synth.loop_length(f_rot, 48000.0, 8.0)
        assert synth._smooth(L)
        assert abs(1200 * np.log2(n * 48000.0 / L / f_rot)) <= 1.0


def test_render_layers_add_up_and_loop_is_seamless():
    ac = load_aircraft("uh1h")
    r = synth.render(ac, FlightCondition(airspeed_kt=70, climb_fpm=-700), STANDARD_OBSERVERS["front_right"],
                     duration=2.0, quality="draft")
    s = r.signals
    layers = s["layer/rotor"] + s["layer/slap"] + s["layer/swish"] + s["layer/tail"]
    assert np.max(np.abs(layers - s["total"])) < 1e-6 * np.max(np.abs(s["total"]))
    assert np.sqrt(np.mean(s["layer/slap"] ** 2)) > 0.1 * np.sqrt(np.mean(s["total"] ** 2))
    x = s["total"]
    assert abs(x[0] - x[-1]) <= np.max(np.abs(np.diff(x)))
    # cockpit: through the cabin wall
    rc = synth.render(ac, FlightCondition(airspeed_kt=70, climb_fpm=-700),
                      synth.Observer("cockpit", position=tuple(ac.cockpit)), duration=1.0, quality="draft", cabin=8.0)
    assert np.all(np.isfinite(rc.total))


def test_flightgear_xml_is_well_formed():
    ac = load_aircraft("uh60a")
    files = [dict(condition=c.name, observer=v, source=src) for c in HELICOPTER_CONDITIONS
             for v in ("side", "cockpit") for src in ("rotor", "slap", "swish", "tail")]
    manifest = dict(rotors={"main": dict(nominal_rpm=257.8, blades=4, role="main"),
                            "tail": dict(nominal_rpm=1190.0, blades=4, role="tail")}, files=files)
    root = ET.fromstring(flightgear_xml(ac, HELICOPTER_CONDITIONS, manifest, BankOptions()))
    sounds = root.find("fx").findall("rotor")
    assert len(sounds) == len(files)
    ige = [snd for snd in sounds if "hover_ige" in snd.find("path").text]
    assert ige and all(any(p.text == "position/altitude-agl-ft" for p in snd.find("volume").iter("property"))
                       for snd in ige)
    for snd in sounds:
        assert snd.find("path").text.endswith(".wav")
        assert snd.find("pitch/property").text in ("rotors/main/rpm", "rotors/tail/rpm")


def test_ground_effect_lowers_hover_power():
    from pywopwop.operating import operate
    ac = load_aircraft("uh60a")
    oge = operate(ac, FlightCondition()).points[0]
    ige = operate(ac, FlightCondition(ground_height_m=0.6 * ac.main.radius)).points[0]
    k = 1.0 - (1.0 / (4.0 * 0.6)) ** 2                  # Cheeseman-Bennett
    assert abs(ige.aero.ground_effect - k) < 0.01
    assert 0.8 < ige.aero.power / oge.aero.power < 0.92
    assert ige.recirculation_rms > 1.0 and oge.recirculation_rms == 0.0
