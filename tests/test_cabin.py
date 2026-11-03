"""The cabin's walls (coincidence, leaks, open doors), the APU's signature, and the doors and APU in the FlightGear
sections."""
import xml.etree.ElementTree as ET

import numpy as np

from pywopwop import engine as en
from pywopwop.model import Cabin, load_aircraft


def test_limp_wall_is_the_mass_law_with_leaks():
    c = Cabin()
    f = np.array([100.0, 1000.0, 10000.0])
    tl = -10 * np.log10(c.tau(f))
    mass = 20 * np.log10(8.0 * f) - 47.0
    composite = -10 * np.log10((1 - c.leak) * 10 ** (-mass / 10) + c.leak)      # the areas' transmission adds
    assert np.all(np.abs(tl - composite) < 1e-9)
    assert abs(tl[0] - mass[0]) < 0.1 and abs(tl[2] - 30.0) < 0.5                # the leaks' floor, -30 dB


def test_coincidence_lets_a_whine_through():
    c = load_aircraft("uh60a").cabin
    f = np.geomspace(200.0, 20000.0, 400)
    limp = -10 * np.log10(Cabin(surface_mass=c.surface_mass).tau(f))
    tl = -10 * np.log10(c.tau(f))
    assert np.all(tl[f < 2000.0] > limp[f < 2000.0] - 4.0)       # the windows' lighter glazing, a little
    fc = c.coincidence_hz
    dip = Cabin.panel_tl(fc, c.surface_mass, fc, c.loss_factor) - Cabin.panel_tl(fc, c.surface_mass, None, 0.0)
    assert abs(dip - 10 * np.log10(2 * c.loss_factor / np.pi)) < 1e-9      # Sharp's dip at coincidence
    whine = np.array([10000.0])
    assert c.tau(whine)[0] > Cabin(surface_mass=c.surface_mass).tau(whine)[0]   # more of a whine gets in
    assert np.all(-10 * np.log10(c.tau(f, doors_open=True)) <= -10 * np.log10(c.open_share) + 1e-9)
    assert {o["property"].split("/")[3] for o in c.openings} == {"bayL", "bayR", "passenger", "crew", "windowsL",
                                                                 "windowsR"}


def test_apu_moves_with_its_speed():
    apu = en.load_apu("uh60a")
    full, half = en.apu_source(apu, 100.0), en.apu_source(apu, 50.0)
    assert abs(10 * np.log10(full.total() / half.total()) - 10 * apu.speed_exponent * np.log10(2.0)) < 0.5
    f8 = [f for f, _ in full.tones]
    assert abs(f8[-1] - 8.006 * apu.shaft_hz) < 1e-6 and abs(half.tones[-1][0] - f8[-1] / 2) < 1e-6


def test_doors_and_apu_in_the_engines_section():
    m = dict(aircraft="X", exterior_distance_m=50.0, lit_ng=[65.0], cold_ng=[], starter_ng=[],
             openings=[dict(share=0.1, property="doors"), dict(share=0.02, property="windows")],
             apu=dict(name="A", property="/apu/n", grid=[50.0, 100.0]),
             engines=[dict(name="left", properties="/e0", starter_property="/s0", idle_ng=65.0, lightoff_ng=15.0)],
             files=[dict(file=f"{w}{v}{n}.wav", engine=w, layer=l, state=s, ng=n, observer=v, fuel_pph=100.0)
                    for v in ("cockpit", "cockpit_open", "side")
                    for w, l, s, ns in (("left", "compressor", "lit", (65.0,)), ("apu", "apu", "run", (50.0, 100.0)))
                    for n in ns])
    fx = ET.fromstring(en.flightgear_engine_xml(m)).find("fx")
    snd = {s.findtext("name"): ET.tostring(s).decode() for s in fx}
    assert len(snd) == len(m["files"])
    assert "doors" in snd["pww-left-compressor-lit-65-cockpit_open"]
    assert "doors" in snd["pww-left-compressor-lit-65-cockpit"]
    assert "doors" not in snd["pww-left-compressor-lit-65-side"]
    assert "/apu/n" in snd["pww-apu-apu-run-100-side"] and "<pitch>" in snd["pww-apu-apu-run-100-side"]


def test_gearbox_follows_the_torque_and_the_rotor():
    tr = en.load_transmission("uh60a")
    full, half = en.gearbox_source(tr, 100.0, 100.0), en.gearbox_source(tr, 100.0, 50.0)
    assert abs(10 * np.log10(full.total() / half.total()) - 20 * np.log10(2.0)) < 1e-9     # amplitude ~ torque
    slow = en.gearbox_source(tr, 50.0, 100.0)
    assert abs(slow.tones[0][0] - 0.5 * full.tones[0][0]) < 1e-9 and abs(full.tones[0][0] - 980.0) < 1e-9
