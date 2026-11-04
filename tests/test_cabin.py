"""The cabin's walls (coincidence, leaks, open doors), the APU's signature, and the doors and APU in the FlightGear
sections."""
import dataclasses
import xml.etree.ElementTree as ET

import numpy as np
import pytest

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


def test_the_cockpit_builds_up_its_lows():
    c = Cabin()
    f = np.array([20.0, 35.0, 125.0, 1000.0, 4000.0])
    g = 10 * np.log10(c.room(f))
    assert g[0] == 0.0 and g[1] == 0.0                        # below its lowest mode no diffuse field builds up
    t = c.tau(f)[2]                                           # what the walls and leaks let out is lost to the room
    assert abs(g[2] + 10 * np.log10(0.15 * (1 - t) + t)) < 1e-9 and g[2] > g[3] > g[4] > 0.0
    assert c.alpha(np.array([50.0]))[0] > 2.0 * 0.15          # the lows leak out more than the trim absorbs
    assert np.allclose(c.inside(f), c.tau(f) * c.room(f))
    assert np.allclose(Cabin(absorption=[]).room(f), 1.0)
    u = load_aircraft("uh60a").cabin
    assert np.all(u.room(f, doors_open=True) < u.room(f) + 1e-12)      # what is open absorbs all


def test_a_convected_field_drives_the_walls_less_than_sound():
    from pywopwop.model import wall_patches
    c = load_aircraft("mv22b").cabin
    f = np.array([50.0, 200.0, 1000.0, 20000.0])
    tip, bl = c.near_field_share(f, 202.0), c.near_field_share(f, 86.0)
    assert np.all(tip <= 1.0) and np.all(bl <= 1.0)
    assert tip[0] < 0.01 and tip[0] < bl[0]                   # the faster footprint couples the more poorly
    assert np.all(np.diff(tip[:3]) > 0)                       # better and better toward its coincidence
    assert np.allclose(Cabin().near_field_share(f, 202.0), np.minimum(
        10 ** (-Cabin.panel_tl(f, 8.0, None, 0.0) / 10) * (1 - 0.0) / Cabin().tau(f), 1.0))   # a limp wall: fully
    P, A = wall_patches([[0.0, 1.0, -1.0, 1.0], [4.0, 1.0, -1.0, 1.0]], dx=0.5, n_around=8)
    assert len(P) == 64 and abs(A.sum() / 4.0 - 7.0) < 0.3     # a 2 x 2 m superellipse's girth, about 7 m
    assert np.all(np.abs(P[:, 1]) <= 1.04) and np.all(np.abs(P[:, 2]) <= 1.04)
    Pb, Ab = wall_patches([[0.0, 1.0, -1.0, 1.0, 12.0], [4.0, 1.0, -1.0, 1.0, 12.0]], dx=0.5, n_around=8)
    assert 7.0 < Ab.sum() / 4.0 < 8.0                          # boxier: nearer the square's 8 m
    corner = lambda Q: np.min(np.hypot(1.0 - np.abs(Q[:, 1]), 1.0 - np.abs(Q[:, 2])))
    fine = lambda ex: wall_patches([[0.0, 1.0, -1.0, 1.0, ex], [4.0, 1.0, -1.0, 1.0, ex]], 0.5, 256, 0.0)[0]
    assert corner(fine(12.0)) < 0.5 * corner(fine(4.0))       # its shoulders reach out toward the corners


def test_a_near_wall_is_heard_directly_above_the_lowest_mode():
    c = Cabin()
    f = np.array([20.0, 1000.0, 4000.0])
    near, far = c.direct(f, 0.5, 0.25, 80.0), c.direct(f, 2.0, 0.25, 80.0)
    assert near[0] == 0.0 and far[0] == 0.0                   # a pressure chamber: no direct field
    assert abs(near[1] / far[1] - 16.0) < 1e-9                # 1 / r^2
    assert far[2] > far[1]                                    # the more the interior absorbs, the more it counts
    assert c.direct(f, 0.01, 0.25, 80.0)[1] == c.direct(f, 0.25, 0.25, 80.0)[1]    # no closer than half the patch


def test_the_drive_signature_moves_with_the_rotor_and_the_torque():
    tr = en.load_transmission("mv22b")
    ref = en.gearbox_source(tr, 100.0, tr.reference_torque_pct)
    half = en.gearbox_source(tr, 100.0, 0.5 * tr.reference_torque_pct)
    assert abs(10 * np.log10(ref.bands.sum() / half.bands.sum()) - 20 * np.log10(2.0)) < 1e-9   # amplitude ~ torque
    slow = en.gearbox_source(tr, 83.7, tr.reference_torque_pct)
    c = lambda s: np.sum(s.bands * en.BANDS) / s.bands.sum()                 # the roar's power centroid
    assert abs(c(slow) / c(ref) - 0.837) < 0.03                              # its frequencies with the rotor speed
    assert abs(slow.tones[0][0] - 0.837 * ref.tones[0][0]) < 1e-9
    assert abs(slow.ripple[0] - 16.61) < 0.01 and slow.ripple[1] == tr.ripple[1]     # the blade rate with it


def test_the_drive_swells_and_fades_with_the_blades():
    from pywopwop.atmosphere import Atmosphere
    from pywopwop.model import Observer
    from scipy.signal import butter, hilbert, sosfiltfilt
    tr = en.load_transmission("mv22b")
    src = en.gearbox_source(tr, 83.7, tr.reference_torque_pct)
    fs = 48000.0
    sig, _ = en._render_sources({"gearbox": (None, src)}, Observer("c", position=(0.0, 0.0, 0.0)), None, fs, 6.0,
                                3, "free", None, 1.3, 25.0, Atmosphere(0.0))
    x = sig["layer/gearbox"]
    e = np.abs(hilbert(sosfiltfilt(butter(4, [2000, 5000], "bandpass", fs=fs, output="sos"), x)))
    E = np.abs(np.fft.rfft(e - e.mean()))
    f = np.fft.rfftfreq(len(e), 1.0 / fs)
    k = np.argmin(np.abs(f - src.ripple[0]))
    depth = 2.0 * E[k - 2:k + 3].max() / len(e) / e.mean()              # the envelope's modulation index
    assert abs(depth - tr.ripple[1]) < 0.02


def test_a_trimmed_sidewall_is_a_double_wall():
    c = Cabin(surface_mass=5.0, trim_mass=3.0, trim_gap=0.075)
    f = np.array([50.0, 2000.0])
    tl = c.opaque_tl(f)
    assert abs(tl[0] - Cabin.panel_tl(50.0, 8.0, None, 0.0)) < 1e-9          # below the resonance: both as one
    assert abs(tl[1] - (Cabin.panel_tl(2000.0, 5.0, None, 0.02) + Cabin.panel_tl(2000.0, 3.0, None, 0.0) + 6.0)) < 1e-9
    assert c.opaque_tl(np.array([1000.0]))[0] > Cabin.panel_tl(1000.0, 8.0, None, 0.0) + 15.0   # far tighter
    assert np.all(Cabin().opaque_tl(f) == Cabin.panel_tl(f, 8.0, None, 0.02))   # untrimmed: one wall


def test_the_glazing_map_picks_the_window_patches():
    from pywopwop.model import glazed_patches
    P = np.array([[4.0, 1.1, -3.0], [4.0, 0.0, -2.5], [1.0, 1.1, -3.0]])
    assert list(glazed_patches(P, [[3.5, 5.6, -4.1, -2.56]])) == [True, False, False]
    c = dataclasses.replace(load_aircraft("mv22b").cabin, leak=0.0)          # the walls alone, no leaks
    f = np.array([2000.0])
    assert c.patch_tau(f, True)[0] > 100 * c.patch_tau(f, False)[0]           # the windows pass the highs
    assert c.convected(f, 86.0, True)[0] <= c.patch_tau(f, True)[0] + 1e-12


def test_the_air_calms_above_the_surface_layer():
    from pywopwop.broadband import air_turbulence
    assert air_turbulence(0.0) == 0.8 and air_turbulence(300.0) == 0.8
    assert air_turbulence(3000.0) == 0.2 and 0.2 < air_turbulence(600.0) < 0.8


def test_the_cabins_modes_scatter_the_harmonics():
    from pywopwop.synth import modal_response
    n = 20000
    g = modal_response("right", 0, n, np.ones(n))
    assert abs(np.mean(np.abs(g) ** 2) - 1.0) < 0.05                        # the diffuse field's power on average
    assert abs(np.std(20 * np.log10(np.abs(g))) - 5.57) < 0.2               # a complex Gaussian's scatter
    assert np.allclose(modal_response("right", 0, n, np.zeros(n)), 1.0)     # none below the lowest mode
    assert not np.allclose(g, modal_response("left", 0, n, np.ones(n)))     # each rotor by its own paths


def test_gearbox_keeps_its_level_at_a_tiltrotors_cruise_rpm():
    """the simulator's rotor speed fade ends at the lowest governed speed: nothing off in airplane mode"""
    mv = load_aircraft("mv22b")
    full = en.gearbox_full_rpm(en.load_transmission("mv22b"), mv)
    assert full == pytest.approx(100.0 * mv.airplane_rpm / mv.main.rpm)
    assert full < 83.7                  # the rotors in the recorded cruise
    uh = load_aircraft("uh60a")
    tr = en.load_transmission("uh60a")
    assert en.gearbox_full_rpm(tr, uh) == (tr.rpm_nominal or uh.main.rpm)
