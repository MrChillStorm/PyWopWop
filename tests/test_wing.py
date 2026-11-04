"""A tiltrotor's wing behind its proprotors: the vortex segments' induced velocity, Sears' response, the blades'
viscous wakes, and the wing's lift heard with the rotor's harmonics."""
import numpy as np

from pywopwop import wing
from pywopwop.aero import AeroOptions
from pywopwop.model import load_aircraft
from pywopwop.operating import FlightCondition, operate
from pywopwop.synth import rotor_sound


def test_a_long_segment_induces_a_line_vortex_velocity_softened_in_its_core():
    g, h = 10.0, 0.5
    far = wing._segment(h, 0.0, 0.0, 0.0, 0.0, -1e4, 0.0, 0.0, 1e4, g, 0.01, 0.0, 1.0, 0.0)
    assert abs(far - g / (2.0 * np.pi * h)) < 1e-4 * far        # along z, at +x: the velocity along +y
    core = wing._segment(0.01, 0.0, 0.0, 0.0, 0.0, -1e4, 0.0, 0.0, 1e4, g, 0.01, 0.0, 1.0, 0.0)
    assert abs(core - g / (2.0 * np.pi * 0.01) / np.sqrt(2.0)) < 1e-3 * core   # Vatistas: 1/sqrt(2) at the core


def test_sears_falls_as_the_root_of_the_reduced_frequency():
    assert abs(wing.sears(0.0) - 1.0) < 1e-12
    s = wing.sears(np.array([10.0, 40.0]))
    assert np.allclose(np.abs(s), 1.0 / np.sqrt(1.0 + 2.0 * np.pi * np.array([10.0, 40.0])))
    assert abs(np.angle(s[1]) + 0.25 * np.pi) < 0.01                           # a quarter period late


def test_the_wings_lift_adds_slowly_falling_harmonics_at_its_root():
    ac = load_aircraft("mv22b")
    st = operate(ac, FlightCondition(airspeed_kt=240.0, nacelle_deg=0.0), AeroOptions(n_radial=24, n_azimuth=360))
    p = st.points[0]
    wl = wing.wing_loads(p, ac.wing, st.pitch, n_t=1024)
    le, chord, width = wing.wing_strips(ac.wing)
    assert wl.force.shape[0] == len(le) and np.allclose(wl.force[:, 0], 0.0)
    # the viscous wake: a pulse a fraction of a millisecond wide, its harmonics flat far up
    root = int(np.argmin(np.abs(le[:, 1] - np.sign(p.hub[1]) * 1.6)))
    up = wing.viscous_wake_upwash(p.aero, wl.pos[root:root + 1], wl.normal[root:root + 1],
                                  np.asarray(p.U, float), 100)[0]
    assert abs(up[90]) > 0.7 * abs(up[3]) > 0.0
    # heard with the rotor near the root, above the cockpit's aft end: its 24th to 30th harmonics stronger with the
    # wing (below them the blades' own near field holds there)
    obs = np.array([2.0, np.sign(p.hub[1]) * 1.0, -2.6])
    plain = rotor_sound(p, obs, "draft", with_broadband=False, cache={})
    cache = {(id(p), "wing"): wl}
    both = rotor_sound(p, obs, "draft", with_broadband=False, cache=cache)
    h = np.arange(24, 31) * 3
    gain = np.abs((both.H_thickness + both.H_loading)[h]) / np.abs((plain.H_thickness + plain.H_loading)[h])
    assert np.median(gain) > 3.0


def test_the_wings_loads_reach_the_cabin_through_its_structure():
    from dataclasses import replace
    from pywopwop.model import Cabin
    ac = load_aircraft("mv22b")
    st = operate(ac, FlightCondition(airspeed_kt=240.0, nacelle_deg=0.0), AeroOptions(n_radial=24, n_azimuth=360))
    p = st.points[0]
    wl = wing.wing_loads(p, ac.wing, st.pitch, n_t=1024, attach_y=1.16)
    assert wl.attached.any() and not wl.attached.all()        # only the loads near the roots
    f = np.arange(400) * wl.f_rot
    cab = ac.cabin
    p2 = wing.structure_borne(wl, cab, 89.0, f)
    h = np.arange(10, 31) * 3
    slope = np.polyfit(h / 3, 10 * np.log10(p2[h]), 1)[0]
    assert -2.0 < slope < 0.0                                 # its harmonics fall slowly, as the walls' don't
    # the power fed in follows the input mobility; a damper structure holds less of it
    assert abs(wing.structure_borne(wl, replace(cab, wing_mobility=2 * cab.wing_mobility), 89.0, f)[30] / p2[30] - 2.0) < 1e-9
    assert wing.structure_borne(wl, replace(cab, structure_loss_factor=0.06), 89.0, f)[30] < p2[30]
    none = replace(wl, attached=np.zeros(len(wl.attached), bool))
    assert not wing.structure_borne(none, cab, 89.0, f).any()
    assert Cabin().wing_mobility is None


def test_the_wing_covers_the_skin_under_it_and_its_near_field_is_no_sound():
    from pywopwop.model import wall_patches
    ac = load_aircraft("mv22b")
    W, _ = wall_patches(ac.fuselage, 0.5, 8)
    c = wing.covered(W, ac.wing)
    assert c.any() and np.all(W[c, 2] > -2.7) and np.all(W[c, 0] < 2.0)     # the roof under it, none of the sides
    assert not wing.covered(W, [s[:4] for s in ac.wing]).any()               # no thickness given: nothing covered
    wl = wing.WingLoads(pos=np.array([[0.0, 0.0, 0.0]]), normal=np.array([[0.0, 0.0, 1.0]]),
                        force=np.array([[0.0, 100.0]], complex), f_rot=10.0, c=340.0)
    close = np.array([0.0, 0.0, 0.3])
    full, sound = abs(wing.harmonics(wl, close, 8)[1]), abs(wing.harmonics(wl, close, 8, radiating=True)[1])
    assert sound < 0.1 * full                                                # kr << 1: nearly all of it hugs the wing
