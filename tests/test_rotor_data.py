"""The rotors against their tests: what no check of the program against itself can see."""
import numpy as np
import pytest

from pywopwop import f1a
from pywopwop.aero import AeroOptions, solve_rotor
from pywopwop.model import Airfoil, Rotor, load_aircraft
from pywopwop.operating import RotorPoint

RHO, C, NU = 1.2, 340.3, 1.46e-5


def _bo105(ct_sigma):
    r = load_aircraft("bo105").main
    om = 425.0 * 2 * np.pi / 60
    T = ct_sigma * 0.07 * RHO * np.pi * r.radius ** 2 * (om * r.radius) ** 2
    return solve_rotor(r, om, 0.0, 0.0, T, RHO, C, NU, AeroOptions(bvi=False))


def test_pitching_moment_is_per_unit_span():
    """the section's moment about the elastic axis is q c^2 (cm + cl (x_ea - 1/4)), once in the chord's square"""
    a = _bo105(0.08)
    bm = a.model
    ld = bm.loads(a.theta0, a.theta1c, a.theta1s, a.lam_i)
    r = a.rotor
    q = 0.5 * RHO * (ld["UT"] ** 2 + ld["UP"] ** 2)
    want = q * bm.chord[None, :] ** 2 * (r.airfoil.moment(ld["alpha"], ld["mach"]) + ld["cl"] * (r.elastic_axis - 0.25))
    assert np.allclose(ld["mea"], want, rtol=1e-9)


def test_bo105_blade_twists_nose_down():
    """NASA TM-103867: the steady torsion moment is nose down at every thrust, and hardly grows with it"""
    lo, hi = (np.degrees(np.mean(_bo105(c).theta_e[:, -1])) for c in (0.02, 0.12))
    assert -2.5 < lo < -0.5 and -2.5 < hi < -0.5 and abs(hi - lo) < 0.7


def test_xv15_hover_has_no_false_stall():
    """NASA TM-86833 to the test's limit: the trim stays on the attached branch, power within 7 %"""
    r = load_aircraft("xv15").main
    vt = 0.69 * C
    A = np.pi * r.radius ** 2
    for cts, cps in ((0.08, 0.00644), (0.14, 0.01419), (0.16, 0.01824), (0.18, 0.02364)):
        a = solve_rotor(r, vt / r.radius, 0.0, 0.0, cts * 0.0891 * 1.225 * A * vt ** 2, 1.225, C, NU, AeroOptions(bvi=False))
        assert a.power / (1.225 * A * vt ** 3 * 0.0891) == pytest.approx(cps, rel=0.07)


def test_bo105_trims_at_high_advance_ratio():
    """NASA TM-110356 fig. 21(d), where the iteration used to run away to 90 deg of cyclic and return it: power within 10 %"""
    r = load_aircraft("bo105").main
    om = 425.0 * 2 * np.pi / 60
    vt = om * r.radius
    A = np.pi * r.radius ** 2
    a = solve_rotor(r, om, 0.3289 * vt, np.radians(-10.0), 0.07 * 0.07 * RHO * A * vt ** 2, RHO, C, NU, AeroOptions(n_azimuth=360, bvi=False))
    assert a.power / (RHO * A * vt ** 3 * 0.07) == pytest.approx(0.00669, rel=0.10)


def test_a_trim_that_fails_says_so():
    """a section that can't make the thrust: an error, not a number"""
    r = load_aircraft("xv15").main
    r.airfoil = Airfoil(name="generic", thickness=0.12, cl_max=1.4, cd0=0.008, mach_dd=0.78)
    vt = 0.69 * C
    with pytest.raises(ValueError, match="the trim failed"):
        solve_rotor(r, vt / r.radius, 0.0, 0.0, 0.18 * 0.0891 * 1.225 * np.pi * r.radius ** 2 * vt ** 2, 1.225, C, NU, AeroOptions(bvi=False))


def _harmonic_level(pt, o, blades):
    src = f1a.build_sources(pt, chord_points=4, surface_chord=24, fs_target=96000.0)
    Ht, Hl, n = f1a.single_blade_harmonics(src, o)
    return src, Ht, Hl, n


def test_xv15_near_field_against_nasas_stand():
    """NASA CR-177436 fig. 9.1, 2 ft out of the tip path and 8 ft under the disc: the harmonics' level at high thrust, nothing
    fitted, and its dip as the twisted blade's tip unloads"""
    r = load_aircraft("xv15").main
    vt = 0.69 * C
    om = vt / r.radius
    o = np.array([r.radius + 0.61, 0.0, -2.44])
    k = np.arange(1, 160) * r.blades
    lv = {}
    for lb in (1000, 2500, 8000):
        T = lb * 4.448222
        a = solve_rotor(r, om, 0.0, 0.0, T, 1.225, C, NU, AeroOptions())
        pt = RotorPoint(r, om, 0.0, 0.0, T, np.eye(3), np.zeros(3), np.zeros(3), aero=a)
        _, Ht, Hl, n = _harmonic_level(pt, o, r.blades)
        lv[lb] = 10 * np.log10(np.sum((r.blades * 2 * np.abs((Ht + Hl)[k]) / n) ** 2 / 2) / 4e-10)
    assert 121.0 < lv[8000] < 125.5                # the harmonics alone, under the measured overall level of 125.5
    assert lv[2500] < lv[1000] - 3 and lv[2500] < lv[8000] - 10


def test_hart2_advancing_lobe():
    """HART II baseline (Lim et al. 2003, fig. 7): the 6-40 BPF level at the advancing lobe, the effective core's calibration"""
    r = Rotor(name="hart2", radius=2.0, blades=4, rpm=1041.0, chord=0.121, root_cutout=0.22, twist=-8.0, hub="hingeless", precone_deg=2.5,
              flap_frequency=1.11, lock_number=5.0, torsion_frequency=3.65, elastic_axis=0.25, cg_axis=0.25, clamp_radius=0.1,
              torsion_damping=0.2, airfoil=Airfoil(name="NACA 23012", table="naca23012_c81", shape="naca23012", cl_max=1.5))
    om = 1041.0 * 2 * np.pi / 60
    al = np.radians(5.3)
    a = solve_rotor(r, om, 33.0, al, 3250.0, 1.225, 340.0, NU, AeroOptions())
    U = 33.0 * np.array([np.cos(al), 0.0, np.sin(al)])
    pt = RotorPoint(r, om, 33.0, al, 3250.0, np.eye(3), np.zeros(3), U, aero=a)
    src = f1a.build_sources(pt, chord_points=4, surface_chord=24, fs_target=96000.0)
    ex, ez, ey = np.array([np.cos(al), 0, np.sin(al)]), np.array([-np.sin(al), 0, np.cos(al)]), np.array([0.0, 1.0, 0.0])
    Ht, Hl, n = f1a.single_blade_harmonics(src, 1.0 * ex + 1.8 * ey - 2.215 * ez)
    k = np.arange(6, 41) * 4
    level = 10 * np.log10(np.sum((4 * 2 * np.abs(Ht[k] + Hl[k]) / n) ** 2 / 2) / 4e-10)
    assert level == pytest.approx(114.0, abs=3.0)
