"""A tiltrotor's wing under its discs in a hover: the inflow over it, the blade load it makes, the turbulence that comes
back up through the disc, and where all that stops."""
import numpy as np
import pytest

from pywopwop.aero import FOUNTAIN_DEFICIT, AeroOptions, Fountain, fountain_inflow, fountain_turbulence
from pywopwop.model import FlightCondition, load_aircraft
from pywopwop.operating import operate

OPTS = AeroOptions(n_radial=24, n_azimuth=360)


class Disc:
    """a blade model's geometry"""
    R = 5.0
    x = np.linspace(0.2, 1.0, 17)
    psi = np.radians(np.arange(0.0, 360.0, 1.0))


def fountain(**kw):
    return Fountain(**{**dict(psi_inboard=np.radians(270.0), aft_sign=-1.0, mid_chord_ahead=0.0, leading_edge_ahead=1.0,
                              chord=2.0, clearance=2.0, v_hover=20.0), **kw})


def test_the_inflow_deficit_lies_over_the_wing_inboard():
    d = fountain_inflow(Disc, 0.1, fountain())
    tip = d[:, -1]
    assert tip.min() == pytest.approx(-FOUNTAIN_DEFICIT * 0.1)
    assert abs(tip[270]) > 0 and tip[90] == 0.0 and tip[0] == 0.0 and tip[180] == 0.0
    # 0.8 of the chord wide, across the wing: at the tip within asin(0.8 / R) of the inboard azimuth
    a = np.degrees(np.arcsin(0.8 / Disc.R))
    assert tip[int(270 + a) + 1] == 0.0 and tip[int(270 - a) - 1] == 0.0
    assert fountain_inflow(Disc, 0.1, fountain(strength=0.5)).min() == pytest.approx(-0.5 * FOUNTAIN_DEFICIT * 0.1)


def test_the_turbulence_lies_on_the_outer_disc_from_the_wing_rearward():
    u = fountain_turbulence(Disc, fountain())
    tip = u[:, -1]
    assert tip.max() > 0.0
    rear = int(270 - 30)            # aft_sign -1: the tail lies at lower azimuths
    ahead = int(270 + 30)           # 30 degrees forward of the rotor axis is past the wing's leading edge, asin(1 / 5)
    assert tip[rear] == tip.max() and tip[ahead] == 0.0 and tip[90] == 0.0
    assert np.all(u[:, Disc.x < 0.6] == 0.0)


def test_the_blades_load_jumps_over_the_wing_in_the_hover():
    ac = load_aircraft("mv22b")
    a = operate(ac, FlightCondition(airspeed_kt=0.0), OPTS).points[0].aero
    k = np.argmin(np.abs(a.x - 0.9))
    fz = a.fz[:, k]
    assert fz.max() > 1.1 * np.median(fz)                     # 20 % more angle of attack, lagged by Wagner
    psi = np.degrees(a.psi[np.argmax(fz)])
    assert 240.0 < psi < 300.0                                # inboard, over the wing
    assert np.max(a.bwi_rms) > 3.0                            # the fountain's, of the hover's 21 m/s induced velocity


def test_it_is_gone_in_cruise_and_on_rotors_without_a_wing():
    ac = load_aircraft("mv22b")
    hover = operate(ac, FlightCondition(airspeed_kt=0.0), OPTS).points[0].aero
    slow = operate(ac, FlightCondition(airspeed_kt=10.0), OPTS).points[0].aero
    assert 0.0 < np.abs(slow.dlambda_fountain).max() < np.abs(hover.dlambda_fountain).max()
    assert operate(ac, FlightCondition(airspeed_kt=40.0), OPTS).points[0].aero.dlambda_fountain is None
    assert operate(load_aircraft("uh60a"), FlightCondition(airspeed_kt=0.0), OPTS).points[0].aero.dlambda_fountain is None
    air = operate(ac, FlightCondition(airspeed_kt=220.0, nacelle_deg=0.0), OPTS).points[0].aero
    assert air.dlambda_fountain is None
