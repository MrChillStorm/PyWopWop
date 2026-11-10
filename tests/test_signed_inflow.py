"""a hovering rotor's annuli loaded downward (a twisted proprotor at flat pitch): the mirrored momentum balance"""
import warnings

import numpy as np

from pywopwop.aero import AeroOptions
from pywopwop.bank import RUNUP_CONDITIONS, flight_condition
from pywopwop.model import load_aircraft
from pywopwop.operating import operate


def test_flat_pitch_tips_get_upwash():
    """the MV-22B on the gear at flat pitch: clamped, its tips' downward loads take no inflow and the net thrust nearly
    cancels; mirrored, they drive air up, their angle of attack eases, and the rotor makes the flight model's thrust"""
    warnings.simplefilter("ignore")
    c = next(c for c in RUNUP_CONDITIONS if c.rpm_percent == 65.0)
    out = {}
    for signed in (False, True):
        a = load_aircraft("mv22b")
        for r in a.rotors:
            r.signed_hover_inflow = signed
        s = operate(a, flight_condition(a, c, 0.0), AeroOptions(n_radial=14, n_azimuth=24))
        pts = s.points if hasattr(s, "points") else s
        out[signed] = sum(p.thrust for p in pts) / (a.mass_kg * 9.80665)
    assert out[True] > out[False] + 0.02
    assert 0.03 < out[True] < 0.12          # the flight model: 0.060 at 66 % with the lever down
    assert load_aircraft("mv22b").main.signed_hover_inflow and not load_aircraft("uh60a").main.signed_hover_inflow
