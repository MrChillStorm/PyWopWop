"""Tiltrotors in airplane mode: the trim follows the aircraft's level-flight trims, the hubs swing about the pivot."""
import numpy as np

from pywopwop.aero import AeroOptions
from pywopwop.model import FlightCondition, load_aircraft
from pywopwop.operating import operate


def test_hub_swings_forward_about_the_pivot():
    ac = load_aircraft("mv22b")
    r = ac.main
    hub, axis = r.tilted(0.0)
    # in helicopter mode 1.732 m above the pivot and 0.306 m aft of it; in airplane mode as far ahead and above
    assert np.allclose(hub - np.asarray(r.pivot), [1.732, 0.0, 0.306], atol=1e-9)
    assert np.allclose(axis, [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(r.tilted(90.0)[0], r.position)


def test_airplane_trim_follows_the_table():
    ac = load_aircraft("mv22b")
    tab = np.asarray(ac.airplane_trim)
    opts = AeroOptions(n_radial=24, n_azimuth=180, bvi=False, elastic=False)
    st = operate(ac, FlightCondition(airspeed_kt=220.0, nacelle_deg=0.0), opts)
    alpha = np.interp(220.0, tab[:, 0], tab[:, 1])
    assert abs(np.degrees(st.pitch) - alpha) < 1e-9                    # level: the attitude is the angle of attack
    for p in st.points:
        assert abs(p.aero.thrust / np.interp(220.0, tab[:, 0], tab[:, 2]) - 1.0) < 1e-3
        assert abs(p.aero.omega * 60 / (2 * np.pi) - ac.airplane_rpm) < 1e-9
        assert abs(np.degrees(p.alpha_tpp) + 90.0 - alpha) < 0.01       # the flow nearly along the shaft
    # climbing, the rotors also lift the weight's component along the path
    up = operate(ac, FlightCondition(airspeed_kt=220.0, climb_fpm=1500.0, nacelle_deg=0.0), opts)
    assert up.points[0].aero.thrust > st.points[0].aero.thrust
