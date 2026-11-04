"""The cockpit's wind: the wall pressure's level, the altitude's equivalence, the FlightGear section's volumes."""
import xml.etree.ElementTree as ET

import numpy as np

from pywopwop.atmosphere import Atmosphere
from pywopwop.model import KT, load_aircraft
from pywopwop.wind import equivalent, flightgear_wind_xml, wall_pressure_psd

from test_bank import _expr


def test_wall_pressure_is_farabee_and_casarellas():
    a = Atmosphere(0.0)
    f = np.logspace(-1, 5, 20000)
    for kt in (60.0, 150.0, 300.0):
        S = wall_pressure_psd(f, kt * KT, 2.0, a)
        p_rms = np.sqrt(np.trapezoid(S, f))
        q = 0.5 * a.rho * (kt * KT) ** 2
        assert 0.003 < p_rms / q < 0.012          # Bies's flight and wind-tunnel compilation: about 0.0066 q


def test_altitude_equivalence():
    a = Atmosphere(7600.0)
    veq, amp = equivalent(250.0, a)
    a0 = Atmosphere(0.0)
    sigma, nu_r = a.rho / a0.rho, a.nu / a0.nu
    assert abs(veq - 250.0 * nu_r ** (-1 / 6)) < 1e-9 and abs(amp - np.sqrt(sigma * sigma * nu_r)) < 1e-12


def _props(eas, h):
    a = Atmosphere(h)
    return {"/velocities/equivalent-kt": eas, "/environment/density-slugft3": a.rho / 515.379,
            "/environment/temperature-degc": a.T - 273.15}


def test_flightgear_section():
    speeds = (30.0, 60.0, 100.0, 150.0)
    m = dict(aircraft="test", files=[dict(file=f"wind_{v:g}kt_cockpit.wav", airspeed_kt=v) for v in speeds])
    fx = ET.fromstring(flightgear_wind_xml(m)).find("fx")
    vol = lambda s, p: _expr(s.find("volume/expression")[0], p)
    sounds = fx.findall("wind")
    # at sea level each loop plays alone at its own speed, at full volume
    for i, v in enumerate(speeds):
        vols = [vol(s, _props(v, 0.0)) for s in sounds]
        assert abs(vols[i] - 1.0) < 2e-3 and sum(x ** 2 for x in vols) - vols[i] ** 2 < 1e-5
    # aloft: the loops' powers add up to the amplitude's square, around the equivalent airspeed
    h = 6000.0
    a = Atmosphere(h)
    eas = 120.0 * np.sqrt(a.rho / Atmosphere(0.0).rho)                 # 120 kt true
    veq, amp = equivalent(120.0, a)
    vols = [vol(s, _props(eas, h)) for s in sounds]
    assert abs(sum(x ** 2 for x in vols) - amp ** 2) < 1e-3 * amp ** 2
    assert all(x == 0.0 for x, v in zip(vols, speeds) if abs(v - veq) > 50.0)


def test_wind_passes_the_windscreen_not_its_leaks():
    from pywopwop.wind import transmission, windscreen
    ac = load_aircraft("mv22b")
    f = np.array([100.0, 1000.0, 10000.0])
    U = 250.0 * KT
    T = transmission(f, U, windscreen(ac), 340.0)
    mass = 10 ** (-(20 * np.log10(ac.cabin.window_mass * f) - 47.0) / 10)
    fa = ac.cabin.window_coincidence_hz * (0.7 * U / 340.0) ** 2           # hydrodynamic coincidence
    assert np.allclose(T, mass * (f / fa) ** 2 / (1 + (f / fa) ** 2))
    assert T[2] < 0.1 * ac.cabin.leak                         # far under the leaks' floor: they don't carry the eddies
