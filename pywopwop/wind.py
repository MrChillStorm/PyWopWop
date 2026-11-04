"""Wind noise in the cockpit: the turbulent boundary layer's pressure fluctuations on the windscreen, heard through it.

The boundary layer at the windscreen is a turbulent flat plate's from the nose (the 1/7-power law: delta = 0.37 x
Re_x^-1/5, C_f = 0.0592 Re_x^-1/5), at the true airspeed and the air's density and viscosity. The wall pressure's
spectrum is Goody's (AIAA J. 42(9), 2004) shape,

    phi(omega) U / (tau_w^2 delta) = 3 w^2 / ((w^0.75 + 0.5)^3.7 + (1.1 R_T^-0.57 w)^7),   w = omega delta / U,
    R_T = (delta / U) / (nu / u_tau^2),

at Farabee and Casarella's measured level (Phys. Fluids A 3, 1991), p_rms^2 / tau_w^2 = 6.5 + 1.86 ln(Re_tau / 333):
Goody's own integral overpredicts the variance 1.5-2 times (Pargal et al. 2023), and Bies's compilation of flight and
wind-tunnel data (NASA CR-626), p_rms = 0.0066 q, is within about 1 dB of it. Into the cockpit as Levine and DeFelice
predict a helicopter's cabin noise (NASA 19790002686: within 1.5 dB of the measured speech interference level of
NASA's and Sikorsky's research helicopter): the wall pressure as the level on the skin, less the windscreen's
transmission loss by the mass law (the cabin's window panel; its skin without windows; or a limp wall of a given
surface mass). Above the wall pressure's peak (omega delta / U about 1) Goody's spectrum falls about 2 dB an octave and
the mass law 6 more, as a cockpit recorded in cruise does (an MV-22B's, 400 Hz to 5 kHz); a coincidence dip there, which
sound arriving from all directions meets, would let the highs through, and the recording has none. The boundary
layer's convected eddies pass no gap the way sound does, so the cabin's leaks don't carry them: their floor under the
walls' loss, which lets an outside source's sound in, would leave the wind flat to 10 kHz. Below hydrodynamic
coincidence (the panel's bending waves slower than the eddies, convected at 0.7 of the flow's speed) the panel couples
poorly to them: a second-order roll-off, (f/f_a)^2 / (1 + (f/f_a)^2), at f_a = f_c (0.7 U / c)^2 from its acoustic
coincidence f_c, as for a glider's canopy (glider.py). Inside, it builds up as the cockpit's absorption lets it
(model.Cabin.room), as the outside sources' sound does. Not in it: separated flow (the windscreen's frame, wipers),
the leaks' own flow noise, compressibility (past Mach 0.5 or so).

At altitude, the spectrum's frequency scale U/delta goes as U^1.2 nu^-0.2 and the wall shear stress as rho U^1.8
nu^0.2: a true airspeed V with kinematic viscosity nu sounds as the sea level's at V_eq = V (nu/nu_0)^-1/6, louder in
amplitude by sqrt(sigma mu/mu_0) (sigma the density ratio). So a bank of loops at sea level, cross-faded by V_eq and
scaled, covers any altitude: a simulator computes V_eq and the scale from the equivalent airspeed, the air's density
and temperature (equivalent())."""
import json
from pathlib import Path

import numpy as np

from .atmosphere import Atmosphere
from .model import KT
from .synth import highpass, write_wav

SPEEDS_KT = (30.0, 60.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)


def boundary_layer(U, x, atm):
    """the turbulent boundary layer x (m) from the nose at U (m/s): thickness (m), wall shear stress (Pa), Re_tau, R_T"""
    nu = atm.nu
    re = U * x / nu
    delta = 0.37 * x * re ** -0.2
    tau = 0.5 * atm.rho * U ** 2 * 0.0592 * re ** -0.2
    ut = np.sqrt(tau / atm.rho)
    return delta, tau, ut * delta / nu, (delta / U) / (nu / ut ** 2)


def _goody(w, rt):
    return 3.0 * w ** 2 / ((w ** 0.75 + 0.5) ** 3.7 + (1.1 * rt ** -0.57 * w) ** 7)


def wall_pressure_psd(f, U, x, atm):
    """the wall pressure's one-sided spectrum (Pa^2/Hz) at the frequencies f (Hz)"""
    delta, tau, re_tau, rt = boundary_layer(U, x, atm)
    p2 = tau ** 2 * (6.5 + 1.86 * np.log(max(re_tau, 333.0) / 333.0))
    return goody_psd(f, U, delta, p2, rt)


def goody_psd(f, U, delta, p2, rt):
    """Goody's shape for a layer delta (m) thick under the flow speed U (m/s), with the variance p2 (Pa^2) and the
    ratio of outer to inner time scales rt: the one-sided spectrum (Pa^2/Hz) at the frequencies f (Hz)"""
    s = np.linspace(-12.0, 12.0, 4801)
    integral = np.trapezoid(_goody(np.exp(s), rt) * np.exp(s), s)        # over w, 0 to infinity
    w = 2.0 * np.pi * np.asarray(f, float) * delta / U
    return p2 * _goody(w, rt) / integral * delta / U * 2.0 * np.pi


def windscreen(aircraft, surface_mass=None):
    """the panel the wind drives: (surface mass kg/m^2, coincidence Hz or None, loss factor), the cabin's windows'
    (its skin's without windows), or a limp wall of surface_mass"""
    c = aircraft.cabin
    if surface_mass is not None:
        return float(surface_mass), None, 0.02
    if c.window_share > 0:
        return c.window_mass, c.window_coincidence_hz, c.window_loss_factor
    return c.surface_mass, c.coincidence_hz, c.loss_factor


def transmission(f, U, panel, c0):
    """the share of the wall pressure's power heard inside through the panel at the flow speed U: its coupling to the
    convected eddies below hydrodynamic coincidence and its transmission (no leaks)"""
    from .model import Cabin
    m, fc, eta = panel
    T = 10.0 ** (-Cabin.panel_tl(f, m, None, eta) / 10.0)        # the mass law: the coincidence dip is sound's
    if not fc:
        return T
    fa = fc * (0.7 * U / c0) ** 2
    r = (np.asarray(f, float) / fa) ** 2
    return T * r / (1.0 + r)


def skin_wind_psd(aircraft, f, U, atm, listener=None):
    """the boundary layer over the whole skin round the cockpit and cabin (the aircraft's fuselage sections): each wall
    patch's wall pressure at its run from the nose (the foremost section), by its area and its direct field at the
    listener (default the cockpit's seat), through its walls (the windows where the glazing map has them, the opaque
    walls elsewhere; without a map both by their shares; model.Cabin.convected; no leaks): the power heard inside
    before the interior's build-up"""
    from .model import glazed_patches, wall_patches
    c = aircraft.cabin
    P, A = wall_patches(aircraft.fuselage)
    G = glazed_patches(P, aircraft.glazing) if aircraft.glazing else [None] * len(P)
    L = np.asarray(listener if listener is not None else aircraft.cockpit, float)
    nose = max(s[0] for s in aircraft.fuselage)
    T = {g: c.convected(f, 0.7 * U, g, atm.c) for g in set(G)}
    return sum(a * wall_pressure_psd(f, U, max(nose - p[0], 0.1), atm) * T[g]
               * (1.0 + c.direct(f, np.linalg.norm(p - L), a, A.sum())) for p, a, g in zip(P, A, G)) / A.sum()


def wind_loop(aircraft, airspeed_kt, altitude_m=0.0, fs=48000.0, duration=15.0, surface_mass=None, seed=1,
              f_highpass=25.0, listener=None):
    """a seamless loop of the wind at the pilot's head (Pa): random phases with the wall pressure's spectrum, one
    spectrum for the whole loop (exactly periodic), through the windscreen (surface_mass: a limp one instead), or for
    an aircraft with a fuselage description the boundary layer over all of its skin round the cockpit and cabin"""
    if not aircraft.windscreen_m and not aircraft.fuselage:
        raise ValueError(f"{aircraft.name} has no windscreen_m (its distance from the nose) or fuselage")
    atm = Atmosphere(altitude_m)
    n = int(round(duration * fs))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    U = airspeed_kt * KT
    if aircraft.fuselage and surface_mass is None:
        S = skin_wind_psd(aircraft, np.maximum(f, 1e-6), U, atm, listener)
    else:
        S = wall_pressure_psd(np.maximum(f, 1e-6), U, aircraft.windscreen_m, atm)
        S = S * transmission(np.maximum(f, 1e-6), U, windscreen(aircraft, surface_mass), atm.c)
    S = S * aircraft.cabin.room(f)
    S[0] = 0.0
    rng = np.random.default_rng(seed)
    X = n * np.sqrt(S * fs / (2 * n)) * np.exp(2j * np.pi * rng.random(len(f)))
    return highpass(np.fft.irfft(X, n), fs, f_highpass)


def equivalent(airspeed_kt, atm):
    """the sea level's airspeed whose wind has the same spectrum (kt), and the amplitude against it"""
    a0 = Atmosphere(0.0)
    nu_r = atm.nu / a0.nu
    sigma = atm.rho / a0.rho
    return airspeed_kt * nu_r ** (-1.0 / 6.0), np.sqrt(sigma * nu_r * sigma)


# ---------------------------------------------------------------- a bank of wind loops and its FlightGear section

def _e(tag, *kids):
    return f"<{tag}>" + "".join(kids) + f"</{tag}>"


def _v(x):
    return f"<value>{x:.7g}</value>"


def _ramp_e(e, a, b):
    """0 at a, 1 at b of the expression e (either order), clipped"""
    k = 1.0 / (b - a)
    return _e("clip", _e("sum", _e("product", e, _v(k)), _v(-a * k)), "<clipMin>0</clipMin><clipMax>1</clipMax>")


def open_weights(openings):
    """FlightGear expressions weighting the shut cockpit's loops and the all-open cockpit's ([{share, property}]):
    the area open as a share u of all that can open; the two loops are one signal through two walls, the power
    through the openings grows with their area, so by equal power: sqrt(1 - u) and sqrt(u)"""
    total = sum(o["share"] for o in openings)
    u = _e("clip", _e("sum", *[_e("product", f"<property>{o['property']}</property>", _v(o["share"] / total))
                               for o in openings]), "<clipMin>0</clipMin><clipMax>1</clipMax>")
    return _e("sqrt", _e("difference", _v(1.0), u)), _e("sqrt", u)


def flightgear_wind_xml(manifest, sound_dir="Sounds/pywopwop"):
    """FlightGear <fx> entries for the cockpit's wind: each sea level loop's weight by the equivalent sea level airspeed
    V_eq (powers added: the loops are unrelated noise), times the amplitude at altitude, from FlightGear's equivalent
    airspeed, the air's density and temperature (Sutherland's viscosity)"""
    sigma = _e("div", "<property>/environment/density-slugft3</property>", _v(0.00237689))
    T = _e("sum", "<property>/environment/temperature-degc</property>", _v(273.15))
    mu_r = _e("product", _e("pow", _e("div", T, _v(288.15)), _v(1.5)), _e("div", _v(398.55), _e("sum", T, _v(110.4))))
    nu_r = _e("div", mu_r, sigma)
    tas = _e("div", "<property>/velocities/equivalent-kt</property>", _e("sqrt", sigma))
    veq = _e("product", tas, _e("pow", nu_r, _v(-1.0 / 6.0)))
    amp = _e("sqrt", _e("product", sigma, mu_r))
    speeds = [e["airspeed_kt"] for e in manifest["files"]]
    out = []
    for i, e in enumerate(manifest["files"]):
        w = []
        w.append(_ramp_e(veq, speeds[i - 1] if i else 0.0, speeds[i]))
        if i < len(speeds) - 1:
            w.append(_ramp_e(veq, speeds[i + 1], speeds[i]))
        vol = _e("product", _e("sqrt", _e("product", *w) if len(w) > 1 else w[0]), amp)
        out.append(f"\t\t<wind>\n\t\t\t<name>pww-wind-{e['airspeed_kt']:g}</name>\n\t\t\t<mode>looped</mode>\n"
                   f"\t\t\t<path>{sound_dir}/{e['file']}</path>\n"
                   "\t\t\t<condition><property>sim/current-view/internal</property></condition>\n"
                   f"\t\t\t<volume><expression>{vol}</expression></volume>\n\t\t</wind>\n")
    return ("<?xml version=\"1.0\"?>\n<!-- generated by pywopwop: the cockpit's wind for " + manifest["aircraft"]
            + "; merge into the aircraft's sound file -->\n<PropertyList>\n\t<fx>\n" + "".join(out)
            + "\t</fx>\n</PropertyList>\n")


def make_wind_bank(aircraft, out_dir, speeds=SPEEDS_KT, fs=48000.0, duration=15.0, surface_mass=None, gain_db=None,
                   seed=1, progress=print):
    """sea level wind loops in the cockpit at the speeds, manifest.json and pywopwop-wind.xml. gain_db: the loops'
    gain (a rotor bank's cockpit gain, so that their loudness against its loops is the physics'); default: the
    loudest at -1 dBFS"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loops = [(v, wind_loop(aircraft, v, 0.0, fs, duration, surface_mass, seed + i)) for i, v in enumerate(speeds)]
    peak = max(np.abs(x).max() for _, x in loops)
    gain = 10 ** (gain_db / 20.0) if gain_db is not None else 10 ** (-1.0 / 20.0) / peak
    if peak * gain > 10 ** (-0.1 / 20.0):
        raise ValueError(f"the loudest wind loop would peak at {20 * np.log10(peak * gain):+.1f} dBFS at this gain")
    files = []
    from .synth import a_weighted_ms
    for v, x in loops:
        fn = f"wind_{v:g}kt_cockpit.wav"
        write_wav(out / fn, x, fs, 16, gain, seed=int(v))
        oaspl = 10 * np.log10(np.mean(x ** 2) / 4e-10)
        dba = 10 * np.log10(a_weighted_ms(x, fs) / 4e-10)
        files.append(dict(file=fn, airspeed_kt=v, oaspl_db=round(oaspl, 2), dba=round(dba, 2)))
        progress(f"  wind {v:5.0f} kt (sea level) cockpit {oaspl:6.1f} dB {dba:6.1f} dBA")
    m, fc, eta = windscreen(aircraft, surface_mass)
    manifest = dict(aircraft=aircraft.name, generator="pywopwop wind", sample_rate=fs, windscreen_m=aircraft.windscreen_m,
                    windscreen=dict(surface_mass=m, coincidence_hz=fc, loss_factor=eta),
                    gain_db={"cockpit": round(20 * np.log10(gain), 3)}, files=files)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "pywopwop-wind.xml").write_text(flightgear_wind_xml(manifest))
    return manifest
