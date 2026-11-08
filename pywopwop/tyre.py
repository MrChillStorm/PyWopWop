"""Tyres rolling on a runway: a measured tyre's rolling noise, scaled to an aircraft's tyre by the physics of its mechanisms.

A tyre rolling on a paved surface sounds by two kinds of mechanism (Hoever 2014, Winroth, Hoever and Kropp 2013). The road's
texture shakes the tread and belt, mostly below about 1 kHz; their sound power grows as the speed squared, and with the
texture's amplitude at the wavelength the speed maps to each frequency (lambda = V / f). Air squeezed and sucked in and
out of the contact radiates as monopoles, mostly above 1 kHz, its power growing as the speed to the fourth. The horn
between tread and ground amplifies both. A full model of the belt's vibration (Kropp's plate, Hoever's waveguide finite
elements) needs the belt's layered stiffness, mass and damping, which no maker publishes for a bias-ply aircraft tyre; so
here the spectrum is one that was measured, scaled by those mechanisms:

- The reference: Hoever's 315/80 R22.5 truck tyre on a 0/8 stone mastic asphalt at 60 km/h, nine microphones on a half
  circle 1 m from the contact (his fig. 6.22), within a hemisphere over the ground, so its sound power follows.
- Its two parts by frequency: vibration below and air flow above a crossover at 1 kHz, each by its own speed law.
- The vibration part moves with the runway's texture against the reference's, at the wavelength the speed maps to.
- The power scales with the contact's width; the spectrum's features move as the tyre's size, frequency as one over the
  square root of its diameter (that turns the truck's 630 Hz peak into a car tyre's 800 Hz).
- In the cockpit: the airborne part through the cabin's walls, plus what the wheels put into the structure, which no
  model here computes: recorded landings give that, an ASK 21's and a P2006T's (STRUCTURES). It is 20 to 30 dB over the
  airborne part, so a cockpit is what its structure makes it: the glider's wheel, unsprung under the seat, puts in 4 to
  14 dB more above 250 Hz than the P2006T's sprung gear does.

Checked: the reference itself, 85.8 dB(A) at 1 m against the measured 85.1; a car with four tyres at 80 km/h on an ISO
10844 surface, 73.9 dB(A) at 7.5 m, about 4 dB over the 70 typical of the EU's tyre labels for such tyres; a light
aircraft towed at 55 mph on smooth asphalt on three 5.00-5 ribbed tyres, 60.5 dB(A) 30 m to the side, against a
measurement of 60 quoted (without a source) in US patent 11198519. In a cockpit the roll sounds as the recordings' do:
Gaussian noise, no tone and nothing once a wheel's turn. Good to perhaps 5 dB outside on asphalt; in a cockpit the
structure's transfer rests on one landing per airframe, and how it grows with speed is the model's alone. Not in it:
grass and concrete (no texture data here), the tyre's load and pressure (rolling noise changes little with either), wet
runways, the touchdown's chirp, brakes and skids.
"""
import json
from pathlib import Path

import numpy as np

from .glider import BANDS

BAND_WIDTH = 2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0)
RHO_C = 1.2 * 343.0

# The reference tyre (Hoever 2014, fig. 6.22, measured): A-weighted third-octave bands at 1 m, 60 km/h, 1000 kg on the tyre
REFERENCE = dict(width_m=0.315, diameter_m=1.07, kmh=60.0, surface="sma08", distance_m=1.0,
                 hz=[315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 2000.0],
                 dba=[70.2, 69.3, 75.7, 79.1, 78.1, 77.3, 77.1, 71.6])
CROSSOVER_HZ = 1000.0          # vibration below, air flow above (Hoever 2014, section 2.2)

# Road texture, dB re 1 um by third-octave wavelength (Hoever 2014, fig. 6.6a, read to about 2 dB)
TEXTURE = {
    "wavelength_mm": [1.0, 2.0, 4.0, 8.0, 16.0, 31.5, 63.0, 125.0, 200.0],
    "iso": [31.0, 31.0, 34.0, 36.0, 37.0, 38.0, 38.0, 37.0, 36.0],       # ISO 10844 reference surface: smooth asphalt
    "sma08": [43.0, 44.0, 46.5, 49.0, 51.0, 50.5, 48.0, 45.0, 42.0],     # stone mastic asphalt 0/8: a runway's asphalt
}
SURFACES = {"smooth asphalt": "iso", "asphalt": "sma08"}

# What a wheel puts into a cockpit through the structure, by airframe: the cockpit's level per third octave less the
# tyre's sound power level (dB re 20 uPa less dB re 1 pW), 126 Hz to 2 kHz (tools/tyre_recordings.py). Outside
# those bands no recording tells the wheel from the airflow and the engines, so it falls away there by 12 dB an octave.
# Both were recorded by cameras fixed to the airframe, whose microphones may hear its shaking too.
STRUCTURES = {
    # An ASK 21B's front seat, its main wheel unsprung 2 m away under the seat, in a thin glass-fibre shell; a
    # touchdown on a paved runway at 44 kt. The mean of two ways of setting the camera's gain, which agree to 1.6 dB
    # a band on average, 4 at most: the airflow just before the touchdown, against glider.py's, for the roll's first
    # 0.3 s, before the camera's gain control acts (the 2-5 kHz airflow doesn't change as the roll starts, the
    # roll's bands jump 15 dB); and glider.py's 2-5 kHz airflow in the next 4 s. Lower speeds can't be calibrated:
    # there the 2-5 kHz is the roll's own and the gain control holds the level. Through the shell the tyre's
    # airborne sound comes 30 dB under this.
    "ask21": dict(hz=[126.0, 158.0, 200.0, 251.0, 316.0, 398.0, 501.0, 631.0, 794.0, 1000.0, 1259.0, 1585.0, 1995.0],
                  db=[29.4, 18.7, 13.4, 12.1, 7.6, 1.8, -5.6, -4.7, -8.0, -7.7, -13.1, -17.6, -20.6]),
    # A Tecnam P2006T's cabin (the X-57's airframe), sprung gear, a camera at the right seat: the first 3 s after a
    # touchdown at about 58 kt on an asphalt runway, the two main wheels down, before braking (3 s later, all three
    # down, gives about 1 dB less). The camera's gain set by wind.py's windscreen noise for the X-57 at 65 kt in the
    # 2-5 kHz just before the touchdown (an idling engine's line at 557 Hz moves under 1 dB through it, so the gain
    # held); so the roll stands to the X-57's wind loops as it did in that cabin. At the touchdown the cabin's
    # 150-1700 Hz rises 1.2 dB against its low bands and 3.6 against its high ones, both of which fall as the
    # engines and the wind slow: so the roll is at most a quarter to a half of that band, and is taken as half of
    # each band's level, the upper end. In the 126 and 158 Hz bands the cabin's level fell at the touchdown, so half
    # is likely too much there. 55 or 62 kt would move it by about 1 dB.
    "p2006t": dict(hz=[126.0, 158.0, 200.0, 251.0, 316.0, 398.0, 501.0, 631.0, 794.0, 1000.0, 1259.0, 1585.0, 1995.0],
                   db=[27.5, 16.7, 14.4, 2.5, -1.0, -2.7, -9.2, -11.8, -16.7, -21.6, -23.0, -26.0, -28.9]),
}


def a_weight(f):
    f2 = np.asarray(f, float) ** 2
    ra = 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
    return 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def _reference_shape():
    """The reference's unweighted levels at 1 m: a parabola in octaves fitted to the measured bands from 500 Hz up (below,
    NASA-style near-field effects at 1 m make the measurement less certain, Hoever says), steepening no further than
    12 dB an octave beyond its last band. Returns a function of frequency, dB."""
    f = np.asarray(REFERENCE["hz"])
    L = np.asarray(REFERENCE["dba"]) - a_weight(f)
    use = f >= 500.0
    x = np.log2(f[use] / 1000.0)
    c2, c1, c0 = np.polyfit(x, L[use], 2)

    def level(fq):
        o = np.log2(np.asarray(fq, float) / 1000.0)
        lo, hi = np.log2(f[0] / 1000.0) - 1.0, np.log2(f[-1] / 1000.0) + 1.0
        q = lambda t: c2 * t ** 2 + c1 * t + c0
        dq = lambda t: 2 * c2 * t + c1
        out = q(np.clip(o, lo, hi))
        s_lo, s_hi = min(max(dq(lo), 0.0), 12.0), max(min(dq(hi), 0.0), -12.0)
        return np.where(o < lo, out + s_lo * (o - lo), np.where(o > hi, out + s_hi * (o - hi), out))
    return level


_SHAPE = _reference_shape()


def texture_db(surface, wavelength_m):
    """the surface's texture level at the wavelengths, dB re 1 um"""
    key = SURFACES.get(surface, surface)
    lam = np.log(np.clip(np.asarray(wavelength_m, float) * 1e3, 1.0, 200.0))
    return np.interp(lam, np.log(TEXTURE["wavelength_mm"]), TEXTURE[key])


def tyre_power(width_m, diameter_m, speed_ms, surface="asphalt"):
    """Sound power per third-octave band (W) of one tyre rolling at the speed: the reference's, scaled (see the module)"""
    if speed_ms <= 0.0:
        return np.zeros(len(BANDS))
    f = BANDS
    v_ref = REFERENCE["kmh"] / 3.6
    # the features move as one over the square root of the diameter: the reference's band for each of ours
    shift = np.sqrt(diameter_m / REFERENCE["diameter_m"])
    L = _SHAPE(f * shift)
    w_air = 1.0 / (1.0 + (CROSSOVER_HZ / (f * shift)) ** 4)
    tex = texture_db(surface, speed_ms / f) - texture_db(REFERENCE["surface"], v_ref / (f * shift))
    vib = (1.0 - w_air) * (speed_ms / v_ref) ** 2 * 10.0 ** (tex / 10.0)
    air = w_air * (speed_ms / v_ref) ** 4
    p2_1m = 4e-10 * 10.0 ** (L / 10.0) * (vib + air) * (width_m / REFERENCE["width_m"])
    return p2_1m * 2.0 * np.pi * REFERENCE["distance_m"] ** 2 / RHO_C      # over a hemisphere


def tyre_noise(tyres, speed_ms, listener, surface="asphalt"):
    """mean-square pressure per third-octave band (Pa^2) at the listener (m, the aircraft's frame) from the tyres
    ({position (the contact, m), width_m, diameter_m}), each over a hard ground (a hemisphere)"""
    out = np.zeros(len(BANDS))
    for t in tyres:
        r = max(float(np.linalg.norm(np.asarray(listener, float) - np.asarray(t["position"], float))), 0.3)
        out = out + tyre_power(t["width_m"], t["diameter_m"], speed_ms, surface) * RHO_C / (2.0 * np.pi * r ** 2)
    return out


def structure_db(f, airframe="p2006t"):
    """the airframe's transfer (STRUCTURES) at the frequencies, falling 12 dB an octave outside its bands"""
    hz, db = np.log2(STRUCTURES[airframe]["hz"]), np.asarray(STRUCTURES[airframe]["db"])
    o = np.log2(np.asarray(f, float))
    return np.interp(o, hz, db) - 12.0 * (np.maximum(hz[0] - o, 0.0) + np.maximum(o - hz[-1], 0.0))


def cockpit_noise(tyres, speed_ms, head, walls=None, surface="asphalt", airframe="p2006t"):
    """mean-square pressure per third-octave band (Pa^2) at the pilot's head: (airborne, through the structure). walls: the
    power the cabin's walls let in by frequency (model.Cabin.inside); the structure's part is the airframe's transfer
    (STRUCTURES) for every wheel"""
    air = tyre_noise(tyres, speed_ms, head, surface)
    if walls is not None:
        air = air * walls(BANDS)
    structure = sum(tyre_power(t["width_m"], t["diameter_m"], speed_ms, surface) for t in tyres) / 1e-12 * 4e-10 \
        * 10.0 ** (structure_db(BANDS, airframe) / 10.0)
    return air, structure


# ---------------------------------------------------------------- loops, a bank and its FlightGear section

SPEEDS_KT = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0)     # held at 90 kt beyond


def noise_loop(p2, fs=48000.0, duration=15.0, seed=1, f_highpass=25.0):
    """A seamless loop of Gaussian noise with the third-octave band powers p2 (Pa^2): a rolling tyre's sound is that, no
    tones and no thump once a revolution in the recording (its envelope is Rayleigh's, 0.45 to 0.51 of its mean)"""
    from .synth import highpass
    n = int(round(duration * fs))
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / fs), 1e-6)
    S = np.exp(np.interp(np.log(f), np.log(BANDS), np.log(np.maximum(p2 / (BAND_WIDTH * BANDS), 1e-30))))
    S[0] = 0.0
    rng = np.random.default_rng(seed)
    X = n * np.sqrt(S * fs / (2 * n)) * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(2.0)
    return highpass(np.fft.irfft(X, n), fs, f_highpass)


def soft_limit(y, ceiling_db=-0.1, knee=0.85):
    """y (full scale 1) with what is over knee of the ceiling bent smoothly under it, and how many samples that took.
    Gaussian noise peaks rarely: allowing a thousandth of the samples, those beyond 3.3 standard deviations, and bending
    them changes nothing heard (its power by under 0.01 dB)."""
    c = 10 ** (ceiling_db / 20.0)
    a = np.abs(y)
    over = a > knee * c
    if not over.any():
        return y, 0
    k0 = knee * c
    out = y.copy()
    out[over] = np.sign(y[over]) * (k0 + (c - k0) * np.tanh((a[over] - k0) / (c - k0)))
    return out, int(over.sum())


def low_speed_power(width_m, diameter_m, v_low_ms, surface="asphalt"):
    """the exponent of the A-weighted sound power against speed below the lowest loop (for the fade there)"""
    a = 10.0 ** (a_weight(BANDS) / 10.0)
    v = v_low_ms * np.array([0.2, 0.5, 1.0])
    P = [np.sum(tyre_power(width_m, diameter_m, x, surface) * a) for x in v]
    return float(np.polyfit(np.log(v), np.log(P), 1)[0])


def make_tyre_bank(aircraft, out_dir, tyres, gain_db, airframe, speeds=SPEEDS_KT, exterior_m=30.0, surface="asphalt",
                   fs=48000.0, duration=15.0, seed=1, progress=print):
    """Sea level loops of each tyre rolling at the ground speeds: outside exterior_m to the side, and in the cockpit (its
    airborne part and its part through the structure in one loop; the manifest has each). tyres: {name, gear (FlightGear's
    gear index), position (the contact, m, the aircraft's frame), width_m, diameter_m, static_compression_ft (the flight
    model's at rest, for the load)}. gain_db: a bank's {cockpit, exterior}, so that their loudness against its loops is the
    physics'. airframe: whose structure's transfer the cockpit takes (STRUCTURES). Writes manifest.json and
    pywopwop-tyres.xml."""
    import zlib
    from .model import Observer, STANDARD_OBSERVERS
    from .synth import a_weighted_ms, write_wav
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    so = STANDARD_OBSERVERS["side"]
    side = Observer("side", so.azimuth_deg, so.elevation_deg, exterior_m).offset()
    walls = aircraft.cabin.inside if aircraft.cabin is not None else None
    A = 10.0 ** (a_weight(BANDS) / 10.0)
    dba_of = lambda p2: round(float(10 * np.log10(max(np.sum(p2 * A), 1e-30) / 4e-10)), 2)
    files = []
    for t in tyres:
        for i, v in enumerate(speeds):
            V = v * 0.514444
            air, structure = cockpit_noise([t], V, aircraft.cockpit, walls, surface, airframe)
            for view, p2, g, parts in (("side", tyre_noise([t], V, side, surface), gain_db["exterior"], {}),
                                       ("cockpit", air + structure, gain_db["cockpit"],
                                        dict(airborne_dba=dba_of(air), structure_dba=dba_of(structure)))):
                fn = f"tyre_{t['name']}_{v:g}kt_{view}.wav"
                x = noise_loop(p2, fs, duration, seed + zlib.crc32(fn.encode()) % 100003)
                y, limited = soft_limit(x * 10 ** (g / 20.0))
                if limited > 1e-3 * len(x):
                    raise ValueError(f"{fn} would peak at {20 * np.log10(np.abs(x).max() * 10 ** (g / 20.0)):+.1f} dBFS "
                                     "at this gain: too loud to limit")
                write_wav(out / fn, y, fs, 16, 1.0, seed=int(v) + i)
                oaspl = 10 * np.log10(np.mean(x ** 2) / 4e-10)
                dba = 10 * np.log10(a_weighted_ms(x, fs) / 4e-10)
                files.append(dict(file=fn, tyre=t["name"], view=view, groundspeed_kt=v, oaspl_db=round(float(oaspl), 2),
                                  dba=round(float(dba), 2), limited_samples=int(limited), **parts))
                progress(f"  {t['name']:11s} {v:4.0f} kt {view:8s} {oaspl:6.1f} dB {dba:6.1f} dBA"
                         + (f"  (airborne {parts['airborne_dba']:.1f}, structure {parts['structure_dba']:.1f})" if parts else ""))
    power = {t["name"]: round(low_speed_power(t["width_m"], t["diameter_m"], speeds[0] * 0.514444, surface), 2) for t in tyres}
    manifest = dict(aircraft=aircraft.name, generator="pywopwop tyre", sample_rate=fs, exterior_distance_m=exterior_m,
                    gain_db=gain_db, surface=surface, structure=airframe, speeds_kt=list(speeds), low_speed_power=power,
                    tyres=[{k: (list(map(float, v)) if k == "position" else v) for k, v in t.items()} for t in tyres],
                    properties=PROPERTIES, files=files, max_sources=2 * len(tyres),
                    note="positions: m from the moment reference, the aircraft's frame (x forward, y left, z up); outside: the tyre's airborne sound, scaled from a measured truck tyre (Hoever 2014) and checked "
                         "against one aircraft measurement; cockpit: mostly what the wheels put in through the structure, "
                         f"from one recorded landing in the {airframe} airframe's cabin; asphalt only")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "pywopwop-tyres.xml").write_text(flightgear_tyre_xml(manifest))
    return manifest


# what the flight model gives, per wheel (n: FlightGear's gear index): the wheel's rolling speed (m/s), its compression
# (JSBSim's in feet, whatever the name says) and weight on wheels
PROPERTIES = dict(speed="/gear/gear[{n}]/rollspeed-ms", speed_scale=1.0 / 0.514444, load="/gear/gear[{n}]/compression-norm",
                  wow="/gear/gear[{n}]/wow")
LOAD_FULL = 0.25    # the share of its static load from which a wheel sounds in full: rolling noise changes little with load


def flightgear_tyre_xml(manifest, sound_dir="Sounds/pywopwop", exterior="side"):
    """FlightGear <fx> entries: each tyre's loops cross-faded by equal power over its rolling speed (knots), below the lowest
    loop falling as the speed to its low_speed_power. A wheel lifting off fades out by equal power as its compression falls
    under LOAD_FULL of its static compression. Each loop's condition is only true inside its own speeds and with weight on
    the wheel: a looped sound holds one of SimGear's 128 sources while its condition is true."""
    from .wind import _e, _v, _ramp_e
    P = manifest["properties"]
    speeds = manifest["speeds_kt"]
    have = {(f["tyre"], f["groundspeed_kt"], f["view"]): f for f in manifest["files"]}
    out = []
    for t in manifest["tyres"]:
        n = t["gear"]
        kt = _e("product", f"<property>{P['speed'].format(n=n)}</property>", _v(P["speed_scale"]))
        c0 = LOAD_FULL * t["static_compression_ft"]
        load = _e("sqrt", _e("clip", _e("product", f"<property>{P['load'].format(n=n)}</property>", _v(1.0 / c0)),
                             "<clipMin>0</clipMin><clipMax>1</clipMax>"))
        k = manifest["low_speed_power"][t["name"]]
        pos = t["position"]
        for view in ("cockpit", exterior):
            for i, v in enumerate(speeds):
                f = have.get((t["name"], v, view))
                if f is None:
                    continue
                rise = _ramp_e(kt, speeds[i - 1] if i else 0.0, v)
                fall = _ramp_e(kt, speeds[i + 1], v) if i < len(speeds) - 1 else None
                w = _e("pow", rise, _v(0.5 * k)) if i == 0 else _e("sqrt", rise)
                w = _e("product", w, _e("sqrt", fall)) if fall else w
                lo = f"<greater-than>{kt}<value>{(speeds[i - 1] if i else 0.0):g}</value></greater-than>"
                hi = f"<less-than>{kt}<value>{speeds[i + 1]:g}</value></less-than>" if fall else ""
                view_c = ("<property>sim/current-view/internal</property>" if view == "cockpit"
                          else "<not><property>sim/current-view/internal</property></not>")
                cond = _e("and", view_c, f"<property>{P['wow'].format(n=n)}</property>", lo + hi)
                out.append(f"\t\t<tyre>\n\t\t\t<name>pww-tyre-{t['name']}-{v:g}kt-{view}</name>\n"
                           f"\t\t\t<mode>looped</mode>\n\t\t\t<path>{sound_dir}/{f['file']}</path>\n"
                           f"\t\t\t<condition>{cond}</condition>\n"
                           f"\t\t\t<volume><expression>{_e('product', w, load)}</expression></volume>\n"
                           f"\t\t\t<position>\n\t\t\t\t<x>{-pos[0] + 0.0:.3f}</x>\n\t\t\t\t<y>{pos[1] + 0.0:.3f}</y>\n"
                           f"\t\t\t\t<z>{pos[2]:.3f}</z>\n\t\t\t</position>\n"
                           + ("" if view == "cockpit" else
                              f"\t\t\t<reference-dist>{manifest['exterior_distance_m']:g}</reference-dist>\n"
                              "\t\t\t<max-dist>1500</max-dist>\n")
                           + "\t\t</tyre>\n")
    return ("<?xml version=\"1.0\"?>\n<!-- generated by pywopwop: the tyres of the " + manifest["aircraft"]
            + "; merge into the aircraft's sound file -->\n<PropertyList>\n\t<fx>\n" + "".join(out)
            + "\t</fx>\n</PropertyList>\n")
