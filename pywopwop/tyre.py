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
runways, brakes and skids, a runway's joints.

At touchdown (below the loops): the wheel's squeal, from the energy it loses spinning up, and the strut's thump; the
squeal's share of that loss from a P2006T's landing, which an ASK 21's comes within 3 dB of, the thump's level from one
touchdown at the edge of what its recording shows.
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


# ---------------------------------------------------------------- touchdown: a wheel spinning up, a strut's thump
#
# A wheel meets the runway not turning. Its tread slides until friction spins it up to the ground speed V: the load N on
# it gives a torque mu N r, so the slide lasts until mu r^2 times the integral of N reaches I V. Whatever N does, the slide
# turns half of the work the friction does into the wheel's spin and loses the other half, I (V / r)^2 / 2: 3.5 kJ for a
# P2006T's two mains at 58 kt. A squeal is a fixed share of that loss, so a firm touchdown gives a short loud squeal and a
# gentle one a long quiet one of the same energy. The load in the first instants is the strut's damping on the sink rate,
# rising over the time the unsprung wheel takes to squash its tyre, then its spring; and the weight the wing sheds.

# Tyres and wheels (Goodyear's databook 2022 for the sizes and loaded radii; distributor catalogues for the tyres'
# weights, 11 and 7 lb; the wheels, tubes, brakes and the tyres' vertical stiffness, from their rated loads at a third
# of their section height, estimated)
TYRE_TYPES = {
    "6.00-6": dict(diameter_m=0.4445, width_m=0.16, rolling_radius_m=0.207, tyre_kg=5.6, wheel_kg=3.0, unsprung_kg=11.6,
                   tyre_stiffness=164e3),
    "5.00-5": dict(diameter_m=0.3607, width_m=0.1257, rolling_radius_m=0.168, tyre_kg=3.65, wheel_kg=2.5, unsprung_kg=8.0,
                   tyre_stiffness=153e3),
}

# What a squeal is, recorded (tools/tyre_recordings.py): its own spectrum by third octave (dB, 500 Hz to 8 kHz: the
# cockpit's less the airframe's transfer where that was measured, to 2 kHz, falling above as the cockpit's does), and
# a line: its frequency, glide and wander, and the share of the energy it takes so that the squeal, heard in its
# cockpit, has as much within 60 Hz of its strongest line as the recording (47 and 62 %). A 6.00-6: a P2006T's two
# mains at about 58 kt, 0.10 s, the line wandering 1.08-1.27 kHz. A 5.00-5: an ASK 21's main wheel at 47 kt, its
# second touch after a skip, 0.17 s, the line gliding from 1.78 to 1.69 kHz.
SQUEAL = {
    "6.00-6": dict(hz=[501.0, 630.0, 794.0, 1000.0, 1259.0, 1585.0, 1995.0, 2512.0, 3162.0, 3981.0, 5012.0, 6310.0, 7943.0],
                   db=[-44.1, -25.3, -14.4, -6.4, -3.3, -8.7, -12.6, -13.1, -17.9, -25.6, -30.8, -33.0, -40.1],
                   tone_hz=1170.0, glide=-0.02, wander=0.06, tone_share=0.1, source="P2006T"),
    "5.00-5": dict(hz=[501.0, 630.0, 794.0, 1000.0, 1259.0, 1585.0, 1995.0, 2512.0, 3162.0, 3981.0, 5012.0, 6310.0, 7943.0],
                   db=[-16.3, -21.2, -21.8, -28.7, -13.7, -3.1, -4.3, -15.8, -17.7, -33.1, -19.8, -23.9, -27.6],
                   tone_hz=1730.0, glide=-0.05, wander=0.01, tone_share=0.6, source="ASK 21"),
}
# The squeal's sound energy against the energy the wheel loses spinning up (tools/tyre_recordings.py): the P2006T's
# two mains at 58 kt, the sound power taken from the cockpit through its rolling transfer (STRUCTURES) in the bands it
# was measured in, 500 Hz to 2 kHz, assuming a squeal's contact forces shake the structure as a rolling tyre's do. The
# ASK 21's two touches, through their own camera's gain and their own transfer, come to 3 dB under it. The slide's
# 0.101 s puts that touchdown at 0.78 fps on the X-57's struts.
SQUEAL_EFFICIENCY_DB = -48.1

# A strut's thump in the cockpit: the force step its damping puts into the airframe at contact, damping times the sink
# rate, its level as the force squared. Measured at the P2006T's main touchdown, whose sink rate the slide gives
# (tools/tyre_recordings.py): on the X-57's main struts, 8071 N s/m at 0.78 fps, 1.92 kN. What the 0.15 s from the
# touchdown hold over the approach's 0.15 s windows stands 2 to 5 dB over their scatter, so this is at the edge of
# what the recording shows, good to perhaps 5 dB, and the camera, fixed to the airframe, may hear the shaking itself.
# Third-octave levels over those 0.15 s, dB re 20 uPa, 79-630 Hz: the 126 and 158 Hz bands, where the engines' slowing
# outweighs it, interpolated; below 79 Hz the engines leave nothing to read. An event 1.8 s later, 3 to 9 dB louder,
# is likely the nose wheel coming down harder; its sink rate can't be read.
THUMP = dict(hz=[79.0, 100.0, 126.0, 158.0, 200.0, 251.0, 316.0, 398.0, 501.0, 631.0],
             db=[95.8, 101.7, 99.3, 97.0, 94.6, 89.2, 78.1, 83.5, 75.6, 80.9], window_s=0.15,
             force_n=8071.0 * 0.78 * 0.3048, decay_s=0.05)


def wheel_inertia(tyre_type):
    """the wheel's moment of inertia about its axle (kg m^2): the tyre and tube as mass at 0.8 of its radius, the wheel
    and brake at 0.35"""
    t = TYRE_TYPES[tyre_type]
    R = t["diameter_m"] / 2
    return t["tyre_kg"] * (0.8 * R) ** 2 + t["wheel_kg"] * (0.35 * R) ** 2


def spin_up(speed_ms, tyre_type, damping, spring, sink_ms, static_load_n, mu=0.5, settle_s=1.0, dt=1e-4):
    """the wheel's slide from touching down to rolling (speed_ms: its slip then, the ground speed less its rim speed; the
    ground speed for a wheel not turning): times (s), slip speed (m/s) and friction power (W). The load is
    the strut's damping on the sink rate (rising over the tyre's squash) plus its spring as it compresses, and at least
    the wheel's static load coming on over settle_s as the wing sheds it. mu: the flight model's sliding friction."""
    t = TYRE_TYPES[tyre_type]
    I, r = wheel_inertia(tyre_type), t["rolling_radius_m"]
    tau = 0.5 * np.pi * np.sqrt(t["unsprung_kg"] / t["tyre_stiffness"])
    n = int(5.0 / dt)
    tt = np.arange(n) * dt
    N = np.maximum(damping * sink_ms * np.minimum(tt / tau, 1.0) + spring * sink_ms * tt, static_load_n * tt / settle_s)
    w = np.concatenate(([0.0], np.cumsum(mu * N * r / I * dt)))[:n]
    slip = speed_ms - w * r
    end = int(np.argmax(slip <= 0.0)) if np.any(slip <= 0.0) else n
    return tt[:end], slip[:end], mu * N[:end] * slip[:end]


def _shape(tab, f):
    o = np.log2(np.asarray(f, float))
    return np.interp(o, np.log2(tab["hz"]), tab["db"], left=tab["db"][0] - 60.0, right=tab["db"][-1] - 60.0) \
        - 12.0 * (np.maximum(np.log2(tab["hz"][0]) - o, 0.0) + np.maximum(o - np.log2(tab["hz"][-1]), 0.0))


def _burst(band_energy, tone_energy, tone_hz, glide, envelope, fs, seed, pre=0.01, post=0.05, wander=0.0):
    """a one-shot of noise with these band energies (Pa^2 s per band) and a tone of this energy at tone_hz (gliding by
    the fraction glide, wandering by the fraction wander over some 20 ms), both following envelope (power, the slide's
    samples at fs, mean 1)"""
    rng = np.random.default_rng(seed)
    m = len(envelope)
    n = int(round((pre + post) * fs)) + m
    nf = 1 << int(np.ceil(np.log2(max(n, 2048))))
    f = np.maximum(np.fft.rfftfreq(nf, 1.0 / fs), 1e-6)
    dur = m / fs
    S = np.exp(np.interp(np.log(f), np.log(BANDS), np.log(np.maximum(band_energy / dur / (BAND_WIDTH * BANDS), 1e-30))))
    S[0] = 0.0
    X = nf * np.sqrt(S * fs / (2 * nf)) * np.exp(2j * np.pi * rng.random(len(f)))
    noise = np.fft.irfft(X, nf)[:m]
    noise = noise / np.sqrt(np.mean(noise ** 2) + 1e-30) * np.sqrt(np.sum(band_energy) / dur)
    k = np.arange(m) / fs
    drift = np.convolve(rng.normal(size=m + int(0.04 * fs)), np.hanning(int(0.04 * fs)), "same")[:m]
    drift = wander * drift / max(np.abs(drift).max(), 1e-12)
    ph = 2 * np.pi * np.cumsum(tone_hz * (1.0 + glide * k / max(dur, 1e-9) + drift)) / fs
    tone = np.sqrt(2 * tone_energy / dur) * np.sin(ph + 2 * np.pi * rng.random())
    edge = np.minimum(1.0, np.minimum(k / 0.002, (dur - k) / 0.004).clip(0, None))
    x = (noise + tone) * np.sqrt(envelope) * edge
    return np.concatenate((np.zeros(int(round(pre * fs))), x, np.zeros(n - m - int(round(pre * fs)))))


def squeal_shot(tyre_type, speed_ms, damping, spring, sink_ms, static_load_n, transfer_db=None, distance_m=None, mu=0.5,
                fs=48000.0, seed=1):
    """a wheel's touchdown squeal (Pa): in a cockpit through the airframe's transfer (transfer_db: dB by BANDS, as
    STRUCTURES) or outside at distance_m over a hard ground. Returns (signal, its sound energy J, the slide's s)."""
    tt, slip, P = spin_up(speed_ms, tyre_type, damping, spring, sink_ms, static_load_n, mu)
    lost = np.sum(P) * (tt[1] - tt[0]) if len(tt) > 1 else 0.0
    E = 10 ** (SQUEAL_EFFICIENCY_DB / 10) * lost
    sq = SQUEAL[tyre_type]
    sh = 10 ** (_shape(sq, BANDS) / 10)
    sh = sh / sh.sum()
    if transfer_db is not None:
        # above 2 kHz no transfer was measured and the squeal's spectrum already falls as its cockpit's did: held there
        Tq = np.where(BANDS > 2000.0, np.asarray(transfer_db)[BANDS <= 2000.0][-1], transfer_db)
        to_p2 = 4e-10 / 1e-12 * 10 ** (Tq / 10)
    else:
        to_p2 = np.full(len(BANDS), RHO_C / (2 * np.pi * distance_m ** 2))
    band = E * (1 - sq["tone_share"]) * sh * to_p2
    tone = E * sq["tone_share"] * float(np.interp(np.log(sq["tone_hz"]), np.log(BANDS), to_p2))
    env = np.interp(np.arange(int(round(tt[-1] * fs)) + 1) / fs, tt, P) if len(tt) > 1 else np.ones(1)
    env = env / max(np.mean(env), 1e-30)
    return _burst(band, tone, sq["tone_hz"], sq["glide"], env, fs, seed, wander=sq["wander"]), E, \
        float(tt[-1]) if len(tt) else 0.0


def thump_shot(damping, sink_ms, fs=48000.0, seed=1):
    """a strut's thump in the cockpit (Pa): the measured touchdown's band energies scaled by the square of the force step"""
    e = 10 ** (_shape(THUMP, BANDS) / 10) * 4e-10 * THUMP["window_s"] * (damping * sink_ms / THUMP["force_n"]) ** 2
    m = int(round(4 * THUMP["decay_s"] * fs))
    env = np.exp(-np.arange(m) / fs / THUMP["decay_s"])
    return _burst(e, 0.0, 100.0, 0.0, env / env.mean(), fs, seed, pre=0.005, post=0.02)


TOUCHDOWN_SLIPS_KT = (30.0, 45.0, 60.0, 75.0, 90.0, 105.0)
SINKS_FPS = (0.25, 0.5, 1.0, 2.0, 4.0)      # harder plays the 4 fps squeal at full volume
# the flight model's touchdown, latched per wheel from the moment its weight comes on until it lifts off again: its sink
# rate, and its slip, the ground speed less the wheel's rim speed (a wheel still turning from the last touch slides less;
# the spin-up depends on nothing else, so squeal_shot's speed_ms is the slip). The X-57 provides them.
TOUCHDOWN_PROPERTIES = dict(sink="/fdm/jsbsim/gear/unit[{n}]/touchdown-sink-fps",
                            slip="/fdm/jsbsim/gear/unit[{n}]/touchdown-slip-kt", wow="/gear/gear[{n}]/wow")


def make_touchdown_bank(aircraft, out_dir, groups, gain_db, airframe, slips=TOUCHDOWN_SLIPS_KT, sinks=SINKS_FPS,
                        exterior_m=30.0, mu=0.5, fs=48000.0, seed=1, progress=print):
    """One-shots of each wheel group's touchdown: squeals by slip speed and sink rate, in the cockpit (through the
    airframe's transfer, STRUCTURES) and exterior_m to the side; thumps by sink rate, in the cockpit only. groups: {name,
    tyre_type, damping (N s/m), spring (N/m), static_load_n, wheels: [{gear, position}]}. A one-shot keeps its length, so
    what its volume has to follow is the event's energy: a squeal's goes as the slip squared and doesn't depend on the
    sink rate (which only sets how long it lasts), a thump's as the sink rate squared. Each group's files are fitted to
    that law and each file carries its own difference from it (correction_db), so that on both sides of a window's edge
    the level is the law's and only the timbre and the slide's length step. Writes touchdown.json and
    pywopwop-touchdown.xml."""
    from .synth import a_weighted_ms, write_wav
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T_db = structure_db(BANDS, airframe)
    a_exposure = lambda x: float(10 * np.log10(max(a_weighted_ms(x, fs) * len(x) / fs, 1e-30) / 4e-10))
    shots = []
    for g in groups:
        for v in slips:
            for s in sinks:
                for view in ("cockpit", "side"):
                    x, E, dur = squeal_shot(g["tyre_type"], v * 0.514444, g["damping"], g["spring"], s * 0.3048,
                                            g["static_load_n"], T_db if view == "cockpit" else None,
                                            None if view == "cockpit" else exterior_m, mu, fs, seed + len(shots))
                    shots.append(dict(x=x, file=f"touchdown_{g['name']}_squeal_{v:g}kt_{s:g}fps_{view}.wav", group=g["name"],
                                      kind="squeal", view=view, slip_kt=v, sink_fps=s, slide_s=round(dur, 4), sound_j=E))
                progress(f"  {g['name']:5s} squeal {v:4.0f} kt {s:4g} fps  slide {dur * 1e3:5.0f} ms  {E * 1e3:6.1f} mJ")
        for s in sinks:
            shots.append(dict(x=thump_shot(g["damping"], s * 0.3048, fs, seed + len(shots)),
                              file=f"touchdown_{g['name']}_thump_{s:g}fps_cockpit.wav", group=g["name"], kind="thump",
                              view="cockpit", sink_fps=s))
    # the laws: exposure = K + 20 log10(axis), the axis the slip for a squeal and the sink rate for a thump
    axis = lambda sh: sh["slip_kt"] if sh["kind"] == "squeal" else sh["sink_fps"]
    for sh in shots:
        sh["exposure_dba"] = round(a_exposure(sh["x"]), 2)
    K = {}
    for key in {(sh["group"], sh["kind"], sh["view"]) for sh in shots}:
        same = [sh for sh in shots if (sh["group"], sh["kind"], sh["view"]) == key]
        K[key] = float(np.mean([sh["exposure_dba"] - 20 * np.log10(axis(sh)) for sh in same]))
    # each file written at the top of its window on its axis (the volume, which the sound system holds to 1, then never
    # needs more); the last windows run on, written as far over their centres as the ones before
    top = {}
    for name, centres, lo, hi in (("squeal", slips, 10.0, 1000.0), ("thump", sinks, 0.05, 100.0)):
        r = [w[1] / c for c, w in zip(centres, _windows(centres, lo, hi))]
        r[-1] = r[-2]
        top[name] = dict(zip(centres, r))
    for sh in shots:
        law = K[(sh["group"], sh["kind"], sh["view"])] + 20 * np.log10(axis(sh))
        sh["law_dba"] = round(float(law), 2)
        sh["correction_db"] = round(float(law - sh["exposure_dba"]), 3)
        sh["boost"] = float(top[sh["kind"]][axis(sh)] * 10 ** (sh["correction_db"] / 20))
    # the thumps at a gain of their own: as much under the cockpit's as lets the loudest fit, so that a firm touchdown
    # still thumps harder than a soft one (the bank's gain is the propellers', and the estimate is an upper end)
    loudest = max(np.abs(sh["x"]).max() * sh["boost"] for sh in shots if sh["kind"] == "thump") * 10 ** (gain_db["cockpit"] / 20)
    thump_offset = float(min(0.0, -np.ceil(20 * np.log10(loudest) + 1.0)))
    files = []
    for sh in shots:
        g = gain_db["cockpit" if sh["view"] == "cockpit" else "exterior"] + (thump_offset if sh["kind"] == "thump" else 0.0)
        # a burst of tens of milliseconds can have a hundredth of its samples bent under full scale unheard
        y, limited = soft_limit(sh["x"] * sh["boost"] * 10 ** (g / 20.0))
        if limited > 1e-2 * len(y):
            raise ValueError(f"{sh['file']} would peak at {20 * np.log10(np.abs(y).max()):+.1f} dBFS at this gain")
        write_wav(out / sh["file"], y, fs, 16, 1.0, seed=len(files))
        files.append({k: v for k, v in sh.items() if k != "x"} | dict(limited_samples=int(limited)))
    manifest = dict(aircraft=aircraft.name, generator="pywopwop touchdown", sample_rate=fs, exterior_distance_m=exterior_m,
                    gain_db=gain_db, thump_offset_db=thump_offset, structure=airframe, slips_kt=list(slips),
                    sinks_fps=list(sinks), mu=mu,
                    groups=[{k: (v if k != "wheels" else [dict(gear=w["gear"], position=list(map(float, w["position"])))
                                                             for w in v]) for k, v in g.items()} for g in groups],
                    properties=TOUCHDOWN_PROPERTIES, files=files,
                    note="squeal: the wheel's spin-up loss by physics, its sound share and spectrum from recorded landings; "
                         "thump: one recorded touchdown, at the edge of detection, written thump_offset_db under its "
                         "estimate so that the loudest fits the cockpit's gain. exposure_dba: each file's own, at its "
                         "window's centre; law_dba: what it plays at there (exposure = K + 20 log10 of the slip or sink)")
    (out / "touchdown.json").write_text(json.dumps(manifest, indent=1))
    (out / "pywopwop-touchdown.xml").write_text(flightgear_touchdown_xml(manifest))
    return manifest


def _windows(centres, lo, hi):
    c = list(centres)
    edges = [lo] + [float(np.sqrt(a * b)) for a, b in zip(c, c[1:])] + [hi]
    return list(zip(edges[:-1], edges[1:]))


def flightgear_touchdown_xml(manifest, sound_dir="Sounds/pywopwop", exterior="side"):
    """FlightGear <fx> entries, played once: each wheel's squeal for the slip and sink rate it touched down at, and its
    strut's thump. The windows pick the file (its timbre and the slide's length); the volume makes the energy the law's,
    the squeal's amplitude in proportion to the slip, the thump's to the sink rate, each file's own difference from the
    law taken out, so a touchdown on a window's edge sounds as loud from either side. Each condition holds from the moment
    the wheel's weight comes on until it lifts off, so it plays once a touchdown; the latched values
    (TOUCHDOWN_PROPERTIES) are the flight model's to provide. Each file is written at the top of its window (its boost),
    so the volume stays at 1 or under; past the last window it holds."""
    from .wind import _e, _v
    P = manifest["properties"]
    have = {f["file"]: f for f in manifest["files"]}
    vw = _windows(manifest["slips_kt"], 10.0, 1000.0)
    sw = _windows(manifest["sinks_fps"], 0.05, 100.0)
    out = []
    for g in manifest["groups"]:
        for w in g["wheels"]:
            n, pos = w["gear"], w["position"]
            sink = f"<property>{P['sink'].format(n=n)}</property>"
            slip = f"<property>{P['slip'].format(n=n)}</property>"
            within = lambda p, lo, hi: (f"<greater-than>{p}<value>{lo:g}</value></greater-than>"
                                        f"<less-than>{p}<value>{hi:g}</value></less-than>")
            gain = lambda f, centre: _v(10 ** (f["correction_db"] / 20) / (centre * f["boost"]))
            entries = []
            for view in ("cockpit", exterior):
                for v, (v0, v1) in zip(manifest["slips_kt"], vw):
                    for s, (s0, s1) in zip(manifest["sinks_fps"], sw):
                        f = have.get(f"touchdown_{g['name']}_squeal_{v:g}kt_{s:g}fps_{view}.wav")
                        if f:
                            entries.append((f["file"], view, within(slip, v0, v1) + within(sink, s0, s1),
                                            _e("product", slip, gain(f, v))))
                for s, (s0, s1) in zip(manifest["sinks_fps"], sw):
                    f = have.get(f"touchdown_{g['name']}_thump_{s:g}fps_{view}.wav")
                    if f:
                        entries.append((f["file"], view, within(sink, s0, s1), _e("product", sink, gain(f, s))))
            for fn, view, win, vol in entries:
                view_c = ("<property>sim/current-view/internal</property>" if view == "cockpit"
                          else "<not><property>sim/current-view/internal</property></not>")
                cond = _e("and", view_c, f"<property>{P['wow'].format(n=n)}</property>", win)
                out.append(f"\t\t<touchdown>\n\t\t\t<name>pww-{fn[:-4]}-gear{n}</name>\n\t\t\t<mode>once</mode>\n"
                           f"\t\t\t<path>{sound_dir}/{fn}</path>\n\t\t\t<condition>{cond}</condition>\n"
                           f"\t\t\t<volume><expression>{_e('clip', vol, '<clipMin>0</clipMin><clipMax>1</clipMax>')}"
                           "</expression></volume>\n"
                           f"\t\t\t<position>\n\t\t\t\t<x>{-pos[0] + 0.0:.3f}</x>\n\t\t\t\t<y>{pos[1] + 0.0:.3f}</y>\n"
                           f"\t\t\t\t<z>{pos[2] + 0.0:.3f}</z>\n\t\t\t</position>\n"
                           + ("" if view == "cockpit" else
                              f"\t\t\t<reference-dist>{manifest['exterior_distance_m']:g}</reference-dist>\n"
                              "\t\t\t<max-dist>1500</max-dist>\n")
                           + "\t\t</touchdown>\n")
    return ("<?xml version=\"1.0\"?>\n<!-- generated by pywopwop: the touchdowns of the " + manifest["aircraft"]
            + "; merge into the aircraft's sound file -->\n<PropertyList>\n\t<fx>\n" + "".join(out)
            + "\t</fx>\n</PropertyList>\n")
