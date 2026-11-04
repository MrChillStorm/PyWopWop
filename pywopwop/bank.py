"""Sound banks for flight simulators: seamless loops over flight conditions and listening positions, with a
manifest and a FlightGear sound configuration that blends them.

All loops of a bank share one gain, so their relative loudness is the physics' (a descent's blade slap is louder
than the cruise because it is). The loops are recorded at the rotor's nominal speed: a simulator plays them with
pitch = rpm / nominal rpm, which the band limit leaves room for (no aliasing up to the pitch headroom).
"""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

import numpy as np

from .aero import AeroOptions
from .atmosphere import Atmosphere
from .model import FlightCondition, Observer, STANDARD_OBSERVERS
from .operating import operate
from .synth import Quality, Variability, render, write_wav


@dataclass
class BankCondition:
    name: str
    airspeed_kt: float
    climb_fpm: float = 0.0
    description: str = ""
    ground_clearance_m: float = None    # wheels or skids above the ground: in ground effect, listeners on the ground
    listener_distance_m: float = None   # their horizontal distance (default: the positions' own 100 m)
    nacelle_deg: float = 90.0           # tiltrotors: 0 airplane mode
    rpm_percent: float = 100.0
    ground_run: bool = False            # standing on the gear at flat pitch (the rotor's flat_pitch_deg)


HELICOPTER_CONDITIONS = [
    BankCondition("hover", 0.0, 0.0, "hover out of ground effect"),
    BankCondition("hover_ige", 0.0, 0.0, "hover in ground effect, 1.5 m up; listeners 50 m away on the ground",
                  ground_clearance_m=1.5, listener_distance_m=50.0),
    BankCondition("transition", 40.0, 0.0, "translational lift"),
    BankCondition("cruise_slow", 80.0, 0.0, "level, near minimum power"),
    BankCondition("cruise", 120.0, 0.0, "level cruise"),
    BankCondition("climb", 70.0, 1000.0, "climb"),
    BankCondition("bvi", 65.0, -800.0, "approach descent: blade-vortex interaction"),
    BankCondition("descent_steep", 50.0, -1200.0, "steep approach: the wake is blown above the disc, less slap"),
]

MULTIROTOR_CONDITIONS = [
    BankCondition("hover", 0.0, 0.0),
    BankCondition("hover_ige", 0.0, 0.0, "hover 0.5 m up; listeners 15 m away", ground_clearance_m=0.5,
                  listener_distance_m=15.0),
    BankCondition("cruise_slow", 15.0, 0.0),
    BankCondition("cruise", 30.0, 0.0),
    BankCondition("climb", 10.0, 600.0),
    BankCondition("descent", 15.0, -400.0),
]

# the rotors run up on the ground at flat pitch: a start (and a shutdown, run backwards), listeners 50 m away
RUNUP_RPM = (20.0, 35.0, 50.0, 65.0, 80.0, 90.0, 100.0)
RUNUP_CONDITIONS = [BankCondition(f"runup_{p:g}", 0.0, 0.0, f"on the ground at flat pitch, rotor at {p:g} %",
                                  ground_clearance_m=0.0, listener_distance_m=50.0, rpm_percent=p, ground_run=True)
                    for p in RUNUP_RPM]

# tiltrotors in airplane mode (bank --mode airplane: the aircraft's airplane_trim and airplane_rpm)
AIRPLANE_CONDITIONS = [
    BankCondition("airplane_180", 180.0, 0.0, "airplane mode, slow cruise", nacelle_deg=0.0),
    BankCondition("airplane_220", 220.0, 0.0, "airplane mode, cruise", nacelle_deg=0.0),
    BankCondition("airplane_260", 260.0, 0.0, "airplane mode, fast cruise", nacelle_deg=0.0),
    BankCondition("airplane_climb", 180.0, 1500.0, "airplane mode, climb", nacelle_deg=0.0),
    BankCondition("airplane_descent", 230.0, -1500.0, "airplane mode, descent", nacelle_deg=0.0),
]


AIRPLANE_ALTITUDE_M = 3048.0        # a tiltrotor's airplane mode: thin, smooth air, as in its cruise


@dataclass
class BankOptions:
    conditions: list = None
    observers: list = field(default_factory=lambda: ["side", "front_right", "rear", "below"])
    cockpit: bool = True
    duration: float = 15.0          # s: long enough that the loop point and its repetition go unnoticed
    fs: float = 48000.0
    bits: int = 16
    quality: str = "normal"
    pitch_headroom: float = 1.25
    peak_dbfs: float = -1.0
    layers: bool = True            # the FlightGear snippet plays the layers (else the mixes)
    seed: int = 1
    environment: str = "open"
    only: list = None               # render only these conditions (names)
    update: bool = False            # add them to the bank in out_dir at its gains, keeping the rest
    listener_height: float = 1.7
    mode: str = None                # 'airplane': a tiltrotor's airplane mode conditions (default: its helicopter mode)
    altitude_m: float = None        # the air it flies in, at the conditions' calibrated airspeeds (default: sea level,
                                    # a tiltrotor's airplane mode at cruise)
    runup: bool = False             # add the ground run-up conditions (with update: render only them)


def nominal_rpm(aircraft, rotor, conds):
    """the rotor speed the loops are rendered at (a simulator pitches them by its rpm over this): the airplane
    mode's for an airplane mode bank"""
    if any(c.nacelle_deg < 90.0 for c in conds) and rotor.role in ("main", "lift"):
        return aircraft.airplane_rpm
    return rotor.rpm


def flight_condition(aircraft, c, altitude_m=0.0):
    """The bank condition as flown at altitude_m. A condition's airspeed is calibrated: a simulator fades the loops by
    its calibrated airspeed, and the thrust and the wing's loads follow the dynamic pressure that stands for, so the
    true airspeed is the one with sea level's dynamic pressure at that speed."""
    true_kt = c.airspeed_kt * np.sqrt(Atmosphere(0.0).rho / Atmosphere(altitude_m).rho)
    return FlightCondition(airspeed_kt=float(true_kt), climb_fpm=c.climb_fpm, nacelle_deg=c.nacelle_deg,
                           ground_height_m=None if c.ground_clearance_m is None
                           else aircraft.hub_height_m + c.ground_clearance_m, rpm_percent=c.rpm_percent,
                           collective_deg=aircraft.main.flat_pitch_deg if c.ground_run else None,
                           altitude_m=altitude_m)


def _observers(aircraft, c, opts):
    """Listening positions for a condition: in ground effect, on the ground at the condition's distance, level with
    the rotor (no position below it); otherwise the standard ones."""
    obs = []
    for o in opts.observers:
        o = STANDARD_OBSERVERS[o] if isinstance(o, str) else o
        if c.ground_clearance_m is not None:
            if o.elevation_deg < -80.0:
                continue
            d = c.listener_distance_m or 50.0
            dh = aircraft.hub_height_m + c.ground_clearance_m - opts.listener_height
            o = Observer(o.name, o.azimuth_deg, -float(np.degrees(np.arctan2(dh, d))), float(np.hypot(d, dh)))
        obs.append(o)
    if opts.cockpit:
        obs.append(Observer("cockpit", position=tuple(aircraft.cockpit)))
        if aircraft.cabin.openings:
            obs.append(Observer("cockpit_open", position=tuple(aircraft.cockpit)))     # every opening open
    return obs


def make_bank(aircraft, out_dir, opts=None, progress=print):
    """Render the bank into out_dir: WAV loops, manifest.json and a FlightGear sound snippet. With opts.update,
    render only opts.only's conditions into the existing bank, at its gains (refusing if they would clip)."""
    opts = opts or BankOptions()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conds = opts.conditions
    if conds is None and opts.mode == "airplane":
        conds = AIRPLANE_CONDITIONS
    elif conds is None:
        conds = MULTIROTOR_CONDITIONS if aircraft.tail is None and len(aircraft.rotors) > 2 else HELICOPTER_CONDITIONS
    if opts.runup:
        conds = list(conds) + RUNUP_CONDITIONS
        if opts.update and opts.only is None:
            opts.only = [c.name for c in RUNUP_CONDITIONS]
    old = None
    if opts.update:
        mf = out / "manifest.json"
        if not mf.exists():
            raise FileNotFoundError(f"no bank to update in {out}")
        old = json.loads(mf.read_text())
    todo = [c for c in conds if opts.only is None or c.name in opts.only]
    if opts.only and len(todo) < len(opts.only):
        raise ValueError(f"unknown conditions: {set(opts.only) - {c.name for c in todo}}")
    q = Quality.get(opts.quality)
    renders = []
    state = None
    altitude = opts.altitude_m
    if altitude is None:
        altitude = AIRPLANE_ALTITUDE_M if opts.mode == "airplane" else 0.0
    for c in todo:
        ci = [x.name for x in conds].index(c.name)
        if c.ground_run and aircraft.main.flat_pitch_deg is None:
            raise ValueError(f"{aircraft.name}'s main rotor has no flat_pitch_deg (its collective fully down)")
        fc = flight_condition(aircraft, c, altitude)
        state = operate(aircraft, fc, AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
        for o in _observers(aircraft, c, opts):
            cabin = aircraft.cabin if o.position is not None else None
            r = render(aircraft, fc, o, fs=opts.fs, duration=opts.duration, quality=q, cabin=cabin, state=state,
                       doors_open=o.name == "cockpit_open",
                       variability=Variability(seed=opts.seed + 7 * ci), pitch_headroom=opts.pitch_headroom,
                       environment=opts.environment)
            renders.append((c, o, r))
            lv = r.levels()["total"]
            progress(f"  {c.name:12s} {o.name:12s} {lv['oaspl']:6.1f} dB  {lv['dba']:6.1f} dBA")
    # one gain for the outside loops and one for the cockpit's (a simulator plays them as separate sets; the
    # cockpit's near-field rotor thump is far louder than 100 m outside), each covering the mixes and the layers
    # (rotor, slap, swish, tail), which add up to the mix
    def keys(r):
        return ["total"] + [k for k in r.signals if k.startswith("layer/")]

    def group(o):
        return "cockpit" if o.position is not None else "exterior"
    gains = {}
    for g in {group(o) for _, o, _ in renders}:
        peak = max(np.max(np.abs(r.signals[k])) for _, o, r in renders if group(o) == g for k in keys(r))
        if old is not None:
            gains[g] = 10.0 ** (old["gain_db"][g] / 20.0)
            top = 20 * np.log10(peak * gains[g])
            if top > -0.1:
                raise ValueError(f"the new {g} loops would peak at {top:+.1f} dBFS at the bank's gain: rebuild the "
                                 "whole bank instead")
        else:
            gains[g] = 10.0 ** (opts.peak_dbfs / 20.0) / peak
    entries = []
    for c, o, r in renders:
        lv = r.levels()
        gain = gains[group(o)]
        ref = np.sqrt(np.mean(r.signals["total"] ** 2))
        for k in keys(r):
            label = "mix" if k == "total" else k.split("/", 1)[1]
            if np.sqrt(np.mean(r.signals[k] ** 2)) < 1e-4 * ref:
                continue                     # no blade slap in this condition, say
            fn = f"{c.name}_{o.name}_{label}.wav"
            write_wav(out / fn, r.signals[k], r.fs, opts.bits, gain, seed=abs(hash(fn)) % 2 ** 32)
            entries.append(dict(file=fn, condition=c.name, airspeed_kt=c.airspeed_kt, climb_fpm=c.climb_fpm,
                                observer=o.name, source=label, oaspl_db=round(lv[k]["oaspl"], 2),
                                dba=round(lv[k]["dba"], 2), seconds=round(len(r.signals[k]) / r.fs, 4)))
    if old is not None:
        # keep the rest of the bank; its conditions in the standard order
        done = {c.name for c in todo}
        entries = [e for e in old["files"] if e["condition"] not in done] + entries
        previous = {d["name"]: d for d in old["conditions"]}
        known = {c.name: c for c in conds}
        order = [c.name for c in conds]
        names = sorted(set(previous) | done, key=lambda n: order.index(n) if n in order else len(order))
        conds = [known[n] if n in known else BankCondition(**previous[n]) for n in names]
        gains = {**{g: 10.0 ** (v / 20.0) for g, v in old["gain_db"].items()}, **gains}
    observers = _observers(aircraft, BankCondition("", 0.0), opts)
    manifest = dict(aircraft=aircraft.name, generator="pywopwop", sample_rate=opts.fs, bits=opts.bits,
                    gain_db={g: round(20 * np.log10(v), 3) for g, v in gains.items()},
                    note="levels are dB re 20 uPa at the listener before the bank gain; distance "
                         f"{observers[0].distance_m:g} m for the exterior positions",
                    rotors={r.name: dict(nominal_rpm=nominal_rpm(aircraft, r, conds), blades=r.blades, role=r.role)
                            for r in aircraft.rotors},
                    pitch_headroom=opts.pitch_headroom, conditions=[asdict(c) for c in conds],
                    hub_height_m=aircraft.hub_height_m, altitude_m=altitude,
                    observers=[dict(name=o.name, azimuth_deg=o.azimuth_deg, elevation_deg=o.elevation_deg,
                                    distance_m=o.distance_m, position=o.position) for o in observers],
                    files=entries)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "pywopwop-sound.xml").write_text(flightgear_xml(aircraft, conds, manifest, opts))
    return manifest


def _ramp(prop, a, b):
    """A FlightGear expression going linearly from 0 at a to 1 at b (either order), clipped to 0..1."""
    f = 1.0 / (b - a)
    return _clip(prop, f, -a * f)


def _clip(prop, factor, offset):
    """factor * prop + offset, clipped to 0..1. (A <volume>'s own <factor>, <offset>, <min> and <max> can't make a
    fade: FlightGear clips factor * prop alone and adds every entry's offset to the product afterwards, a negative
    factor adds its offset too, and a sound takes at most five <volume> entries (SimGear's xmlsound.cxx).)"""
    return (f"<clip><sum><product><property>{prop}</property><value>{factor:.6g}</value></product>"
            f"<value>{offset:.6g}</value></sum><clipMin>0</clipMin><clipMax>1</clipMax></clip>")


def flightgear_xml(aircraft, conds, manifest, opts, exterior="side", sound_dir="Sounds/pywopwop"):
    """A FlightGear <PropertyList> of looped sounds: the level-flight loops cross-fade with airspeed, the descent
    (blade slap) loops come in with the rate of descent, every loop is pitched with its rotor's rpm."""
    spd = "velocities/airspeed-kt"
    vs = "velocities/vertical-speed-fps"
    lines = ["<?xml version=\"1.0\"?>\n<!-- generated by pywopwop: rotor sound loops for "
             f"{aircraft.name}; merge into the aircraft's sound file -->\n<PropertyList>\n\t<fx>\n"]
    level = sorted([c for c in conds if abs(c.climb_fpm) < 1.0 and c.ground_clearance_m is None],
                   key=lambda c: c.airspeed_kt)
    runup = sorted([c for c in conds if c.ground_run], key=lambda c: c.rpm_percent)
    ige = [c for c in conds if c.ground_clearance_m is not None and not c.ground_run]
    wow = "gear/gear[0]/wow"
    airborne = (f"<difference><value>1</value><property>{wow}</property></difference>" if runup else None)
    agl = "position/altitude-agl-ft"
    R_ft = aircraft.main.radius / 0.3048
    z_lo, z_hi = 0.5 * R_ft, 1.5 * R_ft          # ground effect fully in below, gone above (hub height ~ agl)
    descents = [c for c in conds if c.climb_fpm < -1.0]
    climbs = [c for c in conds if c.climb_fpm > 1.0]
    rotors = manifest["rotors"]
    have = {(e["condition"], e["observer"], e["source"]) for e in manifest["files"]}
    sources = ["rotor", "slap", "swish", "tail"] if opts.layers else ["mix"]
    tail = next((r for r in rotors.values() if r["role"] == "tail"), None)
    main = next(r for r in rotors.values() if r["role"] != "tail")

    open_view = any(e["observer"] == "cockpit_open" for e in manifest["files"]) and aircraft.cabin.openings
    if open_view:
        from .wind import open_weights
        shut_w, open_w = open_weights(aircraft.cabin.openings)

    def entry(c, view, src, vols):
        if (c.name, view, src) not in have:
            return ""
        if view.startswith("cockpit") and open_view:
            # the shut and the all-open cockpit's loops by how much of the doors and windows is open
            vols = vols + [open_w if view == "cockpit_open" else shut_w]
        fn = f"{c.name}_{view}_{src}.wav"
        rot = tail if src == "tail" else main
        prop = f"rotors/{'tail' if src == 'tail' else 'main'}/rpm"
        cond = "<property>sim/current-view/internal</property>" if view.startswith("cockpit") else \
            "<not><property>sim/current-view/internal</property></not>"
        # a run-up loop is recorded at its own rotor speed and plays from the next lower one's
        ref = rot['nominal_rpm'] * c.rpm_percent / 100.0
        lo = 0.2 * rot['nominal_rpm']
        if c.ground_run:
            k = runup.index(c)
            lo = rot['nominal_rpm'] * (runup[k - 1].rpm_percent / 100.0 if k else 0.05)
        s = (f"\t\t<rotor>\n\t\t\t<name>pww-{c.name}-{view}-{src}</name>\n\t\t\t<mode>looped</mode>\n"
             f"\t\t\t<path>{sound_dir}/{fn}</path>\n\t\t\t<condition>\n\t\t\t\t<and>\n\t\t\t\t\t{cond}\n"
             f"\t\t\t\t\t<greater-than><property>{prop}</property><value>{lo:.0f}</value>"
             f"</greater-than>\n\t\t\t\t</and>\n\t\t\t</condition>\n")
        # one <volume>: the product of the fades and (off the ground) the rotor speed's share of its nominal
        terms = vols + ([] if c.ground_run else [_clip(prop, 1.0 / rot['nominal_rpm'], 0.0)])
        s += "\t\t\t<volume><expression><product>\n"
        s += "".join(f"\t\t\t\t{t}\n" for t in terms)
        s += "\t\t\t</product></expression></volume>\n"
        s += (f"\t\t\t<pitch><property>{prop}</property><factor>{1.0 / ref:.6g}</factor>"
              f"<min>{min(0.2, lo / ref):.3g}</min><max>{opts.pitch_headroom:g}</max></pitch>\n")
        if not view.startswith("cockpit"):
            # the distance the loop was recorded at: FlightGear scales it from there
            d = (c.listener_distance_m or 50.0) if c.ground_clearance_m is not None else 100.0
            s += f"\t\t\t<reference-dist>{d:g}</reference-dist>\n\t\t\t<max-dist>4000</max-dist>\n"
        s += "\t\t</rotor>\n"
        return s

    def vertical(group, sign):
        """climbs (sign +1) or descents (-1): cross-faded by the rate (fps) from 5 ft/s on, each over an airspeed
        window around its own speed; their volumes add up to at most 1 at any speed and rate"""
        group = sorted(group, key=lambda c: sign * c.climb_fpm)
        rate = [sign * max(sign * c.climb_fpm / 60.0, 10.0) for c in group]
        out = []
        for i, c in enumerate(group):
            vols = [_ramp(vs, sign * 5.0 if i == 0 else rate[i - 1], rate[i])]
            if i < len(group) - 1:
                vols.append(_ramp(vs, rate[i + 1], rate[i]))
            vols += [_ramp(spd, max(c.airspeed_kt - 40.0, 0.0), c.airspeed_kt),
                     _ramp(spd, c.airspeed_kt + 50.0, c.airspeed_kt)]
            out.append((c, vols))
        return out
    moving = vertical(climbs, 1) + vertical(descents, -1)
    # the level-flight loops take what the climbs and descents leave, so that the volumes always add up to 1 (a fast
    # descent outside the descents' airspeed windows stays on the level loops)
    rest = ("<clip><difference><value>1</value>" + "".join("<product>" + "".join(v) + "</product>" for _, v in moving)
            + "</difference><clipMin>0</clipMin><clipMax>1</clipMax></clip>") if moving else None
    if airborne:
        # on the ground the run-up loops (by rotor speed), off it the flight conditions'
        moving = [(c, v + [airborne]) for c, v in moving]
        rest = f"<product>{rest}{airborne}</product>" if rest else airborne
    rpm = "rotors/main/rpm"
    nom = main["nominal_rpm"]

    for view in [exterior] + ((["cockpit"] + (["cockpit_open"] if open_view else [])) if opts.cockpit else []):
        for i, c in enumerate(level):
            vols = []
            if i > 0:
                vols.append(_ramp(spd, level[i - 1].airspeed_kt, c.airspeed_kt))
            if i < len(level) - 1:
                vols.append(_ramp(spd, level[i + 1].airspeed_kt, c.airspeed_kt))
            if ige and c.airspeed_kt == 0.0:
                vols.append(_ramp(agl, z_lo, z_hi))         # near the ground, the hover in ground effect takes over
            if rest:
                vols.append(rest)
            for src in sources:
                lines.append(entry(c, view, src, vols))
        for c in ige:
            nxt = next((x.airspeed_kt for x in level if x.airspeed_kt > 0), 40.0)
            vols = [_ramp(agl, z_hi, z_lo), _ramp(spd, nxt, 0.0)] + ([rest] if rest else [])
            for src in sources:
                lines.append(entry(c, view, src, vols))
        for c, vols in moving:
            for src in sources:
                lines.append(entry(c, view, src, vols))
        for k, c in enumerate(runup):
            vols = [f"<property>{wow}</property>"]
            # the slowest loop fades out toward a stopped rotor
            below = runup[k - 1].rpm_percent / 100.0 * nom if k > 0 else 0.0
            vols.append(_ramp(rpm, below, c.rpm_percent / 100.0 * nom))
            if k < len(runup) - 1:
                vols.append(_ramp(rpm, runup[k + 1].rpm_percent / 100.0 * nom, c.rpm_percent / 100.0 * nom))
            for src in sources:
                lines.append(entry(c, view, src, vols))
    lines.append("\t</fx>\n</PropertyList>\n")
    return "".join(lines)
