"""Propeller aircraft: discs in groups, each group at its own speed and pitch.

A helicopter's rotors share a trim; a propeller's speed and pitch are its own. A group of discs (a wingtip's cruise
propeller, a contactor's high-lift propellers) has its rotor speed from the condition and its load from the blade pitch or
the thrust: fixed pitch finds the thrust that pitch gives at the speed and airspeed, by the secant method on the trim's
collective.
"""
from dataclasses import dataclass, replace
import itertools
from pathlib import Path
import tomllib

import numpy as np

from .aero import AeroOptions, solve_rotor
from .atmosphere import Atmosphere
from .model import KT, Aircraft, Cabin, FlightCondition, Observer, Rotor
from .operating import OperatingState, RotorPoint, _frame, _pitch_matrix

PROP_DIR = Path(__file__).parent / "presets" / "propellers"
INCH = 0.0254


def _solved_at_pitch(rot, omega, V, a, theta75, atm, opts, t_ref):
    """The rotor at the blade pitch theta75 (rad, at 0.75 R): the thrust for it by the secant method on the trim's collective"""
    quick = replace(opts, bvi=False, n_radial=min(opts.n_radial, 20), n_azimuth=min(opts.n_azimuth, 60))
    ts = [0.5 * t_ref, t_ref]
    solve = lambda t, o=quick: solve_rotor(rot, omega, V, a, t, atm.rho, atm.c, atm.nu, o)
    sols = [solve(t) for t in ts]
    for _ in range(14):
        (t0, t1), (s0, s1) = ts[-2:], sols[-2:]
        e0, e1 = s0.theta0 - theta75, s1.theta0 - theta75
        if abs(e1) < np.radians(0.01) or e1 == e0:
            break
        ts.append(t1 - e1 * (t1 - t0) / (e1 - e0))
        sols.append(solve(ts[-1]))
    return solve(ts[-1], opts), ts[-1]


def _solved_at_thrust(rot, omega, V, a, thrust, atm, opts, stops):
    """The rotor at the thrust, its pitch held between the stops (rad): a propeller on its fine stop makes what it makes. Returns
    the solution and its thrust"""
    quick = replace(opts, bvi=False, n_radial=min(opts.n_radial, 20), n_azimuth=min(opts.n_azimuth, 60))
    t_ref = 0.5 * atm.rho * (omega * rot.radius) ** 2 * np.pi * rot.radius ** 2 * 0.01
    try:
        theta = solve_rotor(rot, omega, V, a, thrust, atm.rho, atm.c, atm.nu, quick).theta0
    except ValueError:                                    # out of its reach: past whichever stop that thrust lies
        try:
            fine = _solved_at_pitch(rot, omega, V, a, stops[0], atm, opts, t_ref)
            if thrust < fine[0].thrust:
                return fine
        except ValueError:                                # so fast that the fine stop is far past windmilling
            pass
        return _solved_at_pitch(rot, omega, V, a, stops[1], atm, opts, t_ref)
    if not stops[0] <= theta <= stops[1]:
        return _solved_at_pitch(rot, omega, V, a, min(max(theta, stops[0]), stops[1]), atm, opts, t_ref)
    return _solved_at_pitch(rot, omega, V, a, theta, atm, opts, max(abs(thrust), t_ref))


@dataclass
class PropCondition:
    """A flight condition of a propeller aircraft"""
    name: str = "cruise"
    airspeed_kt: float = 100.0          # true airspeed
    altitude_m: float = 0.0
    alpha_deg: float = 2.0              # the body's angle of attack
    wing_cl: float = 0.5                # the wing's lift coefficient: its upwash at the discs
    cruise_rpm: float = 0.0             # 0: the cruise propellers stopped
    cruise_pitch_deg: float = 22.0      # at 0.75 R
    high_lift_rpm: float = 0.0          # 0: the high-lift propellers folded

    def flight(self):
        return FlightCondition(airspeed_kt=self.airspeed_kt, altitude_m=self.altitude_m, nacelle_deg=0.0)


def load_prop_aircraft(name):
    """A preset of presets/propellers: an Aircraft whose rotors are its discs, with .groups ({group: [rotor names]}) and
    .pitch_deg (of the fixed-pitch discs)."""
    p = Path(name)
    p = p if p.suffix else PROP_DIR / f"{name}.toml"
    with open(p, "rb") as f:
        d = tomllib.load(f)
    a = d["aircraft"]
    o = np.asarray(a["origin_in"], float)
    frame = lambda s: tuple(np.array([o[0] - s[0], -s[1], s[2] - o[2]]) * INCH)
    common = dict(cyclic=False, hub="hingeless", flap_frequency=2.0, elastic=False, axis=(1.0, 0.0, 0.0), role="lift")

    def rotor(k, name, hub, rotation):
        keep = ("radius", "blades", "root_cutout", "chord", "twist", "airfoil")
        return Rotor(name=name, rpm=2000.0, position=frame(hub), rotation=rotation, **common, **{q: d[k][q] for q in keep})
    c, h = d["cruise"], d["high_lift"]
    rotors = [rotor("cruise", n, s, r) for n, s, r in zip(c["names"], c["hubs_in"], c["rotation"])]
    groups = {n: [n] for n in c["names"]}
    right = h["hubs_in"]
    # disc n's station: right side inboard to outboard from 7, left side its mirror
    station = {7 + i: s for i, s in enumerate(right)}
    station.update({6 - i: [s[0], -s[1], s[2]] for i, s in enumerate(right)})
    for g, ns in h["groups"].items():
        groups[g] = [f"hl_{n}" for n in ns]
        # neighbours counter-rotate, the right wing the left's mirror: a group's discs all turn the same way
        rotors += [rotor("high_lift", f"hl_{n}", station[n], "cw" if n % 2 == 1 else "ccw") for n in ns]
    wing = [[s[0] * INCH, (o[0] - s[1]) * INCH, s[2] * INCH, (s[3] - o[2]) * INCH, s[4] * INCH] for s in a["wing_in"]]
    ac = Aircraft(name=a["name"], mass_kg=a["mass_kg"], description=a.get("description", ""), rotors=rotors,
                  cockpit=frame(a["cockpit_in"]), wing=wing, cabin=Cabin(**d["cabin"]) if "cabin" in d else None)
    ac.groups = groups
    ac.pitch_deg = h["pitch_deg"]
    ac.pitch_stops = tuple(np.radians(c["pitch_stops_deg"]))
    ac.motors = d.get("motor", {})
    ac.hubs = d.get("hub", {})
    return ac


def wing_upwash_deg(ac, rot, cl):
    """The wing's upwash at a disc ahead of it: the bound vortex of its section, at the quarter chord, at the hub, in degrees"""
    s = np.asarray(sorted(ac.wing), float)
    y = abs(rot.position[1])
    chord, le = np.interp(y, s[:, 0], s[:, 2]), np.interp(y, s[:, 0], s[:, 1])
    d = max(rot.position[0] - (le - 0.25 * chord), 0.25 * chord)
    return float(np.degrees(cl * chord / (4.0 * np.pi * d)))


def operate_props(ac, cond, opts=None, discs=None, spread=None, thrust=None):
    """The discs in the condition (discs: their names; default all that turn): an OperatingState for synth.render.
    spread (s): each disc a revolution per that long off the last, unsynchronised motors whose tones beat over a loop.
    thrust (N, each): the load, the pitch following it, in place of the blade pitch."""
    opts = opts or AeroOptions(bvi=False, elastic=False)
    atm = Atmosphere(cond.altitude_m)
    V = max(cond.airspeed_kt * KT, 0.5)
    vel = np.array([V, 0.0, 0.0])
    pitch = np.radians(cond.alpha_deg)
    Rbh = _pitch_matrix(pitch)
    U_h = -vel
    aft_h = Rbh @ np.array([-1.0, 0.0, 0.0])
    points, solved = [], {}
    k = 0
    for rot in ac.rotors:
        high = rot.name.startswith("hl_")
        rpm = cond.high_lift_rpm if high else cond.cruise_rpm
        if rpm <= 0.0 or (discs is not None and rot.name not in discs):
            continue
        if spread:
            rpm = rpm + 60.0 / spread * ((k + 1) // 2) * (-1) ** k
            k += 1
        omega = rpm * 2.0 * np.pi / 60.0
        # the wing's upwash tilts the flow into the disc, as the angle of attack does
        up = wing_upwash_deg(ac, rot, cond.wing_cl)
        Rd = _pitch_matrix(np.radians(cond.alpha_deg + up))
        M = _frame(Rd @ np.asarray(rot.axis), U_h, aft_h, rot.spin_sign)
        Uc = M @ U_h
        a = float(np.arcsin(np.clip(Uc[2] / V, -1.0, 1.0)))
        theta = np.radians(ac.pitch_deg if high else cond.cruise_pitch_deg)
        key = (high, round(rpm), round(float(np.degrees(a)), 0))
        if key not in solved and thrust is not None:
            solved[key] = _solved_at_thrust(rot, omega, V, a, thrust, atm, opts, ac.pitch_stops)
        elif key not in solved:
            t_ref = 0.5 * atm.rho * (omega * rot.radius) ** 2 * np.pi * rot.radius ** 2 * 0.01
            solved[key] = _solved_at_pitch(rot, omega, V, a, theta, atm, opts, t_ref)
        pt = RotorPoint(rot, omega, V, a, solved[key][1], M, Rbh @ np.asarray(rot.position), Uc)
        pt.aero = solved[key][0]
        points.append(pt)
    return OperatingState(ac, cond.flight(), atm, pitch, pitch, points, vel)


def render_group(ac, cond, group, observer, fs=48000.0, duration=15.0, quality="normal", variability=None, seed=1,
                 environment="open", opts=None, thrust=None, state=None):
    """A seamless loop of one group's discs at an observer riding with the aircraft (synth.render's Render). state: their
    OperatingState, to use again for another observer."""
    from .broadband import BroadbandOptions, air_turbulence
    from .synth import Quality, Variability, render
    names = ac.groups[group]
    q = Quality.get(quality)
    state = state or operate_props(ac, cond, opts or AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth, bvi=False, elastic=False),
                                   discs=names, spread=duration, thrust=thrust)
    bo = BroadbandOptions(n_azimuth=q.bb_azimuth, turbulence_ms=air_turbulence(cond.altitude_m), from_zero_lift=True)
    var = variability or Variability(seed=seed)
    r = render(ac, cond.flight(), observer, rotors=names, fs=fs, duration=duration, quality=q, variability=var,
               broadband_opts=bo, cabin=ac.cabin if observer.position is not None else None, state=state,
               environment=environment)
    r.state = state
    return r


# ---------------------------------------------------------------- motors

def tone_loop(tones, rpm, L, fs, seed, wander=0.002):
    """Tones at orders of the rotation (order, rms Pa), periodic over the L samples: each a whole number of cycles a loop,
    the whole breathing and wandering slowly by a share of a percent, as a motor under a real load does."""
    from .synth import _slow_noise, _warp
    rng = np.random.default_rng(seed)
    t = np.arange(L) / L
    x = np.zeros(L)
    f_rev = rpm / 60.0
    for order, rms in tones:
        cycles = max(int(round(order * f_rev * L / fs)), 1)
        x += np.sqrt(2.0) * rms * np.sin(2.0 * np.pi * (cycles * t + rng.uniform()))
    x *= 1.0 + 0.1 * _slow_noise(rng, L, fs, 1.0)[0]
    d = wander * _slow_noise(rng, L, fs, 0.5)[0]
    warp = np.cumsum(d - np.mean(d)) / fs
    return _warp(x, warp - np.mean(warp), fs)


def motor_loop(ac, kind, group, observer, cond, source="motor", seed=1, fs=48000.0, duration=15.0, environment="open"):
    """A group's motors (or, source 'hub', its spinners) at an observer: their tones at the reference speed and full torque,
    from the preset, at each disc's distance, through what the rotors' sound goes through. Returns (loop in Pa, tones)."""
    from .environment import Environment, apply as apply_environment
    from .synth import cabin_filter, highpass, loop_length, observer_position, propagate_static
    table = ac.motors[kind] if source == "motor" else ac.hubs[kind]
    rpm = table["rpm"]
    state_pos = observer_position(_Attitude(np.radians(cond.alpha_deg)), observer)
    rotors = [next(r for r in ac.rotors if r.name == n) for n in ac.groups[group]]
    Rbh = _pitch_matrix(np.radians(cond.alpha_deg))
    d = [float(np.linalg.norm(state_pos - Rbh @ np.asarray(r.position))) for r in rotors]
    if source == "motor":
        at1m = [(o, 2e-5 * 10.0 ** ((table["level_db"] + rel) / 20.0)) for o, rel, _ in table["tones"]]
    else:
        at1m = [(o, 2e-5 * 10.0 ** (db / 20.0) * table["distance_m"]) for o, db in table["tones"]]
    atm = Atmosphere(cond.altitude_m)
    L, _ = loop_length(rpm / 60.0, fs, duration)
    x = np.zeros(L)
    for i, dist in enumerate(d):
        x += tone_loop([(o, a / max(dist, 0.3)) for o, a in at1m], rpm, L, fs, seed + i)
    mean_d = float(np.mean(d))
    if observer.position is None:
        x = propagate_static(x, fs, mean_d, atm.T, atm.humidity, atm.p)
        x = apply_environment(x, fs, -state_pos[2], mean_d, Environment(environment))
    else:
        x = cabin_filter(x, fs, ac.cabin)
    return highpass(x, fs, 25.0), at1m


@dataclass
class _Attitude:
    pitch: float


# ---------------------------------------------------------------- a bank and its FlightGear section

CRUISE_SPEEDS_KT = (0.0, 60.0, 100.0, 150.0)
CRUISE_RPM = (1500.0, 2250.0, 2700.0)
CRUISE_CT = (-0.03, 0.0, 0.05, 0.12)     # regenerating, windmilling, cruising, climbing: thrust / (rho n^2 D^4)
HIGH_LIFT_SPEEDS_KT = (0.0, 40.0, 60.0, 80.0, 100.0, 120.0)
HIGH_LIFT_RPM = (1500.0, 3000.0, 4000.0, 4800.0, 5400.0)   # Fixed mode's 4800, Airspeed mode's top
VIEWS = ("cockpit", "side")


def _wing_cl(ac, kt, area=6.19):
    """The wing's lift coefficient at the speed: the weight's, at most what the blown wing reaches. None standing still"""
    if kt <= 0.0:
        return 0.0
    q = 0.5 * 1.225 * (kt * KT) ** 2
    return float(min(ac.mass_kg * 9.81 / (q * area), 2.5))


def make_prop_bank(ac, out_dir, distance=30.0, fs=48000.0, duration=15.0, bits=16, peak_dbfs=-1.0, seed=1, progress=print,
                   cruise_speeds=CRUISE_SPEEDS_KT, cruise_rpm=CRUISE_RPM, cruise_ct=CRUISE_CT,
                   high_lift_speeds=HIGH_LIFT_SPEEDS_KT, high_lift_rpm=HIGH_LIFT_RPM, views=VIEWS):
    """Loops of each group of discs over airspeed and rotor speed (the cruise propellers also over their load, a thrust
    coefficient from regenerating to climbing), in the cockpit and outside, with their motors' tones and the high-lift
    spinners' as layers of their own, manifest.json and pywopwop-props.xml. One gain for the cockpit's loops and one for the
    outside's. The groups are separate so that a stopped motor or a contactor's group can drop out."""
    import json
    from .synth import write_wav, _db, a_weighted_ms
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    so = Observer("side", 90.0, -20.0, distance)
    place = dict(cockpit=Observer("cockpit", position=tuple(ac.cockpit)), side=so)
    loops = []

    def keep(name, group, source, view, x, **kw):
        ms = float(np.mean(x ** 2))
        loops.append(dict(file=f"{name}_{group}_{view}_{source}.wav", condition=name, group=group, source=source, observer=view,
                          oaspl_db=round(_db(ms), 2), dba=round(_db(a_weighted_ms(x, fs)), 2), x=x.astype(np.float32), **kw))
    i = 0
    for g in ac.groups:
        cruise = g.startswith("cruise")
        speeds, rpms = (cruise_speeds, cruise_rpm) if cruise else (high_lift_speeds, high_lift_rpm)
        for kt in speeds:
            for rpm in rpms:
                for ct in (cruise_ct if cruise else (None,)):
                    if ct is not None and ct < 0.0 and kt <= 0.0:
                        continue                          # standing still nothing turns a propeller
                    i += 1
                    rot = next(r for r in ac.rotors if r.name == ac.groups[g][0])
                    D = 2.0 * rot.radius
                    T = None if ct is None else ct * 1.225 * (rpm / 60.0) ** 2 * D ** 4
                    c = PropCondition(f"{kt:g}kt_{rpm:g}rpm" + ("" if ct is None else f"_ct{ct:g}"), kt, 0.0,
                                      2.0 if cruise else 4.0, _wing_cl(ac, kt) * (0.5 if cruise else 1.0),
                                      rpm if cruise else 0.0, 22.0, 0.0 if cruise else rpm)
                    state = None
                    for v in views:
                        r = render_group(ac, c, g, place[v], fs=fs, duration=duration, seed=seed + 13 * i, thrust=T, state=state)
                        state = r.state
                        p = r.meta["rotors"][ac.groups[g][0]]
                        keep(c.name, g, "prop", v, r.signals["total"], airspeed_kt=kt, rpm=rpm, ct=ct, thrust_n=round(p["thrust_N"], 1),
                             power_kw=round(p["power_kW"], 2))
                        progress(f"  {g:13s} {c.name:24s} {v:8s} {loops[-1]['dba']:5.1f} dBA, thrust {p['thrust_N']:7.1f} N, {p['power_kW']:5.1f} kW")
        kind = "cruise" if cruise else "high_lift"
        ref = PropCondition("ref", 75.0, 0.0, 2.0)
        for v in views:
            keep("motor", g, "motor", v, motor_loop(ac, kind, g, place[v], ref, "motor", seed + 7, fs, duration)[0], rpm=ac.motors[kind]["rpm"])
            if kind in ac.hubs:
                keep("hub", g, "hub", v, motor_loop(ac, kind, g, place[v], ref, "hub", seed + 11, fs, duration)[0], rpm=ac.hubs[kind]["rpm"])

    def group_of(o):
        return "cockpit" if o == "cockpit" else "exterior"
    gains = {}
    for gr in {group_of(q["observer"]) for q in loops}:
        peak = max(float(np.max(np.abs(q["x"]))) for q in loops if group_of(q["observer"]) == gr)
        gains[gr] = 10.0 ** (peak_dbfs / 20.0) / peak
    files = []
    for q in loops:
        x = q.pop("x")
        write_wav(out / q["file"], x.astype(float), fs, bits, gains[group_of(q["observer"])], seed=abs(hash(q["file"])) % 2 ** 32)
        files.append(q)
    manifest = dict(aircraft=ac.name, generator="pywopwop", sample_rate=fs, bits=bits, exterior_distance_m=distance,
                    gain_db={k: round(20.0 * np.log10(v), 3) for k, v in gains.items()},
                    cruise_speeds_kt=list(cruise_speeds), cruise_rpm=list(cruise_rpm), cruise_ct=list(cruise_ct),
                    high_lift_speeds_kt=list(high_lift_speeds), high_lift_rpm=list(high_lift_rpm),
                    groups=ac.groups, positions=_positions(ac), properties=PROPERTIES, files=files,
                    note="levels: dB re 20 uPa at the listener before the gain. Motor tones are assumed; the spinners' are NASA's "
                         "tunnel array's; the cruise propeller has no acoustic data behind it")
    manifest["max_sources"] = prop_sources(manifest)
    progress(f"  at most {manifest['max_sources']['cockpit']} loops at once inside, {manifest['max_sources']['exterior']} outside")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "pywopwop-props.xml").write_text(flightgear_prop_xml(manifest))
    return manifest


# what the flight model is asked for, per disc (n: 0 and 1 the cruise propellers, left and right; 1 to 12 the high-lift discs)
PROPERTIES = dict(airspeed="/velocities/airspeed-kt",
                  cruise_rpm="/fdm/jsbsim/x57/cruise[{n}]/rpm", cruise_ct="/fdm/jsbsim/x57/cruise[{n}]/ct",
                  cruise_torque="/fdm/jsbsim/x57/cruise[{n}]/torque-norm",
                  hl_rpm="/fdm/jsbsim/x57/hl[{n}]/rpm", hl_torque="/fdm/jsbsim/x57/hl[{n}]/torque-norm")


def _positions(ac):
    """Each group's middle, in FlightGear's sound frame (x aft, y left, z up), m from the moment reference"""
    out = {}
    for g, names in ac.groups.items():
        p = np.mean([next(r for r in ac.rotors if r.name == n).position for n in names], axis=0)
        out[g] = dict(x=round(-float(p[0]), 3), y=round(float(p[1]), 3), z=round(float(p[2]), 3))
    return out


def _section(manifest, sound_dir="Sounds/pywopwop", exterior="side"):
    from .wind import _e, _v, _ramp_e
    have = {(f["condition"], f["group"], f["source"], f["observer"]): f for f in manifest["files"]}
    P = manifest["properties"]
    spd = f"<property>{P['airspeed']}</property>"
    entries, loops = [], []

    def hat(prop, grid, i):
        """the equal-power weight of grid point i on prop (a list of ramps), and the window it runs in"""
        w = []
        if i > 0:
            w.append(_ramp_e(prop, grid[i - 1], grid[i]))
        if i < len(grid) - 1:
            w.append(_ramp_e(prop, grid[i + 1], grid[i]))
        return w, (grid[i - 1] if i else None, grid[i + 1] if i < len(grid) - 1 else None)

    def condition(view, gates):
        c = ["<property>sim/current-view/internal</property>" if view == "cockpit"
             else "<not><property>sim/current-view/internal</property></not>"]
        for prop, (lo, hi) in gates:
            if lo is not None:
                c.append(f"<greater-than>{prop}<value>{lo:g}</value></greater-than>")
            if hi is not None:
                c.append(f"<less-than>{prop}<value>{hi:g}</value></less-than>")
        return _e("and", *c)

    def entry(name, f, view, vol, pitch, group, gates, kinds):
        pos = manifest["positions"][group]
        entries.append(f"\t\t<{f['source']}>\n\t\t\t<name>pww-{name}</name>\n\t\t\t<mode>looped</mode>\n"
                       f"\t\t\t<path>{sound_dir}/{f['file']}</path>\n\t\t\t<condition>{condition(view, gates)}</condition>\n"
                       f"\t\t\t<volume><expression>{vol}</expression></volume>\n{pitch}"
                       f"\t\t\t<position>\n\t\t\t\t<x>{pos['x']:g}</x>\n\t\t\t\t<y>{pos['y']:g}</y>\n\t\t\t\t<z>{pos['z']:g}</z>\n\t\t\t</position>\n"
                       + ("" if view == "cockpit" else f"\t\t\t<reference-dist>{manifest['exterior_distance_m']:g}</reference-dist>\n\t\t\t<max-dist>3000</max-dist>\n")
                       + f"\t\t</{f['source']}>\n")
        loops.append((view, gates, kinds))

    for g, names in manifest["groups"].items():
        cruise = g.startswith("cruise")
        n = {"cruise_left": 0, "cruise_right": 1}.get(g, int(names[0].split("_")[1]) if not cruise else 0)
        rpm_p = f"<property>{P['cruise_rpm' if cruise else 'hl_rpm'].format(n=n)}</property>"
        tq_p = f"<property>{P['cruise_torque' if cruise else 'hl_torque'].format(n=n)}</property>"
        ct_p = f"<property>{P['cruise_ct'].format(n=n)}</property>"
        speeds = manifest["cruise_speeds_kt" if cruise else "high_lift_speeds_kt"]
        rpms = manifest["cruise_rpm" if cruise else "high_lift_rpm"]
        cts = manifest["cruise_ct"] if cruise else [None]
        for view in ("cockpit", exterior):
            for a, kt in enumerate(speeds):
                for b, rpm in enumerate(rpms):
                    for c, ct in enumerate(cts):
                        name = f"{kt:g}kt_{rpm:g}rpm" + ("" if ct is None else f"_ct{ct:g}")
                        f = have.get((name, g, "prop", view))
                        if f is None and ct is not None and ct < 0.0:   # standing still nothing regenerates: the windmilling
                            f = have.get((f"{kt:g}kt_{rpm:g}rpm_ct{cts[c + 1]:g}", g, "prop", view))  # loop holds the grid's corner
                        if f is None:
                            continue
                        wv, gv = hat(spd, speeds, a)
                        wr, gr = hat(rpm_p, rpms, b)
                        if b == 0:
                            gr = (rpms[0] * 0.1, gr[1])   # down to a tenth of the lowest loop's speed, its level with the speed
                            wr = wr + [_ramp_e(rpm_p, rpms[0] * 0.1, rpms[0])]
                        w = wv + wr
                        gates = [(spd, gv), (rpm_p, gr)]
                        if ct is not None:
                            wc, gc = hat(ct_p, cts, c)
                            w += wc
                            gates.append((ct_p, gc))
                        vol = _e("sqrt", _e("product", *w)) if w else _v(1.0)
                        pitch = f"\t\t\t<pitch>{rpm_p}<factor>{1.0 / rpm:.6g}</factor><min>0.3</min><max>1.5</max></pitch>\n"
                        entry(f"{name}-{g}-{view}", f, view, vol, pitch, g, gates, ("kt", "rpm") + (("ct",) if ct is not None else ()))
            f = have.get(("motor", g, "motor", view))
            if f:
                ref = f["rpm"]
                vol = _e("product", _e("clip", _e("sum", _e("abs", tq_p), _v(0.15)), "<clipMin>0</clipMin><clipMax>1.3</clipMax>"),
                         _ramp_e(rpm_p, 0.0, 0.2 * ref))
                pitch = f"\t\t\t<pitch>{rpm_p}<factor>{1.0 / ref:.6g}</factor><min>0.05</min><max>1.5</max></pitch>\n"
                entry(f"motor-{g}-{view}", f, view, vol, pitch, g, [(rpm_p, (0.0, None))], ("rpm",))
            f = have.get(("hub", g, "hub", view))
            if f:
                ref = f["rpm"]
                vol = _e("pow", _e("div", rpm_p, _v(ref)), _v(3.0))
                pitch = f"\t\t\t<pitch>{rpm_p}<factor>{1.0 / ref:.6g}</factor><min>0.05</min><max>1.5</max></pitch>\n"
                entry(f"hub-{g}-{view}", f, view, vol, pitch, g, [(rpm_p, (0.0, None))], ("rpm",))
    head = ("<?xml version=\"1.0\"?>\n<!-- generated by pywopwop: the propellers and motors of the " + manifest["aircraft"]
            + "; merge into the aircraft's sound file -->\n<PropertyList>\n\t<fx>\n")
    return head + "".join(entries) + "\t</fx>\n</PropertyList>\n", loops


def flightgear_prop_xml(manifest, sound_dir="Sounds/pywopwop", exterior="side"):
    """FlightGear <fx> entries. Each group's loops cross-fade by equal power over airspeed and rotor speed (the cruise
    propellers also over their thrust coefficient), each pitched by the disc's rotor speed over its own; below the lowest
    speed the level falls with it. The motors' tones follow the torque and the rotor speed, the spinners' the cube of the
    speed. Every loop's condition is true only inside its fades: a looped sound holds a source while it is."""
    return _section(manifest, sound_dir, exterior)[0]


def prop_sources(manifest, sound_dir="Sounds/pywopwop", exterior="side"):
    """The most of the section's loops running at once, inside and outside, every group turning: each fade's axis has at
    most two loops between its grid points."""
    out = {}
    seen = {}
    for v, gates, kinds in _section(manifest, sound_dir, exterior)[1]:
        seen.setdefault(v, []).append(gates)
    for group, view in (("cockpit", "cockpit"), ("exterior", exterior)):
        out[group] = _worst(seen.get(view, []))
    return out


def _worst(all_gates):
    """The largest number of gate sets true together, at the midpoints between every property's grid points"""
    props = sorted({p for gates in all_gates for p, _ in gates})
    edges = {p: sorted({e for gates in all_gates for q, w in gates if q == p for e in w if e is not None}) for p in props}
    mids = {p: [e[0] - 1.0] + [0.5 * (a + b) for a, b in zip(e, e[1:])] + [e[-1] + 1.0] for p, e in edges.items()}
    # every rotor speed and thrust coefficient property moves on its own: take each one at its worst, airspeed being shared
    speed = next(p for p in props if "airspeed" in p)
    best = 0
    for v in mids[speed]:
        total = 0
        groups = {}
        for gates in all_gates:
            w = dict(gates).get(speed)
            if w is not None and not ((w[0] is None or v > w[0]) and (w[1] is None or v < w[1])):
                continue
            key = tuple(sorted(p for p, _ in gates if p != speed))
            groups.setdefault(key, []).append(dict(gates))
        for key, sets in groups.items():
            worst = 0
            for point in itertools.product(*[mids[p] for p in key]):
                at = dict(zip(key, point))
                n = sum(all((g[p][0] is None or at[p] > g[p][0]) and (g[p][1] is None or at[p] < g[p][1]) for p in key) for g in sets)
                worst = max(worst, n)
            total += worst
        best = max(best, total)
    return best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="render a propeller aircraft's sound bank and its FlightGear section")
    ap.add_argument("aircraft", help="preset or TOML file (presets/propellers)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--distance", type=float, default=30.0)
    a = ap.parse_args()
    make_prop_bank(load_prop_aircraft(a.aircraft), a.output, a.distance)
