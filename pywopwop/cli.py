"""pywopwop: rotor sounds from first principles, for flight simulators."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .model import FlightCondition, Observer, STANDARD_OBSERVERS, list_presets, load_aircraft


def _condition(a):
    return FlightCondition(airspeed_kt=a.speed, climb_fpm=a.climb, rpm_percent=a.rpm, mass_kg=a.mass,
                           altitude_m=a.altitude, ground_height_m=a.hub_agl, nacelle_deg=a.nacelle)


def _aero_opts(a, q):
    from .aero import AeroOptions
    return AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth, bvi=not a.no_bvi, elastic=not a.rigid)


def _add_flight(p):
    p.add_argument("aircraft", help="preset name or TOML file (see 'pywopwop presets')")
    p.add_argument("--speed", type=float, default=0.0, help="true airspeed, kt")
    p.add_argument("--climb", type=float, default=0.0, help="vertical speed, ft/min (negative: descent)")
    p.add_argument("--rpm", type=float, default=100.0, help="rotor speed, percent")
    p.add_argument("--nacelle", type=float, default=90.0,
                   help="tiltrotors: 90 helicopter mode, 0 airplane mode (the preset's airplane trims and rpm)")
    p.add_argument("--mass", type=float, default=None, help="gross mass, kg")
    p.add_argument("--altitude", type=float, default=0.0, help="pressure altitude, m (ISA)")
    p.add_argument("--hub-agl", type=float, default=None,
                   help="main rotor hub height above the ground, m: in ground effect (default: out of it)")
    p.add_argument("--no-bvi", action="store_true", help="no blade-vortex interaction (wake model off)")
    p.add_argument("--rigid", action="store_true", help="rigid blades (elastic model off)")
    p.add_argument("--quality", choices=("draft", "normal", "high"), default="normal")


def cmd_presets(a):
    for name in list_presets():
        ac = load_aircraft(name)
        rot = ", ".join(f"{r.name} {r.blades}x{r.radius:g} m {r.rpm:g} rpm" for r in ac.rotors)
        print(f"{name:12s} {ac.name:10s} {ac.description}\n{'':12s} {rot}")


def cmd_trim(a):
    from .operating import operate
    from .synth import Quality
    ac = load_aircraft(a.aircraft)
    q = Quality.get(a.quality)
    t0 = time.time()
    st = operate(ac, _condition(a), _aero_opts(a, q))
    print(f"{ac.name}: {a.speed:g} kt, {a.climb:+g} ft/min; pitch {np.degrees(st.pitch):+.1f} deg, "
          f"tip-path plane {np.degrees(st.tpp_pitch):+.1f} deg ({time.time() - t0:.2f} s)")
    for p in st.points:
        ae = p.aero
        bvi = f"{np.abs(ae.dlambda_bvi).max():.3f}" if ae.dlambda_bvi is not None else "-"
        print(f"  {p.rotor.name:6s} thrust {ae.thrust / 1e3:7.2f} kN  power {ae.power / 1e3:7.1f} kW  "
              f"CT/sigma {ae.ct_sigma:.3f}  mu {ae.mu_x:.3f}  alpha_TPP {np.degrees(p.alpha_tpp):+5.1f} deg  "
              f"Mat {ae.advancing_tip_mach:.3f}" + (f"  ground effect {ae.ground_effect:.3f}, recirculation "
                                                    f"{p.recirculation_rms:.1f} m/s" if p.ground_height else ""))
        print(f"         collective {np.degrees(ae.theta0):5.2f}  cyclic {np.degrees(ae.theta1c):+5.2f} "
              f"{np.degrees(ae.theta1s):+5.2f} deg  coning {np.degrees(ae.beta0):4.2f} deg  "
              f"elastic tip twist {np.degrees(ae.theta_e[:, -1].min()):+.2f}..{np.degrees(ae.theta_e[:, -1].max()):+.2f}"
              f" deg  BVI peak inflow {bvi}")


def _observer(a, ac):
    if a.cockpit:
        return Observer("cockpit", position=tuple(ac.cockpit))
    if a.az is not None or a.el is not None:
        return Observer("custom", a.az or 0.0, a.el or 0.0, a.dist)
    o = STANDARD_OBSERVERS[a.observer]
    return Observer(o.name, o.azimuth_deg, o.elevation_deg, a.dist)


def cmd_render(a):
    from .synth import Quality, Variability, render, write_wav
    ac = load_aircraft(a.aircraft)
    q = Quality.get(a.quality)
    obs = _observer(a, ac)
    var = Variability.none() if a.no_variation else Variability(seed=a.seed, variants=a.variants)
    t0 = time.time()
    r = render(ac, _condition(a), obs, rotors=a.rotor, fs=a.fs, duration=a.duration, quality=q, variability=var,
               with_broadband=not a.no_broadband,
               cabin=(ac.cabin if a.cabin is None else a.cabin) if a.cockpit else None, doors_open=a.doors_open,
               aero_opts=_aero_opts(a, q), pitch_headroom=a.headroom, f_highpass=a.highpass,
               environment=a.environment, stems=a.stems)
    x = r.total
    gain = 10 ** (a.peak / 20.0) / np.max(np.abs(x))
    out = Path(a.output or f"{Path(a.aircraft).stem}_{a.speed:g}kt_{a.climb:+g}fpm_{obs.name}.wav")
    write_wav(out, x, r.fs, a.bits, gain)
    print(f"wrote {out} ({len(x) / r.fs:.3f} s seamless loop, {time.time() - t0:.1f} s)")
    for k, v in r.levels().items():
        print(f"  {k:20s} {v['oaspl']:6.1f} dB  {v['dba']:6.1f} dBA")
    if a.stems:
        for k, x2 in r.signals.items():
            if k != "total":
                write_wav(out.with_name(out.stem + "_" + k.replace("/", "_") + ".wav"), x2, r.fs, a.bits, gain)
    if a.plot:
        _plot(r, a.plot)


def _plot(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = r.total
    fs = r.fs
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    n = int(min(len(x), 0.5 * fs))
    for k, y in r.signals.items():
        if "/" in k:
            ax[0].plot(np.arange(n) / fs * 1e3, y[:n], lw=0.6, label=k)
    ax[0].plot(np.arange(n) / fs * 1e3, x[:n], "k", lw=0.8, label="total")
    ax[0].set_xlabel("ms")
    ax[0].set_ylabel("Pa")
    ax[0].legend(fontsize=7, ncol=3)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X = np.abs(np.fft.rfft(x)) / len(x) * 2 / np.sqrt(2)
    ax[1].semilogx(f[1:], 20 * np.log10(X[1:] / 2e-5 + 1e-9), lw=0.4)
    ax[1].set_xlim(10, fs / 2)
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel("Hz")
    ax[1].set_ylabel("dB re 20 uPa")
    ax[0].set_title(f"{r.meta['aircraft']} {r.meta['airspeed_kt']:g} kt {r.meta['climb_fpm']:+g} ft/min, "
                    f"{r.meta['observer']} {r.meta['distance_m']:.0f} m")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(f"wrote {path}")


def cmd_wind(a):
    import json
    from .wind import make_wind_bank
    ac = load_aircraft(a.aircraft)
    g = json.loads((Path(a.gain_from) / "manifest.json").read_text())["gain_db"]["cockpit"] if a.gain_from else None
    print(f"{ac.name}: the cockpit's wind into {a.output}")
    make_wind_bank(ac, a.output, fs=a.fs, duration=a.duration, surface_mass=a.surface_mass, gain_db=g)
    print(f"FlightGear section: {Path(a.output) / 'pywopwop-wind.xml'}")


def cmd_bank(a):
    from .bank import BankOptions, make_bank
    ac = load_aircraft(a.aircraft)
    opts = BankOptions(observers=a.observers, cockpit=not a.no_cockpit, duration=a.duration, fs=a.fs,
                       bits=a.bits, quality=a.quality, pitch_headroom=a.headroom, only=a.conditions,
                       update=a.update, mode=a.mode, runup=a.runup, altitude_m=a.altitude)
    t0 = time.time()
    print(f"{ac.name}: rendering the bank into {a.output}")
    m = make_bank(ac, a.output, opts)
    gains = ", ".join(f"{k} {v:+.1f} dB" for k, v in m["gain_db"].items())
    print(f"{len(m['files'])} loops, gains {gains}, {time.time() - t0:.0f} s; "
          f"FlightGear snippet: {Path(a.output) / 'pywopwop-sound.xml'}")


def cmd_flyover(a):
    from .flyover import FlyoverOptions, render_flyover
    from .synth import Quality, write_wav
    ac = load_aircraft(a.aircraft)
    q = Quality.get(a.quality)
    t0 = time.time()
    fo = FlyoverOptions(altitude_m=a.height, offset_m=a.offset, duration=a.duration, fs=a.fs,
                        ground=a.ground, observer_height=1.2)
    r = render_flyover(ac, _condition(a), fo, quality=q, aero_opts=_aero_opts(a, q))
    out = Path(a.output or f"{Path(a.aircraft).stem}_flyover_{a.speed:g}kt.wav")
    gain = 10 ** (a.peak / 20.0) / np.max(np.abs(r.total))
    write_wav(out, r.total, r.fs, a.bits, gain)
    lv = r.levels()["total"]
    print(f"wrote {out} ({a.duration:g} s, max {r.meta['lmax_dba']:.1f} dBA, {time.time() - t0:.0f} s)")


def cmd_play(a):
    from .player import play
    play(a.files, gains=a.gain, volume=10 ** (a.volume / 20.0), seconds=a.seconds)


def cmd_glider(a):
    from .glider import GliderCondition, load_glider, render_glider
    from .synth import write_wav
    g = load_glider(a.glider)
    if a.cockpit or a.rear:
        o = Observer("cockpit_rear" if a.rear else "cockpit", position=tuple(g.cockpit["rear" if a.rear else "front"]))
    elif a.az is not None or a.el is not None:
        o = Observer("custom", a.az or 0.0, a.el or 0.0, a.dist)
    else:
        so = STANDARD_OBSERVERS[a.observer]
        o = Observer(so.name, so.azimuth_deg, so.elevation_deg, a.dist)
    c = GliderCondition(airspeed_kt=a.speed, brakes=a.brakes, load_factor=a.load_factor, altitude_m=a.altitude,
                        turbulence_ms=a.turbulence, aileron=a.aileron, rudder=a.rudder)
    t0 = time.time()
    r = render_glider(g, c, o, a.fs, a.duration, a.seed, a.environment)
    x = r.total
    out = Path(a.output or f"{Path(a.glider).stem}_{a.speed:g}kt_brakes{a.brakes:g}_{o.name}.wav")
    write_wav(out, x, r.fs, a.bits, 10 ** (a.peak / 20.0) / np.max(np.abs(x)))
    m = r.meta
    print(f"wrote {out} ({len(x) / r.fs:.2f} s seamless loop, {time.time() - t0:.1f} s): angle of attack "
          f"{m['alpha_body']:+.1f} deg, CL {m['CL']:.2f}, elevator {m['elevator_deg']:+.1f} deg"
          + (f", roll {m['roll_rate_dps']:+.0f} deg/s" if a.aileron else "")
          + (f", sideslip {m['sideslip_deg']:.0f} deg" if a.rudder else "")
          + (f", {m['separated_share']:.0%} of the chord separated" if m["separated_share"] > 0 else "")
          + (", stalled" if m["stalled_share"] > 0 else ""))
    for k, v in r.levels().items():
        print(f"  {k:16s} {v['oaspl']:6.1f} dB  {v['dba']:6.1f} dBA")


def cmd_glider_bank(a):
    from .glider import load_glider, make_glider_bank
    g = load_glider(a.glider)
    t0 = time.time()
    print(f"{g.name}: rendering the bank into {a.output}")
    m = make_glider_bank(g, a.output, distance=a.dist, fs=a.fs, duration=a.duration, bits=a.bits)
    gains = ", ".join(f"{k} {v:+.1f} dB" for k, v in m["gain_db"].items())
    print(f"{len(m['files'])} loops, stall {m['stall_speed_kt']:.1f} kt, gains {gains}, {time.time() - t0:.0f} s; "
          f"FlightGear section: {Path(a.output) / 'pywopwop-glider.xml'}")


def cmd_engine(a):
    from .engine import EngineCondition, engine_state, load_installations, render_engine
    from .synth import write_wav
    ac = load_aircraft(a.aircraft)
    insts = load_installations(a.aircraft)
    inst = next((x for x in insts if x.name == a.engine), insts[0]) if a.engine else insts[0]
    o = (Observer("cockpit_open" if a.doors_open else "cockpit", position=tuple(ac.cockpit)) if a.cockpit
         else Observer("side", a.az, 0.0, a.dist))
    c = EngineCondition(a.ng, lit=not a.cold, starter=a.starter, fuel_pph=a.fuel)
    print(f"{ac.name} {inst.name} engine ({inst.engine.name}): " + engine_state(inst.engine, a.ng, not a.cold, a.fuel).summary())
    t0 = time.time()
    r = render_engine(inst, c, o, a.fs, a.duration, a.seed, a.environment, cabin=ac.cabin, hub_height=ac.hub_height_m)
    x = r.total
    out = Path(a.output or f"{Path(a.aircraft).stem}_{inst.name}_ng{a.ng:g}{'_cold' if a.cold else ''}_{o.name}.wav")
    write_wav(out, x, r.fs, a.bits, 10 ** (a.peak / 20.0) / max(np.max(np.abs(x)), 1e-12))
    print(f"wrote {out} ({len(x) / r.fs:.2f} s seamless loop, {time.time() - t0:.1f} s); sources at 1 m: "
          + ", ".join(f"{k} {v:.1f} dB" for k, v in r.meta["source_db_1m"].items()))
    for k, v in r.levels().items():
        print(f"  {k:18s} {v['oaspl']:6.1f} dB  {v['dba']:6.1f} dBA")


def cmd_engine_bank(a):
    from .engine import make_engine_bank
    t0 = time.time()
    print(f"{a.aircraft}: rendering the engines' bank into {a.output}")
    m = make_engine_bank(a.aircraft, a.output, distance=a.dist, fs=a.fs, duration=a.duration, bits=a.bits, like=a.like,
                         nacelle_deg=a.nacelle)
    gains = ", ".join(f"{k} {v:+.1f} dB" for k, v in m["gain_db"].items())
    print(f"{len(m['files'])} loops, gains {gains}, {time.time() - t0:.0f} s; FlightGear section: "
          f"{Path(a.output) / 'pywopwop-engines.xml'}")


def cmd_validate(a):
    from . import validation
    ok = validation.run(print)
    sys.exit(0 if ok else 1)


def cmd_bench(a):
    from . import validation
    validation.benchmark(print)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pywopwop", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("presets", help="list the aircraft presets").set_defaults(fn=cmd_presets)
    s = sub.add_parser("trim", help="trim the aircraft and show its rotors' operating state")
    _add_flight(s)
    s.set_defaults(fn=cmd_trim)
    s = sub.add_parser("render", help="render a seamless rotor sound loop")
    _add_flight(s)
    s.add_argument("--observer", choices=sorted(STANDARD_OBSERVERS), default="side")
    s.add_argument("--az", type=float, help="observer azimuth from the nose, deg (right positive)")
    s.add_argument("--el", type=float, help="observer elevation, deg (below negative)")
    s.add_argument("--dist", type=float, default=100.0, help="observer distance, m")
    s.add_argument("--cockpit", action="store_true", help="listen from the cockpit (with the cabin wall)")
    s.add_argument("--cabin", type=float, default=None,
                   help="for --cockpit: a limp wall of this surface mass, kg/m^2 (default: the preset's cabin)")
    s.add_argument("--doors-open", action="store_true", help="for --cockpit: the cabin's doors open")
    s.add_argument("--rotor", nargs="*", help="only these rotors (names or roles, e.g. main tail)")
    s.add_argument("--duration", type=float, default=15.0, help="loop length, s (rounded to whole revolutions)")
    s.add_argument("--fs", type=float, default=48000.0)
    s.add_argument("--bits", type=int, choices=(16, 24, 32), default=16)
    s.add_argument("--peak", type=float, default=-1.0, help="normalise the peak to this dBFS")
    s.add_argument("--headroom", type=float, default=1.25, help="largest pitch factor the simulator applies")
    s.add_argument("--no-broadband", action="store_true")
    s.add_argument("--no-variation", action="store_true",
                   help="perfectly periodic (no gusts, vortex wander, tracking or rpm wander)")
    s.add_argument("--variants", type=int, default=8, help="revolutions with their own gusts and vortex wander")
    s.add_argument("--highpass", type=float, default=25.0, help="recording-chain high-pass, Hz (0: full physics)")
    s.add_argument("--environment", choices=("free", "ground", "open", "urban"), default="open",
                   help="the outside listener's surroundings: anechoic, ground reflection, plus a light or a strong "
                        "diffuse field")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--stems", action="store_true", help="also write each rotor and component")
    s.add_argument("--plot", help="write a waveform and spectrum plot (PNG)")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_render)
    s = sub.add_parser("bank", help="render a sound bank and a FlightGear sound snippet")
    s.add_argument("aircraft")
    s.add_argument("-o", "--output", required=True, help="output directory")
    s.add_argument("--observers", nargs="*", default=["side", "front_right", "rear", "below"],
                   choices=sorted(STANDARD_OBSERVERS))
    s.add_argument("--no-cockpit", action="store_true")
    s.add_argument("--duration", type=float, default=15.0, help="loop length, s (rounded to whole revolutions)")
    s.add_argument("--fs", type=float, default=48000.0)
    s.add_argument("--bits", type=int, choices=(16, 24, 32), default=16)
    s.add_argument("--headroom", type=float, default=1.25)
    s.add_argument("--quality", choices=("draft", "normal", "high"), default="normal")
    s.add_argument("--conditions", nargs="*", help="only these conditions (e.g. hover_ige)")
    s.add_argument("--update", action="store_true",
                   help="add the conditions to the existing bank in the output directory, at its gains")
    s.add_argument("--mode", choices=("helicopter", "airplane"), default="helicopter",
                   help="tiltrotors: the helicopter mode's conditions, or the airplane mode's (nacelles 0)")
    s.add_argument("--altitude", type=float, help="the air the bank flies in, m (default: sea level; airplane mode "
                   "3048, its cruise)")
    s.add_argument("--runup", action="store_true",
                   help="add the rotor's run-up on the ground at flat pitch (with --update: render only it)")
    s.set_defaults(fn=cmd_bank)
    s = sub.add_parser("wind", help="render the cockpit's wind (the windscreen's boundary layer) and its FlightGear section")
    s.add_argument("aircraft")
    s.add_argument("-o", "--output", required=True, help="output directory")
    s.add_argument("--gain-from", help="a bank's directory: its cockpit gain (the wind's loudness against its loops)")
    s.add_argument("--duration", type=float, default=15.0)
    s.add_argument("--fs", type=float, default=48000.0)
    s.add_argument("--surface-mass", type=float, default=None,
                   help="a limp windscreen of this surface mass, kg/m^2 (default: the cabin's windows)")
    s.set_defaults(fn=cmd_wind)
    s = sub.add_parser("flyover", help="render a fly-by past a listener on the ground")
    _add_flight(s)
    s.add_argument("--height", type=float, default=150.0, help="height above the listener at closest approach, m")
    s.add_argument("--offset", type=float, default=0.0, help="sideline distance of the track, m")
    s.add_argument("--duration", type=float, default=20.0)
    s.add_argument("--ground", choices=("hard", "grass", "none"), default="grass")
    s.add_argument("--fs", type=float, default=48000.0)
    s.add_argument("--bits", type=int, choices=(16, 24, 32), default=16)
    s.add_argument("--peak", type=float, default=-1.0)
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_flyover)
    s = sub.add_parser("play", help="loop WAV files gaplessly (as a simulator does), several at once")
    s.add_argument("files", nargs="+", help="WAV files, e.g. a bank's layers")
    s.add_argument("--gain", type=float, nargs="*", help="linear gain per file")
    s.add_argument("--volume", type=float, default=0.0, help="overall volume, dB")
    s.add_argument("--seconds", type=float, help="stop after this long (default: Ctrl-C)")
    s.set_defaults(fn=cmd_play)
    s = sub.add_parser("glider", help="render a glider's airframe sound (wing, tail, canopy wind, airbrakes)")
    s.add_argument("glider", help="glider preset or TOML file (presets/gliders)")
    s.add_argument("--speed", type=float, default=60.0, help="true airspeed, kt")
    s.add_argument("--brakes", type=float, default=0.0, help="airbrake opening, 0..1")
    s.add_argument("--load-factor", type=float, default=1.0, help="g, a steady pull-up")
    s.add_argument("--aileron", type=float, default=0.0, help="-1..1 (+ rolls right), a steady roll")
    s.add_argument("--rudder", type=float, default=0.0, help="-1..1, a steady sideslip")
    s.add_argument("--altitude", type=float, default=0.0)
    s.add_argument("--turbulence", type=float, default=1.0, help="rms turbulence the glider meets, m/s")
    s.add_argument("--cockpit", action="store_true", help="the front pilot's head, under the canopy")
    s.add_argument("--rear", action="store_true", help="the rear pilot's head")
    s.add_argument("--observer", choices=sorted(STANDARD_OBSERVERS), default="side")
    s.add_argument("--az", type=float)
    s.add_argument("--el", type=float)
    s.add_argument("--dist", type=float, default=30.0)
    s.add_argument("--environment", choices=("free", "ground", "open", "urban"), default="open")
    s.add_argument("--duration", type=float, default=15.0)
    s.add_argument("--fs", type=float, default=48000.0)
    s.add_argument("--bits", type=int, choices=(16, 24, 32), default=16)
    s.add_argument("--peak", type=float, default=-1.0)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_glider)
    s = sub.add_parser("glider-bank", help="render a glider's sound bank and its FlightGear section")
    s.add_argument("glider")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--dist", type=float, default=30.0, help="outside listeners' distance, m")
    s.add_argument("--duration", type=float, default=15.0)
    s.add_argument("--fs", type=float, default=48000.0)
    s.add_argument("--bits", type=int, choices=(16, 24, 32), default=16)
    s.set_defaults(fn=cmd_glider_bank)
    s = sub.add_parser("engine", help="render one turbine engine's sound (compressor, combustion, turbine, starter)")
    s.add_argument("aircraft", help="aircraft preset or TOML file with [[engine]] entries")
    s.add_argument("--engine", help="which of its engines (default: the first)")
    s.add_argument("--ng", type=float, default=100.0, help="gas generator speed, percent")
    s.add_argument("--cold", action="store_true", help="no flame: cranked by the starter, or spooling down")
    s.add_argument("--starter", action="store_true", help="the starter cranking")
    s.add_argument("--fuel", type=float, help="fuel flow, lb/h (default: the operating line; below idle a start's)")
    s.add_argument("--cockpit", action="store_true", help="the pilot's head")
    s.add_argument("--doors-open", action="store_true", help="with --cockpit: the cabin's doors open")
    s.add_argument("--az", type=float, default=90.0, help="outside: azimuth from the nose, deg (+ right)")
    s.add_argument("--dist", type=float, default=50.0, help="outside: distance on the ground, m")
    s.add_argument("--environment", choices=("free", "ground", "open", "urban"), default="open")
    s.add_argument("--duration", type=float, default=15.0)
    s.add_argument("--fs", type=float, default=48000.0)
    s.add_argument("--bits", type=int, choices=(16, 24, 32), default=16)
    s.add_argument("--peak", type=float, default=-1.0)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_engine)
    s = sub.add_parser("engine-bank", help="render an aircraft's engines' sound bank and its FlightGear section")
    s.add_argument("aircraft")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--like", help="a rotor bank (folder or manifest.json) whose gains to use")
    s.add_argument("--dist", type=float, default=50.0, help="outside listener's distance on the ground, m")
    s.add_argument("--nacelle", type=float, default=90.0,
                   help="tiltrotors: the nacelles' angle, deg (90 helicopter mode, 0 airplane mode)")
    s.add_argument("--duration", type=float, default=15.0)
    s.add_argument("--fs", type=float, default=48000.0)
    s.add_argument("--bits", type=int, choices=(16, 24, 32), default=16)
    s.set_defaults(fn=cmd_engine_bank)
    sub.add_parser("validate", help="check the models against published and closed-form results").set_defaults(
        fn=cmd_validate)
    sub.add_parser("bench", help="time the solvers").set_defaults(fn=cmd_bench)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
