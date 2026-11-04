"""The blade slap's level over the rotor's operating states, for a flight model to drive a bank's slap loops.

    python tools/slap_map.py mv22b -o out/mv22b/slap_map.json

trims the aircraft over a grid of airspeed and vertical speed, and records for each state the main rotor's advance
ratio mu and inflow through the tip-path plane lambda (lambda0 - mu_z: induced less upflow, positive down through
the disc, both over the tip speed), its thrust, and the slap layer's level (dB and dBA re 20 uPa) at the bank's
listening positions. The blade-vortex interaction depends on where the tip vortices pass the blades, which mu and
lambda set (lambda near zero: the wake stays in the disc), and on their strength, which follows the thrust: so a
flight model that computes mu and lambda for its rotors can look the slap's level up in any manoeuvre, not only in
the steady descents the bank was rendered at.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pywopwop.aero import AeroOptions                     # noqa: E402
from pywopwop.model import FlightCondition, Observer, STANDARD_OBSERVERS, load_aircraft   # noqa: E402
from pywopwop.operating import operate                    # noqa: E402
from pywopwop.synth import Quality, Variability, render   # noqa: E402


def point(args):
    name, kt, fpm, duration, quality, observers = args
    ac = load_aircraft(name)
    q = Quality.get(quality)
    fc = FlightCondition(airspeed_kt=kt, climb_fpm=fpm)
    try:
        st = operate(ac, fc, AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
    except Exception as e:                                  # a state the trim can't reach
        return dict(airspeed_kt=kt, climb_fpm=fpm, error=str(e))
    p = st.points[0]
    a = p.aero
    out = dict(airspeed_kt=kt, climb_fpm=fpm, mu=float(a.mu_x), lam=float(a.lambda0 - a.mu_z),
               lambda0=float(a.lambda0), mu_z=float(a.mu_z), thrust_N=float(a.thrust),
               ct=float(a.thrust / (a.rho * np.pi * a.R ** 2 * (a.omega * a.R) ** 2)), levels={})
    obs = [Observer("cockpit", position=tuple(ac.cockpit)) if o == "cockpit" else STANDARD_OBSERVERS[o] for o in observers]
    for o in obs:
        r = render(ac, fc, o, duration=duration, quality=q, state=st, variability=Variability(seed=1),
                   cabin=ac.cabin if o.name == "cockpit" else None)
        lv = r.levels()
        out["levels"][o.name] = {k.split("/")[-1]: dict(oaspl=round(v["oaspl"], 2), dba=round(v["dba"], 2))
                                 for k, v in lv.items() if k in ("layer/slap", "total")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("aircraft")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--speeds", default="0:130:10", help="kt, start:stop:step (stop included)")
    ap.add_argument("--climbs", default="-3000:1500:250", help="ft/min, start:stop:step (stop included)")
    ap.add_argument("--duration", type=float, default=2.0, help="s of sound per state")
    ap.add_argument("--quality", default="normal")
    ap.add_argument("--observers", default="side,cockpit",
                    help="listening positions (the bank's FlightGear section plays side's outside, cockpit's inside)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--append", action="store_true",
                    help="add the states to the map in the output file (made with the same settings), keeping its others")
    a = ap.parse_args()
    rng = lambda s: np.arange(float(s.split(":")[0]), float(s.split(":")[1]) + 1e-9, float(s.split(":")[2]))
    obs = a.observers.split(",")
    old = []
    if a.append and os.path.exists(a.output):
        prev = json.load(open(a.output))
        if (prev["duration"], prev["quality"]) != (a.duration, a.quality):
            sys.exit(f"{a.output} was made with other settings ({prev['duration']} s, {prev['quality']})")
        old = prev["points"]
    have = {(q["airspeed_kt"], q["climb_fpm"]) for q in old}
    jobs = [(a.aircraft, float(v), float(c), a.duration, a.quality, obs) for v in rng(a.speeds) for c in rng(a.climbs)
            if (float(v), float(c)) not in have]
    with ProcessPoolExecutor(a.workers) as ex:
        res = []
        for i, r in enumerate(ex.map(point, jobs)):
            res.append(r)
            if "error" in r:
                print(f"{r['airspeed_kt']:5.0f} kt {r['climb_fpm']:+6.0f} ft/min: {r['error']}", flush=True)
            elif i % 20 == 0:
                print(f"{i + 1}/{len(jobs)}: {r['airspeed_kt']:.0f} kt {r['climb_fpm']:+.0f} ft/min mu {r['mu']:.3f} "
                      f"lambda {r['lam']:+.4f} slap {r['levels']['side']['slap']['dba']:.1f} dBA (side)", flush=True)
    ac = load_aircraft(a.aircraft)
    res = old + res
    json.dump(dict(aircraft=ac.name, rotor=ac.main.name, rpm=ac.main.rpm, radius=ac.main.radius,
                   mass_kg=ac.mass_kg, duration=a.duration, quality=a.quality, points=res),
              open(a.output, "w"), indent=1)
    print(f"wrote {a.output}: {len(res)} states ({len(res) - len(old)} new)")


if __name__ == "__main__":
    main()
