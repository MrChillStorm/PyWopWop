"""The X-57's main wheels touching down in JSBSim, against tyre.spin_up.

    python tools/jsbsim_touchdown.py PATH/TO/X-57/X-57 [KT SINK_FPS ...]

Needs the jsbsim Python module and the X-57's flight model (github.com/MrChillStorm/X-57; its aircraft folder, holding
x57-mod3.xml and Systems/). Trims the aircraft at the airspeed for a steady descent at the sink rate, flaps down, then
holds the controls and lets its mains meet the runway: the slide lasts until the X-57's own rim model (gear/unit[1]/
spin-kt, spun up by 0.5 times the strut's force) reaches the ground speed. Prints each slide and spin_up's at the sink
rate and slip the flight model latched, with the X-57's mass at the wheel (tools/tyre_recordings.py's X57_MASS).
Defaults: 75 kt at 0.5, 1, 2.6 and 4 fps."""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pywopwop import tyre as T                                       # noqa: E402
from tyre_recordings import X57_MASS                                 # noqa: E402

MODEL = "x57-mod3"
FPS_PER_KT = 1.68781


def load(aircraft_dir, tree):
    import jsbsim
    d = os.path.join(tree, "aircraft", MODEL)
    if not os.path.isdir(d):
        os.makedirs(d)
        os.symlink(os.path.join(aircraft_dir, MODEL + ".xml"), os.path.join(d, MODEL + ".xml"))
        os.symlink(os.path.join(aircraft_dir, "Systems"), os.path.join(d, "Systems"))
    fdm = jsbsim.FGFDMExec(tree, None)
    fdm.set_debug_level(0)
    fdm.load_model(MODEL)
    fdm.set_dt(1e-3)
    fdm["ctl/instant"] = 1               # flaps where they are commanded, at once
    fdm["ctl/flap-cmd-deg"] = 30
    return fdm


def balance(fdm, kt, sink_fps, alpha, elev, h=40.0):
    """the vertical (earth, down) and pitch accelerations of this state"""
    fdm["ic/vt-kts"] = kt
    fdm["ic/gamma-deg"] = -np.degrees(np.arcsin(sink_fps / (kt * FPS_PER_KT)))
    fdm["ic/alpha-deg"] = alpha
    fdm["ic/h-agl-ft"] = h
    fdm["fcs/elevator-cmd-norm"] = elev
    fdm.run_ic()
    fdm.run()
    fdm.run()
    th = np.radians(fdm["attitude/theta-deg"])
    return np.array([-np.sin(th) * fdm["accelerations/udot-ft_sec2"] + np.cos(th) * fdm["accelerations/wdot-ft_sec2"],
                     fdm["accelerations/qdot-rad_sec2"]])


def trim(fdm, kt, sink_fps):
    x = np.array([8.0, 0.2])
    for _ in range(12):
        f = balance(fdm, kt, sink_fps, *x)
        if np.abs(f).max() < 1e-3:
            break
        J = np.column_stack([(balance(fdm, kt, sink_fps, *(x + d)) - f) / 1e-3 for d in np.eye(2) * 1e-3])
        x = x - np.linalg.solve(J, f)
    return x


def touchdown(fdm, kt, sink_fps, gap_ft=0.02):
    """the trimmed state with its mains gap_ft over the runway, run with the controls held: (latched sink fps, slip
    kt, slide s)"""
    alpha, elev = trim(fdm, kt, sink_fps)
    fdm["ic/alpha-deg"], fdm["fcs/elevator-cmd-norm"] = alpha, elev
    lo, hi = 0.0, 10.0
    for _ in range(30):                  # the height at which the mains just touch
        fdm["ic/h-agl-ft"] = 0.5 * (lo + hi)
        fdm.run_ic()
        if fdm["gear/unit[1]/compression-ft"] > 0:
            lo = 0.5 * (lo + hi)
        else:
            hi = 0.5 * (lo + hi)
    fdm["ic/h-agl-ft"] = hi + gap_ft
    fdm.run_ic()
    t, on = 0.0, None
    while t < 2.0:
        fdm.run()
        t += 1e-3
        if on is None and fdm["gear/unit[1]/WOW"] > 0.5:
            on = t
        if on is not None and fdm["gear/unit[1]/spin-kt"] >= fdm["velocities/vg-fps"] * 0.592484 - 1e-6:
            return fdm["gear/unit[1]/touchdown-sink-fps"], fdm["gear/unit[1]/touchdown-slip-kt"], t - on
    raise RuntimeError("no touchdown")


def main(argv):
    aircraft_dir = os.path.abspath(argv[0])
    pairs = [float(a) for a in argv[1:]] or [75, 0.5, 75, 1.0, 75, 2.6, 75, 4.0]
    with tempfile.TemporaryDirectory() as tree:
        for kt, fps in zip(pairs[::2], pairs[1::2]):
            fdm = load(aircraft_dir, tree)
            sink, slip, slide = touchdown(fdm, kt, fps)
            model = T.spin_up(slip * 0.514444, "6.00-6", 553 * 14.5939, 5228 * 14.5939, sink * 0.3048,
                              3000 * 0.85 / 2 * 4.44822, mass_kg=X57_MASS["main"])[0][-1]
            print(f"{kt:g} kt, {fps:g} fps: latched {sink:.2f} fps and {slip:.1f} kt; slide {slide * 1e3:.0f} ms in "
                  f"JSBSim, {model * 1e3:.0f} ms by spin_up")


if __name__ == "__main__":
    main(sys.argv[1:])
