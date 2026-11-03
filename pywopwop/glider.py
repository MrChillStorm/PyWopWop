"""Glider sound: the airframe's aerodynamic noise, in the cockpit and outside.

A glider has no engine; what is heard is the air. Its sources here:

- The wing's and tail's self-noise (BPM): the turbulent boundary layer passing the trailing edge, the separating
  flow as the stall nears, laminar vortex shedding where the pressure side stays laminar to the trailing edge. The
  boundary layers are the real sections' (FX S 02-196, FX 60-126, FX 71-L-150): their displacement thicknesses at
  the trailing edge and their transition points come from NeuralFoil at each strip's Reynolds number and angle of
  attack, instead of BPM's NACA 0012 correlations.
- Turbulent inflow (Amiet): the atmosphere's (and a thermal's) turbulence meeting the leading edges.
- The canopy's wind: the boundary layer's wall pressure on the canopy, heard through the acrylic (the mass law),
  less what a panel misses below hydrodynamic coincidence (its bending waves slower than the boundary layer's
  eddies: a second-order roll-off from that frequency down). The flow over the fuselage is a potential flow (a
  panel method on the preset's contours) at the body's angle of attack, with the wing's lifting line; the boundary
  layer runs along the canopy's top line, laminar from the nose to the canopy frame, turbulent from there, gathering
  the crossflow's boundary layer on the leeward line as the angle grows, and separating where it can no longer
  stand the pressure rise. Each panel of the glass gets Goody's wall pressure on its local layer, each heard by the
  solid angle it fills at the ear; separated, the wall's own fine turbulence is gone and only the separated shear
  layer's large, slow eddies remain.
- Airbrakes: each plate sheds vortices like a fence on a wall, a compact dipole by Curle's analogy (Strouhal number
  0.12 on the plate's height, rms force coefficient 0.25, spanwise correlation over three heights: assumed values
  of the bluff-plate literature); the separated flow behind it makes the whole section's lift fluctuate (rms lift
  coefficient 0.04, assumed); and its wake thickens the boundary layer at the trailing edge behind it (BPM's
  trailing-edge noise with the plate's height as the displacement thickness). Spoiler studies find such noise
  broadband and growing as about the sixth power of the speed, which these give.
- Separation and stall: each section's suction side separates from its trailing edge forward as its angle grows
  (where NeuralFoil's boundary layer's skin friction falls below zero), and BPM's separation noise follows that:
  none while attached, growing as the separated share grows to the share at the maximum lift, deep stall past
  it. The attached layer's fine hiss goes, the separated flow's low rumble comes. (BPM's own stages are a NACA
  0012's angles in its tunnel, which at a glider's speeds would call a cambered section stalled at 6 degrees.)

Lift: the weight (times the load factor) shared along the span by Schrenk's approximation, less where the brakes
spoil it. Manoeuvres are steady: a pull-up at the load factor, a roll at the aileron, a sideslip at the rudder.
The control surfaces deflect the sections (NeuralFoil's flapped aerofoils), the roll rate balances the wing's
rolling moment, the tailplane balances the pitching moments about the centre of gravity with the elevator, the fin
the fuselage's yawing moment with the rudder; a deflected aileron's ends shed vortices like a wing's tips. Gusts
move every source's level slowly (the turbulence's velocity over the airspeed), so a loop breathes the way flight
in moving air does.
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
import tomllib

import numpy as np
import scipy.fft as sfft

from . import bpm
from .atmosphere import Atmosphere, G0
from .model import KT, STANDARD_OBSERVERS, Observer
from .synth import (Render, Variability, _slow_noise, highpass, loop_length, propagate_static, scintillate)
from .wind import wall_pressure_psd

GLIDER_DIR = Path(__file__).parent / "presets" / "gliders"
BANDS = bpm.third_octave_bands(20.0, 20000.0)


def list_gliders():
    return sorted(p.stem for p in GLIDER_DIR.glob("*.toml"))


@dataclass
class Glider:
    name: str
    mass_kg: float
    wing_area: float
    span: float
    dihedral_deg: float
    fuselage_half_width: float
    wing: dict
    airbrakes: dict
    tailplane: dict
    fin: dict
    fuselage: dict
    cockpit: dict
    description: str = ""
    cg_x: float = None                  # centre of gravity, m aft of the datum
    ailerons: dict = None

    def chord(self, y):
        return np.interp(np.abs(y), self.wing["y"], self.wing["chord"])

    def x_le(self, y):
        return np.interp(np.abs(y), self.wing["y"], self.wing["x_le"])

    def incidence(self, y):
        return np.interp(np.abs(y), self.wing["incidence_y"], self.wing["incidence_deg"])


def load_glider(name_or_path):
    p = Path(name_or_path)
    if not p.suffix:
        p = GLIDER_DIR / f"{name_or_path}.toml"
    if not p.exists():
        raise FileNotFoundError(f"no glider '{name_or_path}' (presets: {', '.join(list_gliders())})")
    with open(p, "rb") as f:
        d = tomllib.load(f)
    g = dict(d["glider"])
    return Glider(**g, wing=d["wing"], airbrakes=d["airbrakes"], tailplane=d["tailplane"], fin=d["fin"],
                  fuselage=d["fuselage"], cockpit=d["cockpit"], ailerons=d.get("ailerons", {}))


def to_body(X, Y, Z):
    """The preset's datum frame (X aft, Y right, Z up) to the aircraft frame (x forward, y left, z up)."""
    return np.array([-X, -Y, Z], float)


@dataclass
class GliderCondition:
    airspeed_kt: float = 60.0           # true airspeed
    load_factor: float = 1.0            # a steady pull-up: the pitch rate (n - 1) g / V; negative: inverted
    brakes: float = 0.0                 # airbrake opening, 0..1
    aileron: float = 0.0                # -1..1, + rolls right (the FDM's aileron-pos-norm): a steady roll
    rudder: float = 0.0                 # -1..1 (the FDM's rudder-pos-norm): a steady sideslip
    altitude_m: float = 0.0
    turbulence_ms: float = 1.0          # rms vertical turbulence the glider meets (a thermal's edge: 2 to 3)
    turbulence_scale: float = 30.0      # m

    @property
    def V(self):
        return self.airspeed_kt * KT


# ---------------------------------------------------------------- strips

@dataclass
class Strip:
    kind: str                           # wing, tailplane, fin
    te: np.ndarray                      # trailing edge midpoint, aircraft frame
    span_dir: np.ndarray
    normal: np.ndarray
    chord: float
    span: float
    y: float                            # spanwise station |y| (wing) for the lift distribution
    section: str
    blend: tuple = None                 # (other section, fraction)
    brake: float = 0.0                  # share of the strip under an airbrake
    tip: bool = False
    control: str = None                 # aileron, elevator, rudder
    flap: float = 0.0                   # the control surface's share of the chord
    side: int = 0                       # +1 right, -1 left


def strips(g):
    out = []
    G = np.radians(g.dihedral_deg)
    e = np.asarray(g.wing["strip_edges"], float)
    kink, tip_y = g.wing["kink_y"], e[-1]
    by0, by1 = g.airbrakes["y"]
    ay0, ay1 = (g.ailerons or {}).get("y", (0.0, 0.0))
    for side in (+1, -1):                       # +1 the right wing (Y > 0, aircraft y < 0)
        for i in range(len(e) - 1):
            y0, y1 = e[i], e[i + 1]
            ys = np.linspace(y0, y1, 11)
            cs = g.chord(ys)
            ym = float(np.trapezoid(cs * ys, ys) / np.trapezoid(cs, ys))
            c = float(np.trapezoid(cs, ys) / (y1 - y0))
            z = (ym - g.fuselage_half_width) * np.tan(G)
            X_te = float(g.x_le(ym)) + c
            frac = 0.0 if ym < kink else (ym - kink) / (tip_y - kink)
            span_dir = np.array([0.0, -side * np.cos(G), np.sin(G)])
            normal = np.array([0.0, side * np.sin(G), np.cos(G)])
            brk = max(0.0, min(y1, by1) - max(y0, by0)) / (y1 - y0)
            out.append(Strip("wing", to_body(X_te, side * ym, z), span_dir, normal, c, y1 - y0, ym,
                             g.wing["root_section"], (g.wing["tip_section"], frac) if frac > 0 else None, brk,
                             tip=(i == len(e) - 2), side=side,
                             **(dict(control="aileron", flap=g.ailerons["chord_fraction"]) if ay0 <= ym <= ay1
                                else {})))
    t = g.tailplane
    he = np.linspace(0.0, t["span"] / 2, 4)
    for side in (+1, -1):
        for i in range(3):
            ym = 0.5 * (he[i] + he[i + 1])
            f = ym / (t["span"] / 2)
            c = t["root_chord"] + (t["tip_chord"] - t["root_chord"]) * f
            X_te = t["x_le_root"] + (t["x_le_tip"] - t["x_le_root"]) * f + c
            out.append(Strip("tailplane", to_body(X_te, side * ym, t["z"]), np.array([0.0, -side, 0.0]),
                             np.array([0.0, 0.0, 1.0]), c, he[i + 1] - he[i], ym, t["section"], tip=(i == 2),
                             control="elevator", flap=t.get("elevator_chord_fraction", 0.0), side=side))
    fn = g.fin
    ze = np.linspace(fn["z_bottom"], fn["z_top"], 4)
    for i in range(3):
        zm = 0.5 * (ze[i] + ze[i + 1])
        xle = fn["x_le_bottom"] + (fn["x_le_top"] - fn["x_le_bottom"]) * (zm - 0.15) / 1.10
        c = fn["x_te"] - xle
        out.append(Strip("fin", to_body(fn["x_te"], 0.0, zm), np.array([0.0, 0.0, 1.0]),
                         np.array([0.0, 1.0, 0.0]), c, ze[i + 1] - ze[i], zm, fn["section"],
                         control="rudder", flap=fn.get("rudder_chord_fraction", 0.0)))
    return out


# ---------------------------------------------------------------- sections (NeuralFoil)

ALPHAS = np.arange(-20.0, 22.01, 0.5)


RE_STEP = 1.03                          # Reynolds numbers are rounded to 3 % steps (the polars are cached)


@lru_cache(maxsize=4096)
def _section_polar(section, blend, re_k, flap=None):
    """NeuralFoil over ALPHAS at Reynolds number RE_STEP ** re_k, with a control surface (hinge, deflection in deg,
    trailing edge down positive) deflected: cl, cm, and per side the trailing edge's displacement thickness / chord
    and shape factor, the transition point and the share of the chord separated."""
    import aerosandbox as asb
    import neuralfoil as nf
    af = asb.Airfoil(section)
    if blend is not None:
        af = af.blend_with_another_airfoil(asb.Airfoil(blend[0]), blend[1])
    if flap is not None:
        af = af.add_control_surface(deflection=flap[1], hinge_point_x=flap[0])
    Re = RE_STEP ** re_k
    r = nf.get_aero_from_airfoil(af, alpha=ALPHAS, Re=Re, model_size="xlarge")
    k = len(nf.bl_x_points) - 1
    du = np.asarray(r[f"upper_bl_theta_{k}"]) * np.asarray(r[f"upper_bl_H_{k}"])
    dl = np.asarray(r[f"lower_bl_theta_{k}"]) * np.asarray(r[f"lower_bl_H_{k}"])
    xtr_l, xtr_u = np.asarray(r["Bot_Xtr"]), np.asarray(r["Top_Xtr"])
    return dict(cl=np.asarray(r["CL"]), cm=np.asarray(r["CM"]), du=du, dl=dl, Hu=np.asarray(r[f"upper_bl_H_{k}"]),
                Hl=np.asarray(r[f"lower_bl_H_{k}"]), xtr_l=xtr_l, xtr_u=xtr_u,
                sep_u=_separated(r, "upper", Re, xtr_u), sep_l=_separated(r, "lower", Re, xtr_l))


def _separated(r, side, Re, xtr):
    """The share of the chord separated ahead of the trailing edge on one side, per angle: the stations back to the
    edge where the turbulent boundary layer's skin friction (XFOIL's closure, Swafford's profiles, on NeuralFoil's
    momentum thickness and shape factor) has fallen below zero."""
    import neuralfoil as nf
    xb = np.asarray(nf.bl_x_points)
    th, H, ue = (np.array([np.asarray(r[f"{side}_bl_{q}_{i}"], float) for i in range(len(xb))])
                 for q in ("theta", "H", "ue/vinf"))
    lrt = np.log10(np.maximum(np.abs(ue) * th * Re, 20.0))
    cf = 0.3 * np.exp(-1.33 * H) / lrt ** (1.74 + 0.31 * H) + 1.1e-4 * (np.tanh(4.0 - H / 0.875) - 1.0)
    back = ((cf < 0.0) & (xb[:, None] > xtr[None, :]))[::-1]
    return np.argmin(np.vstack([back, np.zeros((1, back.shape[1]), bool)]), axis=0) / len(xb)


def section_state(strip, Re, cl_target=None, defl=0.0, alpha=None):
    """The strip's flow at the lift coefficient (or at the angle of attack alpha, deg, to its chord), its control
    surface deflected defl (deg, trailing edge down, rounded to a degree): the angle, the lift and moment
    coefficients, whether it is stalled, its zero-lift angle, and the trailing edge's boundary layers (m):
    pressure-side thickness, suction- and pressure-side displacement thicknesses; whether the pressure side is
    laminar to the trailing edge; the share of the chord separated, and the share at the maximum lift. The side
    that separates more is the suction side (behind a deflected control surface it can be the lower one)."""
    blend = None if strip.blend is None else (strip.blend[0], round(strip.blend[1], 2))
    flap = (round(1.0 - strip.flap, 2), int(round(defl))) if strip.flap and round(defl) != 0 else None
    P = _section_polar(strip.section, blend, int(round(np.log(Re) / np.log(RE_STEP))), flap)
    cl = P["cl"]
    ipk = int(np.argmax(np.where(ALPHAS <= 18.0, cl, -9)))
    imn = int(np.argmin(np.where((ALPHAS >= -18.0) & (ALPHAS < ALPHAS[ipk]), cl, 9)))
    clmax, clmin = cl[ipk], cl[imn]
    lin = slice(imn, ipk + 1)                             # between the stalls, lift rising with the angle
    a0 = float(np.interp(0.0, cl[lin], ALPHAS[lin]))
    if alpha is not None:
        a = float(alpha)
        stalled = not ALPHAS[imn] <= a <= ALPHAS[ipk]
    elif clmin < cl_target < clmax:
        a = float(np.interp(cl_target, cl[lin], ALPHAS[lin]))
        stalled = False
    else:
        # past the maximum (or the minimum, inverted): up to 2.5 deg beyond it for a demand 10 % over it
        pk, lim = (ipk, clmax) if cl_target > 0 else (imn, clmin)
        a = float(ALPHAS[pk] + np.sign(lim) * (1.0 + 15.0 * min(cl_target / lim - 1.0, 0.1)))
        stalled = True
    c_l = float(np.interp(a, ALPHAS, cl))
    pk = ipk if (c_l if alpha is not None else cl_target) >= 0 else imn
    at = lambda q: float(np.interp(a, ALPHAS, P[q]))
    su, sl = at("sep_u"), at("sep_l")
    lower = sl > su or (sl == su and (c_l if alpha is not None else cl_target) < 0)
    ds, dp = (at("dl"), at("du")) if lower else (at("du"), at("dl"))
    Hp, xtr_p = (at("Hu"), at("xtr_u")) if lower else (at("Hl"), at("xtr_l"))
    return dict(alpha=a, a0=a0, cl=c_l, cm=at("cm"), stalled=stalled, clmax=clmax, a_peak=float(ALPHAS[pk]),
                delta_p=dp * strip.chord * (2.9 if Hp > 2.2 else 8.0), dstr_s=ds * strip.chord,
                dstr_p=dp * strip.chord, laminar=xtr_p > 0.95, sep=max(su, sl),
                sep_stall=float((P["sep_l"] if lower else P["sep_u"])[pk]), defl=float(round(defl)))


def bpm_angle(st, M):
    """The angle of attack BPM is given (deg): its stages keyed to the section's own separation instead of a
    NACA 0012's angle. BPM's separation noise begins at gamma0 - gamma (its K2 term), peaks at gamma0 and turns to
    deep stall past it (its switch; gamma0 = 23.43 M + 4.651 is a NACA 0012's in BPM's tunnel, about 6 deg at a
    glider's Mach numbers, where a cambered section's flow is still attached). Here the attached flow stays below
    the first, the trailing edge's separation creeping forward takes BPM across the bell as the share separated
    grows to the share at the maximum lift, and only a section past its maximum is in deep stall."""
    g0 = 23.43 * M + 4.651
    gw = 27.094 * M + 3.31
    if st["stalled"]:
        return g0 + abs(st["alpha"] - st["a_peak"])
    if st["sep"] > 0.0:
        return g0 - gw + gw * min(st["sep"] / max(st["sep_stall"], st["sep"], 1e-3), 0.999)
    return min(abs(st["alpha"] - st["a0"]), g0 - gw)


def wing_cl(g, CL, brakes, ys, chords):
    """Section lift coefficients along the span (Schrenk), with the lift the opened brakes spoil (half of it at full
    opening, over their span) made up by the rest of the wing."""
    b = g.span
    c_ell = 4.0 * g.wing_area / (np.pi * b) * np.sqrt(np.clip(1.0 - (2.0 * ys / b) ** 2, 0.0, 1.0))
    load = 0.5 * (chords + c_ell)                      # cl c / CL
    by0, by1 = g.airbrakes["y"]
    spoil = np.where((ys > by0) & (ys < by1), 1.0 - 0.5 * brakes, 1.0)
    load = load * spoil
    return CL * load / chords


def stall_speed(g, load_factor=1.0, altitude_m=0.0, onset=False):
    """True airspeed (kt) where the first wing strip reaches its maximum lift (onset: where the first strip's flow
    begins to separate from its trailing edge)."""
    atm = Atmosphere(altitude_m)
    ws = [s for s in strips(g) if s.kind == "wing" and s.te[1] < 0]
    ys = np.array([s.y for s in ws])
    cs = np.array([s.chord for s in ws])
    lo, hi = 20.0, 120.0
    for _ in range(30):
        v = 0.5 * (lo + hi)
        V = v * KT
        CL = load_factor * g.mass_kg * G0 / (0.5 * atm.rho * V ** 2 * g.wing_area)
        cl = wing_cl(g, CL, 0.0, ys, cs)
        key = "sep" if onset else "stalled"
        over = any(section_state(s, V * s.chord / atm.nu, c)[key] > 0 for s, c in zip(ws, cl))
        lo, hi = (v, hi) if over else (lo, v)
    return hi


# ---------------------------------------------------------------- loads in a manoeuvre

def _fuselage_munk(g):
    """The fuselage's volume (m^3, the contours' super-ellipse sections) times k2 - k1, the apparent-mass factor
    of a prolate spheroid of its fineness (Lamb): its Munk moment is 2 q (k2 - k1) vol sin(angle), turning it
    further into the flow, in pitch and in yaw. 0 without contours."""
    from scipy.special import gamma
    fu = g.fuselage
    if "top_z" not in fu:
        return 0.0
    n = fu.get("section_exponent", 2.0)
    x = np.asarray(fu["x"], float)
    h = np.asarray(fu["top_z"], float) - np.asarray(fu["bottom_z"], float)
    w = np.asarray(fu["width"], float)
    A = h * w * gamma(1 + 1 / n) ** 2 / gamma(1 + 2 / n)
    vol = float(np.trapezoid(A, x))
    e = np.sqrt(max(1.0 - (np.sqrt(4 * A.max() / np.pi) / (x[-1] - x[0])) ** 2, 1e-6))
    lg = np.log((1 + e) / (1 - e))
    a0 = 2 * (1 - e ** 2) / e ** 3 * (lg / 2 - e)
    b0 = 1 / e ** 2 - (1 - e ** 2) / (2 * e ** 3) * lg
    return vol * (b0 / (2 - b0) - a0 / (2 - a0))


def _bisect(f, lo, hi, n=30):
    """The root of f, decreasing, on lo..hi (the end it never crosses to otherwise)."""
    if f(lo) <= 0:
        return lo
    if f(hi) >= 0:
        return hi
    for _ in range(n):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if f(m) > 0 else (lo, m)
    return 0.5 * (lo + hi)


def strip_states(g, cond, atm=None):
    """Each strip's flow (section_state's) in the condition, and the glider's: a steady pull-up at the load factor
    (pitch rate (n - 1) g / V; negative, inverted, from inverted flight: (n + 1) g / V), a steady roll at the
    aileron, a steady sideslip at the rudder.

    The wing carries the weight times the load factor (Schrenk) at its sections' angles. The ailerons deflect with
    the FDM's differential, and the roll rate grows until the wing's rolling moment vanishes (strip theory: the
    descending wing's angle grows by the rate times the station over the airspeed). The tailplane carries what
    balances the wing's and the fuselage's pitching moments about the centre of gravity, at its angle (the body's,
    its setting, less the wing's downwash, plus the pitch rate's over its arm, less its own induced angle), the
    elevator deflecting to the lift it needs, to its stops. The fin with the rudder deflected turns the glider to
    the sideslip where its yawing moment balances the fuselage's (its end plate, the tailplane, counted as half its
    height again; the fin's section symmetric). Returns the states (in strips' order) and dict(CL, alpha_body: the root section's angle less its
    incidence, alpha_geo: with the wing's induced angle, roll_rate_dps, elevator_deg, sideslip_deg, tail_cl)."""
    atm = atm or Atmosphere(cond.altitude_m)
    V, nu = cond.V, atm.nu
    q = 0.5 * atm.rho * V ** 2
    n = cond.load_factor
    CL = n * g.mass_kg * G0 / (q * g.wing_area)
    AR = g.span ** 2 / g.wing_area
    S = strips(g)
    wing = [s for s in S if s.kind == "wing"]
    Re = {id(s): V * s.chord / nu for s in S}
    cl_w = wing_cl(g, CL, cond.brakes, np.array([s.y for s in wing]), np.array([s.chord for s in wing]))
    base = [section_state(s, Re[id(s)], float(c)) for s, c in zip(wing, cl_w)]
    root_alpha = base[0]["alpha"] - float(g.incidence(wing[0].y))
    a_geo = root_alpha + np.degrees(CL / (np.pi * AR))
    st = {}
    # roll
    a = float(np.clip(cond.aileron, -1.0, 1.0))
    ail = g.ailerons or {}

    def aileron(s):
        x = -a * s.side                                   # + down: the right one goes up as the stick goes right
        return x * (ail["down_deg"] if x > 0 else ail["up_deg"]) if s.control == "aileron" else 0.0

    def rolled(p):
        return [section_state(s, Re[id(s)], defl=aileron(s), alpha=b["alpha"] + np.degrees(s.side * p * s.y / V))
                for s, b in zip(wing, base)]
    p = 0.0
    if a != 0.0 and ail:
        p = _bisect(lambda p: -sum(s.side * w["cl"] * s.chord * s.span * s.y for s, w in zip(wing, rolled(p))),
                    -3.0, 3.0)
        for s, w in zip(wing, rolled(p)):
            st[id(s)] = w
    else:
        st.update({id(s): b for s, b in zip(wing, base)})
    # pitch: the tailplane balances the moments about the centre of gravity
    tail = [s for s in S if s.kind == "tailplane"]
    munk = _fuselage_munk(g)
    xq = lambda s: -s.te[0] - 0.75 * s.chord             # quarter chord, datum frame
    de, CLt = 0.0, 0.1
    if g.cg_x is not None:
        xc = g.cg_x
        M = sum(-q * st[id(s)]["cl"] * s.chord * s.span * (xq(s) - xc) + q * st[id(s)]["cm"] * s.chord ** 2 * s.span
                for s in wing)
        M += 2.0 * q * munk * np.sin(np.radians(a_geo))
        S_t = sum(s.chord * s.span for s in tail)
        x_t = sum(xq(s) * s.chord * s.span for s in tail) / S_t
        CLt = M / (q * S_t * (x_t - xc))
        a_t = (a_geo + g.tailplane["incidence_deg"] - np.degrees(2.0 * CL / (np.pi * AR))
               + np.degrees((n - np.sign(n)) * G0 / V * (x_t - xc) / V)
               - np.degrees(CLt / (np.pi * g.tailplane["span"] ** 2 / S_t)))
        t0 = tail[0]
        de = _bisect(lambda d: CLt - section_state(t0, Re[id(t0)], defl=d, alpha=a_t)["cl"],
                     -g.tailplane.get("elevator_up_deg", 0.0), g.tailplane.get("elevator_down_deg", 0.0))
        st.update({id(s): section_state(s, Re[id(s)], defl=de, alpha=a_t) for s in tail})
    else:
        st.update({id(s): section_state(s, Re[id(s)], CLt) for s in tail})
    # yaw: the rudder's sideslip
    fin = [s for s in S if s.kind == "fin"]
    dr = abs(float(np.clip(cond.rudder, -1.0, 1.0))) * g.fin.get("rudder_deg", 0.0)
    beta = 0.0
    if dr > 0 and munk > 0 and g.cg_x is not None:
        S_v = sum(s.chord * s.span for s in fin)
        h_v = g.fin["z_top"] - g.fin["z_bottom"]
        AR_v = 1.5 * h_v ** 2 / S_v
        l_v = sum(xq(s) * s.chord * s.span for s in fin) / S_v - g.cg_x
        f0 = fin[len(fin) // 2]

        def fin_state(s, b):
            # mirrored (the fin's section symmetric): the sideslip's angle positive, the rudder against it
            c = 0.0
            for _ in range(5):
                w = section_state(s, Re[id(s)], defl=-dr, alpha=b - np.degrees(c / (np.pi * AR_v)))
                c = w["cl"]
            return w
        beta = _bisect(lambda b: -fin_state(f0, b)["cl"] * S_v * l_v - 2.0 * munk * np.sin(np.radians(b)), 0.0, 30.0)
        st.update({id(s): fin_state(s, beta) for s in fin})
    else:
        st.update({id(s): section_state(s, Re[id(s)], 0.0) for s in fin})
    info = dict(CL=CL, alpha_body=float(root_alpha), alpha_geo=float(a_geo), roll_rate_dps=float(np.degrees(p)),
                elevator_deg=float(de), sideslip_deg=float(beta), tail_cl=float(CLt))
    return [st[id(s)] for s in S], info


# ---------------------------------------------------------------- the fuselage's flow and the canopy's boundary layer

_FLOWS = {}


def _filament(p, a, b, gam, core):
    """Velocity at the points p (n, 3) of a straight vortex filament from a to b of circulation gam (Biot-Savart,
    with a core of radius core)."""
    r1, r2, r0 = p - a, p - b, b - a
    c = np.cross(r1, r2)
    cn = np.maximum(np.linalg.norm(c, axis=-1), 1e-12)
    l0 = np.linalg.norm(r0)
    h = cn / l0
    k = gam / (4.0 * np.pi) * (r1 @ r0 / np.linalg.norm(r1, axis=-1) - r2 @ r0 / np.linalg.norm(r2, axis=-1)) / l0
    return c / cn[:, None] * (k * h / (h ** 2 + core ** 2))[:, None]


def _wing_field(g, p, n=20):
    """The wing's velocity at the points p (datum frame) per unit airspeed and lift coefficient: a lifting line of
    horseshoe vortices on the quarter-chord line, the circulation Schrenk's (1/2 c cl), trailing aft."""
    G = np.tan(np.radians(g.dihedral_deg))
    yb = np.linspace(0.0, g.span / 2, n + 1)
    ym = 0.5 * (yb[1:] + yb[:-1])
    gam = 0.5 * wing_cl(g, 1.0, 0.0, ym, g.chord(ym)) * g.chord(ym)
    core = 0.1 * float(g.chord(0.0))
    v = np.zeros_like(p)
    for side in (1.0, -1.0):
        for j in range(n):
            ends = [np.array([float(g.x_le(y)) + 0.25 * float(g.chord(y)), side * y,
                              max(y - g.fuselage_half_width, 0.0) * G]) for y in (yb[j], yb[j + 1])]
            far = [e + np.array([60.0, 0.0, 0.0]) for e in ends]
            for a, b in ((far[0], ends[0]), (ends[0], ends[1]), (ends[1], far[1])):
                v += _filament(p, a, b, side * gam[j], core)
    return v


def fuselage_flow(g, nx=72, nt=24):
    """Potential flow about the fuselage: sources inside its surface (desingularised, a panel's size in), their
    strengths such that no flow crosses it, for four onset flows per unit airspeed: along the body axis (aft),
    across it upward (the angle of attack's share), the wing's lifting line per unit lift coefficient, and across
    it to the right (the sideslip's). The
    sections are super-ellipses on the preset's contours. Returns the top line (the plane of symmetry, where the
    canopy's boundary layer runs in straight flight) with each onset flow's surface velocity and the surface flow's
    lateral divergence there; the surface grid (stations, angles round from the top, each onset flow's velocities,
    the tangents along and around, the area per unit of both); and the canopy's panels (grid indices, points,
    normals, areas, velocities). Datum frame; cached per glider."""
    if g.name in _FLOWS:
        return _FLOWS[g.name]
    from scipy.interpolate import PchipInterpolator
    fu = g.fuselage
    xc = np.asarray(fu["x"], float)
    top, bot, wid = (PchipInterpolator(xc, np.asarray(fu[k], float)) for k in ("top_z", "bottom_z", "width"))
    ex = 2.0 / fu.get("section_exponent", 2.0)

    def surf(x, p):
        a, b, zc = np.maximum(wid(x) / 2, 1e-3), (top(x) - bot(x)) / 2, (top(x) + bot(x)) / 2
        sp, cp = np.sin(p), np.cos(p)
        return np.stack([x + 0 * p, a * np.sign(sp) * np.abs(sp) ** ex, zc + b * np.sign(cp) * np.abs(cp) ** ex], -1)
    xe = xc[0] + (xc[-1] - xc[0]) * (1.0 - np.cos(np.pi * np.linspace(0, 1, nx + 1))) / 2   # panel edges
    pe = (np.arange(nt + 1) - 0.5) * 2 * np.pi / nt                                         # 0: the top
    xm, pm = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (pe[1:] + pe[:-1])
    X, PH = np.meshgrid(xm, pm, indexing="ij")
    P = surf(X, PH)
    C = [surf(*np.meshgrid(xe[i:i + nx], pe[j:j + nt], indexing="ij")) for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))]
    N = np.cross(C[2] - C[0], C[3] - C[1]) / 2
    A = np.linalg.norm(N, axis=-1)
    N = N / np.maximum(A, 1e-12)[..., None]
    zc = (top(X) + bot(X)) / 2
    N *= np.sign(np.sum(N * (P - np.stack([X, 0 * X, zc], -1)), -1))[..., None]
    p_, n_, a_ = P.reshape(-1, 3), N.reshape(-1, 3), A.ravel()
    depth = np.minimum(np.sqrt(a_), 0.5 * np.minimum(wid(X) / 2, (top(X) - bot(X)) / 2).ravel())
    src = p_ - n_ * depth[:, None]
    r = p_[:, None, :] - src[None, :, :]
    K = r / (4.0 * np.pi * np.linalg.norm(r, axis=-1)[..., None] ** 3)
    onset = [np.broadcast_to([1.0, 0.0, 0.0], p_.shape), np.broadcast_to([0.0, 0.0, 1.0], p_.shape),
             _wing_field(g, p_), np.broadcast_to([0.0, 1.0, 0.0], p_.shape)]
    sig = np.linalg.solve(np.einsum("ijk,ik->ij", K, n_), -np.stack([np.sum(n_ * o, -1) for o in onset], 1))
    vel = [(o + np.einsum("ijk,j->ik", K, sig[:, k])).reshape(nx, nt, 3) for k, o in enumerate(onset)]
    # the lateral divergence on the top line: the flow along each neighbour's circumferential tangent, over the arc
    t_up, t_dn = P[:, 2] - P[:, 0], P[:, 0] - P[:, -2]
    t_up, t_dn = (t / np.linalg.norm(t, axis=-1, keepdims=True) for t in (t_up, t_dn))
    arc = np.linalg.norm(P[:, 1] - P[:, 0], axis=-1) + np.linalg.norm(P[:, 0] - P[:, -1], axis=-1)
    x0, x1 = fu["canopy_x"]
    around = np.minimum(PH, 2 * np.pi - PH) <= np.radians(fu.get("canopy_half_angle_deg", 70.0))
    cm = (X >= x0) & (X <= x1) & around
    e = 1e-4                                              # the surface's tangents along x and around, at the centres
    Tx, Tp = (surf(X + e, PH) - surf(X - e, PH)) / (2 * e), (surf(X, PH + e) - surf(X, PH - e)) / (2 * e)
    flow = dict(x=xm, z=P[:, 0, 2], v=np.array([v[:, 0] for v in vel]),
                div=np.array([(np.sum(v[:, 1] * t_up, -1) - np.sum(v[:, -1] * t_dn, -1)) / arc for v in vel]),
                xm=xm, pm=pm, vel=np.array(vel), Tx=Tx, Tp=Tp, J=np.linalg.norm(np.cross(Tx, Tp), axis=-1),
                canopy_i=np.nonzero(cm)[0], canopy_j=np.nonzero(cm)[1],
                canopy_p=P[cm], canopy_n=N[cm], canopy_a=A[cm], canopy_v=np.array([v[cm] for v in vel]))
    _FLOWS[g.name] = flow
    return flow


_HS = np.linspace(1.11, 4.0, 2000)
_H1S = np.where(_HS <= 1.6, 3.3 + 0.8234 * np.maximum(_HS - 1.1, 1e-9) ** -1.287, 3.3 + 1.5501 * (_HS - 0.6778) ** -3.064)


def _march(U, dlnh, x, ds, x_tr, nu):
    """Boundary layers along m surface streamlines at once: Thwaites's laminar method from each one's start (where
    its edge speed U (m, n) is first finite) to the transition station x_tr (datum x, from the stations x) or to
    laminar separation, then Head's entrainment method; dlnh (m, n) is the streamlines' spreading (1/h dh/ds:
    converging, the layer thickens), ds (m,) the steps. Separated where the shape factor reaches 2.4. Returns
    theta, H, cf, delta (m, n) and the index where each separated (-1)."""
    m, n = U.shape
    live = np.isfinite(U)
    U = np.where(live, U, 1.0)
    dlnh = np.where(live, dlnh, 0.0)
    dU = np.gradient(U, axis=1) / ds[:, None]
    th, H, cf = np.zeros((m, n)), np.full((m, n), 2.6), np.zeros((m, n))
    integ, turb, H1 = np.zeros(m), np.zeros(m, bool), np.zeros(m)
    sep = np.full(m, -1)
    for i in range(n):
        on = live[:, i]
        lam = on & ~turb
        if i:
            # Thwaites's integral of U^5 h^2, kept over the current h^2 (the spacing's ratio step to step: no
            # underflow where the streamlines fan out of the nose's stagnation)
            r2 = np.exp(-2.0 * dlnh[:, i] * ds)
            integ = np.where(lam & live[:, i - 1], integ * r2 + 0.5 * (U[:, i] ** 5 + U[:, i - 1] ** 5 * r2) * ds, integ)
            t = on & turb
            tp, hp = th[:, i - 1], H[:, i - 1]
            c = 0.246 * 10.0 ** (-0.678 * hp) * np.maximum(U[:, i - 1] * tp / nu, 100.0) ** -0.268
            g1, g2 = dU[:, i - 1] / U[:, i - 1], dlnh[:, i - 1]
            ent = 0.0306 * np.maximum(H1 - 3.0, 1e-3) ** -0.6169
            th1 = tp * H1 + (ent - tp * H1 * (g1 + g2)) * ds
            tn = np.maximum(tp + (c / 2 - tp * ((hp + 2) * g1 + g2)) * ds, 1e-6)
            H1 = np.where(t, np.maximum(th1 / tn, 3.31), H1)
            th[:, i] = np.where(t, tn, th[:, i])
            H[:, i] = np.where(t, np.interp(-H1, -_H1S, _HS), H[:, i])
            cf[:, i - 1] = np.where(t, c, cf[:, i - 1])
            sep = np.where(t & (sep < 0) & (H[:, i] >= 2.4), i, sep)
        tl = np.sqrt(0.45 * nu * integ / U[:, i] ** 6)
        th[:, i] = np.where(lam, tl, th[:, i])
        trip = lam & ((x[:, i] >= x_tr) | (tl ** 2 / nu * dU[:, i] < -0.09))
        H[:, i] = np.where(trip, 1.4, H[:, i])
        H1 = np.where(trip, float(np.interp(1.4, _HS, _H1S)), H1)
        turb |= trip
    cf[:, -1] = np.where(turb, 0.246 * 10.0 ** (-0.678 * H[:, -1]) * np.maximum(U[:, -1] * th[:, -1] / nu, 100.0)
                         ** -0.268, 0.0)
    return th, H, cf, th * (np.interp(H, _HS, _H1S) + H), sep


def canopy_boundary_layer(g, V, alpha_deg, CL, atm, n=1500):
    """The boundary layer along the canopy's top line, the plane of symmetry in straight flight, from the nose to the
    canopy's rear frame (_march: laminar to the canopy's front frame, where the preset says it trips, turbulent
    after), with the surface flow's lateral convergence (the leeward line gathers the crossflow's boundary layer as
    the angle of attack grows: a three-dimensional boundary layer's plane-of-symmetry equations). Returns per
    station: x (datum), edge speed, momentum thickness, shape factor, skin friction, thickness (theta (H1 + H)); and
    where it separated (None)."""
    F = fuselage_flow(g)
    w = _onsets(alpha_deg, 0.0, CL)
    Ue = V * np.linalg.norm(sum(c * vk for c, vk in zip(w, F["v"])), axis=-1)
    K = V * sum(c * dk for c, dk in zip(w, F["div"]))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(F["x"]), np.diff(F["z"])))])
    end = np.interp(g.fuselage["canopy_x"][1] + 0.05, F["x"], s)
    sf = np.linspace(s[0], end, n)
    U, xf = np.interp(sf, s, Ue), np.interp(sf, s, F["x"])
    th, H, cf, delta, sep = _march(U[None], (np.interp(sf, s, K) / U)[None], xf[None], np.array([sf[1] - sf[0]]),
                                   g.fuselage["transition_x"], atm.nu)
    return dict(x=xf, Ue=U, theta=th[0], H=H[0], cf=cf[0], delta=delta[0],
                sep=None if sep[0] < 0 else float(xf[sep[0]]))


def _onsets(alpha_deg, beta_deg, CL):
    """The onset flows' weights (fuselage_flow's): the airspeed along the axis and upward, the wing per unit lift
    coefficient, and the sideslip's (the relative wind from the right)."""
    a, b = np.radians(alpha_deg), np.radians(beta_deg)
    return np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), CL, -np.sin(b)


def canopy_layers(g, V, alpha_deg, beta_deg, CL, atm, step=0.01):
    """Each canopy panel's boundary layer, along the surface streamline that reaches it: traced upstream through the
    potential flow's surface velocity to where it leaves the nose's stagnation region (a fifth of the airspeed, or 5
    cm from the nose's tip), the layer marched down it (_march) with
    the streamlines' spreading (the surface divergence of the flow's direction: they converge where the crossflow
    gathers, on the leeward side, and the layer thickens). In sideslip the streamlines cross the canopy at an angle,
    the windward side's layer thin, the leeward side's gathering. Returns per panel: edge speed, theta, H, cf,
    delta, separated, and for the separated ones the largest shear stress their layer carried (at its fastest flow
    before separating) and its thickness there."""
    from scipy.interpolate import RegularGridInterpolator
    F = fuselage_flow(g)
    w = _onsets(alpha_deg, beta_deg, CL)
    v = sum(c * vk for c, vk in zip(w, F["vel"]))
    Tx, Tp, J = F["Tx"], F["Tp"], F["J"]
    G11, G12, G22 = (np.sum(a * b, -1) for a, b in ((Tx, Tx), (Tx, Tp), (Tp, Tp)))
    r1, r2 = np.sum(v * Tx, -1), np.sum(v * Tp, -1)
    det = G11 * G22 - G12 ** 2
    sp = np.maximum(np.linalg.norm(v, axis=-1), 1e-6)
    ua, ub = (G22 * r1 - G12 * r2) / det / sp, (G11 * r2 - G12 * r1) / det / sp   # d(x, phi)/ds
    xm, pm = F["xm"], F["pm"]
    wrap = lambda q: np.concatenate([q[:, -1:], q, q[:, :1]], axis=1)
    dpm = pm[1] - pm[0]
    div = (np.gradient(J * ua, xm, axis=0) + (np.roll(J * ub, -1, 1) - np.roll(J * ub, 1, 1)) / (2 * dpm)) / J
    grid = (xm, np.concatenate([[pm[0] - dpm], pm, [pm[-1] + dpm]]))
    f_ua, f_ub, f_sp, f_div = (RegularGridInterpolator(grid, wrap(q), bounds_error=False, fill_value=None)
                               for q in (ua, ub, sp, div))
    at = lambda f, X, P: f(np.stack([X, P % (2 * np.pi)], -1))
    X, P = F["xm"][F["canopy_i"]].astype(float), F["pm"][F["canopy_j"]].astype(float)
    m, nmax = len(X), 800
    path = np.full((nmax, 2, m), np.nan)
    on = np.ones(m, bool)
    for k in range(nmax):
        path[k, 0, on], path[k, 1, on] = X[on], P[on]
        Xh = X - 0.5 * step * at(f_ua, X, P)
        Ph = P - 0.5 * step * at(f_ub, X, P)
        X, P = np.where(on, X - step * at(f_ua, Xh, Ph), X), np.where(on, P - step * at(f_ub, Xh, Ph), P)
        on &= (X > xm[0] + 0.05) & (at(f_sp, X, P) > 0.2)  # stop where the flow leaves the nose's stagnation
        if not on.any():
            break
    path = path[:k + 1][::-1]                             # forward, every path ending at its panel
    xs, ps = path[:, 0].T, path[:, 1].T                   # (m, n), NaN before each one's start
    ok = np.isfinite(xs)
    flat = lambda f: np.where(ok, at(f, np.where(ok, xs, xm[0]), np.where(ok, ps, 0.0)), np.nan)
    U, dl = V * flat(f_sp), flat(f_div)
    th, H, cf, delta, sep = _march(U, dl, np.where(ok, xs, -1e9), np.full(m, step), g.fuselage["transition_x"],
                                   atm.nu)
    tau_max, d_sep = np.zeros(m), np.zeros(m)
    for j in np.flatnonzero(sep >= 0):
        u = np.where(np.arange(U.shape[1]) < sep[j], np.nan_to_num(U[j]), 0.0)
        i = int(np.argmax(u))
        tau_max[j] = 0.5 * atm.rho * U[j, i] ** 2 * cf[j, i]
        d_sep[j] = delta[j, sep[j]]
    return dict(Ue=U[:, -1], theta=th[:, -1], H=H[:, -1], cf=cf[:, -1], delta=delta[:, -1], separated=sep >= 0,
                tau_max=tau_max, delta_sep=d_sep)


def canopy_wind(g, V, alpha_deg, CL, atm, ear, beta_deg=0.0):
    """The canopy's boundary layer heard at the ear (datum frame): the wall pressure on each panel of the glass
    (Goody's spectrum on that panel's own layer, canopy_layers'; Farabee and Casarella's level), less what a panel
    misses below hydrodynamic coincidence, averaged by the solid angle each panel fills at the ear (a uniform pressure
    is heard at its own level, as Levine and DeFelice's cabin estimate). Where the layer has separated, the wall's
    own turbulence is gone and the separated shear layer's large eddies remain: Goody's outer part alone (time-scale
    ratio 1) at an rms of 2.75 times the largest shear stress the layer carried into the adverse gradient (Simpson
    et al. 1987, Na and Moin 1998: 2.5 to 3 near and past detachment). Returns the band powers (Pa^2, before the
    glass's mass law) and the share of the solid angle under separated flow."""
    from .wind import goody_psd
    F = fuselage_flow(g)
    bl = canopy_layers(g, V, alpha_deg, beta_deg, CL, atm)
    r = F["canopy_p"] - np.asarray(ear, float)
    rn = np.linalg.norm(r, axis=-1)
    om = F["canopy_a"] * np.abs(np.sum(F["canopy_n"] * r, -1)) / rn ** 3
    om /= om.sum()
    fu = g.fuselage
    D = fu.get("canopy_youngs", 3.2e9) * fu.get("canopy_thickness", 0.003) ** 3 / (12.0 * (1.0 - 0.37 ** 2))
    k_ac = np.sqrt(fu["canopy_surface_mass"] / D) / (2.0 * np.pi)
    sep = bl["separated"]
    parts = []
    for i in range(len(om)):
        U = bl["Ue"][i]
        if sep[i]:
            delta, p2, rt = bl["delta_sep"][i], (2.75 * bl["tau_max"][i]) ** 2, 1.0
        else:
            delta = bl["delta"][i]
            tau = 0.5 * atm.rho * U ** 2 * bl["cf"][i]
            ut = np.sqrt(tau / atm.rho)
            p2 = tau ** 2 * (6.5 + 1.86 * np.log(max(ut * delta / atm.nu, 333.0) / 333.0))
            rt = (delta / U) / (atm.nu / ut ** 2)
        parts.append((om[i], U, delta, p2, rt, (0.7 * U) ** 2 * k_ac))
    fg = np.unique(np.concatenate([np.geomspace(lo, hi, 12) for lo, hi in
                                   zip(BANDS * 2 ** (-1 / 6), BANDS * 2 ** (1 / 6))]))
    S = sum(o * goody_psd(fg, u, d, p2, rt) * (fg / fa) ** 2 / (1.0 + (fg / fa) ** 2) for o, u, d, p2, rt, fa in parts)
    return _band_power(lambda f: np.interp(f, fg, S)), float(om[sep].sum())


# ---------------------------------------------------------------- sound at a listener

def _band_power(fun, lo=BANDS * 2 ** (-1 / 6), hi=BANDS * 2 ** (1 / 6), n=12):
    """Integrate a spectral density (Pa^2/Hz) over the third-octave bands."""
    out = np.zeros(len(BANDS))
    for i, (a, b) in enumerate(zip(lo, hi)):
        f = np.geomspace(a, b, n)
        out[i] = np.trapezoid(fun(f), f)
    return out


def _mass_law(f, m):
    """The canopy's transmission on power, by the mass law and not capped for leaks: in an ASK 21's front cockpit
    recorded in aerobatics the hiss keeps falling with it to 6.3 kHz at every speed (a 30 dB cap left it flat from
    2 kHz up, 6 to 8 dB too bright)."""
    with np.errstate(divide="ignore"):
        tl = np.maximum(20.0 * np.log10(np.maximum(m * f, 1e-9)) - 47.0, 0.0)
    return 10.0 ** (-tl / 10.0)


def spectra(g, cond, listener, inside=False):
    """Mean-square pressure per third-octave band (Pa^2) at the listener (aircraft frame, m), per source:
    'airframe' (wing and tail self-noise and turbulent inflow), 'brakes' (what the opened brakes add), 'ailerons'
    (what the deflected ailerons and the roll add; with the aileron deflected), 'wind' (the canopy's boundary layer;
    inside only). inside: the listener sits under the canopy (its mass law applies to the
    sound from outside). Also returns the state: body angle of attack, lift coefficient, the span's share stalled,
    the chord's share separated (the span's mean), and inside the canopy's share (as heard) under separated flow."""
    atm = Atmosphere(cond.altitude_m)
    V = cond.V
    rho, c0, nu = atm.rho, atm.c, atm.nu
    S = strips(g)
    wing = [s for s in S if s.kind == "wing"]
    states, info = strip_states(g, cond, atm)
    CL = info["CL"]
    p2 = lambda spl: 10.0 ** (spl / 10.0) * 4e-10
    air = np.zeros(len(BANDS))
    brakes = np.zeros(len(BANDS))
    L = np.asarray(listener, float)
    stalled_span = separated = 0.0
    M = V / c0
    for i, (s, st) in enumerate(zip(S, states)):
        cl = st["cl"]
        if s.kind == "wing":
            stalled_span += s.span * st["stalled"]
            separated += s.span * st["sep"]
        a_load = abs(st["alpha"] - st["a0"])               # the lift's angle: inflow, tip vortex, laminar shedding
        a_star = bpm_angle(st, M)                         # the separation's stage: the trailing edge's noise
        r = L - s.te
        rn = float(np.linalg.norm(r))
        rh = r / rn
        xe = np.array([-1.0, 0.0, 0.0])                   # the flow over the chord
        theta = np.degrees(np.arccos(np.clip(rh @ xe, -1, 1)))
        phi = np.degrees(np.arctan2(abs(rh @ s.normal), rh @ s.span_dir))
        bl = (st["delta_p"], st["dstr_s"], st["dstr_p"])
        _, _, _, spl = bpm.tblte(a_star, s.chord, V, BANDS, s.span, rn, theta, phi, 1, nu, c0, bl=bl)
        air += p2(np.atleast_2d(spl)[0])
        air += p2(np.atleast_2d(bpm.blunt(a_load, s.chord, V, BANDS, s.span, rn, theta, phi, g.wing["te_thickness"],
                                          14.0, 1, nu, c0, bl=bl))[0])
        if st["laminar"] and not st["stalled"]:
            air += p2(np.atleast_2d(bpm.lblvs(a_load, s.chord, V, BANDS, s.span, rn, theta, phi, 0, nu, c0,
                                              bl=bl))[0])
        if cond.turbulence_ms > 0:
            air += p2(np.atleast_2d(bpm.amiet_inflow(s.chord, V, BANDS, s.span, rn, theta, phi, rho, c0,
                                                     cond.turbulence_ms / V, cond.turbulence_scale,
                                                     np.radians(a_load)))[0])
        if s.tip and s.kind == "wing":
            air += p2(np.atleast_2d(bpm.tip(min(a_load, 15.0), s.chord, V, BANDS, rn, theta, phi, c0,
                                            round_tip=True))[0])
        if s.control == "aileron" and st["defl"] != 0.0:
            # each end of a deflected aileron sheds a vortex as a wing's tip does (a flap's side-edge noise), as
            # strong as the lift the deflection adds there: BPM's square-edged tip at the angle that would give a
            # clean section that lift (thin-aerofoil slope)
            ends = sum(1 for j in (i - 1, i + 1) if not (0 <= j < len(S) and S[j].control == "aileron"
                                                          and S[j].side == s.side))
            jump = st["cl"] - section_state(s, V * s.chord / nu, alpha=st["alpha"])["cl"]
            a_edge = min(np.degrees(abs(jump) / (2.0 * np.pi)), 15.0)
            air += ends * p2(np.atleast_2d(bpm.tip(a_edge, s.chord, V, BANDS, rn, theta, phi, c0))[0])
        if s.brake > 0 and cond.brakes > 0:
            h = g.airbrakes["height"] * cond.brakes
            Lb = s.brake * s.span
            # the plate's wake at the trailing edge: BPM's trailing-edge noise with the wake's thickness, the
            # increase over the clean edge's
            blw = (max(st["delta_p"], 2.0 * h), max(st["dstr_s"], h), st["dstr_p"])
            _, _, _, wk = bpm.tblte(max(a_star, 13.0), s.chord, V, BANDS, Lb, rn, theta, phi, 1, nu, c0, bl=blw)
            _, _, _, cl_te = bpm.tblte(a_star, s.chord, V, BANDS, Lb, rn, theta, phi, 1, nu, c0, bl=bl)
            brakes += np.maximum(p2(np.atleast_2d(wk)[0]) - p2(np.atleast_2d(cl_te)[0]), 0.0)
            # vortex shedding from the plate (Curle), at the plate (a chord's 40 % from the leading edge)
            Ue = V * np.sqrt(1.0 + 0.8 * max(cl, 0.0))
            fs = 0.12 * Ue / h
            Fp = 0.5 * rho * Ue ** 2 * h * 0.25
            lc = 3.0 * h
            pb = s.te + np.array([s.chord * (1.0 - g.airbrakes["x_c"]), 0.0, 0.03])
            rb = L - pb
            rbn = float(np.linalg.norm(rb))
            rbh = rb / rbn
            D = 0.6 * rbh[0] ** 2 + 0.4 * (rbh @ s.normal) ** 2 + 0.05
            ptot = Lb * lc * fs ** 2 * Fp ** 2 * D / (4.0 * c0 ** 2 * rbn ** 2)
            shape = lambda f, fs=fs: (f / fs) ** 2 / (1.0 + (f / fs) ** 2) ** 2.25
            norm = np.trapezoid(shape(np.geomspace(0.01 * fs, 1000 * fs, 4000)), np.geomspace(0.01 * fs, 1000 * fs, 4000))
            brakes += ptot * _band_power(lambda f: shape(f) / norm)
            # and the unsteady lift the separated flow puts on the whole spoiled section (spoiler studies find the
            # pressure fluctuating around the whole airfoil): an rms lift coefficient of 0.04 at full opening
            # (assumed), correlated over half a chord, the same spectrum, radiating normal to the wing
            Fl = 0.5 * rho * V ** 2 * s.chord * 0.04 * cond.brakes
            Dl = (rbh @ s.normal) ** 2 + 0.05
            pl = Lb * 0.5 * s.chord * fs ** 2 * Fl ** 2 * Dl / (4.0 * c0 ** 2 * rbn ** 2)
            brakes += pl * _band_power(lambda f: shape(f) / norm)
    out = dict(airframe=air, brakes=brakes)
    if cond.aileron != 0.0:
        # what the deflected ailerons and the roll add: the airframe's sound over the same glider's without them
        clean, _ = spectra(g, replace(cond, aileron=0.0), listener)
        out = dict(airframe=clean["airframe"], brakes=clean["brakes"], ailerons=np.maximum(air - clean["airframe"], 0.0))
    cstate = {}
    if inside and "top_z" in g.fuselage:
        T = _mass_law(BANDS, g.fuselage["canopy_surface_mass"])
        out = {k: v * T for k, v in out.items()}
        # the canopy's own boundary layer, on the fuselage's flow at the body's angle (the wing's own included:
        # its induced angle, CL / (pi AR), on the root section's)
        wind, shared = canopy_wind(g, V, info["alpha_geo"], CL, atm, to_body(*L), info["sideslip_deg"])
        out["wind"] = wind * T
        cstate = dict(canopy_separated=shared)
    elif inside:
        T = _mass_law(BANDS, g.fuselage["canopy_surface_mass"])
        out = {k: v * T for k, v in out.items()}
        # the canopy's own boundary layer, from where the nose's laminar flow ends
        run = abs(g.fuselage["transition_x"] - (-L[0]))
        Uc = 1.1 * V                                       # the flow speeds up over the canopy's bulge
        # below hydrodynamic coincidence (the canopy's bending waves slower than the boundary layer's convected
        # eddies) the panel couples poorly to the boundary layer: a second-order roll-off there, the convection
        # speed 0.7 of the flow's, acrylic's stiffness
        fu = g.fuselage
        D = fu.get("canopy_youngs", 3.2e9) * fu.get("canopy_thickness", 0.003) ** 3 / (12.0 * (1.0 - 0.37 ** 2))
        f_ac = (0.7 * Uc) ** 2 * np.sqrt(fu["canopy_surface_mass"] / D) / (2.0 * np.pi)
        cpl = lambda f: (f / f_ac) ** 2 / (1.0 + (f / f_ac) ** 2)
        out["wind"] = _band_power(lambda f: wall_pressure_psd(f, Uc, run, atm) * cpl(f)) * T
    wing_span = sum(s.span for s in wing)
    state = dict(CL=CL, alpha_body=info["alpha_body"], stalled_share=stalled_span / wing_span,
                 separated_share=separated / wing_span, roll_rate_dps=info["roll_rate_dps"],
                 elevator_deg=info["elevator_deg"], sideslip_deg=info["sideslip_deg"], **cstate)
    return out, state


# ---------------------------------------------------------------- loops

def _band_loop(msp, L, fs, rng):
    """Gaussian noise with the third-octave band powers msp (Pa^2), power-complementary bands, periodic over L."""
    k = np.fft.rfftfreq(L, 1.0 / fs)
    with np.errstate(divide="ignore"):
        u = 3.0 * np.log2(np.maximum(k, 1e-9) / 1000.0)
    nb = np.round(3.0 * np.log2(BANDS / 1000.0))
    X = np.zeros(len(k), complex)
    for b in range(len(BANDS)):
        if msp[b] <= 0 or BANDS[b] > 0.45 * fs:
            continue
        w = np.cos(0.5 * np.pi * np.clip(u - nb[b], -1.0, 1.0)) * (np.abs(u - nb[b]) < 1.0)
        n = int(np.count_nonzero(w))
        if n == 0:
            continue
        Xb = (rng.normal(size=len(k)) + 1j * rng.normal(size=len(k))) * w
        Xb[0] = 0.0
        # this draw's mean square (one-sided spectrum: Parseval counts it twice), scaled to the band's exactly
        p = 2.0 * np.sum(np.abs(Xb) ** 2) / L ** 2
        X += Xb * np.sqrt(msp[b] / p)
    X[0] = 0.0
    return sfft.irfft(X, L, workers=-1)


def render_glider(g, cond, observer, fs=48000.0, duration=15.0, seed=1, environment="open", f_highpass=20.0):
    """A seamless loop of the glider's sound at an observer riding with it: Render with layers airframe, brakes
    and (inside) wind, and their total."""
    from .environment import Environment, apply as apply_environment
    inside = observer.position is not None
    pos = to_body(*observer.position) if inside else observer.offset()
    sp, state = spectra(g, cond, pos, inside)
    L, _ = loop_length(1.0, fs, duration)                 # any length: the sound has no period of its own
    rng = np.random.default_rng(seed)
    V = cond.V
    bw = max(0.15, V / 60.0)
    gust = _slow_noise(rng, L, fs, bw)[0] * cond.turbulence_ms / V
    k = dict(airframe=2.5, brakes=3.0, ailerons=2.5, wind=2.0)   # amplitude ~ U^2.5 (edges), U^3 (bluff), U^2 (wall)
    sig = {}
    for name, msp in sp.items():
        x = _band_loop(msp, L, fs, rng)
        sig[f"layer/{name}"] = x * np.maximum(1.0 + k[name] * gust, 0.2)
    dist = float(np.linalg.norm(pos))
    var = Variability()
    atm = Atmosphere(cond.altitude_m)
    env = Environment(environment)
    scint_cache = {}
    srng = np.random.default_rng(seed + 999)
    for key in sig:
        x = sig[key]
        if not inside:
            x = propagate_static(x, fs, dist, atm.T, atm.humidity, atm.p)
            x = scintillate(x, fs, dist, var, srng, scint_cache)
            x = apply_environment(x, fs, -pos[2], dist, env)
        sig[key] = highpass(x, fs, f_highpass)
    sig["total"] = sum(sig.values())
    meta = dict(glider=g.name, airspeed_kt=cond.airspeed_kt, brakes=cond.brakes, load_factor=cond.load_factor,
                observer=observer.name, distance_m=dist, **{k: round(float(v), 4) for k, v in state.items()})
    return Render(fs, sig, [], meta)


# ---------------------------------------------------------------- a bank and its FlightGear section

SPEEDS_KT = (40.0, 50.0, 60.0, 75.0, 90.0, 110.0, 130.0, 150.0)
VIEWS = ("cockpit", "cockpit_rear", "side", "front_right", "rear", "below")
STALL_LOAD_FACTORS = (1.0, 2.0, 3.5, 5.0)


def make_glider_bank(g, out_dir, speeds=SPEEDS_KT, views=VIEWS, distance=30.0, fs=48000.0, duration=15.0,
                     bits=16, peak_dbfs=-1.0, environment="open", seed=1, stall_load_factors=STALL_LOAD_FACTORS,
                     progress=print):
    """Loops of the glider at the airspeeds (brakes fully out and full aileron in a steady roll, their layers
    separate) and at the stall at the load factors (1 g, and accelerated: the wing's sound only), inverted at -1 g
    (the airspeeds from 1.1 times its stall's, and its stall), slipping with full rudder (inside: the canopy's flow
    turned), in the cockpit and outside, with manifest.json and pywopwop-glider.xml. Layers: airframe (wing and tail), wind (the canopy's
    boundary layer, inside), brakes (what the opened airbrakes add), ailerons (what full aileron and its roll add);
    mix: airframe and wind. One gain for the cockpit's loops and one for the outside's."""
    import json
    from .synth import write_wav
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vs, vi = stall_speed(g), stall_speed(g, -1.0)
    _, on = spectra(g, GliderCondition(airspeed_kt=stall_speed(g, onset=True), brakes=0.0), np.zeros(3))
    _, on_i = spectra(g, GliderCondition(airspeed_kt=stall_speed(g, -1.0, onset=True), load_factor=-1.0), np.zeros(3))
    conds = [(f"{v:g}kt", GliderCondition(airspeed_kt=v, brakes=1.0, aileron=1.0)) for v in speeds]
    for n in stall_load_factors:
        conds.append(("stall" if n == 1 else f"stall_{n:g}g",
                      GliderCondition(airspeed_kt=round(0.97 * stall_speed(g, n), 1), load_factor=n)))
    inverted = [v for v in speeds if v >= 1.1 * vi]
    conds += [(f"inv_{v:g}kt", GliderCondition(airspeed_kt=v, load_factor=-1.0)) for v in inverted]
    conds.append(("stall_inv", GliderCondition(airspeed_kt=round(0.97 * vi, 1), load_factor=-1.0)))
    conds += [(f"slip_{v:g}kt", GliderCondition(airspeed_kt=v, rudder=1.0)) for v in speeds]
    beta = strip_states(g, GliderCondition(airspeed_kt=60.0, rudder=1.0))[1]["sideslip_deg"]
    renders = []
    for i, (name, c) in enumerate(conds):
        for v in views:
            if c.rudder and not v.startswith("cockpit"):  # a slip turns the canopy's flow: heard inside
                continue
            if v.startswith("cockpit"):
                o = Observer(v, position=tuple(g.cockpit["rear" if v.endswith("rear") else "front"]))
            else:
                so = STANDARD_OBSERVERS[v]
                o = Observer(v, so.azimuth_deg, so.elevation_deg, distance)
            r = render_glider(g, c, o, fs, duration, seed + 13 * i, environment)
            r.signals["mix"] = sum(x for k, x in r.signals.items() if k in ("layer/airframe", "layer/wind"))
            renders.append((name, c, o, r))
            lv = r.levels()
            progress(f"  {name:10s} {v:12s} mix {lv['mix']['dba']:5.1f} dBA, airframe {lv['layer/airframe']['dba']:5.1f}"
                     + (f", brakes {lv['layer/brakes']['dba']:5.1f}" if c.brakes > 0 else "")
                     + (f", ailerons {lv['layer/ailerons']['dba']:5.1f}" if c.aileron != 0 else ""))
    keys = ["mix", "layer/airframe", "layer/wind", "layer/brakes", "layer/ailerons"]

    def group(o):
        return "cockpit" if o.position is not None else "exterior"
    gains = {}
    for gr in {group(o) for _, _, o, _ in renders}:
        peak = max(np.max(np.abs(r.signals[k])) for _, _, o, r in renders if group(o) == gr for k in keys
                   if k in r.signals)
        gains[gr] = 10.0 ** (peak_dbfs / 20.0) / peak
    files = []
    for name, c, o, r in renders:
        lv = r.levels()
        ref = np.sqrt(np.mean(r.signals["mix"] ** 2))
        for k in keys:
            if k not in r.signals or np.sqrt(np.mean(r.signals[k] ** 2)) < 1e-4 * ref:
                continue
            if c.load_factor > 1 and k != "layer/airframe":        # an accelerated stall: the wing's sound only
                continue
            label = k.split("/")[-1]
            fn = f"{name}_{o.name}_{label}.wav"
            write_wav(out / fn, r.signals[k], r.fs, bits, gains[group(o)], seed=abs(hash(fn)) % 2 ** 32)
            files.append(dict(file=fn, condition=name, airspeed_kt=c.airspeed_kt, load_factor=c.load_factor,
                              observer=o.name, source=label,
                              alpha_body_deg=r.meta["alpha_body"], oaspl_db=round(lv[k]["oaspl"], 2),
                              dba=round(lv[k]["dba"], 2)))
    stall = next(r for n, _, _, r in renders if n == "stall")
    stall_i = next(r for n, _, _, r in renders if n == "stall_inv")
    manifest = dict(glider=g.name, generator="pywopwop", sample_rate=fs, bits=bits,
                    gain_db={k: round(20 * np.log10(v), 3) for k, v in gains.items()},
                    stall_speed_kt=round(vs, 1), stall_alpha_body_deg=round(stall.meta["alpha_body"], 2),
                    separation_alpha_body_deg=round(on["alpha_body"], 2),
                    exterior_distance_m=distance, speeds_kt=list(speeds), files=files,
                    stall_load_factors=list(stall_load_factors), inverted_speeds_kt=inverted,
                    slip_speeds_kt=list(speeds), slip_sideslip_deg=round(beta, 1),
                    inverted_stall_speed_kt=round(vi, 1), inverted_stall_alpha_body_deg=round(stall_i.meta["alpha_body"], 2),
                    inverted_separation_alpha_body_deg=round(on_i["alpha_body"], 2),
                    note="levels: dB re 20 uPa at the listener before the gain; brakes: fully open; ailerons: full, "
                         "in a steady roll")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "pywopwop-glider.xml").write_text(flightgear_glider_xml(manifest))
    return manifest


def flightgear_glider_xml(manifest, sound_dir="Sounds/pywopwop", cockpit="cockpit", exterior="side"):
    """FlightGear <fx> entries. The canopy's wind follows the airspeed alone (its flow stays attached as the wing
    stalls): its loops cross-faded by equal power, the stall's the lowest. The wing's sound cross-fades the same way
    while its flow is attached and turns to the stall loops' with the angle of attack, from where the wing's flow
    begins to separate to the stall (as its separated share grows; over the last 3 degrees without that angle in the
    manifest), the stall loops (1 g and accelerated) cross-faded by airspeed in turn. The brakes' layer with the
    square root of their opening (their noise grows with the plates' height), the ailerons' with their deflection.
    Inverted (in the manifest) the same at -1 g, the upright and the inverted loops cross-faded by the load factor
    the pilot feels from +1 to -1 g. Slipping (in the manifest) the cockpit's wind turns to the full-rudder slip's
    with the sideslip angle. Inside the cockpit's loops, outside the chosen position's."""
    from .wind import _e, _v, _ramp_e
    spd = "<property>/velocities/airspeed-kt</property>"
    alpha = "<property>/orientation/alpha-deg</property>"
    brk = "<property>/surface-positions/speedbrake-pos-norm</property>"
    ail = "<property>/surface-positions/left-aileron-pos-norm</property>"
    a_st = manifest["stall_alpha_body_deg"]
    a_on = min(manifest.get("separation_alpha_body_deg", a_st - 3.0), a_st - 0.5)
    w_stall = _ramp_e(alpha, a_on, a_st)
    not_stall = _e("difference", _v(1.0), w_stall)
    have = {(f["condition"], f["observer"], f["source"]): f["file"] for f in manifest["files"]}
    kt = {f["condition"]: f["airspeed_kt"] for f in manifest["files"] if "airspeed_kt" in f}
    stalls = sorted({f["condition"] for f in manifest["files"] if f["condition"].startswith("stall")
                     and f.get("load_factor", 1.0) > 0}, key=lambda c: kt.get(c, 0.0))
    inverted = manifest.get("inverted_speeds_kt", [])
    out = []

    def entry(name, view, src, vol):
        fn = have.get((name, view, src))
        if fn is None:
            return
        cond = ("<property>sim/current-view/internal</property>" if view.startswith("cockpit")
                else "<not><property>sim/current-view/internal</property></not>")
        out.append(f"\t\t<{src}>\n\t\t\t<name>pww-{name}-{view}-{src}</name>\n\t\t\t<mode>looped</mode>\n"
                   f"\t\t\t<path>{sound_dir}/{fn}</path>\n\t\t\t<condition>{cond}</condition>\n"
                   f"\t\t\t<volume><expression>{vol}</expression></volume>\n"
                   + ("" if view.startswith("cockpit") else
                      "\t\t\t<reference-dist>30</reference-dist>\n\t\t\t<max-dist>1500</max-dist>\n")
                   + f"\t\t</{src}>\n")

    def crossfade(pts, view, src, vol):
        """pts: (condition, airspeed) rising; each loop's equal-power weight by airspeed, times vol(weight)"""
        for i, (name, v) in enumerate(pts):
            w = [_ramp_e(spd, pts[i - 1][1] if i else 0.0, v)]
            if i < len(pts) - 1:
                w.append(_ramp_e(spd, pts[i + 1][1], v))
            entry(name, view, src, vol(_e("sqrt", _e("product", *w) if len(w) > 1 else w[0])))
    # upright and inverted: by the load factor the pilot feels, +1 to -1 g (equal power)
    pg = "<property>/accelerations/pilot-g</property>"
    inv_w = _ramp_e(pg, 1.0, -1.0)
    up = _e("sqrt", _e("difference", _v(1.0), inv_w)) if inverted else _v(1.0)
    inv = _e("sqrt", inv_w)
    if inverted:
        a_sti = manifest["inverted_stall_alpha_body_deg"]
        a_oni = manifest.get("inverted_separation_alpha_body_deg", a_sti + 3.0)
        a_oni = max(a_oni, a_sti + 0.5) if a_sti < 0 else a_oni
        w_stall_i = _ramp_e(alpha, a_oni, a_sti)
        not_stall_i = _e("difference", _v(1.0), w_stall_i)
    slips = [(f"slip_{v:g}kt", v) for v in manifest.get("slip_speeds_kt", [])]
    if slips:
        slip_w = _ramp_e(_e("abs", "<property>/orientation/side-slip-deg</property>"), 0.0,
                         manifest["slip_sideslip_deg"])
        straight, slipping = _e("sqrt", _e("difference", _v(1.0), slip_w)), _e("sqrt", slip_w)
    for view in (cockpit, exterior):
        speeds = [(f"{v:g}kt", v) for v in manifest["speeds_kt"]]
        slipped = slips and any((c, view, "wind") in have for c, _ in slips)
        crossfade(speeds, view, "airframe", lambda w: _e("product", w, not_stall, up))
        crossfade(([("stall", kt["stall"])] if ("stall", view, "wind") in have and "stall" in kt else []) + speeds,
                  view, "wind", lambda w: _e("product", w, up, straight) if slipped else _e("product", w, up))
        if slipped:
            crossfade(slips, view, "wind", lambda w: _e("product", w, up, slipping))
        crossfade(speeds, view, "brakes", lambda w: _e("product", w, _e("sqrt", brk)))
        crossfade(speeds, view, "ailerons", lambda w: _e("product", w, _e("abs", ail)))
        crossfade([(c, kt.get(c, 0.0)) for c in stalls], view, "airframe", lambda w: _e("product", w, w_stall, up))
        if "stall" not in kt:
            entry("stall", view, "wind", w_stall)
        if inverted:
            ispeeds = [(f"inv_{v:g}kt", v) for v in inverted]
            crossfade(ispeeds, view, "airframe", lambda w: _e("product", w, not_stall_i, inv))
            crossfade([("stall_inv", kt["stall_inv"])] + ispeeds, view, "wind", lambda w: _e("product", w, inv))
            crossfade([("stall_inv", kt["stall_inv"])], view, "airframe", lambda w: _e("product", w, w_stall_i, inv))
    return ("<?xml version=\"1.0\"?>\n<!-- generated by pywopwop: the airframe's sound of the " + manifest["glider"]
            + "; merge into the aircraft's sound file -->\n<PropertyList>\n\t<fx>\n" + "".join(out)
            + "\t</fx>\n</PropertyList>\n")
