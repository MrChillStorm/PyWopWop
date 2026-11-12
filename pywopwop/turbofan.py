"""Turbofan engines: their sound from their state, for flight simulator loops.

A two-shaft turbofan whose two flows mix into one nozzle (presets/turbofans/<name>.toml), on a jet aircraft
(presets/jets/<name>.toml). The turboshafts' module (engine.py) has the sources' methods; this one has the turbofan's
state and the sources a turboshaft doesn't have: the fan's sound out of the nozzle and the jet in flight.

The state follows from the net thrust alone, through the published nozzle area: the mixed jet leaves fully expanded,
so Fn = rho_j A V_j (V_j - V_0) gives its velocity and the air flow; the fuel's heat sets its temperature; its total
pressure is the fan's delivery, which gives the fan's pressure ratio and temperature rise. The core takes the flow's
share by the bypass ratio; its turbine passes it as a choked nozzle does (W sqrt(T4) / P3 constant, from the
reference row), which with the fuel's heat gives its pressure ratio and turbine entry temperature. The turbines
drive the compressor and the fan. Fuel: the reference row's specific consumption, growing with the flight Mach number
and the air's temperature as engines' do (an estimate; it sets the jet's temperature and the combustion's sound, not
the thrust).

Sources (dB re 20 uPa at 1 m, by the angle from the inlet's axis):
- the fan from the inlet: Heidmann's method (NASA TM X-71763) in the small-engine revision engine.py has: broadband,
  rotor-stator tones, and with a supersonic relative tip speed the buzz-saw at the shaft's orders;
- the fan from the nozzle: Heidmann's discharge broadband and tones, with ANOPP's directivities (NASA TM-83199 part
  2, table III, read from the manual). Their levels (58 and 60 dB) are the small-engine revision's as remembered:
  no copy of it was at hand, so they stand to be corrected by recordings;
- the compressor's first stage through the fan, from the inlet;
- combustion (SAE ARP 876 with GE's turbine loss), the low pressure turbine's last stage (GE's method), and the
  jet's mixing noise (ARP 876's power), in flight less by the relative velocity, ((V_j - V_0) / V_j)^5.5.
Heidmann's levels are for sea level air: at altitude a source's power is the same for the same flow and temperature
rise, and its pressure at 1 m goes with the air's rho c.

Not modelled: the intake's and the bypass duct's liners (none assumed), the mixer, the engine's structure-borne
path except as the jets preset states it (an assumed level), a windmilling or starting engine, reverse thrust.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import tomllib

import numpy as np

from . import engine as en
from .atmosphere import Atmosphere, GAMMA, R_AIR
from .engine import BANDS, CP_AIR, CP_GAS, P_REF, Source
from .model import PRESET_DIR, Cabin, Observer
from .synth import Render, write_wav

LBF = 4.4482216
FT = 0.3048
RHO_C_0 = 1.225 * 340.294

# the directivities of the fan's discharge, dB below their peak, every 10 deg from the inlet's axis (ANOPP table III:
# 10 (log10 D - its maximum))
_F3DB = 10.0 * (np.array([-3.70, -3.27, -2.84, -2.41, -1.98, -1.55, -1.12, -0.69, -0.34, -0.04, 0.19, 0.34, 0.43, 0.46,
                          0.26, -0.14, -0.54, -1.04, -1.54]) - 0.46)
_F3DT = 10.0 * (np.array([-3.45, -3.05, -2.65, -2.25, -1.85, -1.45, -1.05, -0.65, -0.35, -0.05, 0.15, 0.35, 0.45, 0.45,
                          0.25, -0.10, -0.45, -0.85, -1.35]) - 0.45)
DISCHARGE_BROADBAND_DB = 58.0        # Heidmann's 60, less the small-engine revision's 2 (see the module's text)
DISCHARGE_TONE_DB = 60.0             # Heidmann's 63, less 3

for _k, _v in (("fan", 0.001), ("fanaft", 0.001), ("jet", 0.0), ("structure", 0.001), ("engine", 0.001)):
    en.SPEED_WANDER.setdefault(_k, _v)


def _scaled(src, g):
    """a Source with its power times g"""
    out = Source(bands=src.bands * g, tones=[(f, p * g) for f, p in src.tones])
    out.ripple = src.ripple
    return out


@dataclass
class Turbofan:
    name: str
    n1_rpm: float
    n2_rpm: float
    nozzle_area: float
    bypass_ratio: float
    reference_thrust_n: float
    reference_opr: float
    reference_t4_k: float
    fan: dict
    compressor: dict
    lp_turbine: dict
    reference_n1_pct: float = 100.0
    fan_efficiency: float = 0.88
    compressor_efficiency: float = 0.84
    turbine_efficiency: float = 0.89
    burner_efficiency: float = 0.99
    fuel_lhv: float = 43.2e6
    reference_sfc: float = 0.0           # kg/(N s) at the reference row; 0: from the reference turbine entry temperature
    duct_pressure_recovery: float = 0.95 # the mixed jet's total pressure over the fan's delivery: the bypass duct's and the
                                         # mixer's losses (assumed; without them the reference row's fan pressure ratio
                                         # came out 1.60 for the cycle's 1.76)

    def __post_init__(self):
        self._ref = None
        self._k_turbine = None
        if self.reference_sfc <= 0:
            # the fuel that heats the core's air to the reference turbine entry temperature at the reference pressure ratio
            atm = Atmosphere(0.0)
            W = self._nozzle(self.reference_thrust_n, 0.0, atm, 360.0)[1]
            wc = W / (1.0 + self.bypass_ratio)
            T3 = atm.T * (1.0 + (self.reference_opr ** (1.0 / 3.5) - 1.0) / self.compressor_efficiency)
            wf = wc * CP_GAS * (self.reference_t4_k - T3) / (self.burner_efficiency * self.fuel_lhv)
            self.reference_sfc = wf / self.reference_thrust_n
        r = turbofan_state(self, self.reference_thrust_n, self.reference_n1_pct, _reference=True)
        self._k_turbine = r.Wc * np.sqrt(self.reference_t4_k) / (self.reference_opr * 101325.0)
        self._ref = turbofan_state(self, self.reference_thrust_n, self.reference_n1_pct)

    def _nozzle(self, Fn, V0, atm, Tj):
        """the fully expanded mixed jet's velocity and the air flow for the net thrust: Fn = rho A Vj (Vj - V0)"""
        rho = atm.p / (R_AIR * Tj)
        Vj = 0.5 * (V0 + np.sqrt(V0 * V0 + 4.0 * max(Fn, 0.0) / (rho * self.nozzle_area)))
        return Vj, rho * self.nozzle_area * Vj, rho


@dataclass
class TurbofanState:
    n1: float                    # fan speed, % of n1_rpm
    n2: float                    # high pressure rotor, % of n2_rpm
    thrust_n: float
    mach: float
    W: float                     # air flow, kg/s
    Wc: float                    # the core's
    fuel_kg_s: float
    fpr: float
    dT_fan: float                # the fan's temperature rise, K
    opr: float
    T3: float
    T4: float
    T45: float                   # low pressure turbine inlet
    T5: float
    v_jet: float
    rho_jet: float
    T_jet: float
    m_tip: float                 # the fan's tip Mach number (tangential)
    m_rel: float                 # relative (helical)
    lit: bool = True

    def summary(self):
        return (f"N1 {self.n1:5.1f} % N2 {self.n2:5.1f} %: thrust {self.thrust_n / LBF:6.0f} lbf at Mach {self.mach:.2f}, air "
                f"{self.W:6.1f} kg/s, fan pressure ratio {self.fpr:.3f}, overall {self.opr:5.1f}, T4 {self.T4:5.0f} K, jet "
                f"{self.v_jet:4.0f} m/s at {self.T_jet:4.0f} K, fan tip Mach {self.m_tip:.2f} (relative {self.m_rel:.2f})")


def turbofan_state(e, thrust_n, n1_pct, mach=0.0, atm=None, _reference=False):
    """The engine's state at a net thrust (N) and fan speed (% of n1_rpm), flying at a Mach number in the air atm."""
    atm = atm or Atmosphere(0.0)
    V0 = mach * atm.c
    Tt0 = atm.T * (1.0 + 0.2 * mach * mach)
    Pt0 = 0.98 * atm.p * (1.0 + 0.2 * mach * mach) ** 3.5          # the intake's recovery
    sfc = e.reference_sfc * np.sqrt(atm.T / 288.15) * (1.0 + 1.3 * mach)
    wf = sfc * max(thrust_n, 0.0)
    Tj = 360.0
    for _ in range(30):
        Vj, W, rho = e._nozzle(thrust_n, V0, atm, Tj)
        Tt = Tt0 + e.burner_efficiency * wf * e.fuel_lhv / (max(W, 1e-6) * CP_AIR)
        new = max(Tt - Vj * Vj / (2.0 * CP_AIR), 0.6 * atm.T)
        if abs(new - Tj) < 0.01:
            break
        Tj = new
    fpr = max((atm.p / Pt0) * (1.0 + Vj * Vj / (2.0 * CP_AIR * Tj)) ** 3.5 / e.duct_pressure_recovery, 1.0)
    dT_fan = Tt0 * (fpr ** (1.0 / 3.5) - 1.0) / e.fan_efficiency
    wc = W / (1.0 + e.bypass_ratio)
    T2c = Tt0 + dT_fan                                              # the core behind the fan's root
    if _reference or e._k_turbine is None:
        opr, T3 = e.reference_opr, atm.T * (1.0 + (e.reference_opr ** (1.0 / 3.5) - 1.0) / e.compressor_efficiency)
        T4 = e.reference_t4_k
    else:
        opr = e.reference_opr * min(wc / max(e._ref.Wc, 1e-6), 1.5) if e._ref else e.reference_opr
        for _ in range(60):
            T3 = Tt0 * (1.0 + (opr ** (1.0 / 3.5) - 1.0) / e.compressor_efficiency)
            T4 = T3 + e.burner_efficiency * wf * e.fuel_lhv / (CP_GAS * max(wc, 1e-6))
            new = max(wc * np.sqrt(T4) / (e._k_turbine * Pt0), 1.0)     # the choked turbine's delivery pressure over the intake's
            if abs(new - opr) < 1e-4:
                break
            opr = 0.5 * (opr + new)
    dT_lpt = CP_AIR * dT_fan * W / (CP_GAS * max(wc, 1e-6))
    T5 = T4 - CP_AIR * (T3 - T2c) / CP_GAS - dT_lpt
    T45 = T5 + dT_lpt
    n2 = 100.0
    if e._ref is not None:
        n2 = 100.0 * np.sqrt(max(T3 - T2c, 0.0) / max(e._ref.T3 - (288.15 + e._ref.dT_fan), 1.0))
    f = e.fan
    rt = 0.5 * f["tip_diameter"]
    m_t = e.n1_rpm * n1_pct / 100.0 * 2.0 * np.pi / 60.0 * rt / atm.c
    rho_t = Pt0 / (R_AIR * Tt0)
    m_x = W / (rho_t * np.pi * rt ** 2 * (1.0 - f["hub_tip_ratio"] ** 2) * atm.c)
    return TurbofanState(n1_pct, n2, thrust_n, mach, W, wc, wf, fpr, dT_fan, opr, T3, T4, T45, T5, Vj, rho, Tj, m_t,
                         float(np.hypot(m_t, m_x)))


def fan_discharge(e, s, theta, atm=None, harmonics=6):
    """the fan's sound out of the nozzle at theta (deg from the inlet's axis), 1 m: Heidmann's discharge broadband
    and rotor-stator tones"""
    atm = atm or Atmosphere(0.0)
    out = Source()
    if s.dT_fan <= 0 or s.W <= 0 or s.n1 <= 0:
        return out
    c = e.fan
    B, V = c["rotor_blades"], c["stator_vanes"]
    bpf = B * e.n1_rpm * s.n1 / 100.0 / 60.0
    md = 20.0 * np.log10(max(e._ref.m_rel if e._ref else s.m_rel, 1.0))
    over = 20.0 * np.log10(max(s.m_rel, 1.0))
    base = 10.0 * np.log10((s.dT_fan * 1.8) ** 2 * s.W / en.LB) + 10.0 * np.log10(atm.rho * atm.c / RHO_C_0)
    rss = min(c["rotor_stator_spacing"], 300.0)
    igv = c.get("inlet_guide_vanes", False)
    bb = base + DISCHARGE_BROADBAND_DB + md - over - 5.0 * np.log10(rss / 300.0) + (3.0 if igv else 0.0) \
        + np.interp(theta, en._TH, _F3DB)
    r = BANDS / bpf / 2.0
    flog = np.where(r < 1.0, 2.445096095, 13.97197769) * np.log(r) ** 2      # the inlet broadband's spectrum, as engine.py's
    out.bands = P_REF ** 2 * 10.0 ** (0.1 * (bb - flog))
    tl = base + DISCHARGE_TONE_DB + md - over - 10.0 * np.log10(rss / 300.0) + (6.0 if igv else 0.0) \
        + np.interp(theta, en._TH, _F3DT)
    cut = abs(s.m_tip / (1.0 - V / B)) < 1.05 and s.m_tip < 1.0
    for h in range(1, harmonics + 1):
        fall = (8.0 if cut else 0.0) if h == 1 else (3.0 * (h + 1) if igv else (9.2 if h == 2 else 3.0 * h + 1.8))
        out.tones.append((h * bpf, P_REF ** 2 * 10.0 ** (0.1 * (tl - fall))))
    return out


def fan_inlet(e, s, theta, atm=None, harmonics=6, buzz_db=0.0, broadband_db=0.0):
    """the fan's sound out of the intake (engine.py's Heidmann: broadband, tones, buzz-saw), with the compressor's
    first stage behind it"""
    atm = atm or Atmosphere(0.0)
    if s.dT_fan <= 0 or s.W <= 0 or s.n1 <= 0:
        return Source()
    g = atm.rho * atm.c / RHO_C_0
    f1 = e.n1_rpm * s.n1 / 100.0 / 60.0
    out = en._heidmann(e.fan, s.dT_fan, s.W, s.m_tip, s.m_rel, e._ref.m_rel if e._ref else s.m_rel, f1, theta, harmonics)
    # an installation's measured corrections: the buzz-saw's lines (the shaft's orders that are not the blades') and
    # the broadband, each against the blade passing tones
    B = e.fan["rotor_blades"]
    kb, kn = 10.0 ** (0.1 * buzz_db), 10.0 ** (0.1 * broadband_db)
    out.tones = [(f, m * (1.0 if abs(f / f1 / B - round(f / f1 / B)) < 1e-6 else kb)) for f, m in out.tones]
    out.bands = out.bands * kn
    c = e.compressor
    k = 1.0 / 3.5
    dT1 = 288.15 * (c["stage_pressure_ratio"] ** k - 1.0) / c["stage_efficiency"] * (s.n2 / 100.0) ** 2
    rt = 0.5 * c["tip_diameter"]
    T2c = atm.T + s.dT_fan
    cc = np.sqrt(GAMMA * R_AIR * T2c)
    m_t = e.n2_rpm * s.n2 / 100.0 * 2.0 * np.pi / 60.0 * rt / cc
    rho = atm.rho * s.fpr ** (1.0 / GAMMA)
    m_x = s.Wc / (rho * np.pi * rt ** 2 * (1.0 - c["hub_tip_ratio"] ** 2) * cc)
    m_d = e.n2_rpm * 2.0 * np.pi / 60.0 * rt / 340.0
    two = en._heidmann(c, dT1, s.Wc, m_t, float(np.hypot(m_t, m_x)), max(m_d, 1.0), e.n2_rpm * s.n2 / 100.0 / 60.0, theta,
                       3, False)
    gc = 10.0 ** (0.1 * c.get("through_fan_db", -10.0))
    out.bands = (out.bands + two.bands * gc) * g
    out.tones = [(f, m * g) for f, m in out.tones] + [(f, m * gc * g) for f, m in two.tones]
    return out


def core_noise(e, s, theta, atm=None):
    """combustion through the turbines (ARP 876, as engine.combustion_noise) and the low pressure turbine's last stage"""
    atm = atm or Atmosphere(0.0)
    comb, turb = Source(), Source()
    if not s.lit or s.fuel_kg_s <= 0:
        return comb, turb
    r = e._ref
    g = ((r.T4 - r.T5) / 288.15) ** -4
    power = 8.85e-7 * s.Wc * atm.c ** 2 * ((s.T4 - s.T3) / s.T3) ** 2 * (s.opr * (1.0 + 0.2 * s.mach ** 2) ** 3.5) ** 2 * g
    D = 10.0 ** np.interp(theta, en._TH, en._D_COMB)
    S = 10.0 ** np.interp(np.log10(BANDS / 400.0), *en._S_COMB, left=-9.0, right=-9.0)
    comb.bands = atm.rho * atm.c * power * D * S / (4.0 * np.pi)
    t = e.lp_turbine
    n = e.n1_rpm * s.n1 / 100.0 / 60.0
    rows = t["rotor_blades"] if isinstance(t["rotor_blades"], (list, tuple)) else [t["rotor_blades"]]
    # GE's method at the reference row's work and tip speed, carried to this state by Smith and Bushell's law (ANOPP
    # section 8.9: the power as the tip Mach number cubed times the flow), as engine.py carries a power turbine off
    # its governed speed: GE's correlation holds the stage's work and grows without bound as the tip speed falls
    # (at idle its tones came out 122 dB(A) at 50 m, over the whole engine at take-off)
    k = (s.n1 / r.n1) ** 3 * s.Wc / r.Wc
    n_ref = e.n1_rpm * r.n1 / 100.0 / 60.0
    for i, b in enumerate(rows):
        one = en._turbine(max(r.T45 - r.T5, 0.0) / r.T45, np.pi * t["tip_diameter"] * n_ref, t["inlet_area"], b * n, theta, atm)
        one = _scaled(one, k)
        if i:
            one.bands = 0.0 * one.bands                # the broadband once; each row its tones
        turb += one
    return comb, turb


# A subsonic jet's mixing noise by direction, dB under its loudest, every 10 deg from the inlet's axis (180: straight
# down the jet): loudest 30 to 40 deg off the jet's axis, about 10 dB less ahead of the engine, falling again inside
# the cone of silence on the axis. The shape SAE ARP 876 and the jet-noise literature give for jets near Mach 0.8, AS
# REMEMBERED: no table was at hand, so the figures are good to a few dB. engine.py's jet has no directivity (a
# turboshaft's slow exhaust); a turbofan's is heard to.
_D_JET = np.array([-11.0, -11.0, -10.5, -10.0, -9.5, -9.0, -8.5, -8.0, -7.0, -6.0, -5.0, -3.5, -2.0, -0.7, 0.0, 0.0, -2.0, -6.0, -11.0])
_D_JET_MEAN = float(np.sum(10.0 ** (0.1 * _D_JET) * np.sin(np.radians(np.clip(en._TH, 2.0, 178.0)))) / np.sum(np.sin(np.radians(np.clip(en._TH, 2.0, 178.0)))))


def jet_mixing(e, s, theta=None, atm=None):
    """the mixed jet's noise at theta (deg from the inlet's axis; None: the mean over all directions), in flight less
    by its velocity relative to the air. The directivity keeps the sound power engine.jet_noise gives"""
    atm = atm or Atmosphere(0.0)
    out = en.jet_noise(s.v_jet, s.rho_jet, e.nozzle_area, atm)
    rel = max(s.v_jet - s.mach * atm.c, 0.0) / max(s.v_jet, 1.0)
    d = 1.0 if theta is None else 10.0 ** (0.1 * float(np.interp(theta, en._TH, _D_JET))) / _D_JET_MEAN
    out.bands = out.bands * rel ** 5.5 * d
    return out


# ---------------------------------------------------------------- the aircraft

@dataclass
class Jet:
    name: str
    cockpit: tuple
    cabin: Cabin
    engine: Turbofan
    inlet: tuple
    exhaust: tuple
    idle_rpm: float
    thrust_rpm_span: float
    rated_thrust_n: float
    fan_rpm_property: str
    running_property: str = None         # 1 while the engine is to be heard (None: whenever the fan turns fast enough)
    height_m: float = 1.6
    idle_thrust_fraction: float = 0.07   # the flight model's idle, of the maximum thrust
    inside_db: float = 0.0               # what the airframe between the engine and the inside listener takes of its airborne sound
    thrust_property: str = None
    # [[altitude ft, Mach, maximum net thrust N], ...]: the flight conditions the bank is made at, the first on the ground
    conditions: list = field(default_factory=lambda: [[0.0, 0.0, 0.0]])
    # the engine through the airframe to the inside listener: its shafts' once-a-turn tones and a broadband rumble,
    # dB re 20 uPa at the listener at 100 % fan speed, their amplitude with the speed squared (an unbalance's force).
    # Assumed where no measurement exists: it is what is left to hear inside when the air is too thin to carry sound
    structure_tone_db: float = None
    structure_band_db: float = None
    fan_measured: dict = field(default_factory=dict)     # buzz_db, broadband_db: see fan_inlet
    inside_property: str = None          # true where the inside loops are to play (None: FlightGear's internal views)
    # for the other banks (wind.py, tyre.py, gear.py take the same aircraft): the wall the wind drives, m from the nose
    windscreen_m: float = None
    fuselage: list = None
    glazing: list = None

    def thrust_at(self, rpm, max_thrust_n):
        """the net thrust the flight model has at a fan speed: its speed goes with the square root of the thrust's
        share of the maximum"""
        return max_thrust_n * (np.clip((rpm - self.idle_rpm) / self.thrust_rpm_span, 0.0, 1.0)) ** 2


def load_turbofan(name_or_path):
    p = Path(name_or_path)
    if not p.suffix:
        p = PRESET_DIR / "turbofans" / f"{name_or_path}.toml"
    with open(p, "rb") as f:
        d = tomllib.load(f)
    return Turbofan(**d["engine"], fan=d["fan"], compressor=d["compressor"], lp_turbine=d["lp_turbine"])


def load_jet(name_or_path):
    p = Path(name_or_path)
    if not p.suffix:
        p = PRESET_DIR / "jets" / f"{name_or_path}.toml"
    with open(p, "rb") as f:
        d = tomllib.load(f)
    e = dict(d["engine"][0])
    a = d["aircraft"]
    e.pop("name", None)
    return Jet(name=a["name"], cockpit=tuple(a["cockpit"]), height_m=a.get("height_m", 1.6), cabin=Cabin(**d.get("cabin", {})),
               engine=load_turbofan(e.pop("type")), inlet=tuple(e.pop("inlet")), exhaust=tuple(e.pop("exhaust")),
               conditions=a.get("conditions", [[0.0, 0.0, 0.0]]), structure_tone_db=a.get("structure_tone_db"),
               inside_db=a.get("inside_db", 0.0), fan_measured=d.get("fan_measured", {}), inside_property=a.get("inside_property"), windscreen_m=a.get("windscreen_m"),
               structure_band_db=a.get("structure_band_db"), **e)


# Outside, each source is one loop, made where it is loudest (deg from the nose), and the simulator's own sound cone
# gives it its direction: (full cone angle inside which it plays in full, full angle beyond which it plays at the outer
# gain, that gain), about the source's axis: the intake's forward, the nozzle's aft. Fitted by eye to the methods'
# directivities: Heidmann's inlet tones and broadband (full to 40 deg off the axis, 25 dB down at 110), his discharge
# (within 3 dB from 100 to 150 deg off the nose, 35 dB down ahead), the jet's (10 dB down ahead). A cone is one gain for
# the whole spectrum and has no dip on its axis, so the nozzle's on-axis fall is not there.
OUTSIDE = dict(fan=dict(at=30.0, axis="forward", cone=(80.0, 220.0, 0.056)),
               fanaft=dict(at=125.0, axis="aft", cone=(160.0, 360.0, 0.018)),
               core=dict(at=145.0, axis="aft", cone=(120.0, 360.0, 0.32)))
LAYERS = ("fan", "fanaft", "core", "structure")


def jet_sources(jet, s, X, atm, inside):
    """{layer: (position, Source at 1 m)} at the listener X (aircraft frame): the fan from the intake (with the
    compressor), the fan and the turbine from the nozzle, the combustion and the jet"""
    e = jet.engine
    inlet, exh = np.asarray(jet.inlet, float), np.asarray(jet.exhaust, float)
    fwd = np.array([1.0, 0.0, 0.0])
    th_in = en._deg(fwd, X - inlet)
    th_ex = en._deg(fwd, X - exh)
    aft = fan_discharge(e, s, th_ex, atm)
    comb, turb = core_noise(e, s, th_ex, atm)
    aft += turb
    core = comb
    core += jet_mixing(e, s, th_ex, atm)
    return dict(fan=(inlet, fan_inlet(e, s, th_in, atm, **jet.fan_measured)), fanaft=(exh, aft), core=(exh, core))


def structure_source(jet, s):
    """the engine through the airframe, at the inside listener (see Jet.structure_tone_db)"""
    out = Source()
    if jet.structure_tone_db is None:
        return out
    e = jet.engine
    k = (s.n1 / 100.0) ** 4                                             # amplitude with the speed squared
    f1, f2 = e.n1_rpm * s.n1 / 100.0 / 60.0, e.n2_rpm * s.n2 / 100.0 / 60.0
    p2 = P_REF ** 2 * 10.0 ** (0.1 * jet.structure_tone_db) * k
    # each shaft's once-a-turn tone: what a Citation X's cockpit has of its two AE 3007s in the climb (about 120 and
    # 250 Hz, each 10 to 16 dB over the floor: refs/recordings/citationx_cockpit_takeoff)
    out.tones = [(f1, p2), (f2, P_REF ** 2 * 10.0 ** (0.1 * jet.structure_tone_db) * (s.n2 / 100.0) ** 4)]
    if jet.structure_band_db is not None:
        lg = np.log2(BANDS / 250.0)
        out.bands = P_REF ** 2 * 10.0 ** (0.1 * (jet.structure_band_db - 6.0 * np.abs(lg))) * k
    return out


def render_jet(jet, n1_pct, altitude_ft=0.0, mach=0.0, max_thrust_n=None, observer=None, fs=48000.0, duration=8.0, seed=1,
               pitch_headroom=1.3, f_highpass=25.0, merge=True):
    """A seamless loop of the engine at a fan speed, heard riding with the aircraft: outside (an Observer by azimuth
    and distance, level with the engine in flight, standing on the ground at altitude 0) or inside (an Observer with
    a position). Returns a Render with the layers."""
    atm = Atmosphere(altitude_ft * FT)
    e = jet.engine
    rpm = e.n1_rpm * n1_pct / 100.0
    s = turbofan_state(e, jet.thrust_at(rpm, max_thrust_n or jet.rated_thrust_n), n1_pct, mach, atm)
    observer = observer or Observer("side", 90.0, 0.0, 50.0)
    inside = observer.position is not None
    on_ground = altitude_ft <= 0.0
    if inside:
        X = np.asarray(observer.position, float)
    else:
        az = np.radians(observer.azimuth_deg)
        X = np.array([jet.inlet[0] + observer.distance_m * np.cos(az), -observer.distance_m * np.sin(az),
                      1.7 if on_ground else jet.inlet[2]])
    srcs = jet_sources(jet, s, X, atm, inside)
    if inside:
        # the thin air carries less to the wall and the wall passes less of it: the mass law's transmission goes with
        # (rho c)^2, over the sea level air the cabin's model is for
        g = (atm.rho * atm.c / RHO_C_0) ** 2 * 10.0 ** (0.1 * jet.inside_db)
        srcs = {k: (pos, _scaled(src, g)) for k, (pos, src) in srcs.items()}
        st = structure_source(jet, s)
        if st.total() > 0:
            srcs["structure"] = (None, st)
    sig, lv = en._render_sources(srcs, observer, X, fs, duration, seed, "open" if on_ground else "free", jet.cabin,
                                 pitch_headroom, f_highpass, atm)
    # the fan's two ways out and the turbine play as one layer (all pitched by the fan's speed)
    if merge and "layer/fanaft" in sig:
        sig["layer/fan"] = sig.get("layer/fan", 0.0) + sig.pop("layer/fanaft")
        lv["fan"] = round(float(10.0 * np.log10(10.0 ** (0.1 * lv.get("fan", -200.0)) + 10.0 ** (0.1 * lv.pop("fanaft")))), 1)
    sig = {k: v for k, v in sig.items() if k != "total"}
    meta = dict(engine=e.name, n1=n1_pct, n2=round(s.n2, 1), altitude_ft=altitude_ft, mach=mach, thrust_lbf=round(s.thrust_n / LBF),
                source_db_1m=lv, observer=observer.name, state=s.summary())
    return Render(fs, sig, [], meta)


# ---------------------------------------------------------------- a bank and its FlightGear section

GRID = (33.0, 42.0, 52.0, 62.0, 72.0, 82.0, 90.0)   # fan speed, %, between the idle's and the maximum thrust's


def _grid(jet, k):
    """the fan speeds a condition's loops are made at: the flight model's idle and up (its idle thrust is the same
    share of the maximum at every altitude, so its fan speed is too)"""
    at = lambda tf: round(float(100.0 * (jet.idle_rpm + jet.thrust_rpm_span * np.sqrt(tf)) / jet.engine.n1_rpm), 1)
    idle, top = at(jet.idle_thrust_fraction), at(1.0)
    g = tuple(sorted(set((idle, top) + tuple(n for n in GRID if idle + 3.0 < n < top - 3.0))))
    return g if k == 0 else g[::2] if len(g) % 2 else g[::2] + (g[-1],)      # in the air every other one


def make_jet_bank(jet_name, out_dir, distance=50.0, fs=48000.0, duration=15.0, bits=16, peak_dbfs=-1.0, seed=1, progress=print):
    """Loops of the jet's engine over its fan speed at each of its flight conditions (the ground, and altitudes at
    a cruise Mach number), inside and outside from three directions, with manifest.json and pywopwop-jet.xml. Layers:
    fan (from the intake and the nozzle, with the turbine: pitched by the fan's speed), core (combustion and the jet's
    mixing: not pitched), structure (inside: the engine through the airframe)."""
    jet = load_jet(jet_name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inside = Observer("inside", position=tuple(jet.cockpit))
    renders = []
    for k, (alt, mach, fmax) in enumerate(jet.conditions):
        grid = _grid(jet, k)
        for j, n in enumerate(grid):
            ratio = max(grid[j + 1] / n if j + 1 < len(grid) else 1.0, 1.3)
            made = [(inside, render_jet(jet, n, alt, mach, fmax or jet.rated_thrust_n, inside, fs, duration, seed + 100 * k + 7 * j, ratio))]
            for layer, d in OUTSIDE.items():
                o = Observer("outside", d["at"], 0.0, distance)
                r = render_jet(jet, n, alt, mach, fmax or jet.rated_thrust_n, o, fs, duration, seed + 100 * k + 7 * j, ratio, merge=False)
                r.signals = {x: v for x, v in r.signals.items() if x == "layer/" + layer}       # each source where it is loudest
                made.append((o, r))
            for o, r in made:
                renders.append((k, n, o, r))
                lv = r.levels()
                progress(f"  {alt:6.0f} ft N1 {n:5.1f} {o.name:6s} " + "  ".join(f"{x.split('/')[-1]} {v['dba']:5.1f}" for x, v in lv.items())
                         + " dBA")
    group = lambda o: "inside" if o.position is not None else "outside"
    gains = {}
    for g in ("inside", "outside"):
        peak = max(np.max(np.abs(x)) for *_, o, r in renders if group(o) == g for x in r.signals.values())
        want = peak_dbfs[g] if isinstance(peak_dbfs, dict) else peak_dbfs      # one level, or {inside, outside}
        gains[g] = 10.0 ** (want / 20.0) / peak
    files = []
    for k, n, o, r in renders:
        lv = r.levels()
        for x, sig in r.signals.items():
            if lv[x]["oaspl"] < -20.0:
                continue
            layer = x.split("/")[-1]
            fn = f"engine_{layer}_{int(jet.conditions[k][0])}ft_{n:g}_{o.name}.wav"
            write_wav(out / fn, sig, r.fs, bits, gains[group(o)], seed=abs(hash(fn)) % 2 ** 32)
            files.append(dict(file=fn, layer=layer, condition=k, n1=n, observer=o.name, oaspl_db=round(lv[x]["oaspl"], 2),
                              dba=round(lv[x]["dba"], 2), state=r.meta["state"]))
    manifest = dict(aircraft=jet.name, generator="pywopwop turbofan", engine=jet.engine.name, sample_rate=fs, bits=bits,
                    gain_db={g: round(20.0 * np.log10(v), 3) for g, v in gains.items()}, exterior_distance_m=distance,
                    conditions=[dict(altitude_ft=a, mach=m, max_thrust_lbf=round((f or jet.rated_thrust_n) / LBF),
                                     grid=list(_grid(jet, k))) for k, (a, m, f) in enumerate(jet.conditions)],
                    outside={k: dict(d, position=list(jet.inlet if k == "fan" else jet.exhaust)) for k, d in OUTSIDE.items()},
                    n1_rpm=jet.engine.n1_rpm,
                    fan_rpm_property=jet.fan_rpm_property, running_property=jet.running_property, inside_property=jet.inside_property,
                    note="levels: dB re 20 uPa at the listener before the gain; outside: riding with the aircraft at "
                         f"{distance:g} m, on the ground at altitude 0, each source where it is loudest; its direction is the "
                         "simulator's sound cone", files=files)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "pywopwop-jet.xml").write_text(flightgear_jet_xml(manifest))
    return manifest


def flightgear_jet_xml(manifest, sound_dir="Sounds/pywopwop", inside_property=None,
                       altitude_property="/position/altitude-ft", density_property="/environment/density-slugft3"):
    """FlightGear <fx> entries: each layer's loops cross-faded by equal power over the fan's speed and between the flight
    conditions over the altitude. Outside, each source plays from its own place (the intake, the nozzle) with a sound
    cone about its axis (OUTSIDE), so its loudness follows where the aircraft is heard from. The fan's and the structure's loops are pitched
    by the fan's speed over their own; the core's are not. Above the highest condition the loops fade with the air's
    density over that condition's (a source's pressure goes with rho c). Inside loops play in internal views only,
    outside loops in the others. A loop's condition is true only inside its fades."""
    from .wind import _e, _v, _ramp_e
    have = {(f["layer"], f["condition"], f["n1"], f["observer"]): f for f in manifest["files"]}
    rpm = f"<property>{manifest['fan_rpm_property']}</property>"
    run = f"<property>{manifest['running_property']}</property>" if manifest.get("running_property") else None
    alt = f"<property>{altitude_property}</property>"
    # where the inside loops play: FlightGear's internal views, or a property of the aircraft's own (an aircraft
    # whose inside listener is not a view FlightGear calls internal: a camera on an unmanned one)
    ins = inside_property or manifest.get("inside_property") or "sim/current-view/internal"
    conds = manifest["conditions"]
    cones = manifest["outside"]
    dist = manifest["exterior_distance_m"]
    out = []

    def fade(x, pts, i, lo0=None, hi0=None):
        """the i-th of pts on x: weight expressions (ramps from its neighbours) and its condition's bounds"""
        lo = pts[i - 1] if i else lo0
        hi = pts[i + 1] if i + 1 < len(pts) else hi0
        w, cond = [], []
        if lo is not None:
            w.append(_ramp_e(x, lo, pts[i]))
            cond.append(f"<greater-than>{x}<value>{lo:g}</value></greater-than>")
        if hi is not None:
            w.append(_ramp_e(x, hi, pts[i]))
            cond.append(f"<less-than>{x}<value>{hi:g}</value></less-than>")
        return w, cond, lo, hi

    alts = [c["altitude_ft"] for c in conds]
    for k, c in enumerate(conds):
        wa, ca, _, _ = fade(alt, alts, k)
        extra = []
        if k == len(conds) - 1 and k > 0:
            from .atmosphere import Atmosphere as _A
            rho_top = _A(c["altitude_ft"] * FT).rho / 515.378818           # slug/ft^3
            extra = [_e("clip", _e("div", f"<property>{density_property}</property>", _v(rho_top)),
                        "<clipMin>0</clipMin><clipMax>1</clipMax>")]
        for view in ("inside", "outside"):
            inside = view == "inside"
            wd, cd = [], []
            for layer in ("fan", "fanaft", "core", "structure"):
                pts = [n for n in c["grid"] if (layer, k, n, view) in have]
                for i, n in enumerate(pts):
                    f = have[(layer, k, n, view)]
                    # below the idle's loop it plays on, slower and fading, down to 0.3 of the idle: a start's
                    # spool-up and a shutdown's run-down, as far as loops of a running engine can stand in for them
                    # by the fan's speed in rpm, the property itself: a FlightGear condition compares properties
                    # and values, it takes no expression (with one, no loop ever played)
                    k_rpm = manifest["n1_rpm"] / 100.0
                    wn, cn, lo, hi = fade(rpm, [q * k_rpm for q in pts], i, lo0=0.3 * pts[0] * k_rpm)
                    lo, hi = (lo / k_rpm if lo else lo), (hi / k_rpm if hi else hi)
                    ws = wn + wa + wd
                    vol = _e("product", _e("sqrt", _e("product", *ws) if len(ws) > 1 else ws[0]), *([run] if run else []), *extra)
                    cond = [(f"<property>{ins}</property>" if inside else f"<not><property>{ins}</property></not>")] \
                        + ([f"<greater-than>{run}<value>0.5</value></greater-than>"] if run else []) + cn + ca + cd
                    pt = ""
                    if layer != "core":                      # the fan's and the structure's follow its speed
                        pt = (f"\t\t\t<pitch>{rpm}<factor>{100.0 / manifest['n1_rpm'] / n:.6g}</factor><min>{(lo or 0.3 * n) / n:.4g}</min>"
                              f"<max>{(hi or 1.3 * n) / n:.4g}</max><offset>0</offset></pitch>\n")
                    place = ""
                    if not inside:
                        # FlightGear's sound frame: x aft, y left, z up, m; the cone's angles are whole angles
                        cn = cones[layer]
                        px, py, pz = cn["position"]
                        ax = -1.0 if cn["axis"] == "forward" else 1.0
                        place = (f"\t\t\t<position><x>{-px:.2f}</x><y>{py:.2f}</y><z>{pz:.2f}</z></position>\n"
                                 f"\t\t\t<orientation><x>{ax:g}</x><y>0</y><z>0</z><inner-angle>{cn['cone'][0]:g}</inner-angle>"
                                 f"<outer-angle>{cn['cone'][1]:g}</outer-angle><outer-gain>{cn['cone'][2]:g}</outer-gain></orientation>\n")
                    out.append(f"\t\t<engine>\n\t\t\t<name>pww-jet-{layer}-{int(c['altitude_ft'])}-{n:g}-{view}</name>\n"
                               f"\t\t\t<mode>looped</mode>\n\t\t\t<path>{sound_dir}/{f['file']}</path>\n"
                               f"\t\t\t<condition><and>{''.join(cond)}</and></condition>\n"
                               f"\t\t\t<volume><expression>{vol}</expression></volume>\n" + pt + place
                               + ("" if inside else f"\t\t\t<reference-dist>{dist:g}</reference-dist>\n"
                                                    "\t\t\t<max-dist>5000</max-dist>\n")
                               + "\t\t</engine>\n")
    return ("<?xml version=\"1.0\"?>\n<!-- made by pywopwop (turbofan.py): the engine's loops -->\n<PropertyList>\n\t<fx>\n"
            + "".join(out) + "\t</fx>\n</PropertyList>\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="a jet's engine sound bank (presets/jets)")
    ap.add_argument("jet")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--state", action="store_true", help="print the engine's state at each loop and stop")
    ap.add_argument("--duration", type=float, default=15.0, help="loop length, s (shorter loops are heard to repeat)")
    ap.add_argument("--fs", type=float, default=48000.0)
    a = ap.parse_args(argv)
    if a.state:
        jet = load_jet(a.jet)
        for k, (alt, mach, fmax) in enumerate(jet.conditions):
            for n in _grid(jet, k):
                s = turbofan_state(jet.engine, jet.thrust_at(jet.engine.n1_rpm * n / 100.0, fmax or jet.rated_thrust_n), n, mach,
                                   Atmosphere(alt * FT))
                print(f"{alt:6.0f} ft  {s.summary()}")
        return
    m = make_jet_bank(a.jet, a.out, fs=a.fs, duration=a.duration)
    print(f"{len(m['files'])} loops in {a.out}")


if __name__ == "__main__":
    main()
