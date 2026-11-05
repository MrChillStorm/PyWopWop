"""Gas turbine engines: their sound from their thermodynamic state, for flight simulator loops.

The state (the cycle) follows the gas generator speed Ng along the engine's operating line. The compressor's
temperature rise grows as a power of Ng (its work coefficient grows as the variable stators open), fitted to the
engine's published delivery pressures. The fuel flow is the engine model's: steady, or below idle the start's
acceleration schedule. The turbines pass the flow as a nozzle does (Stodola's ellipse law over the whole expansion,
calibrated on the published power turbine inlet temperature at the design point), which with the combustor's heat
release sets the air flow and the turbine inlet temperature; where the compressor rather than the turbines limits
the flow (low speeds, small pressure ratios), it grows in proportion to Ng from the idle's. The gas generator turbine
drives the compressor, the power turbine gives the shaft power of the engine model's torque line, and the exhaust
leaves through its area. Without flame (the starter cranking, the engine spooling down) the same air flows cold.

The sources are those NASA's helicopter noise system ROTONET predicts a helicopter's engines with (ANOPP's fan,
combustion and turbine modules, NASA TM-83199 parts 2 and 4):
- the compressor's first stage, radiated forward from the inlet: Heidmann's method (NASA TM X-71763) in Hough and
  Weir's revision for small engines (NASA CR-198300): broadband, rotor-stator interaction tones (cut off at low
  speed), the inlet distortion's tones, and with a supersonic relative tip speed the buzz-saw's combination tones,
  here at the shaft's own orders (which they are);
- combustion noise, radiated aft through the turbines and the exhaust: Matta's (GE) method in SAE ARP 876, with GE's
  transmission loss through the turbines;
- the power turbine's last stage, from the exhaust: GE's method (Matta, Sandusky and Doyle, FAA-RD-77-4), where the
  turbine works at its governed speed (from the torque line's first power on);
- the exhaust jet's mixing noise: SAE ARP 876's acoustic power (Lighthill's V^8 with its density exponent) in a
  smooth spectrum peaking at a Strouhal number of 0.2;
- the air turbine starter while it cranks: its turbine's tones and broadband by GE's method at its design point.
  The bleed air reaches it choked, so its blades' loading, and so its sound power, holds as it speeds up; its
  frequencies follow its speed.

Tones are not pure: turbulence in the inlet (a rotor's downwash, the inlet's own distortion) and in the wakes scatters
them into narrow bands around their frequencies ("haystacks", a fraction of a percent wide), the shafts' speeds
wander by a tenth of a percent, and the combustion's roar swells and fades. All of it periodic over the loop.

Not modelled: the gas generator turbine's and the centrifugal stage's sound (the blade rows downstream and upstream
of them block it), the power turbine's while it runs the rotor up (GE's correlation doesn't reach so far off its
design), the igniters' sparks (no data on the exciter), the light-off itself (the heat release's monopole is a pulse
of about 10 Hz, below hearing), the APU (the flight model has none), the engine's structure-borne path, and a
tiltrotor's proprotor chopping the inlet's whine at its blade-passing frequency (a recording beside an MV-22B's
nacelle has it).
"""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import tomllib

import numpy as np
import scipy.fft as sfft

from .atmosphere import Atmosphere, GAMMA, R_AIR
from .bpm import third_octave_bands
from .model import PRESET_DIR, Observer
from .synth import (Render, Variability, _slow_noise, _warp, a_weighted_ms, highpass, loop_length,
                    scintillate, write_wav)

CP_AIR, CP_GAS, GAMMA_GAS = 1005.0, 1150.0, 1.33
P_REF = 2e-5
LB = 0.45359237                  # kg
FTLB = 1.3558179                 # J
BANDS = third_octave_bands(20.0, 20000.0)


def _deg(a, b):
    """angle between two vectors, deg"""
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _tab(t, x):
    t = np.asarray(t, float)
    return float(np.interp(x, t[:, 0], t[:, 1]))


@dataclass
class Engine:
    """A free-turbine turboshaft (presets/engines/<name>.toml)."""
    name: str
    ng_rpm: float
    np_rpm: float
    design_ng: float
    pressure_ratio: float
    pressure_ratio_low: list
    t45_design_k: float
    fuel_line: list
    torque_line: list
    rotor_omega: float
    compressor: dict
    power_turbine: dict
    exhaust: dict
    starter: dict
    compressor_efficiency: float = 0.8
    burner_efficiency: float = 0.99
    fuel_lhv: float = 43.2e6
    idle_ng: float = 65.0
    lightoff_ng: float = 15.0
    starter_ng: float = 24.0
    dropout_ng: float = 56.0
    start_fuel: list = (1.35, 60.0)

    def __post_init__(self):
        # the compressor's temperature rise ~ Ng^k through the two published delivery pressures
        k = 1.0 / 3.5
        dt = lambda pr: 288.15 * (pr ** k - 1.0) / self.compressor_efficiency
        n_lo, pr_lo = self.pressure_ratio_low
        self.work_exponent = np.log(dt(self.pressure_ratio) / dt(pr_lo)) / np.log(self.design_ng / n_lo)
        self.dt_design = dt(self.pressure_ratio)
        c = self.compressor
        dt1 = 288.15 * (c["stage_pressure_ratio"] ** k - 1.0) / c["stage_efficiency"]
        self.stage1_share = dt1 / self.dt_design
        c2 = c.get("stage2")
        self.stage2_share = ((288.15 + dt1) * (c2["stage_pressure_ratio"] ** k - 1.0) / c2["stage_efficiency"]
                             / self.dt_design) if c2 else 0.0
        # the turbines' flow constant (ellipse law) at the design point: its heat release and the published T45 set
        # the turbine inlet temperature and the air flow
        T3 = 288.15 + self.dt_design
        wf = _tab(self.fuel_line, self.design_ng) * LB / 3600.0
        T4 = self.t45_design_k + CP_AIR * self.dt_design / CP_GAS
        W = self.burner_efficiency * wf * self.fuel_lhv / (CP_GAS * (T4 - T3))
        self.design_flow = W
        self.design_t4 = T4
        P3 = self.pressure_ratio
        self.flow_constant = W * np.sqrt(T4) / (P3 * np.sqrt(1.0 - P3 ** -2))
        self._idle_flow = None

    def fuel_pph(self, ng, start=False):
        """steady fuel flow (lb/h) at corrected Ng %, or below idle on a start the acceleration schedule's"""
        wf = _tab(self.fuel_line, ng)
        if start and ng < self.idle_ng:
            wf = self.start_fuel[0] * wf + self.start_fuel[1]
        return wf

    def power_w(self, ng):
        """shaft power at 100 % power turbine speed, W"""
        return max(_tab(self.torque_line, ng), 0.0) * FTLB * self.rotor_omega


def list_engines():
    return sorted(p.stem for p in (PRESET_DIR / "engines").glob("*.toml"))


def load_engine(name_or_path):
    p = Path(name_or_path)
    if not p.suffix:
        p = PRESET_DIR / "engines" / f"{name_or_path}.toml"
    if not p.exists():
        raise FileNotFoundError(f"no engine '{name_or_path}' (presets: {', '.join(list_engines())})")
    with open(p, "rb") as f:
        d = tomllib.load(f)
    return Engine(**d["engine"], compressor=d["compressor"], power_turbine=d["power_turbine"], exhaust=d["exhaust"],
                  starter=d["starter"])


@dataclass
class Installation:
    """One engine on the aircraft (an aircraft preset's [[engine]] entry): positions in the aircraft frame (x forward,
    y left, z up, from the main rotor hub), the FlightGear properties it is played by."""
    name: str
    engine: Engine
    inlet: tuple
    exhaust: tuple
    inlet_axis: tuple = (1.0, 0.0, 0.0)          # out of the inlet, against the flow
    exhaust_axis: tuple = (-1.0, 0.0, 0.0)       # along the jet
    properties: str = "/engines/engine[0]"       # n1-pct, n2-pct, running, fuel-flow-pph
    starter_property: str = None                 # true while the starter cranks (default: <properties>/cranking)
    n2_property: str = None                      # the power turbine's speed, % (default: <properties>/n2-pct)
    pivot: tuple = None                          # tiltrotors: the nacelle's conversion axis (along y); the positions
                                                 # and axes are the helicopter mode's (nacelle 90 deg) and turn about it
    cockpit_db: list = None                      # [[nacelle deg, dB], ...]: what the paths to the cockpit lose that the
                                                 # model doesn't resolve (measured), by the nacelle's angle
    inside_db: float = 0.0                       # that loss at this installation's angle

    def tilted(self, nacelle_deg):
        """the installation with the nacelle at nacelle_deg (90: as given), turned forward about the pivot, with its
        cockpit loss at that angle"""
        t_db = np.asarray(self.cockpit_db, float) if self.cockpit_db else None
        out = replace(self, inside_db=float(np.interp(nacelle_deg, t_db[:, 0], t_db[:, 1])) if t_db is not None else 0.0)
        if self.pivot is None or nacelle_deg >= 90.0:
            return out
        t = np.radians(90.0 - nacelle_deg)
        R = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
        pv = np.asarray(self.pivot, float)
        at = lambda p: tuple(float(v) for v in pv + R @ (np.asarray(p, float) - pv))
        turn = lambda a: tuple(float(v) for v in R @ np.asarray(a, float))
        return replace(out, inlet=at(self.inlet), exhaust=at(self.exhaust), inlet_axis=turn(self.inlet_axis),
                       exhaust_axis=turn(self.exhaust_axis))


def load_installations(aircraft_name_or_path, nacelle_deg=90.0):
    """the aircraft preset's engines (a tiltrotor's with its nacelles at nacelle_deg)"""
    p = Path(aircraft_name_or_path)
    if not p.suffix:
        p = PRESET_DIR / f"{aircraft_name_or_path}.toml"
    with open(p, "rb") as f:
        d = tomllib.load(f)
    out = []
    for e in d.get("engine", []):
        e = dict(e)
        e["engine"] = load_engine(e.pop("type"))
        out.append(Installation(**e).tilted(nacelle_deg))
    if not out:
        raise ValueError(f"{p.stem} has no [[engine]] entries")
    return out


# ---------------------------------------------------------------- the cycle

@dataclass
class EngineState:
    ng: float                    # gas generator speed, % (corrected)
    lit: bool
    fuel_pph: float
    W: float                     # air flow, kg/s
    pr: float                    # compressor pressure ratio
    T3: float                    # compressor delivery, K
    T4: float                    # turbine inlet
    T45: float                   # power turbine inlet (TGT)
    T5: float                    # exhaust
    power_w: float
    dT1: float                   # the first compressor stage's temperature rise, K
    m_tip: float                 # its tip Mach number (tangential)
    m_rel: float                 # its relative (helical) tip Mach number
    m_axial: float
    v_jet: float
    rho_jet: float

    def summary(self):
        return (f"Ng {self.ng:5.1f} % {'lit' if self.lit else 'cold'}: air {self.W:5.2f} kg/s, fuel "
                f"{self.fuel_pph:5.0f} lb/h, PR {self.pr:5.2f}, T3 {self.T3 - 273.15:4.0f} C, T4 {self.T4 - 273.15:5.0f} C, "
                f"TGT {self.T45 - 273.15:4.0f} C, power {self.power_w / 745.7:5.0f} shp, jet {self.v_jet:4.0f} m/s, "
                f"stage 1 rel. tip Mach {self.m_rel:.2f}")


def _flow_from_ellipse(e, P3, T3, Q):
    """the turbines' flow and inlet temperature for the heat release Q (W): W sqrt(T4) = K P3 sqrt(1 - P3^-2) and
    W cp (T4 - T3) = Q"""
    a = e.flow_constant * P3 * np.sqrt(max(1.0 - P3 ** -2, 1e-9))
    b = Q / (CP_GAS * a)                       # (T4 - T3) / sqrt(T4) = b
    s = 0.5 * (b + np.sqrt(b * b + 4.0 * T3))
    return a / s, s * s


def engine_state(e, ng, lit=True, fuel_pph=None, start=None, atm=None):
    """The engine's state at gas generator speed ng (% corrected) at sea level: lit (on its operating line, or below
    idle on a start's acceleration schedule) or cold (cranked or spooling down). fuel_pph overrides the schedule."""
    atm = atm or Atmosphere(0.0)
    T2, p0 = atm.T, atm.p
    th = T2 / 288.15
    n = max(ng, 0.0)
    dT = e.dt_design * (n / e.design_ng) ** e.work_exponent * th
    pr = (1.0 + e.compressor_efficiency * dT / T2) ** 3.5
    T3 = T2 + dT
    if start is None:
        start = n < e.idle_ng
    wf = (e.fuel_pph(n, start) if fuel_pph is None else fuel_pph) if lit else 0.0
    Q = e.burner_efficiency * wf * LB / 3600.0 * e.fuel_lhv
    # where the compressor limits the flow: in proportion to Ng from the idle's
    if e._idle_flow is None:
        wi = e.fuel_pph(e.idle_ng) * LB / 3600.0 * e.burner_efficiency * e.fuel_lhv
        dti = e.dt_design * (e.idle_ng / e.design_ng) ** e.work_exponent
        e._idle_flow = _flow_from_ellipse(e, (1.0 + e.compressor_efficiency * dti / 288.15) ** 3.5, 288.15 + dti, wi)[0]
    floor = e._idle_flow * n / e.idle_ng * np.sqrt(1.0 / th) * (p0 / 101325.0)
    if lit and Q > 0:
        W, T4 = _flow_from_ellipse(e, pr, T3, Q)
        if W < floor:
            W, T4 = floor, T3 + Q / (CP_GAS * floor)
    else:
        W, T4 = (max(_flow_from_ellipse(e, pr, T3, 0.0)[0], floor) if n > 0 else 0.0), T3
    wg = W + wf * LB / 3600.0
    T45 = T4 - (CP_AIR * W * dT / (CP_GAS * wg) if lit else 0.0)
    power = e.power_w(n) if lit else 0.0
    T5 = T45 - (power / (wg * CP_GAS) if wg > 0 else 0.0)
    rho_j = p0 / (R_AIR * T5)
    vj = wg / (rho_j * e.exhaust["area"]) if wg > 0 else 0.0
    c = e.compressor
    rt = 0.5 * c["tip_diameter"]
    m_t = e.ng_rpm * n / 100.0 * 2.0 * np.pi / 60.0 * rt / atm.c
    area = np.pi * rt ** 2 * (1.0 - c["hub_tip_ratio"] ** 2)
    m_x = W / (atm.rho * area * atm.c)
    return EngineState(n, lit, wf, W, pr, T3, T4, T45, T5, power, dT * e.stage1_share, m_t, float(np.hypot(m_t, m_x)),
                       m_x, vj, rho_j)


# ---------------------------------------------------------------- the sources (at 1 m: Pa^2 per third-octave band,
# and tones (frequency, Pa^2))

@dataclass
class Source:
    bands: np.ndarray = field(default_factory=lambda: np.zeros(len(BANDS)))
    tones: list = field(default_factory=list)
    ripple: tuple = None            # (Hz, fraction): its amplitude's ripple, e.g. the drive's load at the blade rate

    def __iadd__(self, o):
        self.bands = self.bands + o.bands
        self.tones = self.tones + o.tones
        self.ripple = self.ripple or o.ripple
        return self

    def total(self):
        return float(self.bands.sum() + sum(m for _, m in self.tones))


def _db(x):
    return 10.0 * np.log10(max(x, 1e-30) / P_REF ** 2)


# Heidmann's method, Hough and Weir's small-engine ("AlliedSignal") revision, as ANOPP has it; levels in dB re
# 20 uPa at 1 m from 20 log10(dT / 1 R) + 10 log10(W / 1 lb/s) (NASA TM X-71763 eq. 1)
_TH = np.arange(0.0, 190.0, 10.0)
_F3IB = (np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 180]),
         np.array([-2, -1, 0, 0, 0, -2, -4.5, -7.5, -11, -15, -19.5, -25, -63.5]))
_F3TI = np.array([-3, -1.5, -1.5, -1.5, -1.5, -2, -3, -4, -6, -9, -12.5, -16, -19.5, -23, -26.5, -30, -33.5, -37, -40.5])
_F2CT = np.array([-5.5, -4.5, -3, -1.5, 0, 0, 0, 0, -2.5, -5, -6, -6.9, -7.9, -8.8, -9.8, -10.7, -11.7, -12.6, -13.6])
_CT = dict(mt=(1.135, 1.135, 1.413), sl1=(5.9036, 5.54769, 2.43439), sl2=(-0.632839, -0.632839, -1.030931),
           y1=(50.0, 50.0, 40.0), y2=(57.1896, 56.7981, 50.4058), sl3=(-15.0, -50.0, -30.0), sl4=(30.0, 50.0, 50.0),
           y3=(-4.51545, -30.103, -27.0927), y4=(9.0309, 30.103, 45.1545))


def _heidmann(c, dT, W, m_t, m_rel, m_d, f_shaft, theta, harmonics=6, buzz=True):
    """one compressor stage (c: its preset table) by Heidmann's method, small-engine revision, radiated from the
    inlet at theta (deg from its axis), 1 m: temperature rise dT (K), air flow W (kg/s), tip Mach numbers
    (tangential, relative, the relative at design)"""
    out = Source()
    if dT <= 0 or W <= 0:
        return out
    B, V = c["rotor_blades"], c["stator_vanes"]
    bpf = B * f_shaft
    Mr = m_rel
    rss = c["rotor_stator_spacing"]
    igv = c.get("inlet_guide_vanes", False)
    base = 10.0 * np.log10((dT * 1.8) ** 2 * W / LB)
    md = 20.0 * np.log10(max(m_d, 1.0))
    # broadband
    f1 = 55.5 + md - (20.0 * np.log10(Mr / 0.9) if Mr > 0.9 else 0.0)
    f2 = -5.0 * np.log10(min(rss, 100.0) / 300.0)           # the inlet's distortion caps the spacing's effect
    bb = base + f1 + f2 + np.interp(theta, *_F3IB)
    r = BANDS / bpf / 2.0
    flog = np.where(r < 1.0, 2.445096095, 13.97197769) * np.log(r) ** 2
    out.bands = P_REF ** 2 * 10.0 ** (0.1 * (bb - flog))
    # rotor-stator interaction and inlet distortion tones
    if Mr <= 0.72:
        f1 = 54.5 + md
    else:
        f1 = min(54.5 + md + 50.0 * np.log10(Mr / 0.72), 53.5 + 80.0 * np.log10(max(m_d, 1.0) / Mr))
    tl = base + f1 - 10.0 * np.log10(min(rss, 100.0) / 300.0) + np.interp(theta, _TH, _F3TI)
    cut = abs(m_t / (1.0 - V / B)) < 1.05 and m_t < 1.0
    for h in range(1, harmonics + 1):
        if h == 1:
            fall = 8.0 if cut else 0.0
        elif igv:
            fall = 3.0 * (h + 1)
        else:
            fall = 9.2 if h == 2 else 3.0 * h + 1.8
        msp = 10.0 ** (0.1 * (tl - fall)) + 10.0 ** (0.1 * tl - h + 1)
        out.tones.append((h * bpf, P_REF ** 2 * msp))
    # combination tones (buzz-saw): band levels around 1/2, 1/4, 1/8 BPF, carried by the shaft's orders in each band
    if buzz and Mr >= 1.0:
        ct = np.zeros(len(BANDS))
        lvl = base + np.interp(theta, _TH, _F2CT) - (5.0 if igv else 0.0)
        for K in range(3):
            m = 20.0 * np.log10(Mr)
            f1 = (_CT["sl1"][K] * m + _CT["y1"][K]) if Mr < _CT["mt"][K] else (_CT["sl2"][K] * m + _CT["y2"][K])
            q = BANDS / bpf
            f3 = np.where(q <= 2.0 ** -(K + 1), _CT["sl4"][K] * np.log10(q) + _CT["y4"][K],
                          _CT["sl3"][K] * np.log10(q) + _CT["y3"][K])
            ct += 10.0 ** (0.1 * (lvl + f1 + f3))
        lo, hi = BANDS * 2 ** (-1 / 6), BANDS * 2 ** (1 / 6)
        for b in range(len(BANDS)):
            orders = [k for k in range(1, B) if lo[b] <= k * f_shaft < hi[b]]
            for k in orders:
                out.tones.append((k * f_shaft, P_REF ** 2 * ct[b] / len(orders)))
    return out


def _stage2(e, s, atm):
    """the second stage's temperature rise, tip Mach numbers (tangential, relative) and air flow at state s"""
    c2 = e.compressor["stage2"]
    dT = s.dT1 / e.stage1_share * e.stage2_share
    rt = 0.5 * c2["tip_diameter"]
    m_t = e.ng_rpm * s.ng / 100.0 * 2.0 * np.pi / 60.0 * rt / atm.c
    rho = atm.rho * (1.0 + s.dT1 / atm.T) ** (1.0 / (GAMMA - 1.0))       # behind the first stage
    m_x = s.W / (rho * np.pi * rt ** 2 * (1.0 - c2["hub_tip_ratio"] ** 2) * atm.c)
    return dT, m_t, float(np.hypot(m_t, m_x))


def compressor_noise(e, s, theta, harmonics=6, atm=None):
    """the compressor's sound radiated from the inlet, at theta (deg from the inlet's axis), 1 m: the first stage,
    and the second heard through it (its preset's through_stage1_db)"""
    atm = atm or Atmosphere(0.0)
    if s.ng <= 0 or s.W <= 0:
        return Source()
    f_shaft = e.ng_rpm * s.ng / 100.0 / 60.0
    d = engine_state(e, e.design_ng)
    out = _heidmann(e.compressor, s.dT1, s.W, s.m_tip, s.m_rel, d.m_rel, f_shaft, theta, harmonics)
    _low_speed(out, e.compressor, s.ng)
    c2 = e.compressor.get("stage2")
    if c2:
        dT, m_t, m_r = _stage2(e, s, atm)
        m_d = _stage2(e, d, atm)[2]
        g = 10.0 ** (0.1 * c2.get("through_stage1_db", 0.0))
        two = _heidmann(dict(c2, inlet_guide_vanes=False), dT, s.W, m_t, m_r, m_d, f_shaft, theta, harmonics, False)
        two.bands = two.bands * g
        two.tones = [(f, m * g) for f, m in two.tones]
        _low_speed(two, c2, s.ng)
        out += two
    return out


def _low_speed(src, c, ng):
    """a stage's measured low-speed correction to its tones (its preset's low_speed_tone_db: [[Ng %, dB], ...])"""
    t = c.get("low_speed_tone_db")
    if t:
        g = 10.0 ** (0.1 * _tab(t, ng))
        src.tones = [(f, m * g) for f, m in src.tones]


# log10 of the directivity and the spectrum: SAE ARP 876E's tables 18 and 17, filled in from ANOPP's (NASA TM-83199
# part 2, section 8.2), whose own 1982 tables differ by up to 0.43 below the peak
_D_COMB = np.array([-0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.53, -0.46, -0.39, -0.16, 0.08, 0.31, 0.5, 0.35, 0.12,
                    -0.19, -0.51, -0.8, -0.9])
_S_COMB = (np.arange(-1.1, 1.61, 0.1),
           np.array([-3.87, -3.47, -3.12, -2.72, -2.32, -1.99, -1.7, -1.41, -1.17, -0.97, -0.82, -0.72, -0.82, -0.97,
                     -1.17, -1.41, -1.7, -1.99, -2.32, -2.72, -3.12, -3.47, -3.87, -4.32, -4.72, -5.22, -5.7, -6.2]))


def combustion_noise(e, s, theta, atm=None):
    """the combustor's sound through the turbines, radiated from the exhaust, at theta (deg from the engine's inlet
    axis: 180 straight down the jet), 1 m (ANOPP section 8.2, GE's turbine transmission loss)"""
    atm = atm or Atmosphere(0.0)
    out = Source()
    if not s.lit or s.fuel_pph <= 0:
        return out
    d = engine_state(e, e.design_ng)
    g = ((d.T4 - d.T5) / atm.T) ** -4
    power = 8.85e-7 * s.W * atm.c ** 2 * ((s.T4 - s.T3) / s.T3) ** 2 * s.pr ** 2 * g
    D = 10.0 ** np.interp(theta, _TH, _D_COMB)
    S = 10.0 ** np.interp(np.log10(BANDS / 400.0), *_S_COMB, left=-9.0, right=-9.0)
    out.bands = atm.rho * atm.c * power * D * S / (4.0 * np.pi)
    return out


_D_TBB = np.array([-0.789, -0.689, -0.599, -0.509, -0.409, -0.319, -0.219, -0.129, -0.029, 0.071, 0.151, 0.221, 0.231,
                   0.211, 0.111, -0.029, -0.229, -0.549, -0.869])
_D_TT = np.array([-1.911, -1.671, -1.471, -1.261, -1.061, -0.851, -0.641, -0.431, -0.231, -0.021, 0.189, 0.389, 0.589,
                  0.259, -0.191, -0.591, -0.931, -1.271, -1.611])
_S_TBB = (np.array([-0.903, -0.796, -0.699, -0.602, -0.502, -0.398, -0.301, -0.201, -0.097, 0.0, 0.097, 0.204, 0.301,
                    0.602]),
          np.array([-1.884, -1.604, -1.444, -1.304, -1.184, -1.084, -1.004, -0.924, -0.844, -0.784, -1.004, -1.204,
                    -1.384, -1.924]))


def _turbine(work_ratio, tip_speed, area, f_b, theta, atm, harmonics=4, directional=True):
    """GE's turbine method (ANOPP section 8.3): broadband and tones of a turbine stage, 1 m"""
    out = Source()
    if work_ratio <= 0 or tip_speed <= 0:
        return out
    u = tip_speed / atm.c
    k = atm.rho ** 2 * atm.c ** 4 * area / (4.0 * np.pi)
    pb = 8.589e-5 * work_ratio ** 1.27 * u ** -1.27
    pt = 1.162e-4 * work_ratio ** 1.46 * u ** -4.02
    db = 10.0 ** np.interp(theta, _TH, _D_TBB) if directional else 1.0
    dt = 10.0 ** np.interp(theta, _TH, _D_TT) if directional else 1.0
    lg = np.log10(BANDS / f_b)
    S = 10.0 ** np.interp(lg, *_S_TBB, left=np.nan, right=np.nan)
    # beyond the table: its end slopes (third-octave band levels, about -28 dB/decade below, -18 above)
    S = np.where(lg < _S_TBB[0][0], 10.0 ** (_S_TBB[1][0] + 2.8 * (lg - _S_TBB[0][0])), S)
    S = np.where(lg > _S_TBB[0][-1], 10.0 ** (_S_TBB[1][-1] - 1.8 * (lg - _S_TBB[0][-1])), S)
    out.bands = k * pb * db * S
    out.tones = [(n * f_b, k * pt * dt * 0.6838 * 10.0 ** (-(n - 1) / 2.0)) for n in range(1, harmonics + 1)]
    return out


def turbine_noise(e, s, theta, np_pct=100.0, atm=None):
    """the power turbine's last stage, radiated from the exhaust, at theta (deg from the engine's inlet axis), 1 m"""
    atm = atm or Atmosphere(0.0)
    if not s.lit or s.power_w <= 0:
        return Source()
    t = e.power_turbine
    n = e.np_rpm * np_pct / 100.0 / 60.0
    return _turbine((s.T45 - s.T5) / s.T45, np.pi * t["tip_diameter"] * n, t["inlet_area"], t["rotor_blades"] * n, theta,
                    atm)


_OMEGA = (np.arange(-0.45, 0.251, 0.05), np.array([-1, -0.9, -0.76, -0.58, -0.41, -0.22, 0, 0.22, 0.5, 0.77, 1.07, 1.39,
                                                   1.74, 1.95, 2]))
_PJ = (np.arange(-0.4, 0.41, 0.05), np.array([-0.13, -0.13, -0.13, -0.13, -0.13, -0.12, -0.1, -0.05, 0, 0.1, 0.21, 0.32,
                                              0.41, 0.43, 0.41, 0.31, 0.14]))


def jet_noise(v, rho_j, area, atm=None):
    """a subsonic jet's mixing noise (SAE ARP 876's power, ANOPP section 8.4 eq. 3), nondirectional, 1 m"""
    atm = atm or Atmosphere(0.0)
    out = Source()
    if v <= 1.0:
        return out
    lv = np.log10(v / atm.c)
    w = np.interp(lv, *_OMEGA)
    P = 10.0 ** np.interp(lv, *_PJ)
    power = 6.67e-5 * (rho_j / atm.rho) ** w * (v / atm.c) ** 8 * P * atm.rho * atm.c ** 3 * area
    eta = BANDS / (0.12 * v / np.sqrt(4.0 * area / np.pi))         # the bands peak at eta 2.45: a Strouhal number of 0.3 (NACA 1329)
    S = eta ** 2 / (1.0 + eta ** 2) ** 1.75 * BANDS
    out.bands = atm.rho * atm.c * power * S / S.sum() / (4.0 * np.pi)
    return out


def starter_noise(e, ng, atm=None):
    """the air turbine starter at gas generator speed ng (%), 1 m, nondirectional"""
    atm = atm or Atmosphere(0.0)
    st = e.starter
    if ng <= 0:
        return Source()
    n = e.ng_rpm * ng / 100.0 / 60.0 * st["speed_ratio"]
    n_d = e.ng_rpm * e.dropout_ng / 100.0 / 60.0 * st["speed_ratio"]
    k = (GAMMA - 1.0) / GAMMA
    work = 0.8 * (1.0 - st["supply_pressure_ratio"] ** -k)
    out = _turbine(work, np.pi * st["tip_diameter"] * n_d, st["inlet_area"], st["rotor_blades"] * n, 90.0, atm,
                   directional=False)
    m = st.get("cowl_surface_mass")
    if m:
        # it sits inside the engine's cowling: its turbine heard through the cowl (mass law), its exhaust ducted out
        tau = lambda f: np.clip(10.0 ** (-(20.0 * np.log10(m * np.maximum(f, 1.0)) - 47.0) / 10.0), 0.0, 1.0)
        out.bands = out.bands * tau(BANDS)
        out.tones = [(f, p * float(tau(f))) for f, p in out.tones]
    # its exhaust: the expanded air dumped overboard
    T = st["supply_temperature_k"] * (1.0 - work)
    rho = atm.p / (R_AIR * T)
    out += jet_noise(st["air_flow"] / (rho * st["exhaust_area"]), rho, st["exhaust_area"], atm)
    return out


# ---------------------------------------------------------------- loops

LAYERS = ("compressor", "core", "turbine", "starter")


def layer_sources(inst, s, listener, starter=False, np_pct=100.0, atm=None):
    """{layer: (position, Source at 1 m)} of one engine in state s, heard at the listener (aircraft frame)"""
    e = inst.engine
    X = np.asarray(listener, float)
    inlet, exh = np.asarray(inst.inlet, float), np.asarray(inst.exhaust, float)
    th_in = _deg(np.asarray(inst.inlet_axis, float), X - inlet)
    th_ex = 180.0 - _deg(np.asarray(inst.exhaust_axis, float), X - exh)
    core = combustion_noise(e, s, th_ex, atm)
    core += jet_noise(s.v_jet, s.rho_jet, e.exhaust["area"], atm) if s.lit else Source()
    out = dict(compressor=(inlet, compressor_noise(e, s, th_in, atm=atm)), core=(exh, core),
               turbine=(exh, turbine_noise(e, s, th_ex, np_pct, atm)))
    if starter:
        out["starter"] = (inlet, starter_noise(e, s.ng, atm))
    return out


def _spectrum(src, L, fs, rng, f_lim, haze=0.004, coherent=0.6):
    """the rfft of a loop of L samples with the Source's band powers (Gaussian noise, power-complementary bands) and
    tones scattered into narrow bands: a share `coherent` stays a pure tone, the rest spreads over a Gaussian line
    of rms width haze * f"""
    k = np.fft.rfftfreq(L, 1.0 / fs)
    with np.errstate(divide="ignore"):
        u = 3.0 * np.log2(np.maximum(k, 1e-9) / 1000.0)
    nb = np.round(3.0 * np.log2(BANDS / 1000.0))
    lim = (k < f_lim) * np.where(k > 0.85 * f_lim, 0.5 * (1.0 + np.cos(np.pi * (k / f_lim - 0.85) / 0.15)), 1.0)
    # one draw of complex Gaussian noise, shaped by the bands' power-complementary (cos^2) weights, each band's
    # expected power its own
    psd = np.zeros(len(k))
    for b in range(len(BANDS)):
        if src.bands[b] <= 0 or BANDS[b] > f_lim:
            continue
        sel = np.abs(u - nb[b]) < 1.0
        w2 = np.cos(0.5 * np.pi * (u[sel] - nb[b])) ** 2
        psd[sel] += w2 * src.bands[b] / max(w2.sum(), 1e-30)
    psd[0] = 0.0
    X = (rng.normal(size=len(k)) + 1j * rng.normal(size=len(k))) * np.sqrt(psd / 4.0) * L
    df = fs / L
    for f, msp in src.tones:
        if f >= f_lim or f <= 0 or msp <= 0:
            continue
        k0 = int(round(f / df))
        X[k0] += np.sqrt(2.0 * coherent * msp) * L / 2.0 * np.exp(2j * np.pi * rng.random())
        sig = max(haze * f, 0.5) / df
        j = np.arange(max(1, int(k0 - 4 * sig)), min(len(k), int(k0 + 4 * sig) + 2))
        w = np.exp(-0.5 * ((j - k0) / sig) ** 2)
        Z = rng.normal(size=len(j)) + 1j * rng.normal(size=len(j))
        Z *= np.sqrt(w / w.sum())
        X[j] += Z * np.sqrt((1.0 - coherent) * msp / 2.0) * L
    return X * lim


def _path_gain(f, r, height_above, env, atm, cabin, doors_open=False):
    """spreading from 1 m, air absorption, the ground's reflection (outside), the cabin's walls and interior
    (inside: a model.Cabin)"""
    from .environment import ground_gain
    g = np.exp(-atm.absorption(np.maximum(f, 1.0)) * r / 8.686) / max(r, 0.1) + 0j
    if cabin is not None:
        g = g * np.sqrt(cabin.inside(f, doors_open))
    elif env is not None:
        g = g * ground_gain(f, height_above, r, env)
    return g


@dataclass
class EngineCondition:
    ng: float
    lit: bool = True
    starter: bool = False
    fuel_pph: float = None
    np_pct: float = 100.0


SPEED_WANDER = dict(compressor=0.001, starter=0.002, turbine=0.002, core=0.0, apu=0.001, gearbox=0.002)


def _render_sources(srcs, observer, X, fs, duration, seed, environment, cabin, pitch_headroom, f_highpass, atm):
    """loops of the sources ({layer: (position, Source at 1 m)}) at the listener X (aircraft frame): spreading,
    absorption, outside the ground and turbulent air, inside the cabin's walls (doors open for 'cockpit_open')"""
    from .environment import Environment
    from .model import Cabin
    cabin = cabin or Cabin()
    inside = observer.position is not None
    L, _ = loop_length(1.0, fs, duration)
    f_lim = 0.5 * fs / pitch_headroom
    k = np.fft.rfftfreq(L, 1.0 / fs)
    rng = np.random.default_rng(seed)
    env = None if inside else Environment(environment)
    var = Variability()
    sig, lv = {}, {}
    for name, (pos, src) in srcs.items():
        if src.total() <= 0:
            continue
        lv[name] = round(_db(src.total()), 1)
        if pos is None:                     # given at the listener
            r, G = 0.0, 1.0
        else:
            r = float(np.linalg.norm(X - pos))
            G = _path_gain(k, r, pos[2] - X[2] if not inside else 0.0, env, atm, cabin if inside else None,
                           observer.name == "cockpit_open")
        x = sfft.irfft(_spectrum(src, L, fs, rng, f_lim) * G, L, workers=-1)
        if src.ripple:
            f_r, frac = src.ripple
            n = max(1, int(round(f_r * L / fs)))           # whole cycles in the loop
            x = x * (1.0 + frac * np.cos(2.0 * np.pi * n * np.arange(L) / L))
        if SPEED_WANDER[name] > 0:
            d = SPEED_WANDER[name] * _slow_noise(rng, L, fs, 0.3)[0]
            w = np.cumsum(d - d.mean()) / fs
            x = _warp(x, w - w.mean(), fs)
        if name == "core":
            x = x * np.maximum(1.0 + 0.15 * _slow_noise(rng, L, fs, 3.0)[0], 0.3)
        if not inside and r > 0:
            x = scintillate(x, fs, r, var, rng, {})
        sig[f"layer/{name}"] = highpass(x, fs, f_highpass)
    sig["total"] = sum(sig.values()) if sig else np.zeros(L)
    return sig, lv


def _listener(observer, hub_height):
    return np.asarray(observer.position if observer.position is not None
                      else _ground_listener(observer, hub_height), float)


def render_engine(inst, cond, observer, fs=48000.0, duration=15.0, seed=1, environment="open", cabin=None,
                  pitch_headroom=1.3, f_highpass=25.0, atm=None, hub_height=3.6):
    """A seamless loop of one engine at a listener riding with the aircraft (an Observer: its position, aircraft
    frame, or for the outside a horizontal distance at an azimuth, on the ground beside the aircraft standing on its
    gear). Inside, the cabin's walls (a model.Cabin; default a limp 8 kg/m^2 wall), with its doors open if the
    observer is 'cockpit_open', and the installation's cockpit loss. Returns a Render with the layers and their
    total."""
    atm = atm or Atmosphere(0.0)
    e = inst.engine
    s = engine_state(e, cond.ng, cond.lit, cond.fuel_pph, atm=atm)
    X = _listener(observer, hub_height)
    srcs = layer_sources(inst, s, X, cond.starter, cond.np_pct, atm)
    sig, lv = _render_sources(srcs, observer, X, fs, duration, seed, environment, cabin, pitch_headroom, f_highpass,
                              atm)
    if observer.position is not None and inst.inside_db:
        sig = {k: x * 10.0 ** (inst.inside_db / 20.0) for k, x in sig.items()}
    meta = dict(engine=e.name, installation=inst.name, ng=cond.ng, lit=cond.lit, starter=cond.starter,
                fuel_pph=round(s.fuel_pph, 1), power_shp=round(s.power_w / 745.7, 1), tgt_c=round(s.T45 - 273.15, 1),
                source_db_1m=lv, observer=observer.name)
    return Render(fs, sig, [], meta)


# ---------------------------------------------------------------- the APU

@dataclass
class APU:
    """An auxiliary power unit by its recorded signature (presets/apu/<type>.toml): its shaft's tones by order and
    its broadband third octaves at full speed (dB re 20 uPa at 1 m), both moving with its speed, their power with
    its speed to the speed_exponent."""
    name: str
    shaft_hz: float
    tones: list
    bands: list
    speed_exponent: float = 5.0
    position: tuple = (0.0, 0.0, 0.0)
    property: str = "/engines/apu/n1-pct"


def load_apu(aircraft_name_or_path):
    """the aircraft preset's APU ([apu]: type, position, property), or None"""
    p = Path(aircraft_name_or_path)
    if not p.suffix:
        p = PRESET_DIR / f"{aircraft_name_or_path}.toml"
    with open(p, "rb") as f:
        a = tomllib.load(f).get("apu")
    if not a:
        return None
    with open(PRESET_DIR / "apu" / f"{a['type']}.toml", "rb") as f:
        d = tomllib.load(f)["apu"]
    return APU(**d, position=tuple(a.get("position", (0.0, 0.0, 0.0))),
               property=a.get("property", "/engines/apu/n1-pct"))


def apu_source(apu, n_pct):
    """the APU at n_pct of its speed, 1 m, nondirectional"""
    out = Source()
    k = n_pct / 100.0
    if k <= 0:
        return out
    g = k ** apu.speed_exponent
    b = np.asarray(apu.bands, float)
    lvl = np.interp(np.log(BANDS / k), np.log(b[:, 0]), b[:, 1], left=-200.0, right=-200.0)
    out.bands = P_REF ** 2 * 10.0 ** (0.1 * lvl) * g
    out.tones = [(order * apu.shaft_hz * k, P_REF ** 2 * 10.0 ** (0.1 * L) * g) for order, L in apu.tones]
    return out


def render_apu(apu, n_pct, observer, fs=48000.0, duration=15.0, seed=1, environment="open", cabin=None,
               pitch_headroom=1.3, f_highpass=25.0, atm=None, hub_height=3.6):
    """A seamless loop of the APU at n_pct of its speed at a listener riding with the aircraft (as render_engine)"""
    atm = atm or Atmosphere(0.0)
    X = _listener(observer, hub_height)
    sig, lv = _render_sources({"apu": (np.asarray(apu.position, float), apu_source(apu, n_pct))}, observer, X, fs,
                              duration, seed, environment, cabin, pitch_headroom, f_highpass, atm)
    return Render(fs, sig, [], dict(apu=apu.name, n_pct=n_pct, source_db_1m=lv, observer=observer.name))


# ---------------------------------------------------------------- the main transmission

@dataclass
class Transmission:
    """The drive as heard inside (an aircraft preset's [transmission]): the gearboxes' mesh tones [Hz at 100 % rotor
    speed, dB at the pilot at the reference torque, sideband spacing Hz] and, measured where it is known, the drive's
    and engines' structure-borne roar, the vibration that reaches the cabin's frames through the airframe: third
    octaves [Hz at 100 % rotor speed, dB at the pilot at the reference torque]. Both move with the rotor speed, their
    amplitude with the torque, and that load's ripple at the rotors' blade-passing rate where it is measured (ripple:
    [Hz at 100 % rotor speed, fraction])."""
    tones: list
    reference_torque_pct: float = 75.0
    rpm_property: str = "rotors/main/rpm"
    torque_properties: list = field(default_factory=list)
    bands: list = field(default_factory=list)
    rpm_nominal: float = None       # the rpm property at 100 % (default: the main rotor's rpm)
    ripple: list = None


def load_transmission(aircraft_name_or_path):
    p = Path(aircraft_name_or_path)
    if not p.suffix:
        p = PRESET_DIR / f"{aircraft_name_or_path}.toml"
    with open(p, "rb") as f:
        t = tomllib.load(f).get("transmission")
    return Transmission(**t) if t else None


def gearbox_source(tr, rotor_pct=100.0, torque_pct=100.0):
    """the mesh tones at the pilot: the fundamental's power and its first two sidebands' (planet passing or shaft
    rate, 10 and 18 dB down: the meshes' load varies as they turn), amplitude with the torque"""
    out = Source()
    k = rotor_pct / 100.0
    g = (torque_pct / tr.reference_torque_pct) ** 2
    if tr.bands:
        b = np.asarray(tr.bands, float)
        lvl = np.interp(np.log(BANDS / k), np.log(b[:, 0]), b[:, 1], left=-200.0, right=-200.0)
        out.bands = P_REF ** 2 * 10.0 ** (0.1 * lvl) * g
    if tr.ripple:
        out.ripple = (tr.ripple[0] * k, tr.ripple[1])
    for f, L, side in tr.tones:
        p = P_REF ** 2 * 10.0 ** (0.1 * L) * g
        out.tones.append((f * k, p))
        for j, d in ((1, -10.0), (2, -18.0)):
            out.tones += [(f * k + j * side * k, p * 10.0 ** (0.1 * d)), (f * k - j * side * k, p * 10.0 ** (0.1 * d))]
    return out


def gearbox_full_rpm(tr, aircraft):
    """the rotor speed from which the gearbox plays at its full level, in the units of tr's rpm property: for a
    tiltrotor its airplane mode's share of the helicopter mode's. Its level follows the torque; this only fades it out
    as the rotor stops, and must not take anything off in a cruise at reduced rotor speed."""
    nominal = tr.rpm_nominal or aircraft.main.rpm
    return nominal * (aircraft.airplane_rpm / aircraft.main.rpm if aircraft.airplane_rpm else 1.0)


def render_gearbox(tr, observer, fs=48000.0, duration=15.0, seed=1, torque_pct=100.0, f_highpass=25.0):
    """a seamless loop of the gearbox's tones at the pilot (the levels are the pilot's: no path)"""
    sig, lv = _render_sources({"gearbox": (None, gearbox_source(tr, 100.0, torque_pct))}, observer, None, fs,
                              duration, seed, "free", None, 1.3, f_highpass, Atmosphere(0.0))
    return Render(fs, sig, [], dict(torque_pct=torque_pct, source_db=lv, observer=observer.name))


def _ground_listener(o, hub_height, ear=1.7):
    """an outside Observer as a listener on the ground at its azimuth and (horizontal) distance from the hub"""
    az = np.radians(o.azimuth_deg)
    return np.array([o.distance_m * np.cos(az), -o.distance_m * np.sin(az), ear - hub_height])


# ---------------------------------------------------------------- a bank and its FlightGear section

LIT_NG = (20.0, 25.0, 30.0, 37.0, 45.0, 55.0, 65.0, 72.0, 80.0, 87.5, 92.0, 96.0, 100.0, 104.0)
COLD_NG = (3.0, 5.0, 8.0, 12.0, 17.0, 24.0, 30.0, 37.0, 45.0, 55.0, 65.0, 80.0, 100.0)
STARTER_NG = (3.0, 5.0, 8.0, 12.0, 17.0, 24.0, 30.0, 37.0, 45.0, 55.0, 62.0)
APU_PCT = (5.0, 10.0, 20.0, 35.0, 50.0, 65.0, 80.0, 90.0, 100.0)


def _views(aircraft, distance):
    v = [Observer("cockpit", position=tuple(aircraft.cockpit))]
    if aircraft.cabin.openings:
        v.append(Observer("cockpit_open", position=tuple(aircraft.cockpit)))
    return v + [Observer("side", 90.0, 0.0, distance)]


def make_engine_bank(aircraft_name, out_dir, distance=50.0, fs=48000.0, duration=15.0, bits=16, peak_dbfs=-1.0,
                     like=None, environment="open", seed=1, nacelle_deg=90.0, progress=print):
    """Loops of every engine of the aircraft over its gas generator speed: lit (the operating line; below idle a
    start's acceleration), cold (cranked or spooling down) and the starter, in the cockpit and outside (on the
    ground beside the aircraft, `distance` m to its right), with manifest.json and pywopwop-engines.xml. Layers:
    compressor (the inlet's whine), core (combustion and the jet), turbine (the power turbine, at 100 % Np),
    starter. like: a rotor bank's manifest.json (or its folder) whose gains these loops take, so that their
    loudness against its loops is the physics'; default: the loudest at peak_dbfs. A tiltrotor's engines turn with
    its nacelles (nacelle_deg: 90 helicopter mode, 0 airplane mode); off 90 the files and sounds are named for it
    (airplane_..., pww-airplane-...), so that the two banks can be faded into each other by the nacelle angle."""
    from .model import load_aircraft
    ac = load_aircraft(aircraft_name)
    insts = load_installations(aircraft_name, nacelle_deg)
    mode = "" if nacelle_deg >= 90.0 or all(x.pivot is None for x in insts) else \
        ("airplane_" if nacelle_deg <= 0.0 else f"nacelle{nacelle_deg:g}_")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    renders = []
    for i, inst in enumerate(insts):
        e = inst.engine
        conds = ([("lit", EngineCondition(n), LIT_NG) for n in LIT_NG]
                 + [("cold", EngineCondition(n, lit=False), COLD_NG) for n in COLD_NG]
                 + [("starter", EngineCondition(n, lit=False, starter=True), STARTER_NG) for n in STARTER_NG])
        for j, (kind, c, grid) in enumerate(conds):
            k = grid.index(c.ng)
            ratio = max(grid[k + 1] / grid[k] if k + 1 < len(grid) else 1.0, 1.3)
            for o in _views(ac, distance):
                r = render_engine(inst, c, o, fs, duration, seed + 1000 * i + 7 * j, environment, cabin=ac.cabin,
                                  pitch_headroom=ratio, hub_height=ac.hub_height_m)
                keep = ["layer/starter"] if kind == "starter" else [x for x in r.signals if x.startswith("layer/")
                                                                     and x != "layer/starter"]
                r.signals = {x: r.signals[x] for x in keep if x in r.signals}
                renders.append((inst, kind, c, o, r))
                lv = r.levels()
                progress(f"  {inst.name:6s} {kind:7s} Ng {c.ng:5.1f} {o.name:8s} "
                         + "  ".join(f"{x.split('/')[-1]} {v['dba']:5.1f}" for x, v in lv.items()) + " dBA")

    apu = load_apu(aircraft_name)
    if apu:
        class _A:
            name = "apu"
        for j, n in enumerate(APU_PCT):
            ratio = max(APU_PCT[j + 1] / n if j + 1 < len(APU_PCT) else 1.0, 1.3)
            for o in _views(ac, distance):
                r = render_apu(apu, n, o, fs, duration, seed + 5000 + 7 * j, environment, cabin=ac.cabin,
                               pitch_headroom=ratio, hub_height=ac.hub_height_m)
                r.signals = {"layer/apu": r.signals["layer/apu"]}
                renders.append((_A, "run", EngineCondition(n, lit=False), o, r))
                progress(f"  apu    {n:5.1f} % {o.name:12s} {r.levels()['layer/apu']['dba']:5.1f} dBA")

    tr = load_transmission(aircraft_name)
    if tr:
        class _G:
            name = "gearbox"
        o = Observer("cockpit", position=tuple(ac.cockpit))
        r = render_gearbox(tr, o, fs, duration, seed + 6000)
        r.signals = {"layer/gearbox": r.signals["layer/gearbox"]}
        renders.append((_G, "run", EngineCondition(100.0, lit=False), o, r))
        progress(f"  gearbox 100 % torque cockpit {r.levels()['layer/gearbox']['dba']:5.1f} dBA")

    def group(o):
        return "cockpit" if o.position is not None else "exterior"
    gains = {}
    old = None
    if like:
        p = Path(like)
        old = json.loads((p / "manifest.json" if p.is_dir() else p).read_text())
    for g in {group(o) for *_, o, _ in renders}:
        peak = max(np.max(np.abs(x)) for *_, o, r in renders if group(o) == g for x in r.signals.values())
        if old is not None:
            gains[g] = 10.0 ** (old["gain_db"][g] / 20.0)
            top = 20.0 * np.log10(peak * gains[g])
            if top > -0.1:
                raise ValueError(f"the {g} loops would peak at {top:+.1f} dBFS at {like}'s gain")
        else:
            gains[g] = 10.0 ** (peak_dbfs / 20.0) / peak
    files = []
    for inst, kind, c, o, r in renders:
        lv = r.levels()
        for x, sig in r.signals.items():
            if lv[x]["oaspl"] < 0.0:
                continue
            layer = x.split("/")[-1]
            fn = f"{mode}{inst.name}_{layer}_{kind}_{c.ng:g}_{o.name}.wav"
            write_wav(out / fn, sig, r.fs, bits, gains[group(o)], seed=abs(hash(fn)) % 2 ** 32)
            files.append(dict(file=fn, engine=inst.name, layer=layer, state=kind, ng=c.ng,
                              fuel_pph=r.meta.get("fuel_pph", 0.0), observer=o.name, oaspl_db=round(lv[x]["oaspl"], 2),
                              dba=round(lv[x]["dba"], 2)))
    manifest = dict(aircraft=ac.name, generator="pywopwop engines", sample_rate=fs, bits=bits,
                    gain_db={g: round(20.0 * np.log10(v), 3) for g, v in gains.items()},
                    exterior_distance_m=distance, nacelle_deg=nacelle_deg, mode=mode.rstrip("_"),
                    engines=[dict(name=x.name, engine=x.engine.name, properties=x.properties,
                                  starter_property=x.starter_property or f"{x.properties}/cranking",
                                  n2_property=x.n2_property or f"{x.properties}/n2-pct",
                                  idle_ng=x.engine.idle_ng, lightoff_ng=x.engine.lightoff_ng) for x in insts],
                    lit_ng=list(LIT_NG), cold_ng=list(COLD_NG), starter_ng=list(STARTER_NG),
                    openings=[dict(share=o["share"], property=o["property"]) for o in ac.cabin.openings],
                    apu=dict(name=apu.name, property=apu.property, grid=list(APU_PCT)) if apu else None,
                    gearbox=dict(rpm_property=tr.rpm_property, nominal_rpm=tr.rpm_nominal or ac.main.rpm,
                                 full_rpm=gearbox_full_rpm(tr, ac),
                                 torque_properties=tr.torque_properties) if tr else None,
                    note="levels: dB re 20 uPa at the listener before the gain; turbine loops at 100 % Np; "
                         f"outside: on the ground {distance:g} m to the right", files=files)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "pywopwop-engines.xml").write_text(flightgear_engine_xml(manifest))
    return manifest


def flightgear_engine_xml(manifest, sound_dir="Sounds/pywopwop"):
    """FlightGear <fx> entries: per engine and layer, the loops cross-faded by equal power over its n1, each pitched
    by n1 over its own (the turbine's by n2 over 100 %; the core's not: the combustor sets its spectrum). The lit
    loops while the engine runs, the cold ones while it doesn't, the starter's while it cranks; the core's with the
    fuel flow over its loop's (combustion noise's amplitude goes with the heat release at a given Ng). The APU's by
    its speed. Inside, the shut and the all-open cockpit's loops by how much of the doors and windows is open."""
    from .wind import _e, _v, _ramp_e
    have = {(f["engine"], f["layer"], f["state"], f["ng"], f["observer"]): f for f in manifest["files"]}
    dist = manifest["exterior_distance_m"]
    views = ["cockpit"] + (["cockpit_open"] if any(f["observer"] == "cockpit_open" for f in manifest["files"])
                           else []) + ["side"]
    openings = manifest.get("openings") if "cockpit_open" in views else None
    if openings:
        from .wind import open_weights
        shut_w, open_w = open_weights(openings)

    def doors(view):
        """the shut and the all-open cockpit's loops by how much of the doors and windows is open"""
        if not openings or not view.startswith("cockpit"):
            return []
        return [open_w if view == "cockpit_open" else shut_w]
    out = []
    mode = f"{manifest['mode']}-" if manifest.get("mode") else ""

    def series(who, layer, state, grid, view, speed, extra, lo0=0.0, pitch="n1", n2=None):
        """one layer's loops over the speed (an expression, percent): equal-power cross-fades, each pitched by the
        speed over its own (pitch 'n2': by n2 over 100 %; None: not pitched)"""
        pts = [n for n in grid if (who, layer, state, n, view) in have]
        for i, n in enumerate(pts):
            f = have[(who, layer, state, n, view)]
            lo = pts[i - 1] if i else lo0
            w = [_ramp_e(speed, lo, n)]
            hi = pts[i + 1] if i + 1 < len(pts) else None
            if hi is not None:
                w.append(_ramp_e(speed, hi, n))
            inside = view.startswith("cockpit")
            vol = _e("product", _e("sqrt", _e("product", *w) if len(w) > 1 else w[0]), *extra(f), *doors(view))
            cond = [("<property>sim/current-view/internal</property>" if inside
                     else "<not><property>sim/current-view/internal</property></not>"),
                    f"<greater-than>{speed}<value>{lo:g}</value></greater-than>"]
            if hi is not None:
                cond.append(f"<less-than>{speed}<value>{hi:g}</value></less-than>")
            if pitch == "n2":
                pt = f"\t\t\t<pitch>{n2}<factor>0.01</factor><min>0.05</min><max>1.3</max></pitch>\n"
            elif pitch == "n1":
                pt = (f"\t\t\t<pitch>{speed}<factor>{1.0 / n:.6g}</factor><min>{lo / n:.4g}</min>"
                      f"<max>{(hi or 1.3 * n) / n:.4g}</max></pitch>\n")
            else:
                pt = ""
            out.append(f"\t\t<engine>\n\t\t\t<name>pww-{mode}{who}-{layer}-{state}-{n:g}-{view}</name>\n"
                       f"\t\t\t<mode>looped</mode>\n\t\t\t<path>{sound_dir}/{f['file']}</path>\n"
                       f"\t\t\t<condition><and>{''.join(cond)}</and></condition>\n"
                       f"\t\t\t<volume><expression>{vol}</expression></volume>\n"
                       + pt
                       + ("" if inside else f"\t\t\t<reference-dist>{dist:g}</reference-dist>\n"
                                            "\t\t\t<max-dist>3000</max-dist>\n")
                       + "\t\t</engine>\n")
    for eng in manifest["engines"]:
        P, who = eng["properties"], eng["name"]
        n1 = f"<property>{P}/n1-pct</property>"
        n2 = f"<property>{eng.get('n2_property') or P + '/n2-pct'}</property>"
        run = f"<property>{P}/running</property>"
        ff = f"<property>{P}/fuel-flow-pph</property>"
        crank = f"<property>{eng['starter_property']}</property>"
        lit_lo = eng["lightoff_ng"] - 5.0
        for view in views:
            lit = [run]
            cold = [_e("difference", _v(1.0), run)]
            series(who, "compressor", "lit", manifest["lit_ng"], view, n1, lambda f: lit, lit_lo)
            series(who, "compressor", "cold", manifest["cold_ng"], view, n1, lambda f: cold)
            series(who, "core", "lit", manifest["lit_ng"], view, n1,
                   lambda f: lit + [_e("clip", _e("div", ff, _v(max(f["fuel_pph"], 1.0))),
                                       "<clipMin>0</clipMin><clipMax>3</clipMax>")], lit_lo, pitch=None)
            series(who, "turbine", "lit", manifest["lit_ng"], view, n1, lambda f: lit, lit_lo, pitch="n2", n2=n2)
            series(who, "starter", "starter", manifest["starter_ng"], view, n1, lambda f: [crank])
    gb = manifest.get("gearbox")
    gfile = have.get(("gearbox", "gearbox", "run", 100.0, "cockpit"))
    if gb and gfile:
        # inside (the closed and the open cockpit alike: it is heard through the roof), with the engines' mean torque
        rpm = f"<property>{gb['rpm_property']}</property>"
        tq = _e("div", _e("sum", *[f"<property>{p}</property>" for p in gb["torque_properties"]]),
                _v(100.0 * max(len(gb["torque_properties"]), 1)))
        vol = _e("clip", tq, "<clipMin>0.03</clipMin><clipMax>1</clipMax>")
        out.append(f"\t\t<engine>\n\t\t\t<name>pww-{mode}gearbox-cockpit</name>\n\t\t\t<mode>looped</mode>\n"
                   f"\t\t\t<path>{sound_dir}/{gfile['file']}</path>\n"
                   f"\t\t\t<condition><and><property>sim/current-view/internal</property><greater-than>{rpm}"
                   f"<value>{0.05 * gb['nominal_rpm']:.3g}</value></greater-than></and></condition>\n"
                   f"\t\t\t<volume><expression>{_e('product', vol, _e('clip', _e('div', rpm, _v(gb.get('full_rpm', gb['nominal_rpm']))), '<clipMin>0</clipMin><clipMax>1</clipMax>'))}</expression></volume>\n"
                   f"\t\t\t<pitch>{rpm}<factor>{1.0 / gb['nominal_rpm']:.6g}</factor><min>0.05</min><max>1.3</max></pitch>\n"
                   "\t\t</engine>\n")
    apu = manifest.get("apu")
    if apu:
        for view in views:
            series("apu", "apu", "run", apu["grid"], view, f"<property>{apu['property']}</property>", lambda f: [])
    return ("<?xml version=\"1.0\"?>\n<!-- generated by pywopwop: the engines of " + manifest["aircraft"]
            + "; merge into the aircraft's sound file -->\n<PropertyList>\n\t<fx>\n" + "".join(out)
            + "\t</fx>\n</PropertyList>\n")
