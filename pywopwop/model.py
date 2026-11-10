"""The rotorcraft description: airfoils, rotors, the aircraft, flight conditions and observers.

Aircraft frame: x forward, y left, z up (right-handed), origin at the main rotor hub. Units are SI except where a
name says otherwise (deg, rpm, kt, fpm).
"""
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

import numpy as np

KT = 0.514444
FPM = 0.00508


@dataclass
class Airfoil:
    """Section aerodynamics (a simple, smooth model good through stall and reverse flow) and the shape and
    trailing-edge details the noise models need."""
    name: str = "NACA 0012"
    thickness: float = 0.12          # t/c, of a NACA 4-digit symmetric thickness distribution
    alpha0_deg: float = 0.0          # zero-lift angle
    cm0: float = 0.0                 # pitching moment about the quarter chord at zero lift
    lift_slope: float = 5.9          # per rad, incompressible
    cl_max: float = 1.4              # at low Mach number
    cd0: float = 0.008
    cd2: float = 0.4                 # per rad^2
    mach_dd: float = 0.78            # drag divergence Mach number at zero lift
    te_thickness: float = 0.002      # trailing-edge bluntness h/c (BPM)
    te_angle_deg: float = 14.0       # trailing-edge solid angle (BPM)
    trip: int = 1                    # BPM boundary layer: 0 untripped, 1 tripped, 2 lightly tripped
    table: str = None                # section table (presets/airfoils/<name>.json): Cl, Cd (and Cm) vs alpha, Mach
    shape: str = None                # section shape: 'naca23012', 'naca0012', ... or a coordinate file

    def __post_init__(self):
        if self.table:
            from .airfoils import load_table
            self._table = load_table(self.table)
            # the smooth model's parameters follow the table (used for estimates: trim start, Lock-number mass,
            # the elastic iteration's Jacobian)
            self.lift_slope, self.alpha0_deg = self._table.slope_and_zero()
        else:
            self._table = None
        if self.shape:
            from .airfoils import section_shape
            self._half, self._camber, self.thickness = section_shape(self.shape)
        else:
            self._half, self._camber = None, None

    def coefficients(self, alpha, mach):
        """Lift and drag coefficients for angle of attack alpha (rad, any range) and Mach number."""
        if self._table is not None:
            return self._table(alpha, mach, "cl"), self._table(alpha, mach, "cd")
        alpha = np.asarray(alpha, float)
        mach = np.minimum(np.asarray(mach, float), 0.95)
        beta = np.sqrt(np.maximum(1.0 - np.minimum(mach, 0.9) ** 2, 0.19))
        a = self.lift_slope / beta
        a_rel = alpha - np.radians(self.alpha0_deg)
        cl_max = self.cl_max * np.clip(1.25 - 0.6 * mach, 0.55, 1.0)
        # attached flow, saturated smoothly at cl_max
        x = a * np.sin(a_rel) / cl_max
        cl_att = cl_max * x / (1.0 + np.abs(x) ** 8) ** 0.125
        # past the stall and in reverse flow: flat plate
        cl_fp = 1.05 * np.sin(2.0 * a_rel)
        cd_fp = 0.02 + 1.9 * np.sin(a_rel) ** 2
        stall_angle = cl_max / a
        w = 1.0 / (1.0 + np.exp((np.abs(np.sin(a_rel)) - np.sin(stall_angle) - 0.05) / 0.02))
        w = np.where(np.cos(a_rel) < 0.0, 0.0, w)
        cl = w * cl_att + (1.0 - w) * cl_fp
        m_crit = self.mach_dd - 0.1 - 0.1 * np.abs(cl_att)
        cd_wave = 20.0 * np.maximum(mach - m_crit, 0.0) ** 4
        cd = w * (self.cd0 + self.cd2 * np.sin(a_rel) ** 2 + cd_wave) + (1.0 - w) * cd_fp
        return cl, cd

    def moment(self, alpha, mach):
        """Pitching moment coefficient about the quarter chord."""
        if self._table is not None and self._table.has("cm"):
            return self._table(alpha, mach, "cm")
        return np.full_like(np.asarray(alpha, float), self.cm0)

    def half_thickness(self, x):
        """Half thickness / chord at x/c: the section's shape, or a NACA 4-digit distribution of its thickness."""
        x = np.asarray(x, float)
        if self._half is not None:
            return self._half(x)
        return 5.0 * self.thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3
                                       - 0.1036 * x ** 4)

    def camber(self, x):
        """Mean line / chord at x/c."""
        x = np.asarray(x, float)
        return self._camber(x) if self._camber is not None else np.zeros_like(x)


def _table(spec, default):
    """A constant, or a [[r/R, value], ...] table, as a function of r/R."""
    if spec is None:
        spec = default
    if np.isscalar(spec):
        return lambda x: np.full_like(np.asarray(x, float), float(spec))
    tab = np.asarray(spec, float)
    return lambda x: np.interp(x, tab[:, 0], tab[:, 1])


@dataclass
class Rotor:
    name: str = "main"
    radius: float = 5.0
    blades: int = 4
    rpm: float = 400.0
    chord: object = 0.3              # m, or [[r/R, chord], ...]
    twist: object = -8.0             # deg: linear twist over the radius (root to tip), or [[r/R, deg], ...]
    root_cutout: float = 0.2         # r/R where the aerodynamic blade starts
    airfoil: Airfoil = field(default_factory=Airfoil)
    rotation: str = "ccw"            # seen from the side the thrust points to (main rotors: from above)
    hub: str = "articulated"         # articulated, hingeless or teetering (sets the coning)
    precone_deg: float = 0.0
    flap_frequency: float = 1.0      # rotating first flap frequency, per rev (hingeless rotors above 1)
    lock_number: float = 8.0
    tip_sweep_deg: float = 0.0       # aft sweep of the tip
    tip_sweep_start: float = 1.0     # r/R where the sweep starts
    tip_loss: float = 0.97           # Prandtl tip loss is used; this is the vortex release radius r/R
    index_deg: float = 0.0           # its blades' azimuth behind the first rotor's, on a shaft tying them together
    round_tip: bool = False
    position: tuple = (0.0, 0.0, 0.0)          # hub, aircraft frame
    axis: tuple = (0.0, 0.0, 1.0)              # thrust direction, aircraft frame
    role: str = "main"               # main, tail, lift (multirotor / tiltrotor)
    pivot: tuple = None              # tiltrotors: the nacelle's conversion axis (along y), aircraft frame; position and
                                     # axis are the helicopter mode's (nacelle 90 deg) and turn about it
    tail_top_aft: bool = True        # tail rotors: the blade at the top moves aft (sets the rotation)
    cyclic: bool = True              # the tip-path plane is trimmed (flapping and cyclic); False: a rigid propeller
    # elastic blade (structure.py): flap modes from a rotating beam, one torsion mode
    elastic: bool = True
    flap2_frequency: float = 2.7     # per rev, first elastic flap mode of an articulated blade
    coning_frequency: float = 1.3    # per rev, a teetering rotor's collective (coning) mode
    clamp_radius: float = 0.05       # r/R of the clamped root (hingeless, teetering, propellers)
    torsion_frequency: float = 4.5   # per rev, first torsion mode with the control system
    torsion_rigid_fraction: float = 0.25   # share of the torsion mode that is rigid pitch (control flexibility)
    torsion_gyration: float = 0.25   # radius of gyration about the elastic axis / chord
    elastic_axis: float = 0.25       # x/c
    cg_axis: float = 0.25            # x/c of the section centre of mass
    structural_damping: float = 0.01
    torsion_damping: float = 0.1     # structural plus aerodynamic, fraction of critical
    blade_mass_per_length: float = None    # kg/m; default from the Lock number
    flat_pitch_deg: float = None     # blade pitch at 0.75 R with the collective fully down (ground runs)
    signed_hover_inflow: bool = False  # annuli loaded downward in a hover get the mirrored momentum balance (aero._hover_inflow)

    def __post_init__(self):
        if isinstance(self.airfoil, dict):
            self.airfoil = Airfoil(**self.airfoil)
        self._chord = _table(self.chord, 0.3)
        if np.isscalar(self.twist):
            tw = float(self.twist)
            self._twist = lambda x: tw * (np.asarray(x, float) - 0.75)
        else:
            tab = np.asarray(self.twist, float)
            t75 = np.interp(0.75, tab[:, 0], tab[:, 1])
            self._twist = lambda x: np.interp(x, tab[:, 0], tab[:, 1]) - t75
        a = np.asarray(self.axis, float)
        self.axis = tuple(a / np.linalg.norm(a))
        self.position = tuple(float(v) for v in self.position)
        if self.pivot is not None:
            self.pivot = tuple(float(v) for v in self.pivot)

    def tilted(self, nacelle_deg):
        """hub position and thrust axis (aircraft frame) with the nacelles at nacelle_deg (90: as given), turned
        forward about the pivot"""
        if self.pivot is None or nacelle_deg >= 90.0:
            return np.asarray(self.position), np.asarray(self.axis)
        t = np.radians(90.0 - nacelle_deg)
        R = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
        pv = np.asarray(self.pivot)
        return pv + R @ (np.asarray(self.position) - pv), R @ np.asarray(self.axis)

    @property
    def omega(self):
        return self.rpm * 2.0 * np.pi / 60.0

    @property
    def tip_speed(self):
        return self.omega * self.radius

    def chord_at(self, x):
        return self._chord(x)

    def twist_at(self, x):
        """Twist, rad, relative to 0.75 R."""
        return np.radians(self._twist(x))

    @property
    def solidity(self):
        x = np.linspace(self.root_cutout, 1.0, 101)
        c = self.chord_at(x)
        c_eq = np.trapezoid(c * x ** 2, x) / np.trapezoid(x ** 2, x)   # thrust-weighted
        return self.blades * c_eq / (np.pi * self.radius)

    @property
    def spin_sign(self):
        """+1 when the rotor turns anticlockwise about its thrust axis (right-hand rule), else -1."""
        if self.role == "tail":
            # the top blade's velocity for +1 is axis x up; its sign along x forward decides
            a = np.asarray(self.axis)
            top_velocity_ccw = np.cross(a, [0.0, 0.0, 1.0])
            return -1 if (top_velocity_ccw[0] > 0) == self.tail_top_aft else 1
        return 1 if self.rotation.lower() == "ccw" else -1


@dataclass
class Cabin:
    """The walls between the outside and the pilot's head, by area: the skin (with its frames and trim) and the
    windows as panels, each by its surface mass, coincidence frequency and loss factor; leaks; and the doors' opening.
    A panel's transmission loss is the field-incidence mass law, TL = 20 log10(m f) - 47 dB, below half its
    coincidence frequency, and Sharp's above it, TL = 20 log10(m f) + 10 log10(2 eta f / (pi f_c)) - 47 dB, joined
    linearly (in log frequency) between: the dip where the panel's bending waves match the sound's trace on it lets a
    turbine's whine through far more than the mass law says. The areas' transmission coefficients add (composite
    transmission): an open door's or window's area passes everything.

    Inside, the transmitted sound builds up in a diffuse field as the cockpit's absorption lets it (inside): p^2 = tau
    p_out^2 / alpha, the walls losing a share alpha of what meets them (absorbed, or let out through them; what is
    open, all of it). A trimmed cabin's blankets, trim, seats and crew absorb little below 250 Hz and most of it above
    1 kHz, so the lows build up and the highs don't, though the lowest leak back out through the walls: without it a cockpit is as bright as an anechoic one would be. Below the cockpit's lowest mode there
    is no diffuse field to build up (faded in over the octave below it)."""
    surface_mass: float = 8.0           # kg/m^2: skin, frames and trim
    coincidence_hz: float = None        # the skin's coincidence frequency (None: a limp wall, the mass law alone)
    loss_factor: float = 0.02
    window_share: float = 0.0           # windows' share of the walls round the cockpit
    window_mass: float = 7.0            # kg/m^2
    window_coincidence_hz: float = 6000.0
    window_loss_factor: float = 0.03
    leak: float = 1e-3                  # area share open with everything shut (seals, gaps): -30 dB
    openings: list = field(default_factory=list)    # [{name, share, property}]: what opens, its share of the
                                                    # walls' area, the FlightGear property of its opening (0..1)
    # the interior's absorption coefficient by frequency ([Hz, alpha]; empty: none, as anechoic): a trimmed cabin's
    # glass-fibre blankets behind trim panels, its seats and crew (estimated)
    absorption: list = field(default_factory=lambda: [[125.0, 0.15], [250.0, 0.30], [500.0, 0.55], [1000.0, 0.70],
                                                      [2000.0, 0.75], [4000.0, 0.75]])
    first_mode_hz: float = 70.0         # the cockpit's lowest acoustic mode, c / 2L for its 2.5 m length
    # a trimmed sidewall: the skin (surface_mass, its coincidence) and a trim panel of trim_mass kg/m^2 trim_gap m
    # inside it, the gap filled with blankets (a double wall); None: one wall of surface_mass
    trim_mass: float = None
    trim_gap: float = 0.075
    # the structure under the skin, for what reaches the cabin through it (wing.structure_borne): its frames' and
    # stringers' pitch (m), the panels between them radiating, and the built-up structure's loss factor (estimated)
    frame_pitch: float = 0.5
    stringer_pitch: float = 0.2
    structure_loss_factor: float = 0.03
    wing_mobility: float = None     # s/kg where a wing's loads enter the structure (None: the skin's, pi f_c / 4 m c^2)
    modes_seed: int = 0             # which draw of the cabin's modal response (synth.wall_field)

    @property
    def open_share(self):
        """the area share open with every opening open"""
        return float(sum(o["share"] for o in self.openings))

    @staticmethod
    def panel_tl(f, m, fc, eta):
        f = np.maximum(np.asarray(f, float), 1e-3)
        mass = 20.0 * np.log10(m * f) - 47.0
        if not fc:
            return np.maximum(mass, 0.0)
        sharp = mass + 10.0 * np.log10(2.0 * eta * f / (np.pi * fc))
        at_half = 20.0 * np.log10(m * 0.5 * fc) - 47.0
        at_fc = 20.0 * np.log10(m * fc) - 47.0 + 10.0 * np.log10(2.0 * eta / np.pi)
        u = np.clip(np.log2(f / (0.5 * fc)), 0.0, 1.0)
        tl = np.where(f < 0.5 * fc, mass, np.where(f < fc, at_half + u * (at_fc - at_half), sharp))
        return np.maximum(tl, 0.0)

    def opaque_tl(self, f, dip=True):
        """the opaque walls' transmission loss: the skin's, or with a trim panel the double wall's (Fahy): the mass
        law of both together below their mass-air-mass resonance f_0, then TL_skin + TL_trim + 20 log10(2 k d) up to
        c / 2 pi d, and TL_skin + TL_trim + 6 dB above (the blankets damping the gap's modes). dip: the skin's
        coincidence dip (sound's; a convected field's eddies meet none)"""
        f = np.maximum(np.asarray(f, float), 1e-3)
        skin = self.panel_tl(f, self.surface_mass, self.coincidence_hz if dip else None, self.loss_factor)
        if not self.trim_mass:
            return skin
        m1, m2, d = self.surface_mass, self.trim_mass, self.trim_gap
        both = self.panel_tl(f, m1 + m2, None, 0.0)
        tl2 = self.panel_tl(f, m2, None, 0.0)
        f0 = np.sqrt(1.2 * 340.0 ** 2 * (m1 + m2) / (d * m1 * m2)) / (2.0 * np.pi)
        fd = 340.0 / (2.0 * np.pi * d)
        mid = np.maximum(skin + tl2 + 20.0 * np.log10(np.maximum(2.0 * (2.0 * np.pi * f / 340.0) * d, 1e-9)), both)
        return np.where(f < f0, both, np.where(f < fd, mid, skin + tl2 + 6.0))

    def tau(self, f, doors_open=False):
        """the walls' transmission coefficient (power) at the frequencies f, with every opening open or shut"""
        skin = 10.0 ** (-self.opaque_tl(f) / 10.0)
        win = 10.0 ** (-self.panel_tl(f, self.window_mass, self.window_coincidence_hz, self.window_loss_factor) / 10.0)
        opening = self.leak + (self.open_share if doors_open else 0.0)
        return (1.0 - self.window_share - opening) * skin + self.window_share * win + opening

    def patch_tau(self, f, glazed, doors_open=False):
        """one wall patch's transmission of sound: the windows' (glazed) or the opaque walls', and what is open"""
        f = np.maximum(np.asarray(f, float), 1e-3)
        T = (10.0 ** (-self.panel_tl(f, self.window_mass, self.window_coincidence_hz, self.window_loss_factor) / 10.0)
             if glazed else 10.0 ** (-self.opaque_tl(f) / 10.0))
        opening = self.leak + (self.open_share if doors_open else 0.0)
        return (1.0 - opening) * T + opening

    def convected(self, f, speed, glazed=None, c0=340.0):
        """what a wall passes of a pressure field convected along it at `speed` (m/s), a boundary layer's eddies or a
        passing blade's footprint: by the mass law (the opaque walls' as a double wall with a trim), coupled poorly
        below its hydrodynamic coincidence f_c (speed / c)^2, where its bending waves are slower than the field (a
        limp panel couples fully); no leaks. glazed: a window (True), the opaque walls (False), or both by their
        shares (None)"""
        f = np.maximum(np.asarray(f, float), 1e-3)

        def coupling(fc):
            if not fc:
                return 1.0
            r = (f / (fc * (speed / c0) ** 2)) ** 2
            return r / (1.0 + r)
        win = coupling(self.window_coincidence_hz) * 10.0 ** (-self.panel_tl(f, self.window_mass, None, 0.0) / 10.0)
        opaque = coupling(self.coincidence_hz) * 10.0 ** (-self.opaque_tl(f, dip=False) / 10.0)
        if glazed is None:
            return (1.0 - self.window_share) * opaque + self.window_share * win
        return win if glazed else opaque

    def alpha(self, f, doors_open=False):
        """the share of the sound's energy lost at each reflection inside: what the interior absorbs and what the walls
        let out (their transmission, what is open all of it), which at low frequencies is the most of it; 1 without
        an absorption table"""
        f = np.maximum(np.asarray(f, float), 1e-3)
        if not self.absorption:
            return np.ones_like(f)
        t = np.asarray(self.absorption, float)
        a = np.interp(np.log(f), np.log(t[:, 0]), t[:, 1])
        out = self.tau(f, doors_open)
        return a * (1.0 - out) + out

    def room(self, f, doors_open=False):
        """the interior's build-up of the transmitted power, 1 / alpha, faded in from half the lowest mode to it"""
        f = np.maximum(np.asarray(f, float), 1e-3)
        return self.alpha(f, doors_open) ** -np.clip(np.log2(f / (0.5 * self.first_mode_hz)), 0.0, 1.0)

    def direct(self, f, r, area, total_area, doors_open=False):
        """a wall patch's direct field at a listener r away (m), over its share of the diffuse field: by Sabine's
        room, p^2 = W rho c (1 / (2 pi r^2) + 4 / (S alpha)), which for a patch of area A passing tau of the
        outside's intensity is A / (8 pi r^2) against A / (S alpha); r no closer than half the patch's size, and none
        below the lowest mode, where the cabin is one pressure chamber"""
        f = np.maximum(np.asarray(f, float), 1e-3)
        r = max(float(r), 0.5 * np.sqrt(area))
        g = total_area * self.alpha(f, doors_open) / (8.0 * np.pi * r * r)
        return g * np.clip(np.log2(f / (0.5 * self.first_mode_hz)), 0.0, 1.0)

    def inside(self, f, doors_open=False):
        """the sound's power at the pilot over the outside's: the walls' transmission and the interior's build-up"""
        return self.tau(f, doors_open) * self.room(f, doors_open)

    def near_field_share(self, f, speed, c0=340.0, glazed=None):
        """what the walls (or one patch: glazed True / False) pass of a convected field over what all of them pass of
        sound (the cabin filter's), at most what the patch passes of sound"""
        f = np.maximum(np.asarray(f, float), 1e-3)
        sound = self.tau(f) if glazed is None else self.patch_tau(f, glazed)
        return np.minimum(self.convected(f, speed, glazed, c0), sound) / np.maximum(self.tau(f), 1e-300)


@dataclass
class Aircraft:
    name: str = "aircraft"
    mass_kg: float = 2500.0
    drag_area_m2: float = 1.2        # equivalent flat plate area of the airframe
    download: float = 0.03           # fuselage download in the hover, fraction of weight
    cockpit: tuple = (2.5, -0.4, -1.5)       # pilot's head, aircraft frame
    hub_height_m: float = 3.0       # main rotor hub above the ground, standing on its gear
    rotors: list = field(default_factory=list)
    description: str = ""
    # tiltrotors in airplane mode (nacelles 0): the wing carries the weight and the rotors pull the airframe's drag.
    # The level-flight trims at mass_kg, [[true airspeed kt, angle of attack deg, thrust N of each lifting rotor], ...]
    # (e.g. from a flight model), and the rotor speed (rpm)
    airplane_trim: list = None
    airplane_trim_m: float = 0.0    # the altitude those trims are at, m
    airplane_rpm: float = None
    windscreen_m: float = None      # the windscreen's distance aft of the nose (m): the cockpit's wind (wind.py)
    cabin: Cabin = None             # the cockpit's walls (default: a limp 8 kg/m^2 wall with -30 dB of leaks)
    # the outer skin round the cockpit and the cabin open to it, by sections [x, half-width, floor z, roof z] and
    # optionally their squareness (a superellipse's exponent, 4 by default: higher boxier) (aircraft frame, m): inside,
    # the rotors are heard through the pressure on it (synth.wall_field); without it, at the listener's head
    fuselage: list = None
    # its windows on that skin, as boxes [x from, x to, z from, z to] (both sides): their patches pass through the
    # cabin's window panel, the rest through its opaque walls (without: every patch both, by window_share)
    glazing: list = None
    # a tiltrotor's wing, by sections [|y|, leading edge x, chord, chord line z, half thickness] (aircraft frame, m): in
    # airplane mode behind the discs, the blades' vortices sweep over it and its unsteady lift adds to their harmonics
    # (wing.py); the skin it sits on is out of the outside air
    wing: list = None

    def __post_init__(self):
        self.rotors = [r if isinstance(r, Rotor) else Rotor(**r) for r in self.rotors]
        self.cabin = Cabin(**self.cabin) if isinstance(self.cabin, dict) else (self.cabin or Cabin())

    @property
    def main(self):
        return next(r for r in self.rotors if r.role == "main")

    @property
    def tail(self):
        return next((r for r in self.rotors if r.role == "tail"), None)


@dataclass
class FlightCondition:
    airspeed_kt: float = 0.0
    climb_fpm: float = 0.0           # vertical speed, negative descending
    rpm_percent: float = 100.0
    mass_kg: float = None            # default: the aircraft's
    altitude_m: float = 0.0
    dT: float = 0.0                  # K, ISA deviation
    humidity: float = 0.7
    load_factor: float = 1.0
    ground_height_m: float = None    # main rotor hub above the ground (in ground effect); None: out of it
    nacelle_deg: float = 90.0        # tiltrotors: 90 helicopter mode, 0 airplane mode (the rpm then the aircraft's
                                     # airplane_rpm times rpm_percent)
    collective_deg: float = None     # on the ground (a run-up): the lifting rotors' pitch at 0.75 R, which sets their
                                     # thrust; the aircraft level on its gear, the air still

    @property
    def V(self):
        return self.airspeed_kt * KT

    @property
    def gamma(self):
        """Flight path angle, rad."""
        vz = self.climb_fpm * FPM
        V = max(self.V, 1e-6)
        if abs(vz) >= V:
            return np.copysign(np.pi / 2, vz)
        return float(np.arcsin(vz / V)) if self.V > 0.5 else (np.copysign(np.pi / 2, vz) if vz != 0 else 0.0)

    @property
    def velocity(self):
        """Aircraft velocity in the heading frame (x forward level, y left, z up), m/s."""
        vh = self.V * np.cos(self.gamma) if self.V > 0.5 else 0.0
        return np.array([vh, 0.0, self.climb_fpm * FPM])


def glazed_patches(P, boxes):
    """which wall patches (centres P, aircraft frame) lie in the glazing's boxes [x from, x to, z from, z to]"""
    P = np.asarray(P, float)
    g = np.zeros(len(P), bool)
    for x0, x1, z0, z1 in boxes or []:
        g |= (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 2] >= z0) & (P[:, 2] <= z1)
    return g


def wall_patches(sections, dx=0.5, n_around=8, offset=0.03):
    """The fuselage's skin round the cockpit and cabin in patches: from its sections [x, half-width, floor z, roof z]
    and optionally the section's squareness (linear between them), each slice dx long a rounded rectangle (a
    superellipse, of exponent 4 unless the sections give theirs: higher is boxier, its shoulders sharper) cut into
    n_around sectors of equal angle. Returns the patches' centres, offset m outside the skin (where the outside field
    is computed), and their areas (m^2)."""
    s = np.asarray(sorted(sections), float)
    if s.shape[1] < 5:
        s = np.c_[s, np.full(len(s), 4.0)]
    x0, x1 = s[0, 0], s[-1, 0]
    n = max(1, int(round((x1 - x0) / dx)))
    xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    th = np.linspace(-np.pi, np.pi, 64 * n_around + 1)
    pos, area = [], []
    for x in xs:
        hw, zf, zr, ex = (np.interp(x, s[:, 0], s[:, k]) for k in (1, 2, 3, 4))
        zc, b = 0.5 * (zf + zr), 0.5 * (zr - zf)
        c, sn = np.cos(th), np.sin(th)
        y = hw * np.sign(c) * np.abs(c) ** (2.0 / ex)
        z = zc + b * np.sign(sn) * np.abs(sn) ** (2.0 / ex)
        seg = np.hypot(np.diff(y), np.diff(z))
        for k in range(n_around):
            j = slice(64 * k, 64 * (k + 1))
            m = 64 * k + 32
            out = np.array([y[m], z[m] - zc])
            out /= max(np.linalg.norm(out), 1e-9)
            pos.append((x, y[m] + offset * out[0], z[m] + offset * out[1]))
            area.append(seg[j].sum() * (x1 - x0) / n)
    return np.array(pos), np.array(area)


@dataclass
class Observer:
    """A listener riding with the aircraft: direction from the main rotor hub in the aircraft's heading frame,
    azimuth from the nose (deg, positive to the right), elevation (deg, negative below), distance (m)."""
    name: str = "side"
    azimuth_deg: float = 90.0
    elevation_deg: float = -20.0
    distance_m: float = 100.0
    position: tuple = None           # explicit aircraft-frame position instead (the cockpit)

    def offset(self):
        if self.position is not None:
            return np.asarray(self.position, float)
        az = np.radians(self.azimuth_deg)
        el = np.radians(self.elevation_deg)
        # azimuth positive to the right = negative y in the aircraft frame
        return self.distance_m * np.array([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])


STANDARD_OBSERVERS = {
    "front": Observer("front", 0.0, -30.0),
    "front_right": Observer("front_right", 45.0, -25.0),
    "side": Observer("side", 90.0, -20.0),
    "rear": Observer("rear", 180.0, -30.0),
    "below": Observer("below", 0.0, -89.0),
    "left": Observer("left", -90.0, -20.0),
}


PRESET_DIR = Path(__file__).parent / "presets"


def list_presets():
    return sorted(p.stem for p in PRESET_DIR.glob("*.toml"))


def load_aircraft(name_or_path):
    """An aircraft from a preset name or a TOML file."""
    p = Path(name_or_path)
    if not p.suffix:
        p = PRESET_DIR / f"{name_or_path}.toml"
    if not p.exists():
        raise FileNotFoundError(f"no aircraft '{name_or_path}' (presets: {', '.join(list_presets())})")
    with open(p, "rb") as f:
        d = tomllib.load(f)
    ac = dict(d.get("aircraft", {}))
    if "cabin" in d:
        ac["cabin"] = dict(d["cabin"])
    rotors = []
    for r in d.get("rotor", []):
        r = dict(r)
        if "airfoil" in r:
            r["airfoil"] = Airfoil(**r["airfoil"])
        rotors.append(Rotor(**r))
    ac["rotors"] = rotors
    return Aircraft(**ac)
