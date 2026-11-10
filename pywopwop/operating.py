"""From an aircraft and a flight condition to each rotor's operating point, loads and frame.

Heading frame H: x forward (level), y left, z up, moving with the aircraft. The lifting rotors' tip-path planes are
perpendicular to the force that balances weight and airframe drag; the body hangs with the main rotor shaft along
it (no hub moment), which sets its pitch attitude. The tail rotor carries the main rotor's torque.

Each rotor's canonical frame (see aero.py) is given by a matrix M whose rows are its x, y, z axes in H, with the
y row negated for a clockwise rotor so that every rotor is computed turning anticlockwise: a mirror image.
"""
from dataclasses import dataclass

import numpy as np

from .aero import Fountain, AeroOptions, solve_rotor
from .atmosphere import G0, Atmosphere
from .model import Aircraft, FlightCondition


@dataclass
class RotorPoint:
    rotor: object
    omega: float
    V: float                    # air speed relative to the aircraft, m/s
    alpha_tpp: float            # rad
    thrust: float               # N
    M: np.ndarray               # 3x3, heading frame -> canonical rotor frame
    hub: np.ndarray             # hub position in the heading frame, m
    U: np.ndarray               # air velocity relative to the rotor, canonical frame
    aero: object = None         # AeroSolution
    ground_height: float = None     # hub above the ground, m (in ground effect)
    recirculation_rms: float = 0.0  # rms turbulence of the downwash recirculating through the rotor near the ground

    def to_rotor(self, p_heading):
        """A heading-frame position (relative to the aircraft origin) in this rotor's canonical frame."""
        return self.M @ (np.asarray(p_heading, float) - self.hub)

    def vec_to_rotor(self, v_heading):
        return self.M @ np.asarray(v_heading, float)


@dataclass
class OperatingState:
    aircraft: Aircraft
    condition: FlightCondition
    atm: Atmosphere
    pitch: float                # body pitch attitude, rad
    tpp_pitch: float            # main rotor tip-path plane pitch, rad (nose down negative)
    points: list
    velocity: np.ndarray        # aircraft velocity, heading frame

    def point(self, name):
        return next(p for p in self.points if p.rotor.name == name)


def _pitch_matrix(theta):
    """Body -> heading for a nose-up pitch theta (x forward, y left, z up)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _frame(z_axis, U_h, fallback_aft, spin):
    z = z_axis / np.linalg.norm(z_axis)
    up = U_h - np.dot(U_h, z) * z
    if np.linalg.norm(up) > 0.05:
        xv = up / np.linalg.norm(up)
    else:
        xv = fallback_aft - np.dot(fallback_aft, z) * z
        if np.linalg.norm(xv) < 1e-6:
            xv = np.cross(z, [0.0, 1.0, 0.0])
        xv = xv / np.linalg.norm(xv)
    yv = np.cross(z, xv)
    return np.array([xv, spin * yv, z])


def operate_airplane(aircraft, cond, opts):
    """A tiltrotor in airplane mode (nacelles 0): the wing carries the weight, the rotors pull the drag. The angle of
    attack and each rotor's thrust from the aircraft's level-flight trims at the speed with the same dynamic pressure
    at the trims' altitude (a level trim holds by dynamic pressure, not true airspeed); in a climb or descent the
    weight's component along the path is added to the thrust, the angle of attack kept. The rotors' shafts along the
    body's x axis, their tip-path planes normal to them."""
    if not aircraft.airplane_trim or not aircraft.airplane_rpm:
        raise ValueError(f"{aircraft.name} has no airplane mode (airplane_trim, airplane_rpm)")
    if cond.nacelle_deg != 0.0:
        raise ValueError("between helicopter and airplane mode (nacelles 0 < x < 90) is not modelled")
    atm = Atmosphere(cond.altitude_m, cond.dT, cond.humidity)
    tab = np.asarray(aircraft.airplane_trim, float)
    v_trim = cond.airspeed_kt * np.sqrt(atm.rho / Atmosphere(aircraft.airplane_trim_m).rho)
    if not tab[0, 0] <= v_trim <= tab[-1, 0]:
        raise ValueError(f"airplane mode trims cover {tab[0, 0]:g}-{tab[-1, 0]:g} kt at {aircraft.airplane_trim_m:g} m")
    mass = cond.mass_kg or aircraft.mass_kg
    W = mass * G0 * cond.load_factor
    vel = cond.velocity
    V = np.linalg.norm(vel)
    alpha = np.radians(np.interp(v_trim, tab[:, 0], tab[:, 1]))
    lifting = [r for r in aircraft.rotors if r.role in ("main", "lift")]
    thrust = np.interp(v_trim, tab[:, 0], tab[:, 2]) + W * np.sin(cond.gamma) / (len(lifting) * np.cos(alpha))
    if thrust <= 0:
        raise ValueError("the descent is too steep: the rotors would windmill")
    pitch = cond.gamma + alpha
    Rbh = _pitch_matrix(pitch)
    U_h = -vel
    aft_h = Rbh @ np.array([-1.0, 0.0, 0.0])
    omega = aircraft.airplane_rpm * cond.rpm_percent / 100.0 * 2.0 * np.pi / 60.0
    points = []
    for rot in lifting:
        hub_b, axis_b = rot.tilted(0.0)
        M = _frame(Rbh @ axis_b, U_h, aft_h, rot.spin_sign)
        Uc = M @ U_h
        a = np.arcsin(np.clip(Uc[2] / V, -1.0, 1.0))
        pt = RotorPoint(rot, omega, V, a, thrust, M, Rbh @ hub_b, Uc)
        pt.aero = solve_rotor(rot, omega, V, a, thrust, atm.rho, atm.c, atm.nu, opts)
        points.append(pt)
    return OperatingState(aircraft, cond, atm, pitch, pitch, points, vel)


def _fountain(aircraft, rot, other, M, Rbh, thrust, rho, V):
    """The wing under a tiltrotor's disc in the rotor's frame (nacelles vertical), None without one: the downwash it
    turns back up through the discs, gone with the download by 15 m/s, as the wake is swept off the wing"""
    if not aircraft.wing or other is None:
        return None
    w = np.asarray(aircraft.wing, float)
    hub = np.asarray(rot.position, float)
    y = max(abs(hub[1]) - 0.75 * rot.radius, w[0, 0])
    le, chord, z = (float(np.interp(y, w[:, 0], w[:, k])) for k in (1, 2, 3))
    to_other = M @ (Rbh @ (np.asarray(other.position, float) - hub))
    aft = M @ (Rbh @ np.array([-1.0, 0.0, 0.0]))
    psi = float(np.arctan2(to_other[1], to_other[0]))
    v_hover = float(np.sqrt(thrust / (2.0 * rho * np.pi * rot.radius ** 2)))
    return Fountain(psi_inboard=psi, aft_sign=float(np.sign(-np.sin(psi) * aft[0] + np.cos(psi) * aft[1])),
                    mid_chord_ahead=le - 0.5 * chord - float(hub[0]), leading_edge_ahead=le - float(hub[0]),
                    chord=chord, clearance=abs(z - float(hub[2])), v_hover=v_hover,
                    strength=max(0.0, 1.0 - V / 15.0))


def _at_collective(rot, omega, theta75, rho, c, nu, opts, ground_height, w_ref):
    """A rotor in still air at the collective theta75 (rad, at 0.75 R): the thrust for it by the secant method on the
    trim's collective"""
    from dataclasses import replace
    # (and a rotor that sets signed_hover_inflow gets it here, on the ground at flat pitch, and nowhere else)
    opts = replace(opts, bvi=False, signed_hover_inflow=True)
    w_ref = w_ref * (omega / rot.omega) ** 2        # what the rotor can make at this speed
    ts = [0.05 * w_ref, 0.2 * w_ref]
    sols = [solve_rotor(rot, omega, 0.0, 0.0, t, rho, c, nu, opts, ground_height) for t in ts]
    for _ in range(12):
        (t0, t1), (s0, s1) = ts[-2:], sols[-2:]
        e0, e1 = s0.theta0 - theta75, s1.theta0 - theta75
        if abs(e1) < np.radians(0.005) or e1 == e0:
            break
        t2 = max(t1 - e1 * (t1 - t0) / (e1 - e0), 1e-3 * w_ref)
        ts.append(t2)
        sols.append(solve_rotor(rot, omega, 0.0, 0.0, t2, rho, c, nu, opts, ground_height))
    return sols[-1]


def operate(aircraft, condition=None, opts=None, rpm_scale=None):
    """Trim the aircraft in the condition and solve every rotor's loads."""
    cond = condition or FlightCondition()
    opts = opts or AeroOptions()
    if cond.nacelle_deg < 90.0:
        return operate_airplane(aircraft, cond, opts)
    atm = Atmosphere(cond.altitude_m, cond.dT, cond.humidity)
    rho, c, nu = atm.rho, atm.c, atm.nu
    mass = cond.mass_kg or aircraft.mass_kg
    W = mass * G0 * cond.load_factor
    vel = cond.velocity
    V = np.linalg.norm(vel)
    D = 0.5 * rho * V ** 2 * aircraft.drag_area_m2
    hover_w = max(0.0, 1.0 - V / 15.0)
    F = np.array([0.0, 0.0, W * (1.0 + aircraft.download * hover_w)])
    if V > 0.1:
        F = F + D * vel / V
    n = F / np.linalg.norm(F)
    tpp_pitch = np.arctan2(-n[0], n[2])
    main = aircraft.main
    shaft_tilt = np.arctan2(main.axis[0], main.axis[2])
    pitch = tpp_pitch + shaft_tilt
    ground_run = cond.collective_deg is not None
    if ground_run:
        # standing on the gear, level: the tip-path planes along the shafts, the thrust the collective gives
        if V > 0.1:
            raise ValueError("a ground run (collective_deg) is in still air")
        pitch, tpp_pitch = 0.0, -shaft_tilt
        n = np.asarray(main.axis, float)
    Rbh = _pitch_matrix(pitch)
    U_h = -vel
    aft_h = Rbh @ np.array([-1.0, 0.0, 0.0])
    lifting = [r for r in aircraft.rotors if r.role in ("main", "lift")]
    points = []
    rpm_frac = cond.rpm_percent / 100.0
    for rot in lifting:
        omega = rot.omega * rpm_frac
        M = _frame(n, U_h, aft_h, rot.spin_sign)
        Uc = M @ U_h
        alpha = np.arcsin(np.clip(Uc[2] / V, -1.0, 1.0)) if V > 0.1 else 0.0
        thrust = np.linalg.norm(F) / len(lifting)
        pt = RotorPoint(rot, omega, V, alpha, thrust, M, Rbh @ np.asarray(rot.position), Uc)
        if cond.ground_height_m is not None:
            pt.ground_height = cond.ground_height_m + pt.hub[2]
        if ground_run:
            pt.aero = _at_collective(rot, omega, np.radians(cond.collective_deg), rho, c, nu, opts, pt.ground_height,
                                     W / len(lifting))
            pt.thrust = thrust = pt.aero.thrust
        else:
            other = next((r for r in lifting if r is not rot), None)
            pt.aero = solve_rotor(rot, omega, V, alpha, thrust, rho, c, nu, opts, pt.ground_height,
                                  _fountain(aircraft, rot, other, M, Rbh, thrust, rho, V))
        if pt.ground_height is not None:
            # the downwash spreads over the ground and comes back up through the rotor: turbulence of an assumed
            # 20 % of the hover induced velocity next to the ground, gone by a hub height of three radii, and
            # swept away with forward speed
            vh = np.sqrt(thrust / (2.0 * rho * np.pi * rot.radius ** 2))
            g = np.clip(1.5 - 0.5 * pt.ground_height / rot.radius, 0.0, 1.0)
            pt.recirculation_rms = float(0.2 * vh * g / (1.0 + (V / (2.0 * vh)) ** 2))
        points.append(pt)
    tail = aircraft.tail
    if tail is not None:
        mp = points[0]
        a_b = np.asarray(tail.axis)
        p_b = np.asarray(tail.position) - np.asarray(main.position)
        arm = p_b[0] * a_b[1] - p_b[1] * a_b[0]           # yaw moment about +z per newton of tail thrust
        need = main.spin_sign * mp.aero.torque              # the reaction on the fuselage is -spin * Q
        thrust_tr = abs(need / arm) if abs(arm) > 1e-6 else 0.0
        omega = tail.omega * rpm_frac
        axis_h = Rbh @ a_b
        M = _frame(axis_h, U_h, aft_h, tail.spin_sign)
        Uc = M @ U_h
        alpha = np.arcsin(np.clip(Uc[2] / V, -1.0, 1.0)) if V > 0.1 else 0.0
        pt = RotorPoint(tail, omega, V, alpha, thrust_tr, M, Rbh @ np.asarray(tail.position), Uc)
        pt.aero = solve_rotor(tail, omega, V, alpha, thrust_tr, rho, c, nu,
                              AeroOptions(n_radial=max(16, opts.n_radial // 2), n_azimuth=opts.n_azimuth, bvi=False,
                                          elastic=opts.elastic))
        points.append(pt)
    return OperatingState(aircraft, cond, atm, pitch, tpp_pitch, points, vel)
