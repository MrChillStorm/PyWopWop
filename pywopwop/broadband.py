"""Broadband rotor noise: airfoil self-noise (BPM) and turbulent inflow noise (Amiet) of every blade segment at
every azimuth, heard at an observer as a spectral time history over one revolution.

Each segment's local flow (speed, angle of attack) comes from the aeroelastic solution; its position, velocity and
the direction to the observer from the same retarded-time geometry as the tonal noise. The BPM directivity
functions take the observer's angles in the segment's frame (x along the flow over the chord, y along the span, z
normal); the spectrum is Doppler shifted to the observer, and each segment's mean-square pressure lands at its
reception time. The result shows the blade-passage modulation that makes a rotor's broadband noise swish.
"""
from dataclasses import dataclass

import numpy as np

from . import bpm


@dataclass
class BroadbandOptions:
    n_azimuth: int = 128            # samples of the revolution (per blade passage: n_azimuth / blades)
    turbulence_ms: float = 0.8      # rms turbulence velocity the blades meet (inflow noise), m/s
    turbulence_scale: float = 2.0   # integral length scale, m
    mechanisms: tuple = ("tbl", "blunt", "tip", "lbl", "inflow", "bwi")
    f_lo: float = 20.0
    f_hi: float = 20000.0


def air_turbulence(altitude_m):
    """The rms turbulence the blades meet (m/s, at BroadbandOptions' 2 m scale) by height: the surface layer's up to
    300 m, a quarter of it above 1000 m, in the smooth air over it (checked in an MV-22B's cruise at 10,000 ft: the
    proprotors' low harmonics stand as far out of their inflow noise as the recording's)."""
    return float(np.interp(altitude_m, [300.0, 1000.0], [0.8, 0.2]))


@dataclass
class SpectralHistory:
    t: np.ndarray                   # (Nt,) over one revolution
    f: np.ndarray                   # (Nf,) band centres
    msp: np.ndarray                 # (Nt, Nf) mean-square pressure per band, Pa^2 (all blades)
    period: float

    @property
    def mean_spl(self):
        return 10.0 * np.log10(np.maximum(np.mean(self.msp, axis=0), 1e-30) / 4e-10)

    @property
    def oaspl(self):
        return 10.0 * np.log10(np.mean(np.sum(self.msp, axis=1)) / 4e-10)


def spectral_history(point, observer, opts=None):
    """Broadband noise of a solved rotor (operating.RotorPoint) at an observer riding with it (canonical frame,
    relative to the hub at t = 0)."""
    opts = opts or BroadbandOptions()
    aero = point.aero
    rot = point.rotor
    af = rot.airfoil
    B = rot.blades
    omega = aero.omega
    c0 = aero.c
    nu = aero.nu
    npsi_a = len(aero.psi)
    nb = max(B, int(round(opts.n_azimuth / B)) * B)
    idx = (np.arange(nb) * npsi_a) // nb
    psi = aero.psi[idx]
    tau = psi / omega
    T = 2.0 * np.pi / omega
    R = aero.R
    r = aero.x * R
    U_t = aero.UT[idx]
    U_p = aero.UP[idx]
    U = np.sqrt(U_t ** 2 + U_p ** 2)
    alpha = np.degrees(np.abs(aero.alpha[idx]))
    alpha = np.minimum(alpha, 25.0)
    forward = U_t > 1.0             # reverse flow: no trailing-edge noise of the usual kind
    chord = aero.chord[None, :] * np.ones_like(U)
    span = (aero.dx * R)[None, :] * np.ones_like(U)
    cs, sn = np.cos(psi)[:, None], np.sin(psi)[:, None]
    Uh = np.asarray(point.U, float)
    zq = aero.z[idx]
    # segment (quarter chord, mid span) in the air-fixed frame, and its velocity
    yx = r[None, :] * cs - Uh[0] * tau[:, None]
    yy = r[None, :] * sn - Uh[1] * tau[:, None]
    yz = zq - Uh[2] * tau[:, None]
    vx = -Uh[0] - omega * r[None, :] * sn
    vy = -Uh[1] + omega * r[None, :] * cs
    vz = -Uh[2] + aero.zdot[idx]
    o = np.asarray(observer, float)
    vo = -Uh
    dx = o[0] + vo[0] * tau[:, None] - yx
    dy = o[1] + vo[1] * tau[:, None] - yy
    dz = o[2] + vo[2] * tau[:, None] - yz
    dv = dx * vo[0] + dy * vo[1] + dz * vo[2]
    a2 = c0 ** 2 - vo @ vo
    s = (dv + np.sqrt(dv ** 2 + a2 * (dx ** 2 + dy ** 2 + dz ** 2))) / a2
    rx, ry, rz = dx + vo[0] * s, dy + vo[1] * s, dz + vo[2] * s
    re = c0 * s
    hx, hy, hz = rx / re, ry / re, rz / re
    t_rec = tau[:, None] + s
    # observer angles in the segment frame: x along the flow over the chord (opposite the blade's motion),
    # y along the span, z normal to the disc
    ex_x, ex_y = sn, -cs
    ey_x, ey_y = cs, sn
    hxe = hx * ex_x + hy * ex_y
    hye = hx * ey_x + hy * ey_y
    hze = hz
    theta = np.degrees(np.arccos(np.clip(hxe, -1.0, 1.0)))
    phi = np.degrees(np.arctan2(np.abs(hze), hye))
    Ms_r = (vx * hx + vy * hy + vz * hz) / c0
    Mo_r = (vo[0] * hx + vo[1] * hy + vo[2] * hz) / c0
    doppler = (1.0 - Mo_r) / (1.0 - Ms_r)
    f_out = bpm.third_octave_bands(opts.f_lo, opts.f_hi)
    f_src = f_out[None, None, :] / doppler[..., None]
    Ue = np.maximum(U, 1.0)
    msp = np.zeros(U.shape + (len(f_out),))
    mech = set(opts.mechanisms)
    p2 = lambda spl: 10.0 ** (spl / 10.0) * 4e-10
    with np.errstate(all="ignore"):
        if "tbl" in mech:
            _, _, _, spl = bpm.tblte(alpha, chord, Ue, f_src, span, re, theta, phi, af.trip, nu, c0)
            msp += p2(spl)
        if "blunt" in mech and af.te_thickness > 0:
            spl = bpm.blunt(alpha, chord, Ue, f_src, span, re, theta, phi, af.te_thickness * chord,
                            af.te_angle_deg, af.trip, nu, c0)
            msp += p2(spl)
        if "lbl" in mech and af.trip == bpm.TRIP_NONE:
            spl = bpm.lblvs(alpha, chord, Ue, f_src, span, re, theta, phi, af.trip, nu, c0)
            msp += p2(spl)
        if "inflow" in mech and opts.turbulence_ms > 0:
            spl = bpm.amiet_inflow(chord, Ue, f_src, span, re, theta, phi, aero.rho, c0,
                                   opts.turbulence_ms / Ue, opts.turbulence_scale, np.radians(alpha))
            msp += p2(spl)
        if "bwi" in mech and aero.bwi_rms is not None:
            # blade-wake interaction: the turbulence of the vortices ahead, through Amiet's inflow noise
            urms = aero.bwi_rms[idx]
            if np.any(urms > 0):
                spl = bpm.amiet_inflow(chord, Ue, f_src, span, re, theta, phi, aero.rho, c0, urms / Ue,
                                       aero.bwi_scale[idx], np.radians(alpha))
                msp += p2(spl)
        if "tip" in mech:
            k = -1
            spl = bpm.tip(alpha[:, k], chord[:, k], Ue[:, k], f_src[:, k, :], re[:, k], theta[:, k], phi[:, k],
                          c0, round_tip=rot.round_tip)
            msp[:, k, :] += p2(spl)
    msp = np.where(np.isfinite(msp), msp, 0.0) * forward[..., None]
    # land each segment's sound at its reception time, over one revolution, all blades
    nt = nb
    tg = np.arange(nt) * T / nt
    out = np.zeros((nt, len(f_out)))
    for i in range(len(r)):
        ti = t_rec[:, i]
        tt = np.concatenate([ti - T, ti, ti + T])
        base = ti[0]
        tw = base + (tg - base) % T
        for_band = np.concatenate([msp[:, i, :]] * 3, axis=0)
        order = np.argsort(tt)
        tt = tt[order]
        for_band = for_band[order]
        # linear interpolation of every band at once
        j = np.clip(np.searchsorted(tt, tw) - 1, 0, len(tt) - 2)
        w = ((tw - tt[j]) / (tt[j + 1] - tt[j]))[:, None]
        out += (1.0 - w) * for_band[j] + w * for_band[j + 1]
    total = np.zeros_like(out)
    for b in range(B):
        total += np.roll(out, b * nt // B, axis=0)
    return SpectralHistory(t=tg, f=f_out, msp=total, period=T)
