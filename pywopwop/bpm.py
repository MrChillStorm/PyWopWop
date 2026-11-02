"""Airfoil self-noise: the Brooks, Pope and Marcolini (BPM) model, NASA RP-1218 (1989).

A vectorised port of the report's Fortran code (appendix D), subroutine by subroutine, so it reproduces the
report's own test output (tables D2 and D3). Every function takes NumPy arrays that broadcast against each other
(segments, azimuths, ...) and returns 1/3-octave sound pressure levels (dB re 20 uPa) with the frequency axis last.

Angles of attack are in degrees, lengths in metres, velocities in m/s. Observer angles follow appendix B: theta
from the chordwise (x, downstream) axis, phi from the spanwise (y) axis, both in degrees.
"""
import numpy as np

# 1/3-octave centre frequencies of the report's code (100 Hz to 40 kHz)
FRCEN_BPM = np.array([100., 125., 160., 200., 250., 315., 400., 500., 630., 800., 1000., 1250., 1600., 2000.,
                      2500., 3150., 4000., 5000., 6300., 8000., 10000., 12500., 16000., 20000., 25000., 31500.,
                      40000.])

TRIP_NONE, TRIP_HEAVY, TRIP_LIGHT = 0, 1, 2


def third_octave_bands(f_lo=20.0, f_hi=20000.0):
    """Nominal-series exact centre frequencies 1000 * 10^(n/10) between f_lo and f_hi."""
    n = np.arange(np.ceil(10 * np.log10(f_lo / 1000.0) - 1e-9), np.floor(10 * np.log10(f_hi / 1000.0) + 1e-9) + 1)
    return 1000.0 * 10.0 ** (n / 10.0)


def _log10(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(np.maximum(x, 1e-300))


def _db(x):
    return 10.0 * _log10(x)


def bl_thickness(c, U, alpha, trip, nu):
    """Boundary-layer thickness on the pressure side and displacement thicknesses (THICK): (delta_p, dstr_s, dstr_p)."""
    c, U, alpha = np.broadcast_arrays(np.asarray(c, float), np.asarray(U, float), np.asarray(alpha, float))
    rc = U * c / nu
    lr = _log10(rc)
    delta0 = 10.0 ** (1.6569 - 0.9045 * lr + 0.0596 * lr ** 2) * c
    if trip == TRIP_LIGHT:
        delta0 = 0.6 * delta0
    delta_p = 10.0 ** (-0.04175 * alpha + 0.00106 * alpha ** 2) * delta0
    if trip in (TRIP_HEAVY, TRIP_LIGHT):
        d0 = np.where(rc <= 0.3e6, 0.0601 * np.maximum(rc, 1.0) ** -0.114 * c,
                      10.0 ** (3.411 - 1.5397 * lr + 0.1059 * lr ** 2) * c)
        if trip == TRIP_LIGHT:
            d0 = 0.6 * d0
    else:
        d0 = 10.0 ** (3.0187 - 1.5397 * lr + 0.1059 * lr ** 2) * c
    dstr_p = 10.0 ** (-0.0432 * alpha + 0.00113 * alpha ** 2) * d0
    if trip == TRIP_HEAVY:
        dstr_s = np.select([alpha <= 5.0, alpha <= 12.5],
                           [10.0 ** (0.0679 * alpha) * d0, 0.381 * 10.0 ** (0.1516 * alpha) * d0],
                           14.296 * 10.0 ** (0.0258 * alpha) * d0)
    else:
        dstr_s = np.select([alpha <= 7.5, alpha <= 12.5],
                           [10.0 ** (0.0679 * alpha) * d0, 0.0162 * 10.0 ** (0.3066 * alpha) * d0],
                           52.42 * 10.0 ** (0.0258 * alpha) * d0)
    return delta_p, dstr_s, dstr_p


def directivity_high(M, theta, phi):
    """High-frequency directivity D_h (DIRECTH), angles in degrees."""
    th = np.radians(theta)
    ph = np.radians(phi)
    mc = 0.8 * M
    return (2.0 * np.sin(th / 2.0) ** 2 * np.sin(ph) ** 2
            / ((1.0 + M * np.cos(th)) * (1.0 + (M - mc) * np.cos(th)) ** 2))


def directivity_low(M, theta, phi):
    """Low-frequency directivity D_l (DIRECTL), angles in degrees."""
    th = np.radians(theta)
    ph = np.radians(phi)
    return (np.sin(th) * np.sin(ph)) ** 2 / (1.0 + M * np.cos(th)) ** 4


def _amin(a):
    x = np.abs(a)
    return np.select([x <= 0.204, x <= 0.244],
                     [np.sqrt(np.maximum(67.552 - 886.788 * x ** 2, 0.0)) - 8.219, -32.665 * x + 3.981],
                     -142.795 * x ** 3 + 103.656 * x ** 2 - 57.757 * x + 6.006)


def _amax(a):
    x = np.abs(a)
    return np.select([x <= 0.13, x <= 0.321],
                     [np.sqrt(np.maximum(67.552 - 886.788 * x ** 2, 0.0)) - 8.219, -15.901 * x + 1.098],
                     -4.669 * x ** 3 + 3.491 * x ** 2 - 16.699 * x + 1.149)


def _bmin(b):
    x = np.abs(b)
    return np.select([x <= 0.13, x <= 0.145],
                     [np.sqrt(np.maximum(16.888 - 886.788 * x ** 2, 0.0)) - 4.109, -83.607 * x + 8.138],
                     -817.81 * x ** 3 + 355.21 * x ** 2 - 135.024 * x + 10.619)


def _bmax(b):
    x = np.abs(b)
    return np.select([x <= 0.1, x <= 0.187],
                     [np.sqrt(np.maximum(16.888 - 886.788 * x ** 2, 0.0)) - 4.109, -31.313 * x + 1.854],
                     -80.541 * x ** 3 + 44.174 * x ** 2 - 39.381 * x + 2.344)


def _a0comp(rc):
    return np.select([rc < 9.52e4, rc < 8.57e5], [0.57, -9.57e-13 * (rc - 8.57e5) ** 2 + 1.13], 1.13)


def tblte(alpha, c, U, f, L, r, theta, phi, trip, nu, c0):
    """Turbulent boundary layer trailing-edge noise and separation noise (TBLTE).

    Returns (spl_p, spl_s, spl_alpha, spl_tbl): pressure side, suction side, separation (angle of attack), total.
    """
    alpha, c, U, L, r, theta, phi = [np.asarray(v, float)[..., None] for v in
                                     np.broadcast_arrays(alpha, c, U, L, r, theta, phi)]
    f = np.asarray(f, float)
    rc = U * c / nu
    M = U / c0
    delta_p, dstr_s, dstr_p = bl_thickness(c, U, alpha, trip, nu)
    dbar_l = directivity_low(M, theta, phi)
    dbar_h = directivity_high(M, theta, phi)
    rdstr_p = dstr_p * U / nu

    st1 = 0.02 * M ** -0.6
    st2 = np.select([alpha <= 1.333, alpha <= 12.5],
                    [st1, st1 * 10.0 ** (0.0054 * (alpha - 1.333) ** 2)], 4.72 * st1)
    st1prim = (st1 + st2) / 2.0

    a0 = _a0comp(rc)
    a02 = _a0comp(3.0 * rc)
    ara0 = (20.0 + _amin(a0)) / (_amin(a0) - _amax(a0))
    ara02 = (20.0 + _amin(a02)) / (_amin(a02) - _amax(a02))
    b0 = np.select([rc < 9.52e4, rc < 8.57e5], [0.30, -4.48e-13 * (rc - 8.57e5) ** 2 + 0.56], 0.56)
    brb0 = (20.0 + _bmin(b0)) / (_bmin(b0) - _bmax(b0))

    stp = f * dstr_p / U
    a = _log10(stp / st1)
    aa = _amin(a) + ara0 * (_amax(a) - _amin(a))
    k1 = np.select([rc < 2.47e5, rc < 8.0e5], [-4.31 * _log10(rc) + 156.3, -9.0 * _log10(rc) + 181.6], 128.5)
    dk1 = np.where(rdstr_p <= 5000.0, -alpha * (5.29 - 1.43 * _log10(rdstr_p)), 0.0)
    spl_p = aa + k1 - 3.0 + _db(dstr_p * M ** 5 * dbar_h * L / r ** 2) + dk1

    gamma = 27.094 * M + 3.31
    beta = 72.650 * M + 10.74
    gamma0 = 23.430 * M + 4.651
    beta0 = -34.190 * M - 13.820
    k2 = np.select([alpha <= gamma0 - gamma, alpha <= gamma0 + gamma],
                   [-1000.0, np.sqrt(np.maximum(beta ** 2 - (beta / gamma) ** 2 * (alpha - gamma0) ** 2, 0.0)) + beta0],
                   -12.0) + k1

    sts = f * dstr_s / U
    switch = (alpha >= gamma0) | (alpha > 12.5)
    a = _log10(sts / st1prim)
    aa = _amin(a) + ara0 * (_amax(a) - _amin(a))
    b = np.abs(_log10(sts / st2))
    bb = _bmin(b) + brb0 * (_bmax(b) - _bmin(b))
    spl_s_attached = aa + k1 - 3.0 + _db(dstr_s * M ** 5 * dbar_h * L / r ** 2)
    spl_a_attached = bb + k2 + _db(dstr_s * M ** 5 * dbar_h * L / r ** 2)
    # past the stall the 'A' contributions are dropped and the separation spectrum uses the A' curve (as the code)
    scale_l = _db(dstr_s * M ** 5 * dbar_l * L / r ** 2)
    bb_st = _amin(b) + ara02 * (_amax(b) - _amin(b))
    spl_s = np.where(switch, scale_l, spl_s_attached)
    spl_p = np.where(switch, scale_l, spl_p)
    spl_a = np.where(switch, bb_st + k2 + scale_l, spl_a_attached)

    spl_p = np.maximum(spl_p, -100.0)
    spl_s = np.maximum(spl_s, -100.0)
    spl_a = np.maximum(spl_a, -100.0)
    spl_tbl = _db(10.0 ** (spl_p / 10.0) + 10.0 ** (spl_s / 10.0) + 10.0 ** (spl_a / 10.0))
    return spl_p, spl_s, spl_a, spl_tbl


def lblvs(alpha, c, U, f, L, r, theta, phi, trip, nu, c0):
    """Laminar boundary layer vortex shedding noise (LBLVS)."""
    alpha, c, U, L, r, theta, phi = [np.asarray(v, float)[..., None] for v in
                                     np.broadcast_arrays(alpha, c, U, L, r, theta, phi)]
    f = np.asarray(f, float)
    M = U / c0
    rc = U * c / nu
    delta_p, _, _ = bl_thickness(c, U, alpha, trip, nu)
    dbar_h = directivity_high(M, theta, phi)
    st1prim = np.select([rc <= 1.3e5, rc <= 4.0e5], [0.18, 0.001756 * rc ** 0.3931], 0.28)
    stpkprm = 10.0 ** (-0.04 * alpha) * st1prim
    rc0 = np.where(alpha <= 3.0, 10.0 ** (0.215 * alpha + 4.978), 10.0 ** (0.120 * alpha + 5.263))
    d = rc / rc0
    ld = _log10(d)
    g2 = np.select([d <= 0.3237, d <= 0.5689, d <= 1.7579, d <= 3.0889],
                   [77.852 * ld + 15.328, 65.188 * ld + 9.125, -114.052 * ld ** 2, -65.188 * ld + 9.125],
                   -77.852 * ld + 15.328)
    g3 = 171.04 - 3.03 * alpha
    scale = _db(delta_p * M ** 5 * dbar_h * L / r ** 2)
    e = f * delta_p / U / stpkprm
    le = _log10(e)
    g1 = np.select([e < 0.5974, e <= 0.8545, e < 1.17, e < 1.674],
                   [39.8 * le - 11.12, 98.409 * le + 2.0, -5.076 + np.sqrt(np.maximum(2.484 - 506.25 * le ** 2, 0.0)),
                    -98.409 * le + 2.0], -39.80 * le - 11.12)
    return g1 + g2 + g3 + scale


def _g5comp(hdstar, eta):
    mu = np.select([hdstar < 0.25, hdstar <= 0.62, hdstar < 1.15],
                   [0.1211, -0.2175 * hdstar + 0.1755, -0.0308 * hdstar + 0.0596], 0.0242)
    m = np.select([hdstar <= 0.02, hdstar < 0.5, hdstar <= 0.62, hdstar <= 1.15, hdstar < 1.2],
                  [0.0, 68.724 * hdstar - 1.35, 308.475 * hdstar - 121.23, 224.811 * hdstar - 69.354,
                   1583.28 * hdstar - 1631.592], 268.344)
    m = np.maximum(m, 0.0)
    eta0 = -np.sqrt(m * m * mu ** 4 / (6.25 + m * m * mu * mu))
    k = 2.5 * np.sqrt(np.maximum(1.0 - (eta0 / mu) ** 2, 0.0)) - 2.5 - m * eta0
    return np.select([eta <= eta0, eta <= 0.0, eta <= 0.03616],
                     [m * eta + k, 2.5 * np.sqrt(np.maximum(1.0 - (eta / mu) ** 2, 0.0)) - 2.5,
                      np.sqrt(np.maximum(1.5625 - 1194.99 * eta ** 2, 0.0)) - 1.25],
                     -155.543 * eta + 4.375)


def blunt(alpha, c, U, f, L, r, theta, phi, h, psi, trip, nu, c0):
    """Trailing-edge bluntness vortex shedding noise (BLUNT); h TE thickness (m), psi TE angle (deg)."""
    alpha, c, U, L, r, theta, phi, h, psi = [np.asarray(v, float)[..., None] for v in
                                             np.broadcast_arrays(alpha, c, U, L, r, theta, phi, h, psi)]
    f = np.asarray(f, float)
    M = U / c0
    _, dstr_s, dstr_p = bl_thickness(c, U, alpha, trip, nu)
    dstravg = (dstr_s + dstr_p) / 2.0
    hdstar = h / dstravg
    dstarh = 1.0 / hdstar
    dbar_h = directivity_high(M, theta, phi)
    aterm = 0.212 - 0.0045 * psi
    stpeak = np.where(hdstar >= 0.2, aterm / (1.0 + 0.235 * dstarh - 0.0132 * dstarh ** 2),
                      0.1 * hdstar + 0.095 - 0.00243 * psi)
    g4 = np.where(hdstar <= 5.0, 17.5 * _log10(hdstar) + 157.5 - 1.114 * psi, 169.7 - 1.114 * psi)
    eta = _log10(f * h / U / stpeak)
    g514 = _g5comp(hdstar, eta)
    hdstarp = 6.724 * hdstar ** 2 - 4.019 * hdstar + 1.107
    g50 = _g5comp(hdstarp, eta)
    g5 = g50 + 0.0714 * psi * (g514 - g50)
    g5 = np.minimum(g5, 0.0)
    g5 = np.minimum(g5, _g5comp(np.asarray(0.25), eta))
    scale = _db(M ** 5.5 * h * dbar_h * L / r ** 2)
    return g4 + g5 + scale


def tip(alpha_tip, c, U, f, r, theta, phi, c0, round_tip=False, alprat=1.0):
    """Tip vortex formation noise (TIPNOIS)."""
    alpha_tip, c, U, r, theta, phi = [np.asarray(v, float)[..., None] for v in
                                      np.broadcast_arrays(alpha_tip, c, U, r, theta, phi)]
    f = np.asarray(f, float)
    alp = alpha_tip * alprat
    M = U / c0
    dbar_h = directivity_high(M, theta, phi)
    if round_tip:
        ell = 0.008 * alp * c
    else:
        ell = np.where(np.abs(alp) <= 2.0, (0.023 + 0.0169 * alp) * c, (0.0378 + 0.0095 * alp) * c)
    mm = (1.0 + 0.036 * alp) * M
    um = mm * c0
    scale = _db(M * M * mm ** 3 * ell ** 2 * dbar_h / r ** 2)
    stpp = f * ell / um
    return 126.0 - 30.5 * (_log10(stpp) + 0.3) ** 2 + scale


def amiet_inflow(c, U, f, L, r, theta, phi, rho, c0, intensity, length_scale, alpha_rad=0.0):
    """Turbulent inflow (leading-edge) noise, Amiet's model with the Moriarty-Guidati low-frequency correction,
    as documented for OpenFAST's AeroAcoustics module. intensity is u'/U of the inflow, length_scale in metres."""
    c, U, L, r, theta, phi, intensity, alpha_rad, length_scale = [
        np.asarray(v, float)[..., None] for v in np.broadcast_arrays(c, U, L, r, theta, phi, intensity, alpha_rad,
                                                                      length_scale)]
    f = np.asarray(f, float)
    M = U / c0
    beta2 = 1.0 - M ** 2
    k1 = 2.0 * np.pi * f / U
    kbar = k1 * c / 2.0
    ke = 3.0 / (4.0 * length_scale)
    khat = k1 / ke
    fco = 10.0 * U / (np.pi * c)
    th = np.radians(theta)
    ph = np.radians(phi)
    d_low = directivity_low(M, theta, phi)
    d_high = 2.0 * np.cos(th / 2.0) ** 2 * np.sin(ph) ** 2 / (1.0 + M * np.cos(th)) ** 3
    dbar = np.where(f < fco, d_low, d_high)
    spl = _db(rho ** 2 * c0 ** 4 * length_scale * L / (2.0 * r ** 2) * M ** 5 * intensity ** 2
              * khat ** 3 / (1.0 + khat ** 2) ** (7.0 / 3.0) * dbar) + 78.4
    spl = spl + _db(1.0 + 9.0 * alpha_rad ** 2)
    s2 = 1.0 / (2.0 * np.pi * kbar / beta2 + 1.0 / (1.0 + 2.4 * kbar / beta2))
    lfc = 10.0 * s2 * M * kbar ** 2 / beta2
    return spl + _db(lfc / (1.0 + lfc))
