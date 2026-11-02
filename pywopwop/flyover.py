"""A fly-by heard by a listener on the ground.

Tonal noise: Formulation 1A straight to the fixed listener over the whole pass (every blade, every emission time),
so Doppler, retarded time and directivity are exact, with no interpolation between emission angles. Broadband:
spectral histories at the emission direction towards the listener (as heard by an observer riding along at a
reference distance), synthesised in emission time and carried to the listener's time with spherical spreading.
Each path (direct, and reflected by the ground through an image listener) is filtered by ISO 9613-1 absorption
over its length; the reflection by a plane-wave coefficient of the ground's impedance (Delany-Bazley) at the
grazing angle.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.fft as sfft

from . import broadband, f1a
from .atmosphere import iso9613_alpha
from .operating import operate
from .synth import Quality, Render, Variability, _band_limit, _slow_noise, a_weighting_db


@dataclass
class FlyoverOptions:
    altitude_m: float = 150.0       # height of the aircraft above the listener at closest approach
    offset_m: float = 0.0           # sideline distance of the track (positive: to the listener's right... left)
    duration: float = 20.0          # centred on the closest approach
    fs: float = 48000.0
    ground: str = "grass"           # hard, grass or none
    flow_resistivity: float = 2.0e5     # Pa s / m^2 (grass)
    observer_height: float = 1.2
    bb_block: float = 0.25          # s of emission time between broadband spectral histories
    r_ref: float = 100.0
    variability: Variability = field(default_factory=Variability)


def _delany_bazley_R(f, sin_psi, sigma, rho=1.2):
    X = np.maximum(rho * f / sigma, 1e-4)
    Z = 1.0 + 0.0571 * X ** -0.754 - 1j * 0.087 * X ** -0.732
    return (Z * sin_psi - 1.0) / (Z * sin_psi + 1.0)


def _stft_filter(x, fs, gain_fn, n=4096):
    """Time-varying filtering by overlap-add (Hann windows, 75 % overlap): gain_fn(t_centre, f) -> complex."""
    hop = n // 4
    w = np.hanning(n + 1)[:-1]
    pad = np.concatenate([np.zeros(n), x, np.zeros(n)])
    out = np.zeros_like(pad)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    for start in range(0, len(pad) - n, hop):
        seg = pad[start:start + n] * w
        tc = (start + n / 2 - n) / fs
        Y = sfft.rfft(seg) * gain_fn(tc, f)
        out[start:start + n] += sfft.irfft(Y, n)
    return out[n:n + len(x)] / 1.5


def _cubic(x, pos):
    """x at fractional sample positions (Catmull-Rom), zero outside."""
    n = len(x)
    i = np.floor(pos).astype(int)
    u = pos - i
    ok = (i >= 1) & (i < n - 2)
    i = np.clip(i, 1, n - 3)
    p0, p1, p2, p3 = x[i - 1], x[i], x[i + 1], x[i + 2]
    y = p1 + 0.5 * u * (p2 - p0 + u * (2 * p0 - 5 * p1 + 4 * p2 - p3 + u * (3 * (p1 - p2) + p3 - p0)))
    return np.where(ok, y, 0.0)


def render_flyover(aircraft, condition, opts=None, quality="draft", aero_opts=None):
    from .aero import AeroOptions
    opts = opts or FlyoverOptions()
    q = Quality.get(quality)
    state = operate(aircraft, condition, aero_opts or AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
    fs = opts.fs
    atm = state.atm
    c = atm.c
    vel = state.velocity
    D = opts.duration
    t0 = -D / 2.0
    nout = int(round(D * fs))
    t_out = t0 + np.arange(nout) / fs
    A0 = np.array([0.0, opts.offset_m, opts.altitude_m])        # aircraft origin at t = 0, listener at 0
    lis = {"direct": np.zeros(3), "image": np.array([0.0, 0.0, -2.0 * opts.observer_height])}
    if opts.ground == "none":
        del lis["image"]
    var = opts.variability
    rng = np.random.default_rng(var.seed)
    paths = {k: np.zeros(nout) for k in lis}
    f_lim = 0.5 * fs / 1.02
    for pt in state.points:
        B = pt.rotor.blades
        f_rot = pt.aero.omega / (2.0 * np.pi)
        n_psi = B
        while n_psi * f_rot < 0.75 * fs:
            n_psi *= 2
        src = f1a.build_sources(pt, n_psi=n_psi, chord_points=max(2, q.chord_points // 2),
                                surface_chord=min(q.surface_chord, 12))
        if var.variants > 0 and pt.aero.model is not None:
            from .aero import load_variants
            V = load_variants(pt.aero, var.variants, var.gust_rms, var.gust_scale, var.wake_wander_chords,
                              var.vortex_strength_rms, seed=var.seed + 101)
            f1a.morph_sequence(src, pt, V, 8, rng, var.morph_concentration)
        hub0 = A0 + pt.hub

        def mod_fn(n_start, n_steps, dtau, B=B):
            # slow per-blade load fluctuation over the emission span, from a coarse random process
            span = (n_steps + 1) * dtau
            m = max(8, int(span * 50.0))
            L = 1 << int(np.ceil(np.log2(m)))
            e = _slow_noise(rng, L, L / span, var.load_bandwidth_hz, B + 1)
            tt = np.arange(n_steps + 1) * dtau
            tc = np.arange(L) * span / L
            out = np.empty((B, n_steps + 1))
            for b in range(B):
                eb = np.sqrt(var.load_common) * e[B] + np.sqrt(1.0 - var.load_common) * e[b]
                out[b] = (1.0 + var.tracking_rms * rng.normal()) * (1.0 + var.load_rms * np.interp(tt, tc, eb))
            return out

        # thickness noise is low-frequency (below the high-speed impulsive regime): about 2048 emission steps per
        # revolution and at least 24 kHz carry it
        stride = 1
        while n_psi // (2 * stride) >= 512 and n_psi * f_rot / (2 * stride) >= 12000.0:
            stride *= 2
        # tonal noise to the listener; the ground-reflected path leaves the rotor within a few degrees of the
        # direct one, so it is the same emission retimed and rescaled by the image path (below)
        o = pt.M @ (lis["direct"] - hub0)
        _, pT, pL, _ = f1a.time_history(src, o, t0, D, fs, mod_fn=mod_fn, thickness_stride=stride)
        tonal_direct = pT + pL
        # broadband in emission time
        tau_lo = t0 - (np.linalg.norm(hub0) + abs(t0) * np.linalg.norm(vel) + 200.0) / c - 1.0
        tau_hi = t0 + D
        nb = int(np.ceil((tau_hi - tau_lo) / opts.bb_block)) + 1
        tau_b = tau_lo + np.arange(nb) * opts.bb_block
        bo = broadband.BroadbandOptions(n_azimuth=64)
        hists = []
        for tb in tau_b:
            Hb = hub0 + vel * tb
            e = -Hb / np.linalg.norm(Hb)
            o = pt.M @ (opts.r_ref * (e - vel / c))
            hists.append(broadband.spectral_history(pt, o, bo))
        S = np.array([h.msp for h in hists])                     # (nb, nt, nf)
        fb = hists[0].f
        nt = S.shape[1]
        T = hists[0].period
        ntau = int(np.ceil((tau_hi - tau_lo) * fs)) + 4
        tau = tau_lo + np.arange(ntau) / fs
        kb = np.clip((tau - tau_lo) / opts.bb_block, 0, nb - 1.000001)
        i0 = np.floor(kb).astype(int)
        w = kb - i0
        ph = (tau % T) / T * nt
        j0 = np.floor(ph).astype(int) % nt
        wj = ph - np.floor(ph)
        j1 = (j0 + 1) % nt
        Lf = sfft.next_fast_len(ntau)
        kf = np.fft.rfftfreq(Lf, 1.0 / fs)
        u = 3.0 * np.log2(np.maximum(kf, 1e-9) / 1000.0)
        nbands = np.round(3.0 * np.log2(fb / 1000.0))
        lim = _band_limit(kf, f_lim) * (kf < f_lim)
        xbb = np.zeros(ntau)
        for b in range(len(fb)):
            if fb[b] > f_lim:
                break
            wb = np.cos(0.5 * np.pi * np.clip(u - nbands[b], -1.0, 1.0)) * lim
            X = (rng.normal(size=len(kf)) + 1j * rng.normal(size=len(kf))) * wb
            X[0] = 0.0
            x = sfft.irfft(X, Lf, workers=-1)[:ntau]
            rms = np.sqrt(np.mean(x ** 2))
            if rms == 0:
                continue
            Sb = S[:, :, b]
            env = ((1 - w) * ((1 - wj) * Sb[i0, j0] + wj * Sb[i0, j1])
                   + w * ((1 - wj) * Sb[i0 + 1, j0] + wj * Sb[i0 + 1, j1]))
            xbb += x / rms * np.sqrt(np.maximum(env, 0.0))
        Hb = hub0[None, :] + vel[None, :] * tau[:, None]
        r_d = np.linalg.norm(Hb - lis["direct"][None, :], axis=1)
        t_d = tau + r_d / c
        for k, L in lis.items():
            r = np.linalg.norm(Hb - L[None, :], axis=1)
            t_rec = tau + r / c
            tau_of_t = np.interp(t_out, t_rec, tau)
            r_of_t = np.interp(t_out, t_rec, r)
            y = _cubic(xbb, (tau_of_t - tau_lo) * fs)
            if k == "direct":
                tonal = tonal_direct
            else:
                # the direct path's tonal signal at the moment the same emission reached the direct listener
                td = np.interp(tau_of_t, tau, t_d)
                tonal = _cubic(tonal_direct, (td - t0) * fs) * np.interp(tau_of_t, tau, r_d) / r_of_t
            paths[k] += tonal + y * opts.r_ref / r_of_t
    # propagation filters per path: absorption over the path length, ground reflection on the image path
    signals = {}
    for k, L in lis.items():
        def geom(tr, L=L):
            """Path length and sine of the grazing angle for sound received at tr (emission time solved)."""
            te = tr
            for _ in range(6):
                d = A0 + vel * te - L
                te = tr - np.linalg.norm(d) / c
            d = A0 + vel * te - L
            return np.linalg.norm(d), max(d[2], 0.0) / np.linalg.norm(d)
        alpha = lambda f: iso9613_alpha(f, atm.T, atm.humidity * 100.0, atm.p)

        bands = np.arange(-5, 5)
        n_fr = int(np.ceil(D * 2.0)) + 8
        scint = [var.scint_level_db * np.sqrt(1000.0 * 2.0 ** b / 1000.0) * _slow_noise(rng, n_fr, 2.0, 0.6)[0]
                 for b in bands]

        def gain(tc, f, k=k, geom=geom, scint=scint):
            r, s = geom(tc)
            g = 10.0 ** (-alpha(f) * r / 20.0)
            if var.scint_level_db > 0:
                fi = np.clip((tc - t0) * 2.0, 0, n_fr - 1)
                ob = np.clip(np.log2(np.maximum(f, 1.0) / 1000.0), -5, 4)
                lev = np.interp(ob, bands, [np.interp(fi, np.arange(n_fr), sc) for sc in scint]) * np.sqrt(r / 100.0)
                g = g * 10.0 ** (lev / 20.0)
            if k == "image":
                g = g * (1.0 if opts.ground == "hard" else _delany_bazley_R(f, s, opts.flow_resistivity))
            return g
        signals[k] = _stft_filter(paths[k], fs, gain)
    total = sum(signals.values())
    fade = int(0.3 * fs)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
    total[:fade] *= ramp
    total[-fade:] *= ramp[::-1]
    signals["total"] = total
    # maximum A-weighted level, 'slow' (1 s) windows
    X = sfft.rfft(total, workers=-1)
    fa = np.fft.rfftfreq(len(total), 1.0 / fs)
    xa = sfft.irfft(X * 10.0 ** (a_weighting_db(fa) / 20.0), len(total), workers=-1)
    win = int(fs)
    cs = np.concatenate([[0.0], np.cumsum(xa ** 2)])
    ms = (cs[win:] - cs[:-win]) / win
    meta = dict(aircraft=aircraft.name, airspeed_kt=condition.airspeed_kt, climb_fpm=condition.climb_fpm,
                altitude_m=opts.altitude_m, offset_m=opts.offset_m, ground=opts.ground,
                lmax_dba=float(10 * np.log10(np.max(ms) / 4e-10)), observer="ground", distance_m=opts.altitude_m)
    return Render(fs, signals, [], meta)
