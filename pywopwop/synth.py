"""From rotor acoustics to audio: seamless loops and flight-condition renders.

Tonal noise is synthesised additively from the exact rotor harmonics (as a wavetable synthesizer does): band
limited by construction, below Nyquist divided by the largest pitch factor the simulator will apply, and periodic
in a sample-exact loop (an integer number of revolutions in an integer number of samples; the rotor speed is nudged
by a fraction of a cent to fit). Broadband noise is band noise shaped by the blade-passage envelope of its spectral
time history (a deterministic plus stochastic model, as in spectral modelling synthesis). Real rotors are never
perfectly periodic: gusts and turbulence move the blade loads, blades differ slightly in track, the governor lets
the rotor speed wander. These are added as small slow random processes, periodic over the loop too.
"""
import zlib
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.fft as sfft

from . import broadband, f1a
from .atmosphere import iso9613_alpha
from .model import Observer
from .operating import _pitch_matrix, operate


@dataclass
class Variability:
    load_rms: float = 0.04          # random blade-load fluctuation, fraction
    load_bandwidth_hz: float = 3.0
    load_common: float = 0.5        # share of it that all blades feel together (gusts)
    tracking_rms: float = 0.01      # fixed blade-to-blade load differences, fraction
    rpm_wander: float = 0.002       # rms rotor speed fluctuation, fraction
    rpm_bandwidth_hz: float = 0.25
    broadband_rms: float = 0.1      # slow level variation of the broadband noise, fraction
    seed: int = 1
    # pass-to-pass unsteadiness: revolutions computed with their own turbulent gusts and tip-vortex wander, each
    # blade morphing between them from one passage to the next
    variants: int = 8
    gust_rms: float = 1.0           # m/s, vertical turbulence the blades meet
    gust_scale: float = 20.0        # m, its integral scale
    wake_wander_chords: float = 0.25    # rms tip-vortex height wander, chords
    vortex_strength_rms: float = 0.1
    morph_concentration: float = 0.4    # Dirichlet concentration of the per-passage mix (small: one variant leads)
    # sound through turbulent air: per octave band, slowly fluctuating delay and level, growing with distance and
    # frequency (scintillation); rms delay at 100 m, and rms level at 1 kHz and 100 m
    scint_delay_ms: float = 0.15
    scint_level_db: float = 1.5
    scint_bandwidth_hz: float = 2.0

    @classmethod
    def none(cls):
        return cls(0.0, 3.0, 0.5, 0.0, 0.0, 0.25, 0.0, variants=0, scint_delay_ms=0.0, scint_level_db=0.0)


@dataclass
class Quality:
    name: str = "normal"
    n_radial: int = 40
    n_azimuth: int = 720
    chord_points: int = 4
    surface_chord: int = 24
    fs_source: float = 96000.0
    bb_azimuth: int = 128
    wall_dx: float = 0.5            # the fuselage's wall patches inside: slices along it, m
    wall_around: int = 8            # and sectors round each

    @classmethod
    def get(cls, name):
        if isinstance(name, Quality):
            return name
        return {"draft": cls("draft", 24, 360, 1, 12, 32000.0, 64, 1.0, 6),
                "normal": cls(),
                "high": cls("high", 60, 1440, 6, 32, 192000.0, 256, 0.25, 12)}[name]


@dataclass
class RotorSound:
    """One rotor's acoustics at one observer, ready for synthesis."""
    name: str
    blades: int
    f_rot: float                    # revolutions per second
    H_thickness: np.ndarray         # single-blade rfft over one revolution
    H_loading: np.ndarray
    n: int                          # samples per revolution of those transforms
    history: object = None          # broadband.SpectralHistory
    distance: float = 0.0
    H_loading_base: np.ndarray = None   # loading without the blade-vortex interaction's impulsive part
    role: str = "main"
    H_var: np.ndarray = None        # (variants, harmonics): loading of revolutions with gusts and vortex wander
    H_var_base: np.ndarray = None   # the same without the blade-vortex interaction
    index_deg: float = 0.0          # the rotor's blades behind the first rotor's (Rotor.index_deg)


@dataclass
class Render:
    fs: float
    signals: dict                   # name -> Pa
    rotors: list
    meta: dict = field(default_factory=dict)

    @property
    def total(self):
        return self.signals["total"]

    def levels(self):
        """OASPL and A-weighted level (dB re 20 uPa) of every signal."""
        out = {}
        for k, x in self.signals.items():
            out[k] = dict(oaspl=_db(np.mean(x ** 2)), dba=_db(a_weighted_ms(x, self.fs)))
        return out


def _rfft(x, *a, **k):
    return sfft.rfft(x, *a, workers=-1, **k)


def _irfft(X, n, *a, **k):
    return sfft.irfft(X, n, *a, workers=-1, **k)


def _db(ms):
    return 10.0 * np.log10(max(ms, 1e-30) / 4e-10)


def a_weighting_db(f):
    f = np.asarray(f, float)
    f2 = f ** 2
    ra = 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def a_weighted_ms(x, fs):
    X = _rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    g = 10.0 ** (a_weighting_db(f) / 20.0)
    return np.mean(_irfft(X * g, len(x)) ** 2)


def rotor_sound(point, observer_h, quality="normal", broadband_opts=None, with_broadband=True, cache=None,
                variability=None, on_wall=False):
    """Acoustics of a solved rotor at an observer riding with the aircraft (heading frame, metres from the main
    hub). on_wall: the observer is on the airframe's own skin."""
    from .aero import load_variants
    q = Quality.get(quality)
    cache = cache if cache is not None else {}
    key = id(point)
    if key not in cache:
        cache[key] = f1a.build_sources(point, chord_points=q.chord_points, surface_chord=q.surface_chord,
                                       fs_target=q.fs_source)
    src = cache[key]
    o = point.to_rotor(observer_h)
    Ht, Hl, n = f1a.single_blade_harmonics(src, o)
    Hb = None
    if point.aero.fz_base is not None:
        kb = (key, "base")
        if kb not in cache:
            cache[kb] = f1a.build_sources(point, n_psi=src.n_psi, chord_points=q.chord_points, thickness=False,
                                          loads="base")
        _, Hb, _ = f1a.single_blade_harmonics(cache[kb], o, n)
    Hv = Hvb = None
    var = variability
    # near the ground the rotor also meets its own recirculating downwash: larger eddies, of about half a radius
    rec = point.recirculation_rms
    R = point.rotor.radius
    if var is not None and var.variants > 0 and point.aero.model is not None:
        kv = (key, "variants")
        if kv not in cache:
            g2 = var.gust_rms ** 2 + rec ** 2
            scale = (var.gust_rms ** 2 * var.gust_scale + rec ** 2 * 0.5 * R) / max(g2, 1e-12)
            V = load_variants(point.aero, var.variants, np.sqrt(g2), scale, var.wake_wander_chords,
                              var.vortex_strength_rms, seed=var.seed + 101)
            cache[kv] = [(f1a.build_sources(point, n_psi=src.n_psi, chord_points=q.chord_points, thickness=False,
                                            loads=(v["ft"], v["fz"])),
                          f1a.build_sources(point, n_psi=src.n_psi, chord_points=q.chord_points, thickness=False,
                                            loads=(v["ft_base"], v["fz_base"]))) for v in V]
        Hv = np.array([f1a.single_blade_harmonics(a, o, n)[1] for a, _ in cache[kv]])
        Hvb = np.array([f1a.single_blade_harmonics(b, o, n)[1] for _, b in cache[kv]])
    # a wing behind the disc: its unsteady lift under the blade's vortices, part of the blade's loading
    wl = cache.get((key, "wing"))
    if wl is not None:
        from .wing import harmonics as wing_harmonics
        Hw = wing_harmonics(wl, o, n, radiating=on_wall)
        Hl = Hl + Hw
        Hb = Hb + Hw if Hb is not None else None
        Hv = Hv + Hw[None, :] if Hv is not None else None
        Hvb = Hvb + Hw[None, :] if Hvb is not None else None
    hist = None
    if with_broadband:
        bo = broadband_opts or broadband.BroadbandOptions(n_azimuth=q.bb_azimuth)
        if rec > 0:
            from dataclasses import replace
            t2 = bo.turbulence_ms ** 2 + rec ** 2
            bo = replace(bo, turbulence_ms=float(np.sqrt(t2)),
                         turbulence_scale=(bo.turbulence_ms ** 2 * bo.turbulence_scale + rec ** 2 * 0.3 * R) / t2)
        hist = broadband.spectral_history(point, o, bo)
    return RotorSound(point.rotor.name, point.rotor.blades, point.aero.omega / (2.0 * np.pi), Ht, Hl, n, hist,
                      float(np.linalg.norm(o)), Hb, point.rotor.role, Hv, Hvb, point.rotor.index_deg)


def modal_response(name, seed, n, fade):
    """One draw of the cabin's response to n harmonics of the rotor `name`, faded in by `fade` (0..1 per harmonic)"""
    rng = np.random.default_rng(zlib.crc32(name.encode()) + seed)
    modes = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2.0)
    return np.abs(modes) ** fade * np.exp(1j * fade * np.angle(modes))


def wall_field(point, listener_h, walls_h, areas, quality="normal", broadband_opts=None, with_broadband=True,
               cache=None, variability=None, first_mode_hz=70.0, c0=340.0, cabin=None, glazed=None, covered=None):
    """One rotor heard inside through the pressure on the walls (walls_h: points just outside the skin, heading
    frame; areas, m^2), as a RotorSound at the listener. Below the cockpit's lowest mode the cabin is one pressure
    chamber: the area-weighted mean of the walls' pressure, each patch delayed by its path to the listener; above it a
    diffuse field: their powers added, each harmonic then a draw of the cabin's modal response (random phase, levels
    scattered), faded in over the octave below the mode. The broadband
    adds by power. The pass-to-pass variants are computed once, at the listener, and their fluctuation carried over.
    What the walls pass and the interior builds up is the cabin filter's, afterwards; but the passing blades'
    hydrodynamic pressure (the terms falling as 1/r^2) is no sound: a footprint sweeping along the skin at the tip's
    speed, it drives the walls as a boundary layer's eddies do, poorly below its hydrodynamic coincidence (cabin's
    near_field_share), and only the radiating terms pass as sound."""
    cache = cache if cache is not None else {}
    ref = rotor_sound(point, listener_h, quality, broadband_opts, with_broadband, cache, variability)
    novar = replace(variability, variants=0) if variability is not None else Variability.none()
    w = np.asarray(areas, float) / np.sum(areas)
    f = np.arange(len(ref.H_loading)) * ref.f_rot
    s = np.clip(np.log2(np.maximum(f, 1e-9) / (0.5 * first_mode_hz)), 0.0, 1.0)
    keys = ("H_thickness", "H_loading", "H_loading_base")
    mean = {k: np.zeros(len(f), complex) for k in keys}
    power = {k: np.zeros(len(f)) for k in keys}
    msp = 0.0
    # each patch's walls against the cabin filter's (applied afterwards to all of it): its sound and the near field's
    # amplitude through them over the composite's, per harmonic
    V_tip = point.aero.omega * point.rotor.radius
    kinds = {}
    for g in ([None] if glazed is None else (True, False)):
        if cabin is None:
            kinds[g] = (np.ones(len(f)), np.ones(len(f)))
        else:
            comp = np.maximum(cabin.tau(f), 1e-300)
            snd_g = cabin.tau(f) if g is None else cabin.patch_tau(f, g)
            kinds[g] = (np.sqrt(snd_g / comp), np.sqrt(cabin.near_field_share(f, V_tip, c0, g)))
    src = cache[id(point)]
    src_b = cache.get((id(point), "base"))
    total = float(np.sum(areas))
    loudest = np.zeros(len(areas))
    for i, (wi, ai, p) in enumerate(zip(w, areas, walls_h)):
        if covered is not None and covered[i]:
            continue                                    # under the wing box: no outside air on it
        g_i = None if glazed is None else bool(glazed[i])
        acoustic, near = kinds[g_i]
        snd = rotor_sound(point, p, quality, broadband_opts, with_broadband, cache, novar, on_wall=True)
        r = np.linalg.norm(np.asarray(p) - np.asarray(listener_h))
        delay = np.exp(-2j * np.pi * f * r / c0)
        # the patch's direct field at the listener on top of its share of the diffuse one
        g = 1.0 + (cabin.direct(f, r, ai, total) if cabin is not None else 0.0)
        o = point.to_rotor(p)
        Ht_n, Hl_n, _ = f1a.single_blade_harmonics(src, o, snd.n, near_only=True)
        Hb_n = f1a.single_blade_harmonics(src_b, o, snd.n, near_only=True)[1] if src_b is not None else None
        for k, Hn in (("H_thickness", Ht_n), ("H_loading", Hl_n), ("H_loading_base", Hb_n)):
            H = getattr(snd, k)
            if H is not None:
                H = (H - Hn) * acoustic + Hn * near
                mean[k] += wi * H * delay
                power[k] += wi * g * np.abs(H) ** 2
                loudest[i] += wi * np.sum((g * np.abs(H) ** 2)[:len(f) // 8])
        if snd.history is not None:
            fb = snd.history.f
            gb = 1.0 + (cabin.direct(fb, r, ai, total) if cabin is not None else 0.0)
            if cabin is not None and g_i is not None:
                gb = gb * cabin.patch_tau(fb, g_i) / np.maximum(cabin.tau(fb), 1e-300)
            msp = msp + wi * snd.history.msp * gb[None, :]
    # a wing bolted to the fuselage: its loads there reach the cabin through the structure, an interior pressure,
    # added as the outside pressure the cabin filter (applied afterwards) turns into it
    wl = cache.get((id(point), "wing"))
    if cabin is not None and wl is not None:
        from .wing import structure_borne
        p2 = structure_borne(wl, cabin, total, f)
        extra = 2.0 * p2 * (0.5 * ref.n) ** 2 / np.maximum(cabin.inside(f), 1e-300)
        for k in keys:
            power[k] = power[k] + extra
    # the diffuse field's modes: each harmonic reaches the listener by the cabin's and the structure's many paths,
    # whose sum is a complex Gaussian (Schroeder), independent from one harmonic to the next, their spacing being wider
    # than a mode's bandwidth: random phases and levels scattered by 5.6 dB rms, so a few lines stand on resonances and
    # the rest don't, and neighbours beat less than equal ones would (one draw per harmonic, the same for its parts),
    # faded in with the diffuse field; the mean below it
    turn = modal_response(point.rotor.name, cabin.modes_seed if cabin is not None else 0, len(f), s)
    out = {}
    for k in keys:
        if getattr(ref, k) is None:
            out[k] = None
            continue
        a = np.abs(mean[k])
        out[k] = mean[k] * np.where(a > 0, (np.sqrt(power[k]) / np.where(a > 0, a, 1.0)) ** s, 1.0) * turn

    # the blades' pass-to-pass fluctuation, as a share of the field, at the patch that carries the most of it: at the
    # listener the free field's mid harmonics are small and a gust's change of them no measure of the walls' field,
    # which their steady near field and the wing hold
    strong = rotor_sound(point, walls_h[int(np.argmax(loudest))], quality, broadband_opts, False, cache, variability,
                         on_wall=True)

    def carried(V, base, new):
        """the variants' fluctuation at that patch, per harmonic, on the walls' field: where its harmonic stands
        within 40 dB of its strongest (in a null of the field the ratio means nothing), at most doubled"""
        if V is None:
            return None
        ok = np.abs(base) > 1e-2 * np.abs(base).max()
        r = np.where(ok[None, :], V / np.where(ok, base, 1.0)[None, :], 1.0)
        r = r * np.minimum(1.0, 2.0 / np.maximum(np.abs(r), 1e-12))
        return new[None, :] * r
    return replace(ref, H_thickness=out["H_thickness"], H_loading=out["H_loading"],
                   H_loading_base=out["H_loading_base"],
                   H_var=carried(strong.H_var, strong.H_loading, out["H_loading"]),
                   H_var_base=carried(strong.H_var_base, strong.H_loading_base, out["H_loading_base"])
                   if ref.H_loading_base is not None else None,
                   history=replace(ref.history, msp=msp) if ref.history is not None else None)


def _slow_noise(rng, L, fs, bandwidth, n=1):
    """n unit-rms Gaussian processes below `bandwidth` Hz, periodic over L samples (made at a low rate and
    interpolated: cheap for long loops)."""
    T = L / fs
    m = max(16, int(np.ceil(T * bandwidth * 8)))           # low-rate samples per loop
    k = np.fft.rfftfreq(m, T / m)
    sel = (k > 0) & (k <= bandwidth)
    out = np.zeros((n, L))
    if not np.any(sel):
        return out
    pos = np.arange(L) * m / L
    i = np.floor(pos).astype(int)
    u = pos - i
    for j in range(n):
        X = np.zeros(len(k), complex)
        X[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
        y = np.fft.irfft(X, m)
        p0, p1, p2, p3 = y[(i - 1) % m], y[i % m], y[(i + 1) % m], y[(i + 2) % m]
        x = p1 + 0.5 * u * (p2 - p0 + u * (2 * p0 - 5 * p1 + 4 * p2 - p3 + u * (3 * (p1 - p2) + p3 - p0)))
        out[j] = x / np.sqrt(np.mean(x ** 2))
    return out


def _band_limit(f, f_lim):
    """Raised-cosine roll-off over the last 15 % below f_lim."""
    f0 = 0.85 * f_lim
    g = np.ones_like(f)
    sel = f > f0
    g[sel] = 0.5 * (1.0 + np.cos(np.pi * np.clip((f[sel] - f0) / (f_lim - f0), 0.0, 1.0)))
    return g


def _smooth(n, primes=(2, 3, 5, 7, 11, 13)):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1


def loop_length(f_rot, fs, duration, max_cents=1.0):
    """Samples L and revolutions n of a loop near `duration`: an integer number of revolutions in L samples, L with
    small prime factors only (fast transforms); the rotor speed moves by less than max_cents."""
    n0 = max(1, int(round(duration * f_rot)))
    best = None
    for n in (n0, n0 + 1, n0 - 1, n0 + 2, n0 - 2):
        if n < 1:
            continue
        exact = n * fs / f_rot
        span = int(exact * (2.0 ** (max_cents / 1200.0) - 1.0)) + 1
        for L in range(int(exact) - span, int(exact) + span + 2):
            if L > 0 and _smooth(L):
                err = abs(1200.0 * np.log2(n * fs / L / f_rot))
                if err <= max_cents and (best is None or err < best[0]):
                    best = (err, L, n)
        if best:
            return best[1], best[2]
    return int(round(n0 * fs / f_rot)), n0


def synthesize(sounds, fs=48000.0, duration=15.0, variability=None, pitch_headroom=1.25, f_max=20000.0,
               L=None):
    """Mix rotor sounds into a seamless loop. The first rotor sets the loop (an integer number of its revolutions);
    the others are fitted to it (their speed moves by at most half a revolution per loop). Returns signals in Pa:
    per rotor and component, and 'total'."""
    var = variability if variability is not None else Variability()
    rng = np.random.default_rng(var.seed)
    if L is None:
        L, _ = loop_length(sounds[0].f_rot, fs, duration)
    k = np.fft.rfftfreq(L, 1.0 / fs)
    f_lim = min(f_max, 0.5 * fs / pitch_headroom)
    signals = {}
    total = np.zeros(L)

    def layer(name, x):
        key = f"layer/{name}"
        signals[key] = signals.get(key, np.zeros(L)) + x
    # rpm wander: one warp for every rotor (they share the drive train)
    warp = None
    if var.rpm_wander > 0:
        dlt = var.rpm_wander * _slow_noise(rng, L, fs, var.rpm_bandwidth_hz)[0]
        dlt -= np.mean(dlt)
        warp = np.cumsum(dlt) / fs                  # seconds of rotor time gained or lost
        warp -= np.mean(warp)
    for s in sounds:
        n_rev = max(1, int(round(s.f_rot * L / fs)))
        m = np.arange(len(s.H_loading))
        bins = m * n_rev
        ok = (bins < len(k)) & (m > 0)
        m, bins = m[ok], bins[ok]
        gain = _band_limit(k[bins], f_lim) * (k[bins] < f_lim)
        scale = L / s.n
        B = s.blades
        # thickness: identical blades, all harmonics not multiple of B cancel
        XT = np.zeros(len(k), complex)
        sel = m % B == 0
        XT[bins[sel]] = B * scale * s.H_thickness[m[sel]] * gain[sel]
        xt = _irfft(XT, L)
        # loading: each blade on its own, with its fluctuation and tracking
        eps = _slow_noise(rng, L, fs, var.load_bandwidth_hz, B + 1)
        xl = np.zeros(L)
        xs = np.zeros(L)                 # blade slap: the impulsive part of the loading noise
        for b in range(B):
            ph = np.exp(2j * np.pi * m * b / B) * gain * scale
            track = 1.0 + var.tracking_rms * rng.normal()
            fluct = var.load_rms * (np.sqrt(var.load_common) * eps[B] + np.sqrt(1.0 - var.load_common) * eps[b])
            mb = track * (1.0 + fluct)
            if s.H_var is not None:
                # the blade morphs from one computed revolution to another, passage by passage
                w = _morph_weights(rng, len(s.H_var), n_rev, L, b / B, var.morph_concentration)
                for v in range(len(s.H_var)):
                    Xb = np.zeros(len(k), complex)
                    Xb[bins] = s.H_var[v][m] * ph
                    xv = _irfft(Xb, L)
                    xl += w[v] * xv * mb
                    if s.H_loading_base is not None:
                        Xb[bins] = s.H_var_base[v][m] * ph
                        xs += w[v] * (xv - _irfft(Xb, L)) * mb
                continue
            Xb = np.zeros(len(k), complex)
            Xb[bins] = s.H_loading[m] * ph
            xl += _irfft(Xb, L) * mb
            if s.H_loading_base is not None:
                Xb[bins] = (s.H_loading[m] - s.H_loading_base[m]) * ph
                xs += _irfft(Xb, L) * mb
        if warp is not None:
            xt = _warp(xt, warp, fs)
            xl = _warp(xl, warp, fs)
            xs = _warp(xs, warp, fs)
        xb = np.zeros(L)
        if s.history is not None:
            xb = _broadband(s.history, L, fs, n_rev, rng, f_lim)
            if var.broadband_rms > 0:
                xb *= 1.0 + var.broadband_rms * _slow_noise(rng, L, fs, 0.5)[0]
        if s.index_deg:
            lag = int(round(s.index_deg / 360.0 * L / n_rev))
            xt, xl, xs, xb = (np.roll(x, lag) for x in (xt, xl, xs, xb))
        signals[f"{s.name}/thickness"] = xt
        signals[f"{s.name}/loading"] = xl
        signals[f"{s.name}/slap"] = xs
        signals[f"{s.name}/broadband"] = xb
        signals[s.name] = xt + xl + xb
        total += signals[s.name]
        # layers for simulators, which add up to the total: rotor (thickness and the loading without the slap),
        # slap, swish (broadband) of the lifting rotors, and the tail rotor
        if s.role == "tail":
            layer("tail", signals[s.name])
        else:
            layer("rotor", xt + xl - xs)
            layer("slap", xs)
            layer("swish", xb)
    signals["total"] = total
    return signals


def _morph_weights(rng, nv, n_rev, L, offset, concentration):
    """(nv, L) weights, a partition of unity: a random mix of the variants per revolution (Dirichlet), cross-faded
    with a raised cosine over each revolution; `offset` (revolutions) staggers the blades; periodic over the loop."""
    W = rng.dirichlet(np.full(nv, concentration), size=n_rev)          # (n_rev, nv)
    u = np.arange(L) * n_rev / L - offset
    k0 = np.floor(u).astype(int)
    f = u - k0
    sm = 0.5 - 0.5 * np.cos(np.pi * f)
    return ((1.0 - sm)[None, :] * W[k0 % n_rev].T + sm[None, :] * W[(k0 + 1) % n_rev].T)


def _warp(x, warp, fs, over=4):
    """x(t + warp(t)) for a periodic band-limited x, by FFT oversampling and cubic interpolation."""
    L = len(x)
    if over > 1:
        X = _rfft(x)
        Xo = np.zeros(L * over // 2 + 1, complex)
        Xo[:len(X)] = X
        xo = _irfft(Xo, L * over) * over
    else:
        xo = x
    pos = (np.arange(L) + warp * fs) * over
    i = np.floor(pos).astype(int)
    u = pos - i
    N = len(xo)
    p0, p1, p2, p3 = xo[(i - 1) % N], xo[i % N], xo[(i + 1) % N], xo[(i + 2) % N]
    return p1 + 0.5 * u * (p2 - p0 + u * (2 * p0 - 5 * p1 + 4 * p2 - p3 + u * (3 * (p1 - p2) + p3 - p0)))


def _broadband(hist, L, fs, n_rev, rng, f_lim):
    """Band noise shaped by the per-band envelope of the spectral history, periodic over the loop."""
    k = np.fft.rfftfreq(L, 1.0 / fs)
    t = np.arange(L) / fs
    T_loop = L / fs
    T = T_loop / n_rev
    tt = np.concatenate([hist.t - hist.period, hist.t, hist.t + hist.period]) * (T / hist.period)
    with np.errstate(divide="ignore"):
        u = 3.0 * np.log2(np.maximum(k, 1e-9) / 1000.0)        # continuous 1/3-octave band number
    nb = np.round(3.0 * np.log2(hist.f / 1000.0))
    out = np.zeros(L)
    lim = _band_limit(k, f_lim) * (k < f_lim)
    for b, fb in enumerate(hist.f):
        if fb > f_lim:
            break
        w = np.cos(0.5 * np.pi * np.clip(u - nb[b], -1.0, 1.0)) * lim
        if not np.any(w > 0):
            continue
        X = (rng.normal(size=len(k)) + 1j * rng.normal(size=len(k))) * w
        X[0] = 0.0
        x = _irfft(X, L)
        rms = np.sqrt(np.mean(x ** 2))
        if rms == 0:
            continue
        env2 = np.interp(t % T, tt, np.concatenate([hist.msp[:, b]] * 3))
        out += x / rms * np.sqrt(np.maximum(env2, 0.0))
    return out


def scintillate(x, fs, distance, var, rng, cache=None):
    """Turbulent-atmosphere fluctuations over a path of `distance` m: each octave band (power-complementary split)
    gets its own slow random delay (rms growing as the square root of distance) and level (rms growing with the
    square roots of distance and frequency). Periodic over the loop. Pass the same cache to give several signals
    the same fluctuations."""
    if distance < 5.0 or (var.scint_delay_ms <= 0 and var.scint_level_db <= 0):
        return x
    L = len(x)
    cache = {} if cache is None else cache
    if "bands" not in cache:
        d = np.sqrt(distance / 100.0)
        bands = []
        for b in range(-5, 5):                               # 31 Hz .. 16 kHz
            fb = 1000.0 * 2.0 ** b
            delay = var.scint_delay_ms * 1e-3 * d * _slow_noise(rng, L, fs, var.scint_bandwidth_hz)[0]
            level = var.scint_level_db * d * np.sqrt(fb / 1000.0) * _slow_noise(rng, L, fs, var.scint_bandwidth_hz)[0]
            bands.append((b, delay, 10.0 ** (level / 20.0)))
        cache["bands"] = bands
    X = _rfft(x)
    f = np.fft.rfftfreq(L, 1.0 / fs)
    with np.errstate(divide="ignore"):
        u = np.log2(np.maximum(f, 1e-9) / 1000.0)            # octaves from 1 kHz
    out = np.zeros(L)
    for b, delay, gain in cache["bands"]:
        w = np.cos(0.5 * np.pi * np.clip(u - b, -1.0, 1.0))
        if b == -5:
            w[u < -5] = 1.0
        if b == 4:
            w[u > 4] = 1.0
        xb = _irfft(X * w, L)
        out += _warp(xb, delay, fs, over=2 if b >= 2 else 1) * gain
    return out


def propagate_static(x, fs, distance, T=288.15, rh=0.7, p=101325.0):
    """Atmospheric absorption over a fixed distance, as a circular filter (keeps a loop seamless). Spherical
    spreading is already in the pressure."""
    X = _rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    g = 10.0 ** (-iso9613_alpha(f, T, rh * 100.0, p) * distance / 20.0)
    return _irfft(X * g, len(x))


def cabin_filter(x, fs, cabin=8.0, doors_open=False):
    """Sound through the cabin's walls and built up inside it (a model.Cabin, or a limp wall's surface mass in
    kg/m^2 with the default interior)."""
    from .model import Cabin
    cabin = cabin if isinstance(cabin, Cabin) else Cabin(surface_mass=float(cabin))
    X = _rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return _irfft(X * np.sqrt(cabin.inside(f, doors_open)), len(x))


def highpass(x, fs, f_hp):
    """Remove DC and (smoothly, fourth order) what is below f_hp, circularly. The default 25 Hz is a recording
    chain's: microphones and speakers don't carry a main rotor's blade-passing fundamental (10 to 30 Hz), which
    would otherwise fill the loop's headroom with something nobody hears."""
    X = _rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[0] = 0.0
    if f_hp > 0:
        X *= 1.0 / np.sqrt(1.0 + (f_hp / np.maximum(f, 1e-9)) ** 8)
    return _irfft(X, len(x))


def observer_position(state, observer):
    """An Observer's position in the heading frame: explicit positions are aircraft (body) frame."""
    if observer.position is not None:
        return _pitch_matrix(state.pitch) @ np.asarray(observer.position, float)
    return observer.offset()


def render(aircraft, condition, observer, rotors=None, fs=48000.0, duration=15.0, quality="normal",
           variability=None, with_broadband=True, broadband_opts=None, cabin=None, doors_open=False, f_highpass=25.0,
           aero_opts=None, state=None, pitch_headroom=1.25, environment="open", stems=False):
    """A seamless loop of the aircraft's rotors in a flight condition at an observer riding with it. Returns the
    total and the layers (rotor, slap, swish, tail); with stems, also every rotor and component. environment: the
    listener's surroundings outside (free, ground, open, urban; see environment.py)."""
    from .environment import Environment, apply as apply_environment
    from .aero import AeroOptions
    q = Quality.get(quality)
    if state is None:
        state = operate(aircraft, condition, aero_opts or AeroOptions(n_radial=q.n_radial, n_azimuth=q.n_azimuth))
    pos = observer_position(state, observer)
    pts = [p for p in state.points if rotors is None or p.rotor.name in rotors or p.rotor.role in rotors]
    cache = state.__dict__.setdefault("_sound_cache", {})     # shared by the renders of one flight state
    var = variability if variability is not None else Variability()
    if broadband_opts is None:
        broadband_opts = broadband.BroadbandOptions(n_azimuth=q.bb_azimuth,
                                                    turbulence_ms=broadband.air_turbulence(condition.altitude_m))
    from .wing import MAX_NACELLE_DEG, covered as wing_covered, wing_loads
    if getattr(aircraft, "wing", None) and condition.nacelle_deg <= MAX_NACELLE_DEG:
        for p in pts:
            if (id(p), "wing") not in cache:
                fu = getattr(aircraft, "fuselage", None)
                cache[(id(p), "wing")] = wing_loads(p, aircraft.wing, state.pitch,
                                                    attach_y=max(s_[1] for s_ in fu) if fu else None)
    if cabin and observer.position is not None and getattr(aircraft, "fuselage", None):
        # inside a described fuselage: heard through the pressure on its walls
        from .model import Cabin, glazed_patches, wall_patches
        walls, areas = wall_patches(aircraft.fuselage, q.wall_dx, q.wall_around)
        glazed = glazed_patches(walls, aircraft.glazing) if getattr(aircraft, "glazing", None) else None
        walls_h = [_pitch_matrix(state.pitch) @ w for w in walls]
        first = cabin.first_mode_hz if isinstance(cabin, Cabin) else 70.0
        cab = cabin if isinstance(cabin, Cabin) else Cabin(surface_mass=float(cabin))
        over = wing_covered(walls, aircraft.wing) if getattr(aircraft, "wing", None) else None
        sounds = [wall_field(p, pos, walls_h, areas, q, broadband_opts, with_broadband, cache, var, first,
                             state.atm.c, cab, glazed, over) for p in pts]
    else:
        sounds = [rotor_sound(p, pos, q, broadband_opts, with_broadband, cache, var) for p in pts]
    sig = synthesize(sounds, fs, duration, var, pitch_headroom)
    dist = float(np.linalg.norm(pos))
    atm = state.atm
    env = Environment(environment) if isinstance(environment, str) else environment
    # propagate the layers (and the stems if asked); the total is the layers' sum, through the same linear path
    keep = [k for k in sig if k.startswith("layer/") or (stems and k != "total")]
    sig = {k: sig[k] for k in keep}
    scint_cache = {}
    scint_rng = np.random.default_rng(var.seed + 999)
    for k in sig:
        x = sig[k]
        if dist > 5.0:
            x = propagate_static(x, fs, dist, atm.T, atm.humidity, atm.p)
            x = scintillate(x, fs, dist, var, scint_rng, scint_cache)
            if observer.position is None:
                x = apply_environment(x, fs, -pos[2], dist, env)
        if cabin:
            x = cabin_filter(x, fs, cabin, doors_open)
        sig[k] = highpass(x, fs, f_highpass)
    sig["total"] = sum(v for k, v in sig.items() if k.startswith("layer/"))
    meta = dict(aircraft=aircraft.name, airspeed_kt=condition.airspeed_kt, climb_fpm=condition.climb_fpm,
                rpm_percent=condition.rpm_percent, observer=observer.name, distance_m=dist,
                rotors={p.rotor.name: dict(rpm=p.aero.omega * 60 / (2 * np.pi), thrust_N=p.aero.thrust,
                                           power_kW=p.aero.power / 1e3, ct_sigma=p.aero.ct_sigma,
                                           mu=p.aero.mu_x, alpha_tpp_deg=float(np.degrees(p.alpha_tpp)),
                                           advancing_tip_mach=p.aero.advancing_tip_mach) for p in pts},
                pitch_deg=float(np.degrees(state.pitch)), loop_samples=len(sig["total"]))
    return Render(fs, sig, sounds, meta)


def write_wav(path, x, fs, bits=16, gain=1.0, dither=True, seed=0):
    """Write mono PCM (16 or 24 bit) or 32-bit float. x is scaled by gain; full scale is 1.0."""
    import wave
    y = np.asarray(x, float) * gain
    if bits == 32:
        from scipy.io import wavfile
        wavfile.write(str(path), int(fs), y.astype(np.float32))
        return
    full = 2 ** (bits - 1) - 1
    v = y * full
    if dither:
        rng = np.random.default_rng(seed)
        v = v + rng.uniform(-0.5, 0.5, len(v)) + rng.uniform(-0.5, 0.5, len(v))
    v = np.clip(np.round(v), -full - 1, full).astype(np.int64)
    if bits == 16:
        data = v.astype("<i2").tobytes()
    else:
        b = v.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
        data = b.tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(bits // 8)
        w.setframerate(int(fs))
        w.writeframes(data)
