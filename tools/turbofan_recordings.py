"""A turbofan's fan in recordings: its shaft's rate from the buzz-saw's comb (tones at every multiple of the shaft's
rate, which a fan makes once its blade tips' relative speed is supersonic) and the blade count from the strongest of
the comb's lines (the blade passing tone).

    python tools/turbofan_recordings.py refs/recordings/citationx_fadec_check/mono.wav [more.wav ...]

For each stretch of `step` seconds: the shaft rate f1 between 70 and 150 Hz whose multiples 6 to 40 carry the most
tone (a harmonic sieve on the spectrum less its smooth floor), how much of the tone in that range the comb holds, and
the multiple that is strongest between 18 and 30: the blade count. Stretches whose comb holds little are not listed.

NOT RELIABLE as it stands: on fly-bys the Doppler shift smears the lines over a stretch and the sieve finds combs that
are not there (it gave 18 to 30 blades on the same recordings). The AE 3007's figures in presets/jets/globalhawk.toml
and presets/turbofans/ae3007h.toml were read off one-second spectra by hand: in globalhawk_takeoff_airboyd at 27.5 s the
lines stand 161.9 Hz apart from the 7th to the 22nd and the strongest is at 3,885 Hz, 24.0 of them. Kept as a start.
"""
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import medfilt, spectrogram


def analyse(path, step=4.0, f_lo=70.0, f_hi=150.0):
    fs, x = wavfile.read(path)
    x = x.astype(float) / 32768.0
    f, t, S = spectrogram(x, fs, nperseg=32768, noverlap=24576, window="hann")
    df = f[1] - f[0]
    rows = []
    for t0 in np.arange(0.0, t[-1], step):
        sel = (t >= t0) & (t < t0 + step)
        if not sel.any():
            continue
        m = 10.0 * np.log10(S[:, sel].mean(axis=1) + 1e-20)
        prom = np.clip(m - medfilt(m, 401), 0.0, None)          # dB over the smooth floor
        best = (0.0, 0.0)
        for f1 in np.arange(f_lo, f_hi, 0.1):
            k = np.arange(6, 41)
            idx = np.round(k * f1 / df).astype(int)
            idx = idx[idx < len(prom) - 2]
            sc = np.maximum.reduce([prom[idx - 1], prom[idx], prom[idx + 1]]).sum()
            if sc > best[0]:
                best = (sc, f1)
        sc, f1 = best
        k = np.arange(6, 41)
        idx = np.round(k * f1 / df).astype(int)
        line = np.maximum.reduce([prom[idx - 1], prom[idx], prom[idx + 1]])
        band = (f > 6 * f1) & (f < 40 * f1)
        share = line.sum() / max(prom[band].sum() * 3.0 * df / f1, 1e-9)      # the comb's lines over what random lines would hold
        present = int((line > 6.0).sum())
        kb = np.arange(18, 31)
        b = int(kb[np.argmax(line[kb - 6])])
        rows.append(dict(t=t0, f1=f1, lines=present, share=share, blades=b, bpf=b * f1, bpf_db=float(line[b - 6]),
                         level=10.0 * np.log10(np.mean(x[int(t0 * fs):int((t0 + step) * fs)] ** 2) + 1e-12)))
    return rows


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("==", p.split("/")[-2])
        for r in analyse(p):
            if r["lines"] >= 10:
                print("  %4.0f s  level %6.1f dB  shaft %6.1f Hz (%4.0f rpm)  %2d of 35 multiples stand 6 dB out  strongest of 18..30: %2d "
                      "(%5.0f Hz, +%4.1f dB)" % (r["t"], r["level"], r["f1"], 60 * r["f1"], r["lines"], r["blades"], r["bpf"], r["bpf_db"]))
