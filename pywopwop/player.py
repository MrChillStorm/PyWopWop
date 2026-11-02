"""Gapless loop playback, for auditioning loops as a simulator plays them.

Media players loop by stopping and restarting playback (often re-initialising a resampler or refilling buffers),
which clicks. Here one audio stream stays open and the sound card's callback reads each file through a position
that wraps around inside the sample buffer: the output is exactly the loop repeated, sample for sample. Several
files (the layers of a bank, say) play together, each wrapping on its own.
"""
import numpy as np
from scipy.io import wavfile


def load(path):
    """A WAV file as float samples in -1..1 (mono: the channels averaged) and its sample rate."""
    fs, x = wavfile.read(str(path))
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max + 1)
    elif x.dtype.kind == "u":
        x = (x.astype(np.float64) - 128.0) / 128.0
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, int(fs)


class LoopMixer:
    """Each signal read cyclically from its own position, scaled and summed."""

    def __init__(self, signals, gains=None):
        self.signals = [np.asarray(s, np.float64) for s in signals]
        self.gains = list(gains) if gains is not None else [1.0] * len(self.signals)
        self.pos = [0] * len(self.signals)

    def fill(self, n):
        out = np.zeros(n)
        for i, x in enumerate(self.signals):
            idx = (self.pos[i] + np.arange(n)) % len(x)
            out += self.gains[i] * x[idx]
            self.pos[i] = (self.pos[i] + n) % len(x)
        return out


def play(paths, gains=None, volume=1.0, seconds=None, device=None, progress=print):
    """Loop the files together, gaplessly, until `seconds` have passed or Ctrl-C."""
    try:
        import sounddevice as sd
    except ImportError as e:
        raise SystemExit("playback needs the sounddevice package: pip install 'pywopwop[play]'") from e
    loaded = [load(p) for p in paths]
    rates = {fs for _, fs in loaded}
    if len(rates) > 1:
        raise SystemExit(f"the files have different sample rates: {sorted(rates)}")
    fs = rates.pop()
    mixer = LoopMixer([x for x, _ in loaded], gains)
    problems = []

    def callback(outdata, frames, time, status):
        if status:
            problems.append(str(status))
        y = np.clip(volume * mixer.fill(frames), -1.0, 1.0).astype(np.float32)
        outdata[:] = y[:, None]                 # the same on every output channel

    channels = min(2, sd.query_devices(device, kind="output")["max_output_channels"])
    lengths = ", ".join(f"{len(x) / fs:.3f} s" for x, _ in loaded)
    progress(f"looping {len(paths)} file(s) ({lengths}) at {fs} Hz; Ctrl-C to stop")
    try:
        with sd.OutputStream(samplerate=fs, channels=channels, dtype="float32", callback=callback,
                             latency="high", device=device):
            if seconds is not None:
                sd.sleep(int(seconds * 1000))
            else:
                while True:
                    sd.sleep(1000)
    except KeyboardInterrupt:
        pass
    if problems:
        progress(f"the sound card reported {len(problems)} buffer problem(s) (e.g. {problems[0]}): those were "
                 "dropouts of the system, not the loop")
