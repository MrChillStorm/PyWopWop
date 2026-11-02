"""Gapless looping: the mixer's output is the loop repeated, sample for sample, whatever the callback sizes."""
import numpy as np

from pywopwop.player import LoopMixer


def test_mixer_wraps_sample_exactly():
    rng = np.random.default_rng(3)
    a = rng.normal(size=4801)
    b = rng.normal(size=1777)
    m = LoopMixer([a, b], [0.5, 2.0])
    sizes = rng.integers(1, 3000, size=200)
    out = np.concatenate([m.fill(int(n)) for n in sizes])
    n = len(out)
    ref = 0.5 * np.tile(a, n // len(a) + 1)[:n] + 2.0 * np.tile(b, n // len(b) + 1)[:n]
    assert np.array_equal(out, ref)
