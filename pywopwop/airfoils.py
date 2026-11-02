"""Blade sections: aerodynamic tables (lift, drag and moment against angle of attack and Mach number) and shapes
(coordinates, or the NACA four- and five-digit definitions).

Tables are stored as JSON in presets/airfoils (made from C-81 decks and similar by tools/import_fdm_airfoils.py)
and resampled once onto a fine grid by monotone cubic interpolation in both angle and Mach number, so the loads stay
smooth as a blade sweeps through Mach number and angle (a table's kinks would otherwise ring in the loading noise).
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

AIRFOIL_DIR = Path(__file__).parent / "presets" / "airfoils"


def _resolve(name, suffixes):
    p = Path(name)
    if p.exists():
        return p
    for s in suffixes:
        q = AIRFOIL_DIR / f"{name}{s}"
        if q.exists():
            return q
    raise FileNotFoundError(f"no airfoil '{name}' in {AIRFOIL_DIR}")


class SectionTable:
    """Cl, Cd, Cm(alpha deg, Mach) on a fine regular grid; bilinear lookup."""
    DA = 0.25
    DM = 0.01

    def __init__(self, d):
        self.name = d.get("name", "")
        self.source = d.get("source", "")
        alpha = np.asarray(d["alpha_deg"], float)
        mach = np.asarray(d["mach"], float)
        self.a = np.arange(-180.0, 180.0 + self.DA / 2, self.DA)
        self.m = np.arange(0.0, 1.0 + self.DM / 2, self.DM)
        self.tab = {}
        for k in ("cl", "cd", "cm"):
            if d.get(k) is None:
                continue
            v = np.asarray(d[k], float)                       # (n_mach, n_alpha)
            fine_a = np.array([PchipInterpolator(alpha, row)(np.clip(self.a, alpha[0], alpha[-1])) for row in v])
            if len(mach) > 1:
                fine = PchipInterpolator(mach, fine_a, axis=0)(np.clip(self.m, mach[0], mach[-1]))
            else:
                fine = np.repeat(fine_a, len(self.m), axis=0)
            self.tab[k] = np.ascontiguousarray(fine)           # (n_m, n_a)

    def __call__(self, alpha_rad, mach, key):
        a = np.degrees(np.asarray(alpha_rad, float))
        a = (a + 180.0) % 360.0 - 180.0
        fa = (a + 180.0) / self.DA
        ia = np.clip(np.floor(fa).astype(int), 0, len(self.a) - 2)
        wa = fa - ia
        fm = np.clip(np.asarray(mach, float), 0.0, 1.0) / self.DM
        im = np.clip(np.floor(fm).astype(int), 0, len(self.m) - 2)
        wm = fm - im
        t = self.tab[key]
        return ((1 - wm) * ((1 - wa) * t[im, ia] + wa * t[im, ia + 1])
                + wm * ((1 - wa) * t[im + 1, ia] + wa * t[im + 1, ia + 1]))

    def has(self, key):
        return key in self.tab

    def slope_and_zero(self, mach=0.3):
        """Lift slope (per rad) and zero-lift angle (deg) near zero lift."""
        a = np.radians(np.linspace(-4.0, 4.0, 81))
        cl = self(a, np.full_like(a, mach), "cl")
        p = np.polyfit(a, cl, 1)
        return p[0] * np.sqrt(1.0 - mach ** 2), float(np.degrees(-p[1] / p[0]))


@lru_cache(maxsize=None)
def load_table(name):
    return SectionTable(json.loads(_resolve(name, (".json",)).read_text()))


# ---------------------------------------------------------------- shapes

def naca4_thickness(x, t):
    x = np.asarray(x, float)
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)


def _naca(spec):
    """(half thickness, camber) functions of x/c for 'nacaMPXX' or 'naca230XX' (five-digit, 230 mean line)."""
    d = spec.lower().replace("naca", "").replace(" ", "")
    if len(d) == 4:
        m, p, t = int(d[0]) / 100.0, int(d[1]) / 10.0, int(d[2:]) / 100.0

        def camber(x):
            x = np.asarray(x, float)
            if m == 0:
                return np.zeros_like(x)
            return np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2),
                            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    elif len(d) == 5 and d[:3] in ("210", "220", "230", "240", "250"):
        # standard five-digit mean lines (Abbott and von Doenhoff, table of r and k1)
        r, k1 = {"210": (0.0580, 361.4), "220": (0.1260, 51.64), "230": (0.2025, 15.957), "240": (0.2900, 6.643),
                 "250": (0.3910, 3.230)}[d[:3]]
        t = int(d[3:]) / 100.0

        def camber(x):
            x = np.asarray(x, float)
            return np.where(x < r, k1 / 6.0 * (x ** 3 - 3 * r * x ** 2 + r ** 2 * (3 - r) * x),
                            k1 * r ** 3 / 6.0 * (1 - x))
    else:
        raise ValueError(f"unknown NACA designation {spec}")
    return (lambda x: naca4_thickness(x, t)), camber, t


def _coordinates(path):
    """(half thickness, camber, t/c) from a Selig-ordered coordinate file (trailing edge, upper surface, leading edge,
    lower surface, trailing edge), one header line."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    xy = np.array(rows)
    le = int(np.argmin(xy[:, 0]))
    up = xy[:le + 1][::-1]
    lo = xy[le:]
    x0, x1 = xy[le, 0], max(up[-1, 0], lo[-1, 0])
    xs = np.linspace(0.0, 1.0, 401) * (x1 - x0) + x0
    yu = np.interp(xs, up[:, 0], up[:, 1])
    yl = np.interp(xs, lo[:, 0], lo[:, 1])
    xn = (xs - x0) / (x1 - x0)
    c = x1 - x0
    half = (yu - yl) / 2.0 / c
    camb = (yu + yl) / 2.0 / c
    camb = camb - (camb[0] + (camb[-1] - camb[0]) * xn)     # mean line from leading to trailing edge
    return (lambda x: np.interp(x, xn, half)), (lambda x: np.interp(x, xn, camb)), float(2 * half.max())


@lru_cache(maxsize=None)
def section_shape(spec):
    """(half thickness, camber, t/c) of a section: a NACA designation or a coordinate file name."""
    if spec.lower().startswith("naca"):
        return _naca(spec)
    return _coordinates(_resolve(spec, (".dat", ".txt", "")))
