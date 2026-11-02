"""Turn section tables of the FlightGear JSBSim rotorcraft models into PyWopWop airfoil tables.

    python tools/import_fdm_airfoils.py ~/git/Bo105 ~/git/UH-60
    python tools/import_fdm_airfoils.py --v22 ~/git/V-22

reads Bo105/fdm/data/c81_23012.py (C-81 airfoil table 666 of USAAMRDL TR-75-50), Bo105/fdm/data/c81_0012.py
(C-81's 0012 equations as the TR-75-50 deck sets them), UH-60/fdm/data/genhel_sc1095.py (GenHel's SC1095 maps,
NASA CR-166309) and UH-60/fdm/data/sc1094r8.dat (NASA TP-2003-212265), and writes pywopwop/presets/airfoils/.
With --v22, V-22/fdm/data/sections.py: the V-22 model's proprotor sections in C-81's equation form (the Boeing
XN-series' tables aren't public), the one at 0.75 R (XN12: 12 % thick, design lift 0.15 from V-22/fdm/v22data.py).
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "pywopwop" / "presets" / "airfoils"


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(m)
    sys.path.pop(0)
    return m


def _grid(tables):
    """Put {coef: (alphas, machs, values[alpha][mach])} on the union of their angle and Mach grids."""
    alpha = sorted(set(a for al, _, _ in tables.values() for a in al))
    mach = sorted(set(m for _, ms, _ in tables.values() for m in ms))
    out = {}
    for k, (al, ms, v) in tables.items():
        v = np.asarray(v, float)                     # (n_alpha, n_mach)
        by_m = np.array([np.interp(mach, ms, row) for row in v])          # (n_alpha, n_mach_union)
        out[k] = np.array([np.interp(alpha, al, by_m[:, j]) for j in range(len(mach))]).round(5).tolist()
    return alpha, mach, out


def write(name, source, tables):
    alpha, mach, t = _grid(tables)
    d = dict(name=name, source=source, alpha_deg=alpha, mach=mach, cl=t.get("cl"), cd=t.get("cd"), cm=t.get("cm"))
    (OUT / f"{name}.json").write_text(json.dumps(d))
    print(f"wrote {OUT / (name + '.json')}: {len(alpha)} angles x {len(mach)} Mach numbers, "
          f"{', '.join(k for k in ('cl', 'cd', 'cm') if d[k] is not None)}")


def c81_dict(m, coef, machs):
    al = sorted(coef)
    return al, machs, [coef[a] for a in al]


def main(bo105, uh60):
    OUT.mkdir(parents=True, exist_ok=True)
    m = _module(Path(bo105) / "fdm" / "data" / "c81_23012.py")
    write("naca23012_c81", "Boeing Vertol C-81 airfoil table 666 (NACA 23012), USAAMRDL TR-75-50 table A-1",
          dict(cl=c81_dict(m, m.CL, m.CL_MACH), cd=c81_dict(m, m.CD, m.CD_MACH), cm=c81_dict(m, m.CM, m.CM_MACH)))
    m = _module(Path(bo105) / "fdm" / "data" / "c81_0012.py")
    al, ms, cl, cd = m.tables()
    write("naca0012_c81", "C-81 rotor airfoil equations for the NACA 0012, USAAMRDL TR-75-50 cards 32A-32E",
          dict(cl=(al, ms, cl), cd=(al, ms, cd)))
    m = _module(Path(uh60) / "fdm" / "data" / "genhel_sc1095.py")
    al = list(m.ALPHA_HI)
    lo = set(m.ALPHA_LO)
    cl, cd = [], []
    for a_i, a in enumerate(al):
        if a in lo:
            j = m.ALPHA_LO.index(a)
            cl.append([m.CL_LO[M][j] for M in m.MACH])
            cd.append([m.CD_LO[M][j] for M in m.MACH])
        else:
            cl.append([m.CL_HI[a_i]] * len(m.MACH))
            cd.append([m.CD_HI[a_i]] * len(m.MACH))
    write("sc1095_genhel", "Sikorsky GenHel SC1095 tables 1.5.2-1.5.3, NASA CR-166309",
          dict(cl=(al, m.MACH, cl), cd=(al, m.MACH, cd)))
    src = Path(uh60) / "fdm" / "data" / "sc1094r8.dat"
    shutil.copy(src, OUT / "sc1094r8.dat")
    print(f"copied {src}")


def main_v22(v22):
    OUT.mkdir(parents=True, exist_ok=True)
    m = _module(Path(v22) / "fdm" / "data" / "sections.py")
    t, cl_i = next((t, c) for r, t, c in _module(Path(v22) / "fdm" / "v22data.py").SECTIONS if r == 0.75)
    al, ms, cl, cd = m.Section(t, cl_i).tables()
    write("xn12_v22", f"V-22 FlightGear model's XN12 section ({t:.0%} thick, design lift {cl_i}): C-81's airfoil "
          "equations with the parameters of a cambered section, fitted to the XV-15 rotor's hover test (NASA TM-86833)",
          dict(cl=(al, ms, cl), cd=(al, ms, cd)))


if __name__ == "__main__":
    if sys.argv[1] == "--v22":
        main_v22(sys.argv[2])
    else:
        main(*sys.argv[1:3])
