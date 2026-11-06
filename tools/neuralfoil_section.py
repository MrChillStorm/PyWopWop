#!/usr/bin/env python3
"""A section table (presets/airfoils/<name>.json) from a coordinate file, by NeuralFoil.

    python tools/neuralfoil_section.py mh114 --reynolds 4e5 --source "MH 114, NeuralFoil, Re 4e5"

NeuralFoil gives Cl, Cd and Cm at one Reynolds number from -20 to 20 degrees; Prandtl and Glauert's factor takes Cl and Cm to
Mach 0.7, with the drag rising past the section's drag divergence as Airfoil's smooth model has it; past the stall the
section turns into a flat plate, as that model's does.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent.parent / "pywopwop" / "presets" / "airfoils"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--reynolds", type=float, default=5e5)
    ap.add_argument("--mach-dd", type=float, default=0.75)
    ap.add_argument("--source", default="NeuralFoil")
    a = ap.parse_args()
    import aerosandbox as asb
    import neuralfoil as nf
    rows = [ln.split() for ln in (DIR / f"{a.name}.dat").read_text().splitlines()]
    xy = np.array([[float(r[0]), float(r[1])] for r in rows if len(r) == 2 and r[0][0] in "0123456789.-"])
    af = asb.Airfoil(name=a.name, coordinates=xy)
    al = np.arange(-20.0, 20.01, 0.5)
    r = nf.get_aero_from_airfoil(af, alpha=al, Re=a.reynolds, model_size="xlarge")
    cl, cd, cm = (np.asarray(r[k], float) for k in ("CL", "CD", "CM"))
    grid = np.concatenate([np.arange(-180.0, -30.0, 5.0), np.arange(-30.0, -20.0, 2.0), al, np.arange(22.0, 31.0, 2.0),
                           np.arange(35.0, 180.01, 5.0)])
    rad = np.radians(grid)
    plate_cl, plate_cd = 1.05 * np.sin(2.0 * rad), 0.02 + 1.9 * np.sin(rad) ** 2
    # from the linear section to the plate over 15 to 25 degrees either side
    w = np.clip((np.abs(grid) - 15.0) / 10.0, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    CL = (1.0 - w) * np.interp(grid, al, cl) + w * plate_cl
    CD = (1.0 - w) * np.interp(grid, al, cd) + w * plate_cd
    CM = (1.0 - w) * np.interp(grid, al, cm)
    mach = np.array([0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    pg = 1.0 / np.sqrt(1.0 - np.minimum(mach, 0.8) ** 2)
    rise = 20.0 * np.maximum(mach - a.mach_dd + 0.1, 0.0) ** 4 + 0.5 * np.maximum(mach - a.mach_dd, 0.0) ** 2
    out = dict(name=a.name, source=a.source, alpha_deg=grid.tolist(), mach=mach.tolist(),
               cl=[(CL * ((1.0 - w) * p + w)).tolist() for p in pg],
               cd=[(CD + d).tolist() for d in rise], cm=[(CM * p).tolist() for p in pg])
    (DIR / f"{a.name}.json").write_text(json.dumps(out))
    print(f"{a.name}: Cl {np.interp(0.0, al, cl):.3f} at 0 deg, zero lift {np.interp(0.0, cl[(al>-10)&(al<8)], al[(al>-10)&(al<8)]):.2f} deg, "
          f"Cl max {cl.max():.2f} at {al[np.argmax(cl)]:.1f} deg, Cd min {cd.min():.4f}")


if __name__ == "__main__":
    main()
