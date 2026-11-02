"""Section tables and shapes."""
import numpy as np

from pywopwop.airfoils import load_table, section_shape
from pywopwop.model import Airfoil


def test_tables_are_smooth_and_wrap():
    t = load_table("naca23012_c81")
    a = np.radians(np.linspace(-10, 12, 2001))
    cl = t(a, np.full_like(a, 0.5), "cl")
    # no kinks: the second difference stays small where the table is attached flow
    assert np.max(np.abs(np.diff(cl, 2))) < 1e-3
    assert abs(float(t(np.radians(181.0), 0.4, "cl")) - float(t(np.radians(-179.0), 0.4, "cl"))) < 1e-9


def test_shapes():
    h, c, tc = section_shape("naca0012")
    assert abs(tc - 0.12) < 1e-3 and np.max(np.abs(c(np.linspace(0, 1, 11)))) == 0.0
    h, c, tc = section_shape("naca23012")
    x = np.linspace(0, 1, 2001)
    assert abs(x[np.argmax(c(x))] - 0.15) < 0.01 and abs(c(x).max() - 0.0184) < 0.001
    h, c, tc = section_shape("sc1094r8")
    assert 0.09 < tc < 0.1


def test_airfoil_uses_table_and_shape():
    af = Airfoil(table="sc1095_genhel", shape="sc1094r8")
    cl, cd = af.coefficients(np.radians(6.0), 0.5)
    assert abs(cl - 0.78) < 0.01
    assert abs(af.thickness - 0.094) < 0.002
    assert 5.5 < af.lift_slope < 7.0
