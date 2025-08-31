import numpy as np

from lj_mc.rdf import compute_rdf


def test_rdf_basic_shape_and_bounds():
    # simple cubic arrangement with small N
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    L = 10.0
    g, r = compute_rdf(coords, L, dr=0.1, density=4 / L**3)
    assert g.ndim == 1 and r.ndim == 1
    assert g.size == r.size
    assert np.all(g >= 0)

