import numpy as np

from lj_mc.energy import total_energy_lj, particle_energy_lj


def test_energy_symmetry_and_cutoff():
    # two particles at r = 2 sigma (within cutoff)
    x = np.array([0.0, 2.0])
    y = np.array([0.0, 0.0])
    z = np.array([0.0, 0.0])
    L = 10.0
    e_total = total_energy_lj(x, y, z, L, epsilon=1.0, sigma=1.0, r2_cutoff=100.0)
    # For LJ at r=2: 4[(1/2)^12 - (1/2)^6] = 4(1/4096 - 1/64) = negative
    assert e_total < 0

    # particle energy should sum to total when both contributions counted once
    e0 = particle_energy_lj(x, y, z, L, 0, epsilon=1.0, sigma=1.0, r2_cutoff=100.0)
    e1 = particle_energy_lj(x, y, z, L, 1, epsilon=1.0, sigma=1.0, r2_cutoff=100.0)
    # In this definition, total_energy sums each pair once; particle_energy sums i vs all others.
    # For two particles, e_total == e0 == e1.
    assert np.isclose(e_total, e0)
    assert np.isclose(e_total, e1)

