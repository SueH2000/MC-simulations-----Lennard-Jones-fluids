import numpy as np

from .pbc import apply_pbc


def _lennard_jones_phi_from_r2(r2, epsilon, sigma):
    """Compute Lennard–Jones pair potential from squared distance r^2.

    Handles r2 == 0 by returning 0 contribution (self-interactions excluded by callers).
    """
    # Avoid division by zero
    safe = r2 > 0
    phi = np.zeros_like(r2, dtype=float)
    if np.any(safe):
        inv_r2 = 1.0 / r2[safe]
        inv_r6 = (sigma ** 6) * (inv_r2 ** 3)
        inv_r12 = inv_r6 ** 2
        phi[safe] = 4.0 * epsilon * (inv_r12 - inv_r6)
    return phi


def total_energy_lj(x, y, z, box_length, epsilon=1.05, sigma=1.0, r2_cutoff=2.5):
    """Total LJ energy using vectorized pair loops and r^2 cutoff as in the notebook."""
    num_particles = len(x)
    total = 0.0
    for i in range(num_particles - 1):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]
        dz = z[i] - z[i + 1 :]
        dx, dy, dz = apply_pbc(dx, dy, dz, box_length)
        r2 = dx * dx + dy * dy + dz * dz
        phi = _lennard_jones_phi_from_r2(r2, epsilon=epsilon, sigma=sigma)
        phi[r2 >= r2_cutoff] = 0.0
        total += np.sum(phi)
    return total


def particle_energy_lj(x, y, z, box_length, index, epsilon=1.05, sigma=1.0, r2_cutoff=2.5):
    """Energy contribution for a single particle against all others (used for dU)."""
    mask = np.arange(len(x)) != index
    dx = x[index] - x[mask]
    dy = y[index] - y[mask]
    dz = z[index] - z[mask]
    dx, dy, dz = apply_pbc(dx, dy, dz, box_length)
    r2 = dx * dx + dy * dy + dz * dz
    phi = _lennard_jones_phi_from_r2(r2, epsilon=epsilon, sigma=sigma)
    inside = r2 < r2_cutoff
    return float(np.sum(phi[inside]))

