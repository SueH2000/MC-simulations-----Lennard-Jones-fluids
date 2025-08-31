import numpy as np

from .pbc import apply_pbc


def compute_rdf(coordinates, box_length, dr=0.05, density=None, rcut_fraction=0.9):
    """Compute the radial distribution function g(r) for a single snapshot.

    coordinates: (N, 3) array
    box_length: simulation box length
    dr: bin width
    density: if None, computed as N / L^3
    rcut_fraction: fraction of half-box to use as maximum r
    """
    num_particles = coordinates.shape[0]
    if density is None:
        density = num_particles / (box_length ** 3)

    r_max = (box_length / 2.0) * rcut_fraction
    radii = np.arange(dr, r_max, dr)
    num_bins = len(radii)
    g_values = np.zeros(num_bins, dtype=float)

    for i in range(num_particles):
        diffs = coordinates[i] - coordinates[i + 1 :]
        if diffs.size == 0:
            continue
        dx, dy, dz = diffs[:, 0], diffs[:, 1], diffs[:, 2]
        dx, dy, dz = apply_pbc(dx, dy, dz, box_length)
        distances = np.sqrt(dx * dx + dy * dy + dz * dz)
        mask = distances < r_max
        bin_indices = (distances[mask] / dr).astype(int)
        for b in bin_indices:
            if 0 <= b < num_bins:
                g_values[b] += 2  # i-j and j-i

    for k, r in enumerate(radii):
        shell_vol = (4.0 / 3.0) * np.pi * ((r + dr) ** 3 - r ** 3)
        ideal = density * shell_vol * num_particles
        if ideal > 0:
            g_values[k] /= ideal

    return g_values, radii


def average_rdf(snapshot_list, box_length, dr=0.05, density=None, rcut_fraction=0.9):
    """Average g(r) across multiple snapshots with consistent binning.

    Returns (g_avg, radii)
    """
    results = [compute_rdf(s, box_length, dr=dr, density=density, rcut_fraction=rcut_fraction) for s in snapshot_list]
    min_len = min(len(g) for g, _ in results) if results else 0
    if min_len == 0:
        return np.array([]), np.array([])
    g_sum = np.zeros(min_len)
    radii_ref = results[0][1][:min_len]
    for g, r in results:
        g_sum += g[:min_len]
    return g_sum / len(results), radii_ref

