import numpy as np

from .energy import total_energy_lj, particle_energy_lj
from .pbc import apply_pbc


def run_nvt(
    num_particles=1000,
    density=0.74,
    epsilon=1.05,
    sigma=1.0,
    temperature=1.0,
    num_iterations=500000,
    max_displacement=0.5,
    energy_interval=1000,
    num_samples=10,
    seed=0,
    r2_cutoff=2.5,
):
    """Run NVT Metropolis Monte Carlo for an LJ fluid.

    Returns a dict with energy history, sampled coordinate snapshots, final positions, and metadata.
    """
    rng = np.random.default_rng(seed)

    box_length = (num_particles / density) ** (1.0 / 3.0)
    x = (rng.random(num_particles) - 0.5) * box_length
    y = (rng.random(num_particles) - 0.5) * box_length
    z = (rng.random(num_particles) - 0.5) * box_length

    last_energy = total_energy_lj(x, y, z, box_length, epsilon, sigma, r2_cutoff)

    energy_values_sparse = []
    energy_history = []
    sampled_snapshots = []

    for iteration in range(num_iterations):
        identity = rng.integers(0, num_particles)
        old_energy_particle = particle_energy_lj(
            x, y, z, box_length, identity, epsilon, sigma, r2_cutoff
        )

        old_x, old_y, old_z = x[identity], y[identity], z[identity]

        x[identity] += (rng.random() - 0.5) * max_displacement
        y[identity] += (rng.random() - 0.5) * max_displacement
        z[identity] += (rng.random() - 0.5) * max_displacement

        x[identity], y[identity], z[identity] = apply_pbc(
            x[identity], y[identity], z[identity], box_length
        )

        new_energy_particle = particle_energy_lj(
            x, y, z, box_length, identity, epsilon, sigma, r2_cutoff
        )
        delta_u = new_energy_particle - old_energy_particle

        if rng.random() < np.exp(-delta_u):
            last_energy += delta_u
        else:
            x[identity], y[identity], z[identity] = old_x, old_y, old_z

        if num_iterations > 0 and iteration % max(1, num_iterations // 10) == 0:
            last_energy = total_energy_lj(x, y, z, box_length, epsilon, sigma, r2_cutoff)

        energy_history.append(last_energy)
        if iteration % energy_interval == 0:
            energy_values_sparse.append(last_energy)

        if iteration >= (9 * num_iterations // 10) and len(sampled_snapshots) < num_samples:
            sampled_snapshots.append(np.vstack((x, y, z)).T.copy())

    return {
        "box_length": box_length,
        "final_energy": float(last_energy),
        "energy_history": np.array(energy_history),
        "energy_values": np.array(energy_values_sparse),
        "final_positions": (x, y, z),
        "sampled_snapshots": sampled_snapshots,
        "params": {
            "num_particles": num_particles,
            "density": density,
            "epsilon": epsilon,
            "sigma": sigma,
            "temperature": temperature,
            "num_iterations": num_iterations,
            "max_displacement": max_displacement,
            "energy_interval": energy_interval,
            "num_samples": num_samples,
            "seed": seed,
            "r2_cutoff": r2_cutoff,
        },
    }

