import numpy as np

from .pbc import apply_pbc


def _ua_ur_split(x, y, z, box_length, epsilon=1.05, sigma=1.0, inv_tstar=1.0):
    """Compute attractive and repulsive contributions (scaled by 1/T*) as in notebook."""
    ua = 0.0
    ur = 0.0
    num_particles = len(x)
    for i in range(num_particles - 1):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]
        dz = z[i] - z[i + 1 :]
        dx, dy, dz = apply_pbc(dx, dy, dz, box_length)
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        # Avoid r=0; pairs exclude self
        inv_r6 = (sigma / r) ** 6
        ua += np.sum(-4.0 * epsilon * inv_tstar * inv_r6)
        ur += np.sum(4.0 * epsilon * inv_tstar * (inv_r6 ** 2))
    return ua, ur


def run_npt(
    x, y, z,
    box_length,
    pressure=0.023,
    epsilon=1.05,
    sigma=1.0,
    t_star=1.0/1.05,
    iterations=100000,
    volume_move_fraction=0.03,
    bins=50,
    seed=0,
):
    """Run NPT Monte Carlo volume moves using Ua/Ur split and acceptance from the notebook.

    Returns a dict with arrays for volume history and histogram P(V).
    """
    rng = np.random.default_rng(seed)
    num_particles = len(x)

    volume_initial = box_length ** 3
    dV_max = volume_initial * volume_move_fraction
    inv_tstar = 1.0 / t_star

    ua, ur = _ua_ur_split(x, y, z, box_length, epsilon=epsilon, sigma=sigma, inv_tstar=inv_tstar)

    v_old = volume_initial
    l_current = box_length
    v_hist = [v_old]
    pv_hist = np.zeros(bins, dtype=float)
    min_v = 0.8 * volume_initial
    max_v = 1.2 * volume_initial

    for it in range(iterations):
        dV = (rng.random() - 0.5) * dV_max
        v_prop = v_old + dV
        if v_prop <= 0.0:
            v_hist.append(v_old)
            continue
        l_prop = v_prop ** (1.0 / 3.0)
        ua_prop = ua * (l_current / l_prop) ** 6
        ur_prop = ur * (l_current / l_prop) ** 12
        dUa = ua_prop - ua
        dUr = ur_prop - ur
        acc = np.exp(-(dUa + dUr + pressure * dV) + num_particles * np.log(v_prop / v_old))
        if rng.random() < acc:
            scale = l_prop / l_current
            x *= scale; y *= scale; z *= scale
            v_old = v_prop
            l_current = l_prop
            ua = ua_prop
            ur = ur_prop

        v_hist.append(v_old)
        b = int(np.floor(((v_old - min_v) / (max_v - min_v)) * bins))
        b = max(0, min(b, bins - 1))
        pv_hist[b] += 1

        if iterations > 0 and it % max(1, iterations // 10) == 0:
            ua, ur = _ua_ur_split(x, y, z, l_current, epsilon=epsilon, sigma=sigma, inv_tstar=inv_tstar)

    if pv_hist.sum() > 0:
        pv_hist = pv_hist / pv_hist.sum()

    return {
        "volume_history": np.array(v_hist),
        "pv_hist": pv_hist,
        "bin_edges": np.linspace(min_v, max_v, bins, endpoint=False),
        "final_box_length": l_current,
        "final_positions": (x, y, z),
        "params": {
            "pressure": pressure,
            "epsilon": epsilon,
            "sigma": sigma,
            "t_star": t_star,
            "iterations": iterations,
            "volume_move_fraction": volume_move_fraction,
            "bins": bins,
        },
    }

