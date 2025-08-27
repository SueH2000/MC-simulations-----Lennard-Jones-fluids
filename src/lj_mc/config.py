import os
import yaml


DEFAULTS = {
    "nvt": {
        "num_particles": 1000,
        "density": 0.74,
        "epsilon": 1.05,
        "sigma": 1.0,
        "temperature": 1.0,
        "num_iterations": 500000,
        "max_displacement": 0.5,
        "energy_interval": 1000,
        "num_samples": 10,
        "seed": 0,
        "r2_cutoff": 2.5,
    },
    "npt": {
        "pressure": 0.023,
        "t_star": 1.0 / 1.05,
        "iterations": 100000,
        "volume_move_fraction": 0.03,
        "bins": 50,
        "seed": 0,
    },
    "rdf": {
        "dr": 0.05,
        "rcut_fraction": 0.9,
    },
}


def load_config(path=None):
    config = DEFAULTS.copy()
    if path is None:
        return config
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config path not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    # deep merge
    for section, values in user_cfg.items():
        if isinstance(values, dict) and section in config and isinstance(config[section], dict):
            config[section].update(values)
        else:
            config[section] = values
    return config

