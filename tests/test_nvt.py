import numpy as np

from lj_mc.nvt import run_nvt


def test_nvt_runs_and_collects_snapshots():
    result = run_nvt(num_particles=64, density=0.5, num_iterations=2000, energy_interval=100, num_samples=3, seed=42)
    assert "final_energy" in result and isinstance(result["final_energy"], float)
    assert result["energy_history"].size == 2000
    assert len(result["sampled_snapshots"]) <= 3
    x, y, z = result["final_positions"]
    assert len(x) == 64 and len(y) == 64 and len(z) == 64

