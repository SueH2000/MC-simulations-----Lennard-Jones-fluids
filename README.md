# Lennard–Jones Monte Carlo (NVT/NPT) — Showcase

Clean, readable, *original-code-preserving* notebook for LJ fluids using Metropolis Monte Carlo.
Organized to mirror the project write‑up: **Imports & Parameters → Methods (PBC, LJ energy, NVT) → Results (RDF, NPT P(V), Density)**.

## What’s inside
- `LJ_MC.ipynb` — main, theory-first notebook (kept as the primary narrative)
- `src/lj_mc/` — small, reusable helpers (`run_nvt`, `run_npt`, `compute_rdf`)
- `configs/example.yaml` — reproducible parameters for CLI and programmatic runs
- `requirements.txt` — Python deps for quick environment setup
- `pyproject.toml` — installable package with CLI entry point `lj-mc`

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook LJ_MC.ipynb
```

### Optional: Install as a package + CLI
```bash
pip install -e .
lj-mc all --config configs/example.yaml --out outputs
```

> Tip: For GitHub preview & smaller diffs, keep `num_iterations` modest (e.g. 5e4) and avoid saving large files in the repo.

## Reproducibility
- The notebook remains the canonical, theory-rich artifact; helpers just remove boilerplate.
- Deterministic RNG via seeds; headless plotting in CLI.
- Equations/acceptance unchanged: `exp(-ΔU)` (NVT), r² cutoff; Ua/Ur with `1/T*` and `exp(-(ΔUa+ΔUr+pΔV) + N log(V’/V))` (NPT).

## Notes
- Gas-phase duplicates removed to keep one clean path (liquid). If you want gas too, add as **Appendix** or a second notebook.
- Plots render inline; `plt.savefig(...)` lines were removed to keep the repo light. Add them back if you want image assets.
.