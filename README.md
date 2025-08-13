# Lennard–Jones Monte Carlo (NVT/NPT) — Showcase

Clean, readable, *original-code-preserving* notebook for LJ fluids using Metropolis Monte Carlo.
Organized to mirror the project write‑up: **Imports & Parameters → Methods (PBC, LJ energy, NVT) → Results (RDF, NPT P(V), Density)**.

## What’s inside
- `LJ_MC.ipynb` — main notebook (your original code + comments, lightly structured)
- `requirements.txt` — minimal Python deps

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook LJ_MC.ipynb
```

> Tip: For GitHub preview & smaller diffs, keep `num_iterations` modest (e.g. 5e4) and avoid saving large files in the repo.

## Reproducibility
- Uses your existing `np.random.seed(seed)` pattern.
- Equations/acceptance unchanged: `exp(-ΔU)` in NVT; r² cutoff; Ua/Ur with `1/T*` and `exp(-(ΔUa+ΔUr+pΔV) + N log(V’/V))` in NPT.

## Notes
- Gas-phase duplicates removed to keep one clean path (liquid). If you want gas too, add as **Appendix** or a second notebook.
- Plots render inline; `plt.savefig(...)` lines were removed to keep the repo light. Add them back if you want image assets.
.