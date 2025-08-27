import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config
from .nvt import run_nvt
from .rdf import average_rdf
from .npt import run_npt


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cmd_nvt(args):
    cfg = load_config(args.config)
    params = cfg["nvt"]
    result = run_nvt(**params)
    outdir = args.out
    _ensure_dir(outdir)
    # save energy plot
    plt.figure(figsize=(10, 5))
    history = result["energy_history"]
    plt.plot(range(len(history)), history)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy vs Iteration (NVT)")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "energy_nvt.png"), dpi=150, bbox_inches="tight")
    plt.close()
    # save metadata
    meta = {
        "final_energy": result["final_energy"],
        "box_length": result["box_length"],
        "params": result["params"],
    }
    with open(os.path.join(outdir, "nvt_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # save a few snapshots
    for idx, snap in enumerate(result["sampled_snapshots"]):
        snap_path = os.path.join(outdir, f"snapshot_{idx:02d}.npy")
        import numpy as np

        np.save(snap_path, snap)
    return result


def cmd_rdf(args, prev=None):
    cfg = load_config(args.config)
    rdf_cfg = cfg["rdf"]
    outdir = args.out
    _ensure_dir(outdir)
    if prev is None:
        # quick NVT to get snapshots
        nvt_cfg = cfg["nvt"].copy()
        nvt_cfg["num_iterations"] = max(20000, nvt_cfg["num_iterations"])  # ensure some tail
        prev = run_nvt(**nvt_cfg)
    g_avg, radii = average_rdf(prev["sampled_snapshots"], prev["box_length"], dr=rdf_cfg["dr"], rcut_fraction=rdf_cfg["rcut_fraction"], density=prev["params"]["density"])
    # plot
    plt.figure(figsize=(8, 5))
    import numpy as np

    if g_avg.size:
        plt.plot(radii, g_avg, label="Averaged g(r)")
        plt.xlabel("r / sigma")
        plt.ylabel("g(r)")
        plt.title("Radial Distribution Function")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(outdir, "rdf.png"), dpi=150, bbox_inches="tight")
        plt.close()
    return {"g": g_avg.tolist() if g_avg.size else [], "r": radii.tolist() if g_avg.size else []}


def cmd_npt(args, prev=None):
    cfg = load_config(args.config)
    npt_cfg = cfg["npt"]
    outdir = args.out
    _ensure_dir(outdir)
    if prev is None:
        prev = run_nvt(**cfg["nvt"])  # use default NVT first
    x, y, z = prev["final_positions"]
    result = run_npt(x, y, z, prev["box_length"], **npt_cfg)
    # Plot P(V)
    plt.figure(figsize=(8, 5))
    import numpy as np

    bins = result["bin_edges"]
    pv = result["pv_hist"]
    plt.plot(bins, pv)
    plt.xlabel("Volume (bin)")
    plt.ylabel("Probability")
    plt.title("P(V) histogram (NPT)")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "pv_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="LJ Monte Carlo CLI")
    parser.add_argument("command", choices=["nvt", "rdf", "npt", "all"], help="What to run")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args(argv)

    if args.command == "nvt":
        cmd_nvt(args)
    elif args.command == "rdf":
        prev = cmd_nvt(args)  # run NVT then RDF by default
        cmd_rdf(args, prev=prev)
    elif args.command == "npt":
        prev = cmd_nvt(args)
        cmd_npt(args, prev=prev)
    elif args.command == "all":
        prev = cmd_nvt(args)
        cmd_rdf(args, prev=prev)
        cmd_npt(args, prev=prev)


if __name__ == "__main__":
    main()

