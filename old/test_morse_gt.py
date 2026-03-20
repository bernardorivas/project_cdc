import pickle
import sys
import time
from datetime import datetime
import numpy as np

sys.path.insert(0, "MorseGraph-L4DC")

import itertools
import os

from MorseGraphL4DC.grids import UniformGrid
from MorseGraphL4DC.dynamics import F_integration
from MorseGraphL4DC.systems import SwitchingSystem
from MorseGraphL4DC.analysis import full_morse_graph_analysis
from MorseGraphL4DC.plot import (plot_morse_sets, plot_basins_of_attraction,
                                  save_morse_graph_only)

# --- Ground-truth system ---

c1_bi, c2_bi = 0.0, 0.0
polynomials = [
    lambda x, c=c1_bi: x[0] + x[1] - c,
    lambda x, c=c2_bi: x[2] + x[3] - c,
]

A0 = np.array([[0,1,1,1],[1,0,1,0],[1,1,0,1],[1,0,1,0]], dtype=float)
A1 = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]], dtype=float)
A2 = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]], dtype=float)

mode_params = [
    {"A": A0, "mu": np.array([1.0, 1.0, 1.0, 1.0]), "alpha":  0.05, "beta": -0.05},
    {"A": A1, "mu": np.array([0.8, 0.6, 0.8, 0.6]), "alpha":  0.08, "beta": -0.04},
    {"A": A2, "mu": np.array([0.6, 0.8, 0.6, 0.8]), "alpha":  0.06, "beta": -0.06},
    {"A": A2, "mu": np.array([0.5, 0.5, 0.5, 0.5]), "alpha":  0.10, "beta": -0.03},
]

def make_bistable_vf(A, mu, alpha, beta):
    deg = A.sum(axis=1)
    def vf(x):
        return mu * x - x**3 + alpha * deg * x + beta * A @ x
    return vf

vfs = [make_bistable_vf(**m) for m in mode_params]
system = SwitchingSystem(polynomials, vfs)

# --- Grid (matches notebook: [-2,2]^4, 8^4 boxes) ---

bound = 2.0
grid = UniformGrid(
    bounds=np.array([np.full(4, -bound), np.full(4, bound)]),
    divisions=np.full(4, 8, dtype=int),
)
print(f"Grid: {len(grid.get_boxes())} boxes, box size: {grid.box_size}")

timings = {}
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Compute ground-truth Morse graph ---

tau = 1.0
F_gt = F_integration(system, tau, epsilon=0.0)

print("Computing Morse graph...")
t0 = time.perf_counter()
box_map, morse_graph, basins = full_morse_graph_analysis(grid, F_gt)
timings["morse_graph_analysis"] = time.perf_counter() - t0

print(f"Morse graph nodes: {morse_graph.number_of_nodes()}")
print(f"Morse graph edges: {morse_graph.number_of_edges()}")
print(f"Basins: {len(basins)}")

t0 = time.perf_counter()
out = {"box_map": box_map, "morse_graph": morse_graph, "basins": basins, "grid": grid}
with open("morse_gt.pkl", "wb") as f:
    pickle.dump(out, f)
timings["pickle_save"] = time.perf_counter() - t0
print("Saved to morse_gt.pkl")

# --- Save plots for all 6 projections ---

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = "results/morse_graph"
os.makedirs(outdir, exist_ok=True)

t0 = time.perf_counter()
save_morse_graph_only(morse_graph, os.path.join(outdir, "gt_morse_graph.png"))
timings["plot_morse_graph"] = time.perf_counter() - t0
print(f"Saved {outdir}/gt_morse_graph.png")

dim_names = ["x0", "x1", "x2", "x3"]
t0 = time.perf_counter()
for d0, d1 in itertools.combinations(range(4), 2):
    tag = f"{dim_names[d0]}_{dim_names[d1]}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Ground truth  —  projection ({dim_names[d0]}, {dim_names[d1]})")

    plot_morse_sets(grid, morse_graph, ax=axes[0], dims=(d0, d1))
    axes[0].set_xlabel(f"${dim_names[d0]}$")
    axes[0].set_ylabel(f"${dim_names[d1]}$")
    axes[0].set_title("Morse sets")

    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=axes[1], dims=(d0, d1))
    axes[1].set_xlabel(f"${dim_names[d0]}$")
    axes[1].set_ylabel(f"${dim_names[d1]}$")
    axes[1].set_title("Basins")

    plt.tight_layout()
    path = os.path.join(outdir, f"gt_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
timings["plot_projections"] = time.perf_counter() - t0

# --- Write timing markdown ---

n_boxes = len(grid.get_boxes())
md_lines = [
    "# Morse Graph Ground Truth — Timing",
    "",
    f"**Run:** {run_timestamp}  ",
    f"**Grid:** `[-{bound}, {bound}]^4`, `{n_boxes}` boxes, box size `{grid.box_size}`  ",
    f"**tau:** `{tau}`  ",
    "",
    "| Step | Time (s) |",
    "|------|----------|",
    f"| `full_morse_graph_analysis` | {timings['morse_graph_analysis']:.2f} |",
    f"| Pickle save | {timings['pickle_save']:.2f} |",
    f"| Morse graph plot | {timings['plot_morse_graph']:.2f} |",
    f"| 6 projection plots | {timings['plot_projections']:.2f} |",
    f"| **Total** | **{sum(timings.values()):.2f}** |",
    "",
    "## Results",
    "",
    f"- Morse graph nodes: {morse_graph.number_of_nodes()}",
    f"- Morse graph edges: {morse_graph.number_of_edges()}",
    f"- Basins: {len(basins)}",
]

md_path = os.path.join(outdir, "gt_timing.md")
with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Saved {md_path}")
