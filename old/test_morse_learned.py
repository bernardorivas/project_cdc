import itertools
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, "MorseGraph-L4DC")

from MorseGraphL4DC.grids import UniformGrid
from MorseGraphL4DC.dynamics import F_integration
from MorseGraphL4DC.systems import SwitchingSystem
from MorseGraphL4DC.analysis import full_morse_graph_analysis
from MorseGraphL4DC.plot import (plot_morse_sets, plot_basins_of_attraction,
                                  save_morse_graph_only)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Load learned Track A parameters (saved by pipeline_4d_bistable.py) ---

params = np.load("learned_params_A.npz")
W_A_np  = params["W"]    # (4, 4, 4) continuous adjacency weights
mu_A_np = params["mu"]   # (4, 4)
a_A_np  = params["alpha"] # (4,)
b_A_np  = params["beta"]  # (4,)

# --- Reconstruct system (same polynomials as ground truth) ---

c1_bi, c2_bi = 0.0, 0.0
polynomials = [
    lambda x, c=c1_bi: x[0] + x[1] - c,
    lambda x, c=c2_bi: x[2] + x[3] - c,
]

def make_bistable_vf(A, mu, alpha, beta):
    deg = A.sum(axis=1)
    def vf(x):
        return mu * x - x**3 + alpha * deg * x + beta * A @ x
    return vf

NUM_MODES = 4
learned_vfs = []
for r in range(NUM_MODES):
    W_bin = (W_A_np[r] > 0.5).astype(float)
    learned_vfs.append(make_bistable_vf(W_bin, mu_A_np[r],
                                        float(a_A_np[r]), float(b_A_np[r])))
sys_A = SwitchingSystem(polynomials, learned_vfs)

# --- Load grid from ground-truth results (same grid, no recomputation) ---

with open("morse_gt.pkl", "rb") as f:
    gt = pickle.load(f)
grid = gt["grid"]

print(f"Grid: {len(grid.get_boxes())} boxes, box size: {grid.box_size}")

# --- Compute learned Morse graph ---

timings = {}
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
tau = 1.0
F_A = F_integration(sys_A, tau, epsilon=0.0)

print("Computing learned Morse graph...")
t0 = time.perf_counter()
box_map, morse_graph, basins = full_morse_graph_analysis(grid, F_A)
timings["morse_graph_analysis"] = time.perf_counter() - t0

print(f"Morse graph nodes: {morse_graph.number_of_nodes()}")
print(f"Morse graph edges: {morse_graph.number_of_edges()}")
print(f"Basins: {len(basins)}")

# --- Save results ---

t0 = time.perf_counter()
out = {"box_map": box_map, "morse_graph": morse_graph, "basins": basins, "grid": grid}
with open("morse_learned_A.pkl", "wb") as f:
    pickle.dump(out, f)
timings["pickle_save"] = time.perf_counter() - t0
print("Saved to morse_learned_A.pkl")

# --- Save plots for all 6 projections ---

outdir = "results/morse_graph"
os.makedirs(outdir, exist_ok=True)

t0 = time.perf_counter()
save_morse_graph_only(morse_graph, os.path.join(outdir, "learned_A_morse_graph.png"))
timings["plot_morse_graph"] = time.perf_counter() - t0
print(f"Saved {outdir}/learned_A_morse_graph.png")

dim_names = ["x0", "x1", "x2", "x3"]
t0 = time.perf_counter()
for d0, d1 in itertools.combinations(range(4), 2):
    tag = f"{dim_names[d0]}_{dim_names[d1]}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Track A learned  —  projection ({dim_names[d0]}, {dim_names[d1]})")

    plot_morse_sets(grid, morse_graph, ax=axes[0], dims=(d0, d1))
    axes[0].set_xlabel(f"${dim_names[d0]}$")
    axes[0].set_ylabel(f"${dim_names[d1]}$")
    axes[0].set_title("Morse sets")

    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=axes[1], dims=(d0, d1))
    axes[1].set_xlabel(f"${dim_names[d0]}$")
    axes[1].set_ylabel(f"${dim_names[d1]}$")
    axes[1].set_title("Basins")

    plt.tight_layout()
    path = os.path.join(outdir, f"learned_A_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
timings["plot_projections"] = time.perf_counter() - t0

# --- Write timing markdown ---

bound = float(grid.bounds[1, 0])
n_boxes = len(grid.get_boxes())
md_lines = [
    "# Morse Graph Track A Learned — Timing",
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
    "",
    "## Learned parameters (Track A)",
    "",
    "| Mode | alpha | beta | mu |",
    "|------|-------|------|----|",
]
for r in range(NUM_MODES):
    md_lines.append(
        f"| {r} | {float(a_A_np[r]):.4f} | {float(b_A_np[r]):.4f} "
        f"| {np.round(mu_A_np[r], 3).tolist()} |"
    )

md_path = os.path.join(outdir, "learned_A_timing.md")
with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Saved {md_path}")
