# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository (`cdc_kaitomaani`) contains research on **hybrid dynamical system identification** combined with **computational topology (Morse graph analysis)**. The two main components are:

1. **Jupyter notebooks** (root level) -- hybrid system identification pipelines that learn switched dynamical systems from data, then validate them topologically.
2. **MorseGraph-L4DC** (subdir) -- a Python library for computing Morse graphs, Morse sets, and basins of attraction via combinatorial Conley-Morse theory on uniform grids.

The scientific goal: given a switched dynamical system with multiple modes (each with its own graph topology, coupling parameters, and local dynamics), learn the mode bank and a discrete selector from trajectory data, then compare the **topological structure** (Morse decomposition) of the learned system against the ground truth.

**`MorseGraph/`** contains the CMGDB library (C++ backend with Python bindings, adaptive subdivision). It is used in Section 11 of `morse_graph.ipynb` via `import CMGDB` for per-mode ODE-based Morse graphs (`CMGDB.BoxMap`, `CMGDB.BoxMapData`, `CMGDB.ComputeMorseGraph`, `CMGDB.PlotMorseGraph/PlotMorseSets`). **`MorseGraph-L4DC/`** is a lightweight Python-only reimplementation used in Sections 10 and earlier.

## Environment and setup

- Python 3.14, venv at `.venv/`
- Key dependencies: torch, numpy, scipy, matplotlib, networkx, joblib, scikit-learn
- MorseGraph-L4DC is not pip-installed; notebooks add it to sys.path via `sys.path.insert(0, os.path.join(os.getcwd(), 'MorseGraph-L4DC'))`
- Run notebooks with the `.venv` kernel
- No test suite or CI. Validation is done through the example scripts and notebooks.

## Notebooks

- **`morse_graph.ipynb`** -- main end-to-end pipeline: 4D bistable network with 4 modes, Tracks A & B identification, then Morse graph comparison (piecewise-constant switching in Sections 1-9, data-driven Morse graph via MorseGraph-L4DC in Section 10, CMGDB-based Morse graph with per-mode ODE analysis in Section 11).
- **`hybrid_checkpoint_standalone_A_B_tracks_fixed copy.ipynb`** -- standalone checkpoint of the hybrid identification pipeline (Tracks A & B), without the Morse graph validation stage.
- Output plots go to `results/`.

## Architecture: hybrid identification notebooks

The notebooks follow a two-track pattern:

- **Track A (physics-informed)**: assumes the local dynamics family is known (e.g., Sprott, bistable). Learns per-mode parameters (adjacency W, coupling alpha/beta, bifurcation mu) via a `LearnableModeBank` with sigmoid-parameterized graphs. Uses supervised mode labels for the DHAL selector.
- **Track B (reduced-structure)**: replaces exact dynamics with shared MLP + per-mode residual MLP, keeping only graph-coupling structure. DHAL selector trained **unsupervised** (no mode labels).

Training is staged: (1) train bank with known mode labels, (2) train GRU selector (with straight-through Gumbel-softmax), (3) optional short joint fine-tuning. The selector architecture (`GRUSelector` + `st_gumbel_onehot`) is shared across tracks.

Key model classes in notebooks:
- `PhysicsInformedBank` / `ReducedStructureBank` -- mode banks
- `GRUSelector` -- history-window encoder producing mode logits
- `HybridSelectorModel` -- combines bank + selector
- `SwitchingSystem` (from MorseGraph-L4DC) -- ground-truth state-dependent switching

## Architecture: MorseGraph-L4DC library

Located at `MorseGraph-L4DC/MorseGraphL4DC/`. Core pipeline:

```
UniformGrid  -->  Dynamics (F_*)  -->  Model.compute_box_map()  -->  compute_morse_graph()  -->  compute_all_morse_set_basins()
```

**Core modules:**
- `grids.py`: `UniformGrid(bounds=np.array([[lo],[hi]]), divisions=np.array([n]))`. Boxes are shape `(N, 2, D)`.
- `dynamics.py`: Four outer-approximation strategies:
  - `F_integration(ode_f, tau, epsilon)` -- ODE integration (ground truth). Auto-detects `SwitchingSystem` and creates event functions for switching surfaces.
  - `F_data(X, Y, grid, ...)` -- data-driven with KD-tree neighborhoods.
  - `F_Lipschitz(map_f, L_tau, box_diameter, epsilon)` -- Lipschitz-based rigorous bounds.
  - `F_gaussianprocess(gp_model, confidence_level)` -- GP regression with Bonferroni-corrected confidence. **Not exported from `__init__.py`**; import directly from `dynamics.py`.
- `systems.py`: `SwitchingSystem(polynomials, vector_fields)`. Mode index via binary encoding of polynomial sign patterns: `sigma(x) = sum((p_i(x) > 0) * 2^i)`. Expects exactly `2^k` vector fields for `k` polynomials. Callable as `__call__(t, x)` for scipy integration.
- `analysis.py`: `full_morse_graph_analysis(grid, F)` returns `(box_map, morse_graph, basins)`. Morse graph nodes are `frozenset` of box indices. Basins are disjoint.
- `core.py`: `Model(grid, dynamics).compute_box_map(n_jobs=-1)` -- parallelized box map computation via joblib.

**Supporting modules:**
- `cache.py`: Persistent caching for box maps, Morse graphs, basins (`save_method_results`, `load_method_results`, `cache_exists`, `clear_cache`).
- `postprocessing.py`: System-specific simplification of computed Morse graphs (`post_processing_example_1`, `post_processing_example_2`).
- `learning.py`: GP regression utilities (`GaussianProcessModel`, `train_gp_from_data`).
- `utils.py`: Trajectory generation (`generate_trajectory_data`), mode encoding conversions (MATLAB/binary), polynomial utilities.
- `plot.py`, `metrics.py`, `comparison.py` -- visualization (3-panel: Morse sets, Hasse diagram, basins) and quantitative comparison (IoU, graph edit distance).

**Important conventions:**
- `SwitchingSystem` binary mode encoding: bit `i` corresponds to `sign(p_i) > 0`. Mode 0 means all polynomials negative.
- `UniformGrid.box_to_indices()` returns a **filled rectangular region** of grid indices (cubical convex closure), not a sparse union.
- Box map computation samples `2^D` corners + center per box for `F_integration`.
- Detailed technical reference: `MorseGraph-L4DC/SUMMARY.md`.

## Running examples

```bash
cd MorseGraph-L4DC
python examples/example1.py   # Toggle switch system
python examples/example2.py   # Piecewise Van der Pol
```

Output goes to `MorseGraph-L4DC/examples/example_{1,2}_tau_1_0/`.

## Conventions

- Adjacency matrices are symmetric with zero diagonal. Sigmoid parameterization enforces symmetry via `0.5 * (logits + logits.T)` before applying sigmoid, then zeros the diagonal.
- Coupling follows the form: `alpha * D(W) @ X + beta * W @ X` where `D(W)` is the degree matrix.
- For unsupervised mode discovery (Track B), latent labels are only identifiable up to permutation. Evaluation requires alignment via best-permutation search over confusion matrices.
- Gumbel-softmax temperature is annealed during selector training (typically from ~1.2 down to ~0.35).
