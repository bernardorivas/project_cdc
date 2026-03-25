# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository (`cdc_kaitomaani`) contains research on **hybrid dynamical system identification** combined with **computational topology (Morse graph analysis)**.

The two main components are:

1. **`scripts/morse_graph/`** -- CMGDB-based Morse graph pipeline for 9 autonomous switched oscillator network examples (6D and 8D). Uses pip-installed CMGDB for adaptive subdivision and SCC-based Morse graph extraction.
2. **`notebooks/`** -- Jupyter notebooks for hybrid system identification (learning switched dynamical systems from data) and topological validation.

The scientific goal: given a switched dynamical system with multiple modes (each with its own graph topology, coupling parameters, and local dynamics), learn the mode bank and a discrete selector from trajectory data, then compare the **topological structure** (Morse decomposition) of the learned system against the ground truth.

## Environment and setup

- Python 3.14, venv at `.venv/`
- Key dependencies: CMGDB (pip install), torch, numpy, scipy, matplotlib, networkx, joblib, scikit-learn
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

## Architecture: CMGDB Morse graph pipeline

Located at `scripts/morse_graph/`. See `scripts/morse_graph/README.md` for full documentation.

Core pipeline: simulate network to estimate domain bounds -> construct time-tau map -> build CMGDB.BoxMap (outer approximation) -> adaptive subdivision + SCC analysis -> Morse graph + optional Conley indices.

**Key modules:**
- `cmgdb_pipeline.py`: Domain estimation, tau-map construction, CMGDB model setup, result serialization.
- `examples_config.py`: 9 example system definitions (oscillator types, coupling, initial conditions).
- `dynamics.py`: Oscillator vector fields (Stuart-Landau, radial polynomial, subcritical Hopf, Van der Pol, FitzHugh-Nagumo, Selkov, toggle surrogate) and RK4 integration.
- `run_example.py`: CLI entry point for single Morse graph computation.
- `run_sweep.py`: Parallel parameter sweep across subdivision configurations.

## Running examples

```bash
# single example
python scripts/morse_graph/run_example.py --example 1 --box-mode corners

# parameter sweep
python scripts/morse_graph/run_sweep.py --example 1 --max-workers 4
```

Output goes to `results/morse_graphs/<example_name>/<run_tag>/`.

## Conventions

- Adjacency matrices are symmetric with zero diagonal. Sigmoid parameterization enforces symmetry via `0.5 * (logits + logits.T)` before applying sigmoid, then zeros the diagonal.
- Coupling follows the form: `alpha * D(W) @ X + beta * W @ X` where `D(W)` is the degree matrix.
- For unsupervised mode discovery (Track B), latent labels are only identifiable up to permutation. Evaluation requires alignment via best-permutation search over confusion matrices.
- Gumbel-softmax temperature is annealed during selector training (typically from ~1.2 down to ~0.35).
