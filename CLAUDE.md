# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository (`cdc_kaitomaani`) contains research on **hybrid dynamical system identification** combined with **computational topology (Morse graph analysis)**.

The two main components are:

1. **`scripts/morse_graph/`** -- CMGDB-based Morse graph pipeline for 9 autonomous switched oscillator network examples (6D and 8D). Uses pip-installed CMGDB for adaptive subdivision and SCC-based Morse graph extraction.
2. **`references/`** -- development notebooks (hybrid system identification, topological validation) and a toy DHAL project. Not needed to run the pipeline.

The scientific goal: given a switched dynamical system with multiple modes (each with its own graph topology, coupling parameters, and local dynamics), learn the mode bank and a discrete selector from trajectory data, then compare the **topological structure** (Morse decomposition) of the learned system against the ground truth.

## Environment and setup

- Python 3.14, venv at `.venv/`
- Key dependencies: CMGDB (pip install), torch, numpy, scipy, matplotlib, networkx, joblib, scikit-learn
- No test suite or CI. Validation is done through the example scripts.

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

- Adjacency matrices are symmetric with zero diagonal.
- Coupling follows the form: `alpha * D(W) @ X + beta * W @ X` where `D(W)` is the degree matrix.
- State-dependent switching: edges connect nodes i, j when ||X_i - X_j|| <= eps.
