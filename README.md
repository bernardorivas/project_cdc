# Morse Graph Analysis for Autonomous Switched Oscillator Networks

Compute Morse graphs for coupled oscillator networks with **state-dependent switching** using [CMGDB](https://github.com/marciogameiro/CMGDB) (Computational Morse Graph Database). Each mode of the hybrid system corresponds to a different coupling graph topology. The Morse decomposition captures the global topological structure (attractors, repellers, connecting orbits) and is used for **topological validation** of learned system approximations.

## Examples

Nine autonomous switched network systems are included, spanning 6D and 8D state spaces:

| # | System | Oscillator | Nodes | Dim |
|---|--------|-----------|-------|-----|
| 1 | Stuart-Landau (detuned) | stuart_landau | 3 | 6 |
| 2 | Stuart-Landau (radius mismatch) | stuart_landau | 4 | 8 |
| 3 | Radial two-cycles | radial_poly | 3 | 6 |
| 4 | Radial two-cycles (mixed basins) | radial_poly | 4 | 8 |
| 5 | Subcritical Hopf (rest + cycle) | subcritical_hopf | 4 | 8 |
| 6 | Van der Pol | van_der_pol | 3 | 6 |
| 7 | FitzHugh-Nagumo | fitzhugh_nagumo | 4 | 8 |
| 8 | Selkov | selkov | 3 | 6 |
| 9 | Toggle-rotational | toggle_osc_surrogate | 3 | 6 |

Each node has 2D local dynamics. Coupling is diffusive: edges connect nodes i, j when ||X_i - X_j|| <= eps, and the coupling graph changes the mode of the switched system.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install CMGDB
pip install -r requirements.txt
```

`CMGDB` provides the adaptive subdivision, SCC-based Morse graph extraction, and optional Conley index computation (requires [CHomP](https://github.com/marciogameiro/CHomP)). Everything else (oscillator dynamics, RK4 integration, domain estimation) is self-contained in `scripts/morse_graph/`.

## Quick start

```bash
# Run Morse graph for example 1 (Stuart-Landau, 6D)
python scripts/morse_graph/run_example.py --example 1 --box-mode corners

# Estimate computation time before committing to a full run
python scripts/morse_graph/run_example.py --example 1 --estimate

# Parameter sweep across subdivision configs (parallel)
python scripts/morse_graph/run_sweep.py --example 1 --max-workers 4

# Quick test (small subdivision, fast)
python scripts/morse_graph/run_sweep.py --test
```

Results go to `results/morse_graphs/<example_name>/<run_tag>/` where the run tag encodes subdivision parameters, tau, box evaluation mode, and whether Conley indices were computed.

## Project structure

```
scripts/morse_graph/
    run_example.py       -- single Morse graph computation (CLI entry point)
    run_sweep.py         -- parameter sweep across subdivision configs
    cmgdb_pipeline.py    -- core pipeline: domain bounds, tau-map, CMGDB setup
    examples_config.py   -- definitions of the 9 example systems
    dynamics.py          -- oscillator vector fields and RK4 integration
    README.md            -- detailed script documentation

notebooks/               -- Jupyter notebooks (hybrid system identification + Morse graph validation)
results/morse_graphs/    -- computed Morse graphs (Hasse diagrams, Morse set projections, metadata)
```

## Pipeline overview

1. **Simulate** the switched network (RK4) to estimate domain bounds in R^{2N}.
2. **Construct a time-tau map** closure that maps a point x to its image under the flow.
3. **Build a CMGDB.BoxMap** that outer-approximates the tau-map on each box (evaluating at corners, center, or random samples).
4. **Compute the Morse graph** via CMGDB's adaptive subdivision and SCC analysis. Optionally compute Conley indices.
5. **Save** the Hasse diagram, projected Morse sets per node, timing data, and serialized Morse graph.

See `scripts/morse_graph/README.md` for detailed CLI options and parameter descriptions.
