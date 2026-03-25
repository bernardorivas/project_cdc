# Morse Graph Pipeline for Switched Network Systems

This pipeline computes Morse graphs for autonomous switched networks where switching is **state-dependent** and changes the **topology of the coupling graph**. Each mode of the hybrid system corresponds to a different graph structure governing how oscillator nodes interact.

## Goal

We have a ground-truth switched dynamical system and a learned approximation from data. The Morse graph (via CMGDB's adaptive subdivision) captures the global topological structure of each system -- Morse sets, their partial order (Hasse diagram), and optionally Conley indices. By comparing the Morse decompositions of the ground-truth and learned systems, we obtain a **topological validation** of the learning approach that is robust to small parametric errors.

## Scripts

- `run_example.py` -- Run a single CMGDB Morse graph computation for one of the 9 example systems. Supports CLI overrides for subdivision parameters, box evaluation mode, and Conley index computation.
- `run_sweep.py` -- Parameter sweep: runs `run_example.py` across multiple `(subdiv_init, subdiv_min, subdiv_max)` configurations with and without Conley indices, in parallel.
- `cmgdb_pipeline.py` -- Core pipeline: domain bound estimation, tau-map construction, CMGDB model setup, and result serialization.
- `examples_config.py` -- Definitions of the 9 example systems (oscillator types, coupling parameters, initial conditions).
- `dynamics.py` -- Oscillator vector fields and RK4 integration.

## Usage

```bash
# single run (ground truth, example 1, corners mode)
.venv/bin/python scripts/morse_graph/run_example.py --example 1 --box-mode corners

# parameter sweep (all configs, with and without Conley)
.venv/bin/python scripts/morse_graph/run_sweep.py --max-workers 4

# quick test (2 small configs, no Conley)
.venv/bin/python scripts/morse_graph/run_sweep.py --test
```

Results are saved to `results/morse_graphs/<example_name>/<run_tag>/` where `run_tag` encodes `subdiv_min-max`, `tau`, box mode, and whether Conley indices were computed.
