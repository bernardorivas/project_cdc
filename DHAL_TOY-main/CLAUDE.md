# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Experiment

```bash
python scripts/run_experiment.py [--epochs N] [--output PATH]
```

Outputs a trajectory figure to `outputs/ground_truth.png` by default and prints mode accuracy + MSE metrics.

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Architecture

**DHAL_TOY** is a toy implementation of unsupervised hybrid automata learning. Ground-truth mode labels are not used during training.

### Pipeline (in execution order)

1. **`slds.py`** — defines a 3-mode 2D Switched Linear Dynamical System (SLDS). `build_slds_system()` returns an `SLDSSystem`; `simulate_slds()` produces labeled trajectories.

2. **`data.py`** — `build_dataset()` simulates 180 trajectories (120 steps each), augments states with estimated acceleration, creates 12-step history + 20-step horizon windows, normalizes, and returns train/test `DataLoader`s wrapped in `DatasetBundle`.

3. **`model.py`** — `HybridAutomataLearner` is a mixture-of-experts model:
   - `ModeSelectorMLP`: maps history → soft mode probabilities (K=3)
   - `VaeDynamicsBank`: K independent VAEs, each predicting state residuals
   - Forward: all VAEs run in parallel; outputs are mixed by mode weights

4. **`train.py`** — `train_model()` minimizes:
   ```
   L = traj_mse + kl_weight·KL + entropy_weight·H + diversity_weight·Div
   ```
   where `H` encourages per-sample mode sharpness and `Div` encourages diverse mode usage across the batch. Evaluation uses the Hungarian algorithm to align predicted modes to ground truth.

5. **`rollout.py`** / **`plot.py`** — generate a 10K-step oracle rollout and visualize state dimensions + mode assignments.

### Configuration

All hyperparameters live in `ExperimentConfig` (`config.py`) — a dataclass with ~24 fields covering trajectory counts, model sizes, loss weights, and training schedule. Modify this to tune experiments.

### Entry point

`scripts/run_experiment.py` → `dhal_toy.run.run_experiment()` → orchestrates all stages above.
