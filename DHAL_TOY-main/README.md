# DHAL_TOY

Transition-aware hybrid automata learning toy project (1D network with ground-truth position/velocity state).

## What you get
- Data generation from a jump-hybrid system
- Mode selector + per-mode VAE dynamics (end-to-end training)
- Supervised evaluation (for diagnostics only)
- A single saved figure with ground-truth vs predicted trajectory and mode

## Quick start

```bash
python scripts/run_experiment.py
```

Optional:

```bash
python scripts/run_experiment.py --epochs 300 --output outputs/experiment.png
```

The figure is saved to `outputs/ground_truth.png` by default.

## Project layout

- `scripts/run_experiment.py` – one-command entrypoint
- `src/dhal_toy/sim.py` – jump-hybrid system + simulator
- `src/dhal_toy/data.py` – dataset construction & normalization
- `src/dhal_toy/model.py` – mode selector + VAE dynamics bank
- `src/dhal_toy/train.py` – losses, training loop, evaluation
- `src/dhal_toy/plot.py` – ground-truth plot

## Notes
- Ground-truth mode/switch labels are **not** used for training.
- Training prints a few checkpoints and final evaluation metrics.
