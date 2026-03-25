from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    # System
    num_nodes: int = 1
    num_modes: int = 3
    dt: float = 0.02

    # Dataset
    num_traj: int = 180
    steps_per_traj: int = 120
    history_len: int = 12
    horizon: int = 20
    seed: int = 0
    train_split: float = 0.8

    # Model
    hidden_dim: int = 64
    latent_dim: int = 8

    # Training
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 200
    kl_weight: float = 1e-4
    entropy_weight: float = 1e-4
    diversity_weight: float = 1e-1

    # Rollout visualization
    rollout_steps: int = 10000
    rollout_seed: int = 0
    align_modes: bool = True

    # Output
    output_dir: str = "outputs"
    figure_name: str = "ground_truth.png"
