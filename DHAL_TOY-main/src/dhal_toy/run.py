from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import build_dataset
from .model import HybridAutomataLearner
from .plot import plot_trajectory_and_modes
from .rollout import build_rollout_bundle
from .slds import build_slds_system
from .train import train_model


def run_experiment(config: ExperimentConfig) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    system = build_slds_system(dt=config.dt)

    dataset = build_dataset(
        system=system,
        num_traj=config.num_traj,
        steps_per_traj=config.steps_per_traj,
        history_len=config.history_len,
        horizon=config.horizon,
        seed=config.seed,
        train_split=config.train_split,
        batch_size=config.batch_size,
    )

    print("State dimension:", dataset.x_mean.shape[0])
    print("Train size:", dataset.train_size, "Test size:", dataset.test_size)

    model = HybridAutomataLearner(
        state_dim=dataset.x_mean.shape[0],
        num_modes=system.num_modes,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        history_len=config.history_len,
        horizon=config.horizon,
        residual_scale=1,
    ).to(device)

    stats = train_model(
        model=model,
        train_loader=dataset.train_loader,
        test_loader=dataset.test_loader,
        device=device,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        kl_weight=config.kl_weight,
        entropy_weight=config.entropy_weight,
        diversity_weight=config.diversity_weight,
    )

    print(
        "Final test metrics: "
        f"loss={stats.test_loss:.4f} acc_z={stats.acc_z:.3f} acc_zn={stats.acc_zn:.3f}"
    )

    rollout = build_rollout_bundle(
        system=system,
        model=model,
        history_len=config.history_len,
        x_mean=dataset.x_mean,
        x_std=dataset.x_std,
        device=device,
        rollout_steps=config.rollout_steps,
        rollout_seed=config.rollout_seed,
        align_modes=config.align_modes,
    )

    output_path = Path(config.output_dir) / config.figure_name
    saved_path = plot_trajectory_and_modes(
        states=rollout.states,
        modes=rollout.modes,
        pred_next=rollout.pred_next,
        pred_modes=rollout.pred_modes,
        pred_steps=rollout.pred_steps,
        output_path=output_path,
    )
    print("Saved figure:", saved_path)

    return saved_path
