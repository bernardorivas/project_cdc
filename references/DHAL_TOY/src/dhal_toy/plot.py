from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_and_modes(
    states: np.ndarray,
    modes: np.ndarray,
    pred_next: np.ndarray,
    pred_modes: np.ndarray,
    pred_steps: np.ndarray,
    output_path: str | Path,
) -> Path:
    steps = np.arange(len(states))
    state_dim = states.shape[1]
    num_rows = state_dim + 2

    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 2.2 * state_dim + 2), sharex=True)

    for i in range(state_dim):
        label = f"x{i}"
        axes[i].plot(steps, states[:, i], linewidth=1.7, label="true")
        axes[i].plot(pred_steps, pred_next[:, i], ":", linewidth=1.6, label="pred")
        axes[i].set_ylabel(label)

    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    axes[-2].step(steps, modes, where="post")
    axes[-2].set_ylabel("true mode")

    pred_mode_steps = np.arange(1, 1 + len(pred_modes))
    axes[-1].step(pred_mode_steps, pred_modes, where="post", color="orange")
    axes[-1].set_ylabel("pred mode")
    axes[-1].set_xlabel("step")

    fig.suptitle("Trajectory and mode: ground truth vs prediction")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    return output_path
