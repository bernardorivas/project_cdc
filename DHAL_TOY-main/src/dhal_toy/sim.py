from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ModeParams:
    name: str
    W: np.ndarray
    L: np.ndarray
    a: np.ndarray
    k: float
    c: np.ndarray


@dataclass(frozen=True)
class JumpHybridSystem:
    num_nodes: int
    num_modes: int
    dt: float
    mode_params: Tuple[ModeParams, ...]
    jump_table: Dict[Tuple[int, int], np.ndarray]

    @property
    def state_dim(self) -> int:
        return 2 * self.num_nodes


def laplacian(W: np.ndarray) -> np.ndarray:
    return np.diag(W.sum(axis=1)) - W


def build_default_system(num_nodes: int = 4, num_modes: int = 3, dt: float = 0.20) -> JumpHybridSystem:
    if num_nodes != 4 or num_modes != 3:
        raise ValueError("Default system is defined for num_nodes=4, num_modes=3.")

    W0 = np.array(
        [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=float
    )
    W1 = np.array(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=float
    )
    W2 = np.array(
        [[0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]], dtype=float
    )

    a0 = np.array([0.84, 0.88, 0.86, 0.82], dtype=float)
    a1 = np.array([0.72, 0.78, 0.74, 0.76], dtype=float)
    a2 = np.array([0.87, 0.83, 0.89, 0.85], dtype=float)

    k0, k1, k2 = 0.08, 0.12, 0.06
    c0 = np.array([0.010, -0.008, 0.006, 0.000], dtype=float)
    c1 = np.array([-0.012, 0.009, -0.006, 0.007], dtype=float)
    c2 = np.array([0.006, 0.008, -0.010, 0.012], dtype=float)

    mode_params = (
        ModeParams("mode0", W0, laplacian(W0), a0, k0, c0),
        ModeParams("mode1", W1, laplacian(W1), a1, k1, c1),
        ModeParams("mode2", W2, laplacian(W2), a2, k2, c2),
    )

    def make_jump(dx, dv) -> np.ndarray:
        return np.concatenate([np.array(dx, dtype=float), np.array(dv, dtype=float)], axis=0)

    jump_table = {
        (0, 1): make_jump([0.05, -0.03, 0.02, -0.02], [0.12, -0.10, 0.08, -0.06]),
        (1, 0): make_jump([-0.04, 0.03, -0.02, 0.02], [-0.10, 0.08, -0.07, 0.05]),
        (0, 2): make_jump([-0.03, 0.04, -0.03, 0.03], [-0.08, 0.10, -0.09, 0.08]),
        (2, 0): make_jump([0.03, -0.04, 0.03, -0.03], [0.07, -0.10, 0.09, -0.08]),
        (1, 2): make_jump([0.04, 0.03, -0.03, -0.03], [0.09, 0.07, -0.08, -0.09]),
        (2, 1): make_jump([-0.04, -0.03, 0.03, 0.03], [-0.09, -0.07, 0.08, 0.09]),
    }

    state_dim = 2 * num_nodes
    for r in range(num_modes):
        jump_table[(r, r)] = np.zeros(state_dim, dtype=float)

    return JumpHybridSystem(num_nodes, num_modes, dt, mode_params, jump_table)


def random_switch_sequence(
    num_steps: int,
    num_modes: int,
    min_dwell: int = 8,
    max_dwell: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    modes = []
    current = int(rng.integers(0, num_modes))
    while len(modes) < num_steps + 1:
        dwell = int(rng.integers(min_dwell, max_dwell + 1))
        modes.extend([current] * dwell)
        choices = [m for m in range(num_modes) if m != current]
        current = int(rng.choice(choices))
    return np.array(modes[: num_steps + 1], dtype=int)


def split_state(state: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return state[:num_nodes], state[num_nodes:]


def step_jump_hybrid(system: JumpHybridSystem, state: np.ndarray, mode: int, next_mode: int) -> np.ndarray:
    params = system.mode_params[int(mode)]
    xk, vk = split_state(state, system.num_nodes)

    acc = -params.k * (params.L @ xk) + params.c
    v_flow = params.a * vk + system.dt * acc
    x_flow = xk + system.dt * v_flow

    s_flow = np.concatenate([x_flow, v_flow], axis=0)
    return s_flow + system.jump_table[(int(mode), int(next_mode))]


def simulate_jump_hybrid(system: JumpHybridSystem, s0: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
    states = [s0.copy()]
    for k in range(len(z_seq) - 1):
        states.append(step_jump_hybrid(system, states[-1], z_seq[k], z_seq[k + 1]))
    return np.stack(states, axis=0)
