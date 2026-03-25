from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SLDSSystem:
    dt: float
    A: Tuple[np.ndarray, ...]
    b: Tuple[np.ndarray, ...]
    init_state: np.ndarray
    init_mode: int = 0

    @property
    def num_modes(self) -> int:
        return len(self.A)

    @property
    def state_dim(self) -> int:
        return int(self.init_state.shape[0])


def build_slds_system(dt: float = 0.02) -> SLDSSystem:
    A = (
        np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
    )
    b = (
        np.array([0.0, 2.0], dtype=float),
        np.array([-1.0, -1.0], dtype=float),
        np.array([-1.0, 1.0], dtype=float),
    )
    init_state = np.array([2.0, -10.0], dtype=float)
    return SLDSSystem(dt=dt, A=A, b=b, init_state=init_state, init_mode=0)


def dynamics(system: SLDSSystem, x: np.ndarray, mode: int) -> np.ndarray:
    return system.A[int(mode)] @ x + system.b[int(mode)]


def event_value(x: np.ndarray, mode: int) -> float:
    if mode == 0:
        return x[1] - 2.0
    if mode == 1:
        return x[0]
    return 2.0 - x[1]


def next_mode(mode: int) -> int:
    return (mode + 1) % 3


def apply_state_update(x: np.ndarray, mode: int) -> np.ndarray:
    x_new = x.copy()
    if mode == 0:
        x_new[1] = x_new[1] - 1e-1
    elif mode == 1:
        x_new[0] = x_new[0] - 1e-1
    else:
        x_new[1] = x_new[1] + 1e-1
    return x_new


def simulate_slds(
    system: SLDSSystem,
    num_steps: int,
    rng: np.random.Generator,
    init_state: np.ndarray | None = None,
    init_mode: int | None = None,
    init_noise: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    if init_state is None:
        init_state = system.init_state.copy()
        if init_noise > 0.0:
            init_state = init_state + rng.normal(scale=init_noise, size=init_state.shape)
    if init_mode is None:
        init_mode = system.init_mode

    x = init_state.astype(float)
    mode = int(init_mode)

    states = [x.copy()]
    modes = [mode]

    for _ in range(num_steps):
        dx = dynamics(system, x, mode)
        x_next = x + system.dt * dx

        ev = event_value(x_next, mode)
        if ev <= 0.0:
            x_next = apply_state_update(x_next, mode)
            mode = next_mode(mode)

        x = x_next
        states.append(x.copy())
        modes.append(mode)

    return np.stack(states, axis=0), np.array(modes, dtype=int)


def augment_states_with_acc(system: SLDSSystem, states: np.ndarray, modes: np.ndarray) -> np.ndarray:
    acc = []
    for x, m in zip(states, modes):
        dx = dynamics(system, x, int(m))
        acc.append(np.array([dx[1]], dtype=float))
    acc = np.stack(acc, axis=0)
    return np.concatenate([states, acc], axis=1)
