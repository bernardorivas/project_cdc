from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from .slds import SLDSSystem, augment_states_with_acc, simulate_slds


@dataclass(frozen=True)
class RolloutBundle:
    states: np.ndarray
    modes: np.ndarray
    pred_next: np.ndarray
    pred_modes: np.ndarray
    pred_steps: np.ndarray


def generate_ground_truth(
    system: SLDSSystem,
    T: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s_seq, z_seq = simulate_slds(system, T, rng=rng)
    s_seq_aug = augment_states_with_acc(system, s_seq, z_seq)
    return s_seq_aug, z_seq


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    C = np.zeros((K, K), dtype=int)
    for a, b in zip(y_true, y_pred):
        C[a, b] += 1
    return C


def best_permutation_from_confusion(C: np.ndarray) -> tuple[tuple[int, ...], int]:
    K = C.shape[0]
    best_perm = None
    best_score = -1
    for perm in itertools.permutations(range(K)):
        score = sum(C[i, perm[i]] for i in range(K))
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_perm, best_score


def apply_perm(y_pred: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    mapping = {old: new for old, new in enumerate(perm)}
    return np.array([mapping[y] for y in y_pred], dtype=int)


def predict_with_true_history(
    model,
    s_seq: np.ndarray,
    history_len: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    z_pred = []
    pred_next = []

    with torch.no_grad():
        for k in range(1, len(s_seq) - 1):
            hist = s_seq[:history_len] if k < history_len else s_seq[k - history_len + 1 : k + 1]
            hist_n = (hist - x_mean[None, :]) / x_std[None, :]

            hb = torch.tensor(hist_n[None], dtype=torch.float32, device=device)
            sk = torch.tensor(s_seq[k][None], dtype=torch.float32, device=device)

            out = model(hb, sk)
            zc = int(out["z_curr_idx"].item())
            z_pred.append(zc)
            resid_first = out["residual_pred"][:, 0].cpu().numpy()[0].copy()
            pred_next.append((s_seq[k] + resid_first).copy())

    pred_next = np.stack(pred_next, axis=0)
    z_pred = np.array(z_pred, dtype=int)

    return pred_next, z_pred


def build_rollout_bundle(
    system: SLDSSystem,
    model,
    history_len: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
    rollout_steps: int,
    rollout_seed: int,
    align_modes: bool,
) -> RolloutBundle:
    states, modes = generate_ground_truth(system, rollout_steps, rollout_seed)

    pred_next, pred_modes = predict_with_true_history(
        model=model,
        s_seq=states,
        history_len=history_len,
        x_mean=x_mean,
        x_std=x_std,
        device=device,
    )

    if align_modes:
        true_curr = modes[1 : 1 + len(pred_modes)]
        C = confusion_matrix_numpy(true_curr, pred_modes, system.num_modes)
        perm_best, _ = best_permutation_from_confusion(C)
        pred_modes = apply_perm(pred_modes, perm_best)

    pred_steps = np.arange(2, 2 + len(pred_next))
    return RolloutBundle(
        states=states,
        modes=modes,
        pred_next=pred_next,
        pred_modes=pred_modes,
        pred_steps=pred_steps,
    )
