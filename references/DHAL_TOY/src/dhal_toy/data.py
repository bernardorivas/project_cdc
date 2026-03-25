from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .slds import SLDSSystem, augment_states_with_acc, simulate_slds


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    x_mean: np.ndarray
    x_std: np.ndarray
    train_size: int
    test_size: int


def build_dataset(
    system: SLDSSystem,
    num_traj: int,
    steps_per_traj: int,
    history_len: int,
    horizon: int,
    seed: int,
    train_split: float,
    batch_size: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)

    H, Xk, Xkp1, Xfuture, Zk, Zkp1, Sk = [], [], [], [], [], [], []
    for _ in range(num_traj):
        s_seq, z_seq = simulate_slds(system, steps_per_traj + horizon, rng=rng)
        s_seq_aug = augment_states_with_acc(system, s_seq, z_seq)

        for k in range(history_len - 1, steps_per_traj):
            H.append(s_seq_aug[k - history_len + 1 : k + 1])
            Xk.append(s_seq_aug[k])
            Xkp1.append(s_seq_aug[k + 1])
            Xfuture.append(s_seq_aug[k + 1 : k + 1 + horizon])
            Zk.append(z_seq[k])
            Zkp1.append(z_seq[k + 1])
            Sk.append(int(z_seq[k + 1] != z_seq[k]))

    H_all = np.stack(H)
    Xk_all = np.stack(Xk)
    Xkp1_all = np.stack(Xkp1)
    Xfuture_all = np.stack(Xfuture)
    Zk_all = np.array(Zk)
    Zkp1_all = np.array(Zkp1)
    Sk_all = np.array(Sk)

    perm = np.random.default_rng(123).permutation(Xk_all.shape[0])
    n_train = int(train_split * Xk_all.shape[0])
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    H_train, H_test = H_all[train_idx], H_all[test_idx]
    Xk_train, Xk_test = Xk_all[train_idx], Xk_all[test_idx]
    Xkp1_train, Xkp1_test = Xkp1_all[train_idx], Xkp1_all[test_idx]
    Xfuture_train, Xfuture_test = Xfuture_all[train_idx], Xfuture_all[test_idx]
    Zk_train, Zk_test = Zk_all[train_idx], Zk_all[test_idx]
    Zkp1_train, Zkp1_test = Zkp1_all[train_idx], Zkp1_all[test_idx]
    Sk_train, Sk_test = Sk_all[train_idx], Sk_all[test_idx]

    x_mean = Xk_train.mean(axis=0)
    x_std = Xk_train.std(axis=0) + 1e-6

    def norm_hist(H: np.ndarray) -> np.ndarray:
        return (H - x_mean[None, None, :]) / x_std[None, None, :]

    H_train_n = norm_hist(H_train)
    H_test_n = norm_hist(H_test)

    train_ds = TensorDataset(
        torch.tensor(H_train_n, dtype=torch.float32),
        torch.tensor(Xk_train, dtype=torch.float32),
        torch.tensor(Xkp1_train, dtype=torch.float32),
        torch.tensor(Xfuture_train, dtype=torch.float32),
        torch.tensor(Zk_train, dtype=torch.long),
        torch.tensor(Zkp1_train, dtype=torch.long),
        torch.tensor(Sk_train, dtype=torch.float32),
    )

    test_ds = TensorDataset(
        torch.tensor(H_test_n, dtype=torch.float32),
        torch.tensor(Xk_test, dtype=torch.float32),
        torch.tensor(Xkp1_test, dtype=torch.float32),
        torch.tensor(Xfuture_test, dtype=torch.float32),
        torch.tensor(Zk_test, dtype=torch.long),
        torch.tensor(Zkp1_test, dtype=torch.long),
        torch.tensor(Sk_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return DatasetBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        x_mean=x_mean,
        x_std=x_std,
        train_size=len(train_ds),
        test_size=len(test_ds),
    )
