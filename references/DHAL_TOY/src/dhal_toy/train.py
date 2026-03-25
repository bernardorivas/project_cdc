from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
import torch

from .model import HybridAutomataLearner


@dataclass(frozen=True)
class TrainStats:
    train_loss: float
    test_loss: float
    acc_z: float
    acc_zn: float


def vae_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1)


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def batch_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean_p = probs.mean(dim=0)
    return -(mean_p * torch.log(mean_p + eps)).sum()


def eval_supervised(
    model: HybridAutomataLearner,
    loader,
    device: torch.device,
    kl_weight: float,
    entropy_weight: float,
    diversity_weight: float,
) -> tuple[float, float, float, dict[str, float]]:
    model.eval()
    total = 0.0
    count = 0
    total_traj = 0.0
    total_kl = 0.0
    total_ent = 0.0
    total_div = 0.0
    y_true_z, y_pred_z = [], []
    y_true_zn, y_pred_zn = [], []

    with torch.no_grad():
        for hb, xk, xkp1, xfuture, zk, zkp1, sk in loader:
            hb = hb.to(device)
            xk = xk.to(device)
            xkp1 = xkp1.to(device)
            xfuture = xfuture.to(device)
            zk = zk.to(device)
            zkp1 = zkp1.to(device)

            out = model(hb, xk)
            xkp1_pred = xk + out["residual_pred"][:, 0]
            total += ((xkp1_pred - xkp1) ** 2).mean().item() * xk.shape[0]
            count += xk.shape[0]
            y_true_z.append(zk.cpu().numpy())
            y_pred_z.append(out["z_curr_idx"].cpu().numpy())
            y_true_zn.append(zkp1.cpu().numpy())
            y_pred_zn.append(out["z_next_idx"].cpu().numpy())

            per_mode_resid = out["per_mode_residual"]
            per_mode_traj = xk[:, None, None, :] + torch.cumsum(per_mode_resid, dim=2)
            traj_mse_per_mode = ((per_mode_traj - xfuture[:, None, :, :]) ** 2).mean(dim=-1).mean(dim=-1)
            traj_loss = (out["mode_probs"] * traj_mse_per_mode).sum(dim=-1).mean()

            kl_per_mode = vae_kl(out["mu"], out["logvar"])
            kl = (out["mode_probs"] * kl_per_mode).sum(dim=-1).mean()

            ent = entropy_from_probs(out["mode_probs"]).mean()
            div = -batch_entropy(out["mode_probs"])

            total_traj += traj_loss.item() * xk.shape[0]
            total_kl += kl.item() * xk.shape[0]
            total_ent += ent.item() * xk.shape[0]
            total_div += div.item() * xk.shape[0]

    y_true_z = np.concatenate(y_true_z)
    y_pred_z = np.concatenate(y_pred_z)
    y_true_zn = np.concatenate(y_true_zn)
    y_pred_zn = np.concatenate(y_pred_zn)

    K = int(max(y_true_z.max(), y_pred_z.max()) + 1)
    C = np.zeros((K, K), dtype=int)
    for a, b in zip(y_true_z, y_pred_z):
        C[int(a), int(b)] += 1

    best_perm = None
    best_score = -1
    for perm in itertools.permutations(range(K)):
        score = sum(C[i, perm[i]] for i in range(K))
        if score > best_score:
            best_score = score
            best_perm = perm

    mapping = {old: new for old, new in enumerate(best_perm)}
    y_pred_z_aligned = np.array([mapping[int(y)] for y in y_pred_z], dtype=int)
    y_pred_zn_aligned = np.array([mapping[int(y)] for y in y_pred_zn], dtype=int)

    acc_z_aligned = float((y_pred_z_aligned == y_true_z).mean())
    acc_zn_aligned = float((y_pred_zn_aligned == y_true_zn).mean())

    metrics = {
        "traj": total_traj / count,
        "kl": total_kl / count,
        "ent": total_ent / count,
        "div": total_div / count,
        "total": (total_traj / count)
        + kl_weight * (total_kl / count)
        + entropy_weight * (total_ent / count)
        + diversity_weight * (total_div / count),
    }

    return total / count, acc_z_aligned, acc_zn_aligned, metrics


def train_model(
    model: HybridAutomataLearner,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    kl_weight: float,
    entropy_weight: float,
    diversity_weight: float,
) -> TrainStats:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for hb, xk, xkp1, xfuture, zk, zkp1, sk in train_loader:
            hb = hb.to(device)
            xk = xk.to(device)
            xkp1 = xkp1.to(device)
            xfuture = xfuture.to(device)

            out = model(hb, xk)

            per_mode_resid = out["per_mode_residual"]
            per_mode_traj = xk[:, None, None, :] + torch.cumsum(per_mode_resid, dim=2)
            traj_mse_per_mode = ((per_mode_traj - xfuture[:, None, :, :]) ** 2).mean(dim=-1).mean(dim=-1)
            traj_loss = (out["mode_probs"] * traj_mse_per_mode).sum(dim=-1).mean()

            kl_per_mode = vae_kl(out["mu"], out["logvar"])
            kl = (out["mode_probs"] * kl_per_mode).sum(dim=-1).mean()

            ent = entropy_from_probs(out["mode_probs"]).mean()
            div = -batch_entropy(out["mode_probs"])

            loss = traj_loss + kl_weight * kl + entropy_weight * ent + diversity_weight * div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += traj_loss.item() * xk.shape[0]
            seen += xk.shape[0]

        train_loss = running / seen

        if epoch == 1 or epoch % 25 == 0:
            test_loss, acc_z, acc_zn, metrics = eval_supervised(
                model,
                test_loader,
                device,
                kl_weight=kl_weight,
                entropy_weight=entropy_weight,
                diversity_weight=diversity_weight,
            )
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} acc_z={acc_z:.3f} acc_zn={acc_zn:.3f} "
                f"traj={metrics['traj']:.4f} kl={metrics['kl']:.6f} "
                f"ent={metrics['ent']:.6f} div={metrics['div']:.6f} "
                f"total={metrics['total']:.4f}"
            )

    test_recon, acc_z, acc_zn, _ = eval_supervised(
        model,
        test_loader,
        device,
        kl_weight=kl_weight,
        entropy_weight=entropy_weight,
        diversity_weight=diversity_weight,
    )
    return TrainStats(train_loss=train_loss, test_loss=test_recon, acc_z=acc_z, acc_zn=acc_zn)
