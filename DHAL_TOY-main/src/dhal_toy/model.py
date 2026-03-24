from __future__ import annotations

import torch
import torch.nn as nn


class ModeSelectorMLP(nn.Module):
    def __init__(self, history_len: int, state_dim: int, hidden_dim: int, num_modes: int) -> None:
        super().__init__()
        self.in_dim = history_len * state_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes),
        )

    def forward(self, H_norm: torch.Tensor) -> torch.Tensor:
        B = H_norm.shape[0]
        x = H_norm.reshape(B, -1)
        return self.net(x)


class DynamicsVAE(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int,
        horizon: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.residual_scale = residual_scale
        self.enc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * state_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        resid_pred = self.dec(z)
        resid_pred = torch.tanh(resid_pred) * self.residual_scale
        resid_pred = resid_pred.view(-1, self.horizon, self.state_dim)
        return resid_pred, mu, logvar


class VaeDynamicsBank(nn.Module):
    def __init__(
        self,
        num_modes: int,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int,
        horizon: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.vaes = nn.ModuleList(
            [
                DynamicsVAE(
                    state_dim=state_dim,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    horizon=horizon,
                    residual_scale=residual_scale,
                )
                for _ in range(num_modes)
            ]
        )
        self.num_modes = num_modes
        self.horizon = horizon

    def forward_all(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds, mus, logvars = [], [], []
        for vae in self.vaes:
            p, m, lv = vae(x)
            preds.append(p)
            mus.append(m)
            logvars.append(lv)
        preds = torch.stack(preds, dim=1)
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        return preds, mus, logvars

    def step(self, x: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
        preds, _, _ = self.forward_all(x)
        return preds[torch.arange(x.shape[0], device=x.device), z_idx]


class HybridAutomataLearner(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_modes: int,
        hidden_dim: int,
        latent_dim: int,
        history_len: int,
        horizon: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.selector = ModeSelectorMLP(
            history_len=history_len,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_modes=num_modes,
        )
        self.bank = VaeDynamicsBank(
            num_modes=num_modes,
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            residual_scale=residual_scale,
        )
        self.num_modes = num_modes
        self.horizon = horizon

    def forward(self, H_norm: torch.Tensor, xk: torch.Tensor) -> dict[str, torch.Tensor]:
        mode_logits = self.selector(H_norm)
        mode_probs = torch.softmax(mode_logits, dim=-1)
        per_mode_resid, mu, logvar = self.bank.forward_all(xk)
        resid_pred = (mode_probs[:, :, None, None] * per_mode_resid).sum(dim=1)
        xkp1_pred = xk + resid_pred[:, 0]
        z_idx = mode_probs.argmax(dim=-1)
        return {
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "z_curr_idx": z_idx,
            "z_next_idx": z_idx,
            "xkp1_pred": xkp1_pred,
            "residual_pred": resid_pred,
            "per_mode_residual": per_mode_resid,
            "mu": mu,
            "logvar": logvar,
        }
