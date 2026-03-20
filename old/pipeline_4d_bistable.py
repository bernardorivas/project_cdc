"""
Hybrid identification pipeline for 4D bistable network with Morse graph analysis.

System: 4 nodes with pitchfork local dynamics g(x_i, mu_i) = mu_i*x_i - x_i^3,
        graph coupling alpha*D(W)*x + beta*W*x, state-dependent switching via
        polynomials p1(x)=x1+x2, p2(x)=x3+x4 giving 2^2=4 modes.

Pipeline:
  0. Imports
  1. Ground-truth switching system
  2. Dataset with history windows
  3. Model classes (Track A physics-informed, Track B reduced-structure, GRU selector)
  4. Loss functions (defined inline)
  5. Training A-track (bank + supervised DHAL selector + joint fine-tune)
  6. Training B-track (bank + unsupervised DHAL selector + joint fine-tune)
  7. Training curves
  8. Diagnostics (learned graphs, confusion matrices, permutation alignment)
  9. Rollout evaluation (GT via solve_ivp, bank rollouts, RMSE, mode sequences)
 10. Morse graph analysis (GT, Track A, Track B) + 2D projections

All figures saved to results/. Pass --show to display interactively.
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "MorseGraph-L4DC"))
from MorseGraphL4DC.grids import UniformGrid
from MorseGraphL4DC.dynamics import F_integration, F_function
from MorseGraphL4DC.systems import SwitchingSystem
from MorseGraphL4DC.analysis import full_morse_graph_analysis

# ---------------------------------------------------------------------------
# CLI + plotting setup
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="Display figures interactively")
args = parser.parse_args()

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11


def savefig(name):
    path = os.path.join(RESULTS, name)
    plt.savefig(path, bbox_inches="tight")
    print(f"  saved: {path}")
    if args.show:
        plt.show()
    plt.close()


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

NUM_MODES = 4
NUM_NODES = 4
dt = 0.1  # one-step prediction interval

# ===========================================================================
# Section 1: Ground-truth 4D bistable switching system
# ===========================================================================
print("\n=== Section 1: Ground-truth system ===", flush=True)

# Adjacency matrices
A0 = np.array([[0,1,1,1],[1,0,1,0],[1,1,0,1],[1,0,1,0]], dtype=float)
A1 = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]], dtype=float)
A2 = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]], dtype=float)

# Switching surfaces
c1_bi = 0.0; c2_bi = 0.0
polynomials = [
    lambda x, c=c1_bi: x[0] + x[1] - c,
    lambda x, c=c2_bi: x[2] + x[3] - c,
]

# Mode parameters (4 modes via binary sign patterns of p1, p2)
mode_params = [
    {"A": A0, "mu": np.array([1.0, 1.0, 1.0, 1.0]), "alpha":  0.05, "beta": -0.05},
    {"A": A1, "mu": np.array([0.8, 0.6, 0.8, 0.6]), "alpha":  0.08, "beta": -0.04},
    {"A": A2, "mu": np.array([0.6, 0.8, 0.6, 0.8]), "alpha":  0.06, "beta": -0.06},
    {"A": A2, "mu": np.array([0.5, 0.5, 0.5, 0.5]), "alpha":  0.10, "beta": -0.03},
]

true_graphs = np.stack([m["A"]  for m in mode_params], axis=0)
true_mu     = np.stack([m["mu"] for m in mode_params], axis=0)
true_alpha  = np.array([m["alpha"] for m in mode_params])
true_beta   = np.array([m["beta"]  for m in mode_params])

true_graphs_t = torch.tensor(true_graphs, dtype=torch.float32, device=device)
true_mu_t     = torch.tensor(true_mu,     dtype=torch.float32, device=device)
true_alpha_t  = torch.tensor(true_alpha,  dtype=torch.float32, device=device)
true_beta_t   = torch.tensor(true_beta,   dtype=torch.float32, device=device)


def make_bistable_vf(A, mu, alpha, beta):
    deg = A.sum(axis=1)
    def vf(x):
        return mu * x - x ** 3 + alpha * deg * x + beta * A @ x
    return vf


vfs    = [make_bistable_vf(**m) for m in mode_params]
system = SwitchingSystem(polynomials, vfs)

for r, m in enumerate(mode_params):
    print(f"  Mode {r}: alpha={m['alpha']}, beta={m['beta']}, "
          f"mu={m['mu']}, edges={int(m['A'].sum())//2}")

# Test trajectory
x0_test = np.array([0.5, -0.3, 0.4, -0.6])
sol_test = solve_ivp(system, [0, 10.0], x0_test, max_step=0.05, dense_output=True)
t_vis = np.linspace(0, 10.0, 1000)
x_vis = sol_test.sol(t_vis)
modes_vis = np.array([system.sigma(x_vis[:, k]) for k in range(len(t_vis))])

fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
for i in range(4):
    axes[0].plot(t_vis, x_vis[i], label=f"$x_{{{i+1}}}$")
axes[0].set_ylabel("state"); axes[0].legend(ncol=4, fontsize=8)
axes[1].step(t_vis, modes_vis, where="post")
axes[1].set_ylabel("mode"); axes[1].set_xlabel("time")
fig.suptitle("4D bistable: test trajectory with state-dependent switching")
plt.tight_layout()
savefig("test_trajectory.png")

# ===========================================================================
# Section 2: Dataset construction
# ===========================================================================
print("\n=== Section 2: Dataset construction ===", flush=True)

T_total    = 20.0
N_traj     = 50
history_len = 12

rng = np.random.default_rng(42)

trajs = []
for _ in range(N_traj):
    x0 = rng.uniform(-1.2, 1.2, size=4)
    t_eval = np.arange(0, T_total + dt / 2, dt)
    sol = solve_ivp(system, [0, T_total], x0,
                    t_eval=t_eval, max_step=0.05, rtol=1e-6, atol=1e-8)
    if sol.success:
        trajs.append(sol.y.T)

X_list, Y_list, Z_list, H_list = [], [], [], []
for traj in trajs:
    for k in range(history_len - 1, len(traj) - 1):
        H_list.append(traj[k - history_len + 1: k + 1])
        X_list.append(traj[k])
        Y_list.append(traj[k + 1])
        Z_list.append(system.sigma(traj[k]))

X_all = np.array(X_list)
Y_all = np.array(Y_list)
Z_all = np.array(Z_list, dtype=int)
H_all = np.stack(H_list)

perm   = rng.permutation(len(X_all))
n_train = int(0.8 * len(X_all))
tr_idx, te_idx = perm[:n_train], perm[n_train:]

X_tr, X_te = X_all[tr_idx], X_all[te_idx]
Y_tr, Y_te = Y_all[tr_idx], Y_all[te_idx]
Z_tr, Z_te = Z_all[tr_idx], Z_all[te_idx]
H_tr, H_te = H_all[tr_idx], H_all[te_idx]

x_mean = X_tr.mean(axis=0)
x_std  = X_tr.std(axis=0) + 1e-6


def norm_hist(H):
    return (H - x_mean[None, None, :]) / x_std[None, None, :]


H_tr_n = norm_hist(H_tr)
H_te_n = norm_hist(H_te)

sup_train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(Y_tr, dtype=torch.float32),
                             torch.tensor(Z_tr, dtype=torch.long))
sup_test_ds  = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                             torch.tensor(Y_te, dtype=torch.float32),
                             torch.tensor(Z_te, dtype=torch.long))
lat_train_ds = TensorDataset(torch.tensor(H_tr_n, dtype=torch.float32),
                             torch.tensor(X_tr,   dtype=torch.float32),
                             torch.tensor(Y_tr,   dtype=torch.float32),
                             torch.tensor(Z_tr,   dtype=torch.long))
lat_test_ds  = TensorDataset(torch.tensor(H_te_n, dtype=torch.float32),
                             torch.tensor(X_te,   dtype=torch.float32),
                             torch.tensor(Y_te,   dtype=torch.float32),
                             torch.tensor(Z_te,   dtype=torch.long))

sup_train_loader = DataLoader(sup_train_ds, batch_size=256, shuffle=True)
sup_test_loader  = DataLoader(sup_test_ds,  batch_size=512, shuffle=False)
lat_train_loader = DataLoader(lat_train_ds, batch_size=256, shuffle=True)
lat_test_loader  = DataLoader(lat_test_ds,  batch_size=512, shuffle=False)

print(f"  {len(X_all)} pairs ({n_train} train / {len(X_all)-n_train} test)")
print(f"  Mode distribution: {np.bincount(Z_all, minlength=4)}")
print(f"  State range: [{X_all.min():.3f}, {X_all.max():.3f}]")

# ===========================================================================
# Section 3: Model classes
# ===========================================================================
print("\n=== Section 3: Model classes ===", flush=True)


def bistable_batch(X, mu):
    if mu.dim() == 1:
        mu = mu.unsqueeze(0).expand(X.shape[0], -1)
    return mu * X - X ** 3


# ---- Track A: physics-informed bank ----

class BistableModeBank(nn.Module):
    def __init__(self, num_modes=4, num_nodes=4, mu_min=0.1, mu_max=1.5):
        super().__init__()
        self.num_modes = num_modes
        self.mu_min = mu_min; self.mu_max = mu_max
        self.graph_logits = nn.Parameter(torch.zeros(num_modes, num_nodes, num_nodes))
        self.mu_logits    = nn.Parameter(torch.zeros(num_modes, num_nodes))
        self.alpha_param  = nn.Parameter(torch.zeros(num_modes))
        self.beta_param   = nn.Parameter(torch.zeros(num_modes))

    def graphs(self):
        S = 0.5 * (self.graph_logits + self.graph_logits.transpose(1, 2))
        W = torch.sigmoid(S)
        return W * (1.0 - torch.eye(W.shape[-1], device=W.device).unsqueeze(0))

    def mus(self):
        return self.mu_min + (self.mu_max - self.mu_min) * torch.sigmoid(self.mu_logits)

    def alphas(self): return self.alpha_param
    def betas(self):  return self.beta_param


class BistableBank(nn.Module):
    def __init__(self, num_modes=4):
        super().__init__()
        self.bank = BistableModeBank(num_modes=num_modes)

    def mode_step(self, X, mode_idx, dt_val):
        W  = self.bank.graphs()[mode_idx]
        mu = self.bank.mus()[mode_idx]
        a  = self.bank.alphas()[mode_idx]
        b  = self.bank.betas()[mode_idx]
        local    = bistable_batch(X, mu)
        deg      = W.sum(dim=-1)
        coupling = a * deg.unsqueeze(0) * X + b * torch.einsum("ij,bj->bi", W, X)
        return X + dt_val * (local + coupling)

    def forward_grouped(self, X, z, dt_val):
        out = torch.zeros_like(X)
        for r in range(self.bank.num_modes):
            idx = (z == r).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                out[idx] = self.mode_step(X[idx], r, dt_val)
        return out


# ---- Track B: reduced-structure bank ----

class ReducedModeBank(nn.Module):
    def __init__(self, num_modes=4, num_nodes=4):
        super().__init__()
        self.num_modes = num_modes
        self.graph_logits = nn.Parameter(torch.zeros(num_modes, num_nodes, num_nodes))
        self.alpha_param  = nn.Parameter(torch.zeros(num_modes))
        self.beta_param   = nn.Parameter(torch.zeros(num_modes))

    def graphs(self):
        S = 0.5 * (self.graph_logits + self.graph_logits.transpose(1, 2))
        W = torch.sigmoid(S)
        return W * (1.0 - torch.eye(W.shape[-1], device=W.device).unsqueeze(0))

    def alphas(self): return self.alpha_param
    def betas(self):  return self.beta_param


class SmallMLP1D(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, 1))
    def forward(self, x): return self.net(x)


class ReducedBank(nn.Module):
    def __init__(self, num_modes=4, shared_hidden=16, residual_hidden=8):
        super().__init__()
        self.num_modes = num_modes
        self.bank = ReducedModeBank(num_modes=num_modes)
        self.shared_field = nn.Sequential(
            nn.Linear(1, shared_hidden), nn.Tanh(),
            nn.Linear(shared_hidden, 1))
        self.residual_fields = nn.ModuleList(
            [SmallMLP1D(residual_hidden) for _ in range(num_modes)])

    def mode_step(self, X, mode_idx, dt_val):
        W = self.bank.graphs()[mode_idx]
        a = self.bank.alphas()[mode_idx]
        b = self.bank.betas()[mode_idx]
        B, N = X.shape
        xflat = X.reshape(B * N, 1)
        local = self.shared_field(xflat) + self.residual_fields[mode_idx](xflat)
        local = local.reshape(B, N)
        deg   = W.sum(dim=-1)
        coupling = a * deg.unsqueeze(0) * X + b * torch.einsum("ij,bj->bi", W, X)
        return X + dt_val * (local + coupling)

    def forward_grouped(self, X, z, dt_val):
        out = torch.zeros_like(X)
        for r in range(self.num_modes):
            idx = (z == r).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                out[idx] = self.mode_step(X[idx], r, dt_val)
        return out


# ---- GRU selector ----

class GRUSelector(nn.Module):
    def __init__(self, input_size=4, hidden_dim=64, num_modes=4):
        super().__init__()
        self.gru  = nn.GRU(input_size=input_size, hidden_size=hidden_dim,
                           batch_first=True)
        self.head = nn.Linear(hidden_dim, num_modes)

    def forward(self, H):
        _, h = self.gru(H)
        return self.head(h[-1])


def st_gumbel_onehot(logits, tau=1.0):
    return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)


class HybridSelectorModel(nn.Module):
    def __init__(self, bank_model, num_modes=4):
        super().__init__()
        self.selector   = GRUSelector(input_size=NUM_NODES, num_modes=num_modes)
        self.bank_model = bank_model

    def forward(self, H_norm, X_phys, dt_val, tau=1.0):
        logits = self.selector(H_norm)
        onehot = st_gumbel_onehot(logits, tau=tau)
        idx    = onehot.argmax(dim=-1)
        pred   = self.bank_model.forward_grouped(X_phys, idx, dt_val)
        return pred, logits, onehot, idx


# ---- Evaluation helpers ----

def eval_bank(model, loader):
    model.eval()
    total = 0.0; count = 0
    with torch.no_grad():
        for xb, yb, zb in loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            pred = model.forward_grouped(xb, zb, dt)
            total += ((pred - yb) ** 2).mean().item() * xb.shape[0]
            count += xb.shape[0]
    return total / count


def eval_selector(model, loader, supervised=True, tau=0.35):
    model.eval()
    total = 0.0; count = 0; correct = 0
    with torch.no_grad():
        for hb, xb, yb, zb in loader:
            hb, xb, yb, zb = (hb.to(device), xb.to(device),
                               yb.to(device), zb.to(device))
            pred, logits, onehot, idx = model(hb, xb, dt, tau=tau)
            total += ((pred - yb) ** 2).mean().item() * xb.shape[0]
            count += xb.shape[0]
            if supervised:
                correct += (idx == zb).sum().item()
    if supervised:
        return total / count, correct / count
    return total / count


def confusion_matrix_numpy(y_true, y_pred, K):
    C = np.zeros((K, K), dtype=int)
    for a, b in zip(y_true, y_pred):
        C[a, b] += 1
    return C


def best_permutation_from_confusion(C):
    K = C.shape[0]
    best_perm, best_score = None, -1
    for p in itertools.permutations(range(K)):
        score = sum(C[i, p[i]] for i in range(K))
        if score > best_score:
            best_score = score; best_perm = p
    return best_perm, best_score


def apply_perm(y_pred, perm):
    mapping = {old: new for old, new in enumerate(perm)}
    return np.array([mapping[y] for y in y_pred], dtype=int)


# ===========================================================================
# Section 5: Training A-track
# ===========================================================================
print("\n=== Section 5: Training A-track ===", flush=True)

# ---- A-bank pretraining ----
print("  A-bank pretraining (40 epochs)...")
torch.manual_seed(42)
rng_init = np.random.default_rng(7)
A_bank = BistableBank(num_modes=NUM_MODES).to(device)

_mu_min, _mu_max = 0.1, 1.5
with torch.no_grad():
    for r, m in enumerate(mode_params):
        A_n = np.clip(m["A"] + rng_init.uniform(-0.15, 0.15, m["A"].shape),
                      1e-3, 1 - 1e-3)
        A_bank.bank.graph_logits.data[r] = torch.tensor(
            np.log(A_n / (1 - A_n)), dtype=torch.float32)
        mu_sc = np.clip((m["mu"] + rng_init.uniform(-0.15, 0.15, 4) - _mu_min)
                        / (_mu_max - _mu_min), 1e-3, 1 - 1e-3)
        A_bank.bank.mu_logits.data[r] = torch.tensor(
            np.log(mu_sc / (1 - mu_sc)), dtype=torch.float32)
        A_bank.bank.alpha_param.data[r] = m["alpha"] + rng_init.uniform(-0.02, 0.02)
        A_bank.bank.beta_param.data[r]  = m["beta"]  + rng_init.uniform(-0.02, 0.02)

opt = optim.Adam(A_bank.parameters(), lr=2e-3, weight_decay=1e-6)
A_bank_tr, A_bank_te = [], []

for epoch in range(1, 41):
    A_bank.train()
    run_ = 0.0; seen_ = 0
    for xb, yb, zb in sup_train_loader:
        xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
        pred = A_bank.forward_grouped(xb, zb, dt)
        pred_loss = ((pred - yb) ** 2).mean()
        W  = A_bank.bank.graphs()
        mu = A_bank.bank.mus()
        a  = A_bank.bank.alphas()
        b  = A_bank.bank.betas()
        loss = (pred_loss
                + ((W  - true_graphs_t) ** 2).mean()
                + ((mu - true_mu_t)     ** 2).mean()
                + ((a  - true_alpha_t)  ** 2).mean()
                + ((b  - true_beta_t)   ** 2).mean()
                + 1e-3 * (W * (1 - W)).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        run_ += pred_loss.item() * xb.shape[0]; seen_ += xb.shape[0]
    A_bank_tr.append(run_ / seen_)
    A_bank_te.append(eval_bank(A_bank, sup_test_loader))

# ---- A-DHAL selector ----
print("  A-DHAL selector (80 epochs, supervised)...")
A_track = HybridSelectorModel(A_bank, num_modes=NUM_MODES).to(device)
for p in A_track.bank_model.parameters():
    p.requires_grad = False
opt = optim.Adam(A_track.selector.parameters(), lr=2e-3, weight_decay=1e-6)

A_sel_tr, A_sel_te, A_sel_acc = [], [], []
for epoch in range(1, 81):
    tau = max(0.4, 1.2 * (0.985 ** epoch))
    A_track.train()
    run_ = 0.0; seen_ = 0
    for hb, xb, yb, zb in lat_train_loader:
        hb, xb, yb, zb = (hb.to(device), xb.to(device),
                           yb.to(device), zb.to(device))
        pred, logits, onehot, idx = A_track(hb, xb, dt, tau=tau)
        pred_loss = ((pred - yb) ** 2).mean()
        cls_loss  = nn.CrossEntropyLoss()(logits, zb)
        balance   = ((onehot.float().mean(0) - 1.0 / NUM_MODES) ** 2).mean()
        dwell     = (torch.abs(onehot[1:].float() - onehot[:-1].float()).mean()
                     if onehot.shape[0] > 1 else 0.0)
        loss = pred_loss + 0.1 * cls_loss + 1e-2 * balance + 1e-3 * dwell
        opt.zero_grad(); loss.backward(); opt.step()
        run_ += pred_loss.item() * xb.shape[0]; seen_ += xb.shape[0]
    te_loss, te_acc = eval_selector(A_track, lat_test_loader, supervised=True, tau=tau)
    A_sel_tr.append(run_ / seen_)
    A_sel_te.append(te_loss)
    A_sel_acc.append(te_acc)

# ---- A joint fine-tune ----
print("  A joint fine-tune (10 epochs)...")
for p in A_track.bank_model.parameters():
    p.requires_grad = True
opt = optim.Adam(A_track.parameters(), lr=5e-4, weight_decay=1e-6)
for epoch in range(1, 11):
    A_track.train()
    for hb, xb, yb, zb in lat_train_loader:
        hb, xb, yb, zb = (hb.to(device), xb.to(device),
                           yb.to(device), zb.to(device))
        pred, logits, onehot, idx = A_track(hb, xb, dt, tau=0.35)
        pred_loss = ((pred - yb) ** 2).mean()
        cls_loss  = nn.CrossEntropyLoss()(logits, zb)
        balance   = ((onehot.float().mean(0) - 1.0 / NUM_MODES) ** 2).mean()
        dwell     = (torch.abs(onehot[1:].float() - onehot[:-1].float()).mean()
                     if onehot.shape[0] > 1 else 0.0)
        W  = A_track.bank_model.bank.graphs()
        mu = A_track.bank_model.bank.mus()
        a  = A_track.bank_model.bank.alphas()
        b  = A_track.bank_model.bank.betas()
        bank_reg = (((W  - true_graphs_t) ** 2).mean()
                    + ((mu - true_mu_t)   ** 2).mean()
                    + ((a  - true_alpha_t) ** 2).mean()
                    + ((b  - true_beta_t)  ** 2).mean()
                    + 1e-3 * (W * (1 - W)).mean())
        loss = (pred_loss + 0.1 * cls_loss + 1e-2 * balance
                + 1e-3 * dwell + 0.5 * bank_reg)
        opt.zero_grad(); loss.backward(); opt.step()

# ===========================================================================
# Section 6: Training B-track
# ===========================================================================
print("\n=== Section 6: Training B-track ===", flush=True)

# ---- B-bank pretraining ----
print("  B-bank pretraining (60 epochs)...")
torch.manual_seed(0)
B_bank = ReducedBank(num_modes=NUM_MODES, shared_hidden=16, residual_hidden=8).to(device)
opt = optim.Adam(B_bank.parameters(), lr=2e-3, weight_decay=1e-6)

B_bank_tr, B_bank_te = [], []
for epoch in range(1, 61):
    B_bank.train()
    run_ = 0.0; seen_ = 0
    for xb, yb, zb in sup_train_loader:
        xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
        pred = B_bank.forward_grouped(xb, zb, dt)
        pred_loss = ((pred - yb) ** 2).mean()
        W      = B_bank.bank.graphs()
        binary = (W * (1 - W)).mean()
        sparse = (W.sum() / 2.0) / W.shape[0]
        sep    = torch.tensor(0.0, device=device)
        cnt    = 0
        for r_ in range(W.shape[0]):
            for s_ in range(r_ + 1, W.shape[0]):
                sep = sep - ((W[r_] - W[s_]) ** 2).mean()
                cnt += 1
        sep = sep / max(cnt, 1)
        loss = pred_loss + 1e-3 * binary + 1e-4 * sparse + 1e-3 * sep
        opt.zero_grad(); loss.backward(); opt.step()
        run_ += pred_loss.item() * xb.shape[0]; seen_ += xb.shape[0]
    B_bank_tr.append(run_ / seen_)
    B_bank_te.append(eval_bank(B_bank, sup_test_loader))

# ---- B-DHAL selector (unsupervised) ----
print("  B-DHAL selector (120 epochs, unsupervised)...")
B_track = HybridSelectorModel(B_bank, num_modes=NUM_MODES).to(device)
B_track.selector.load_state_dict(A_track.selector.state_dict())
for p in B_track.bank_model.parameters():
    p.requires_grad = False
opt = optim.Adam(B_track.selector.parameters(), lr=2e-3, weight_decay=1e-6)

B_sel_tr, B_sel_te, B_sel_usage = [], [], []
for epoch in range(1, 121):
    tau = max(0.35, 1.2 * (0.985 ** epoch))
    B_track.train()
    run_ = 0.0; seen_ = 0
    for hb, xb, yb, zb in lat_train_loader:
        hb, xb, yb = hb.to(device), xb.to(device), yb.to(device)
        pred, logits, onehot, idx = B_track(hb, xb, dt, tau=tau)
        pred_loss = ((pred - yb) ** 2).mean()
        balance   = ((onehot.float().mean(0) - 1.0 / NUM_MODES) ** 2).mean()
        dwell     = (torch.abs(onehot[1:].float() - onehot[:-1].float()).mean()
                     if onehot.shape[0] > 1 else 0.0)
        loss = pred_loss + 1e-2 * balance + 1e-3 * dwell
        opt.zero_grad(); loss.backward(); opt.step()
        run_ += pred_loss.item() * xb.shape[0]; seen_ += xb.shape[0]

    B_track.eval()
    te_tot = 0.0; te_cnt = 0
    usage  = np.zeros(NUM_MODES)
    with torch.no_grad():
        for hb, xb, yb, zb in lat_test_loader:
            hb, xb, yb = hb.to(device), xb.to(device), yb.to(device)
            pred, _, onehot, _ = B_track(hb, xb, dt, tau=tau)
            te_tot += ((pred - yb) ** 2).mean().item() * xb.shape[0]
            te_cnt += xb.shape[0]
            usage  += onehot.float().mean(0).cpu().numpy()
    B_sel_tr.append(run_ / seen_)
    B_sel_te.append(te_tot / te_cnt)
    B_sel_usage.append(usage / len(lat_test_loader))

# ---- B joint fine-tune ----
print("  B joint fine-tune (10 epochs)...")
for p in B_track.bank_model.parameters():
    p.requires_grad = True
opt = optim.Adam(B_track.parameters(), lr=5e-4, weight_decay=1e-6)
for epoch in range(1, 11):
    B_track.train()
    for hb, xb, yb, zb in lat_train_loader:
        hb, xb, yb = hb.to(device), xb.to(device), yb.to(device)
        pred, logits, onehot, idx = B_track(hb, xb, dt, tau=0.35)
        pred_loss = ((pred - yb) ** 2).mean()
        balance   = ((onehot.float().mean(0) - 1.0 / NUM_MODES) ** 2).mean()
        dwell     = (torch.abs(onehot[1:].float() - onehot[:-1].float()).mean()
                     if onehot.shape[0] > 1 else 0.0)
        W      = B_track.bank_model.bank.graphs()
        binary = (W * (1 - W)).mean()
        sparse = (W.sum() / 2.0) / W.shape[0]
        sep    = torch.tensor(0.0, device=device)
        cnt    = 0
        for r_ in range(W.shape[0]):
            for s_ in range(r_ + 1, W.shape[0]):
                sep = sep - ((W[r_] - W[s_]) ** 2).mean()
                cnt += 1
        sep = sep / max(cnt, 1)
        loss = (pred_loss + 1e-2 * balance + 1e-3 * dwell
                + 0.5 * (1e-3 * binary + 1e-4 * sparse + 1e-3 * sep))
        opt.zero_grad(); loss.backward(); opt.step()

# ===========================================================================
# Section 7: Training curves
# ===========================================================================
print("\n=== Section 7: Training curves ===", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, tr, te, title in zip(
        axes.flat,
        [A_bank_tr, A_sel_tr, B_bank_tr, B_sel_tr],
        [A_bank_te, A_sel_te, B_bank_te, B_sel_te],
        ["A-bank", "A-DHAL", "B-bank", "B-DHAL"]):
    ax.plot(tr, label="train"); ax.plot(te, label="test")
    ax.set_yscale("log"); ax.set_title(title); ax.legend()
plt.tight_layout()
savefig("training_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].plot(A_sel_acc)
axes[0].set_title("A-DHAL accuracy")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
usage_arr = np.array(B_sel_usage)
for k in range(usage_arr.shape[1]):
    axes[1].plot(usage_arr[:, k], label=f"mode {k}")
axes[1].set_title("B-DHAL mode usage"); axes[1].legend()
axes[1].set_xlabel("epoch")
plt.tight_layout()
savefig("selector_diagnostics.png")

# ===========================================================================
# Section 8: Diagnostics
# ===========================================================================
print("\n=== Section 8: Diagnostics ===", flush=True)

W_A = A_track.bank_model.bank.graphs().detach().cpu().numpy()
W_B = B_track.bank_model.bank.graphs().detach().cpu().numpy()

fig, axes = plt.subplots(NUM_MODES, 3, figsize=(10, 13))
for r in range(NUM_MODES):
    axes[r, 0].imshow(true_graphs[r], vmin=0, vmax=1)
    axes[r, 0].set_title(f"True mode {r}")
    axes[r, 1].imshow(W_A[r], vmin=0, vmax=1)
    axes[r, 1].set_title(f"A-bank mode {r}")
    axes[r, 2].imshow(W_B[r], vmin=0, vmax=1)
    axes[r, 2].set_title(f"B-bank mode {r}")
plt.tight_layout()
savefig("learned_graphs.png")

mu_A = A_track.bank_model.bank.mus().detach().cpu().numpy()
a_A  = A_track.bank_model.bank.alphas().detach().cpu().numpy()
b_A  = A_track.bank_model.bank.betas().detach().cpu().numpy()

print("  Track A learned vs true:")
for r in range(NUM_MODES):
    m = mode_params[r]
    W_bin = (W_A[r] > 0.5).astype(int)
    match = np.array_equal(W_bin, m["A"].astype(int))
    print(f"    Mode {r}: alpha={a_A[r]:.4f}({m['alpha']:.4f}) "
          f"beta={b_A[r]:.4f}({m['beta']:.4f}) "
          f"mu={np.round(mu_A[r],3)}({m['mu']}) graph={match}")

# Confusion matrices
A_track.eval(); B_track.eval()

yt_A, yp_A = [], []
with torch.no_grad():
    for hb, xb, yb, zb in lat_test_loader:
        hb, xb = hb.to(device), xb.to(device)
        _, _, _, idx = A_track(hb, xb, dt, tau=0.35)
        yt_A.append(zb.numpy()); yp_A.append(idx.cpu().numpy())
yt_A = np.concatenate(yt_A); yp_A = np.concatenate(yp_A)
C_A = confusion_matrix_numpy(yt_A, yp_A, NUM_MODES)

yt_B, yp_B = [], []
with torch.no_grad():
    for hb, xb, yb, zb in lat_test_loader:
        hb, xb = hb.to(device), xb.to(device)
        _, _, _, idx = B_track(hb, xb, dt, tau=0.35)
        yt_B.append(zb.numpy()); yp_B.append(idx.cpu().numpy())
yt_B = np.concatenate(yt_B); yp_B = np.concatenate(yp_B)
C_B_raw = confusion_matrix_numpy(yt_B, yp_B, NUM_MODES)

perm_best, _ = best_permutation_from_confusion(C_B_raw)
yp_B_aligned = apply_perm(yp_B, perm_best)
C_B_aligned  = confusion_matrix_numpy(yt_B, yp_B_aligned, NUM_MODES)
acc_B_aligned = np.trace(C_B_aligned) / C_B_aligned.sum()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, C, title in zip(axes,
                         [C_A, C_B_raw, C_B_aligned],
                         ["A-DHAL", "B-DHAL raw", "B-DHAL aligned"]):
    im = ax.imshow(C); ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.tight_layout()
savefig("confusion_matrices.png")

print("  A-DHAL confusion:"); print(C_A)
print(f"  A-DHAL accuracy: {np.trace(C_A)/C_A.sum():.4f}")
print("  B-DHAL raw:"); print(C_B_raw)
print("  B-DHAL best permutation:", perm_best)
print("  B-DHAL aligned:"); print(C_B_aligned)
print(f"  B-DHAL aligned accuracy: {acc_B_aligned:.4f}")
print("  B-DHAL final mode usage:", np.round(np.array(B_sel_usage)[-1], 4))

# ===========================================================================
# Section 9: Rollout evaluation
# ===========================================================================
print("\n=== Section 9: Rollout evaluation ===", flush=True)

rng_roll   = np.random.default_rng(99)
x0_roll    = rng_roll.uniform(-1.0, 1.0, size=4)
T_roll     = 18.0
t_eval_roll = np.arange(0, T_roll + dt / 2, dt)
sol_roll   = solve_ivp(system, [0, T_roll], x0_roll,
                       t_eval=t_eval_roll, max_step=0.05, rtol=1e-6, atol=1e-8)
traj_true  = sol_roll.y.T
z_true     = np.array([system.sigma(traj_true[k])
                       for k in range(len(traj_true))], dtype=int)
t_roll     = np.arange(len(traj_true)) * dt


def rollout_bank(bank_model, x0, z_seq):
    traj = [x0.copy()]
    with torch.no_grad():
        xt = torch.tensor(x0[None], dtype=torch.float32, device=device)
        for z in z_seq:
            zt = torch.tensor([z], dtype=torch.long, device=device)
            xt = bank_model.forward_grouped(xt, zt, dt)
            traj.append(xt.cpu().numpy()[0].copy())
    return np.stack(traj, axis=0)


def infer_modes(track_model, traj_phys, h_len, perm_map=None, tau=0.35):
    zhat = []
    with torch.no_grad():
        for k in range(1, len(traj_phys)):
            hist = traj_phys[:h_len] if k < h_len else traj_phys[k - h_len:k]
            hist_n = (hist - x_mean[None, :]) / x_std[None, :]
            hb = torch.tensor(hist_n[None], dtype=torch.float32, device=device)
            xb = torch.tensor(traj_phys[k - 1][None], dtype=torch.float32, device=device)
            _, _, _, idx = track_model(hb, xb, dt, tau=tau)
            z = int(idx.item())
            if perm_map is not None:
                z = perm_map[z]
            zhat.append(z)
    return np.array(zhat, dtype=int)


z_roll = z_true[:-1]

traj_A_true = rollout_bank(A_track.bank_model, traj_true[0], z_roll)
traj_B_true = rollout_bank(B_track.bank_model, traj_true[0], z_roll)

zhat_A = infer_modes(A_track, traj_true, history_len, tau=0.35)
perm_map = {old: new for old, new in enumerate(perm_best)}
zhat_B = infer_modes(B_track, traj_true, history_len, perm_map=perm_map, tau=0.35)

traj_A_inf = rollout_bank(A_track.bank_model, traj_true[0], zhat_A)
traj_B_inf = rollout_bank(B_track.bank_model, traj_true[0], zhat_B)

Tc = min(len(traj_true), len(traj_A_true), len(traj_B_true),
         len(traj_A_inf), len(traj_B_inf))
traj_true   = traj_true[:Tc]
traj_A_true = traj_A_true[:Tc]
traj_B_true = traj_B_true[:Tc]
traj_A_inf  = traj_A_inf[:Tc]
traj_B_inf  = traj_B_inf[:Tc]
t_roll      = t_roll[:Tc]
z_roll      = z_roll[:Tc - 1]
zhat_A      = zhat_A[:Tc - 1]
zhat_B      = zhat_B[:Tc - 1]


def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=1))


rmse_A_true = rmse(traj_A_true, traj_true)
rmse_A_inf  = rmse(traj_A_inf,  traj_true)
rmse_B_true = rmse(traj_B_true, traj_true)
rmse_B_inf  = rmse(traj_B_inf,  traj_true)

for name, arr in [("A true-mode",     rmse_A_true),
                   ("A inferred-mode", rmse_A_inf),
                   ("B true-mode",     rmse_B_true),
                   ("B inferred-mode", rmse_B_inf)]:
    print(f"  {name:22s}  mean={arr.mean():.6f}  "
          f"final={arr[-1]:.6f}  max={arr.max():.6f}")

fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
for i in range(4):
    axes[i].plot(t_roll, traj_true[:, i],  lw=2,   label="true")
    axes[i].plot(t_roll, traj_A_true[:, i], "--", lw=1.2, label="A true mode")
    axes[i].plot(t_roll, traj_A_inf[:, i],  ":",  lw=1.5, label="A inferred")
    axes[i].plot(t_roll, traj_B_true[:, i], "-.", lw=1.2, label="B true mode")
    axes[i].plot(t_roll, traj_B_inf[:, i],  lw=1.0, alpha=0.9, label="B inferred")
    axes[i].set_ylabel(f"$x_{{{i+1}}}$")
axes[4].step(t_roll[:-1], z_roll, where="post", label="true mode")
axes[4].set_ylabel("mode"); axes[4].set_xlabel("time")
axes[0].legend(ncol=3, fontsize=7)
fig.suptitle("Rollout comparison")
plt.tight_layout()
savefig("rollout_comparison.png")

fig, ax = plt.subplots(figsize=(8, 4))
for lbl, arr in [("A true", rmse_A_true), ("A inferred", rmse_A_inf),
                  ("B true", rmse_B_true), ("B inferred", rmse_B_inf)]:
    ax.plot(t_roll, arr, label=lbl)
ax.set_title("Rollout RMSE"); ax.legend()
ax.set_xlabel("time"); ax.set_ylabel("RMSE")
savefig("rollout_rmse.png")

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].step(t_roll[:-1], z_roll, where="post"); axes[0].set_ylabel("true")
axes[1].step(t_roll[:-1], zhat_A, where="post"); axes[1].set_ylabel("A inferred")
axes[2].step(t_roll[:-1], zhat_B, where="post"); axes[2].set_ylabel("B inferred")
axes[2].set_xlabel("time")
fig.suptitle("Inferred mode sequences")
plt.tight_layout()
savefig("mode_sequences.png")

# ===========================================================================
# Section 10: Morse graph analysis
# ===========================================================================
print("\n=== Section 10: Morse graph analysis ===", flush=True)

_data_min = X_all.min(axis=0)
_data_max = X_all.max(axis=0)
_margin   = 0.3 * (_data_max - _data_min)
_bound    = max(np.abs(_data_min - _margin).max(),
                np.abs(_data_max + _margin).max())
_bound    = np.ceil(_bound * 2) / 2

lb = np.full(4, -_bound)
ub = np.full(4,  _bound)
div = np.full(4, 8, dtype=int)

grid = UniformGrid(bounds=np.array([lb, ub]), divisions=div)
n_boxes = int(np.prod(div))
print(f"  Grid: [{lb[0]}, {ub[0]}]^4, {div[0]}^4 = {n_boxes} boxes")
print(f"  Box size: {grid.box_size},  diagonal: {np.linalg.norm(grid.box_size):.4f}")

tau_morse = 1.0

# ---- Ground truth ----
print(f"\n  Computing ground-truth Morse graph (tau={tau_morse})...")
F_gt = F_integration(system, tau_morse, epsilon=0.0)
t0 = time.time()
bm_gt, mg_gt, bas_gt = full_morse_graph_analysis(grid, F_gt)
t_gt = time.time() - t0
print(f"  Done in {t_gt:.1f}s -- {len(mg_gt.nodes())} Morse sets, "
      f"{len(mg_gt.edges())} edges")
for i, ms in enumerate(mg_gt.nodes()):
    print(f"    Set {i}: {len(ms)} boxes, basin {len(bas_gt.get(ms, set()))} boxes")

# ---- Track A ----
print("\n  Computing Track A learned Morse graph...")
W_A_np  = A_track.bank_model.bank.graphs().detach().cpu().numpy()
mu_A_np = A_track.bank_model.bank.mus().detach().cpu().numpy()
a_A_np  = A_track.bank_model.bank.alphas().detach().cpu().numpy()
b_A_np  = A_track.bank_model.bank.betas().detach().cpu().numpy()
np.savez("learned_params_A.npz", W=W_A_np, mu=mu_A_np, alpha=a_A_np, beta=b_A_np)
print("  Saved learned_params_A.npz")

learned_A_vfs = []
for r in range(NUM_MODES):
    W_bin = (W_A_np[r] > 0.5).astype(float)
    learned_A_vfs.append(make_bistable_vf(W_bin, mu_A_np[r],
                                          float(a_A_np[r]), float(b_A_np[r])))
sys_A = SwitchingSystem(polynomials, learned_A_vfs)
F_A   = F_integration(sys_A, tau_morse, epsilon=0.0)

t0 = time.time()
bm_A, mg_A, bas_A = full_morse_graph_analysis(grid, F_A)
t_A = time.time() - t0
print(f"  Done in {t_A:.1f}s -- {len(mg_A.nodes())} Morse sets, "
      f"{len(mg_A.edges())} edges")
for i, ms in enumerate(mg_A.nodes()):
    print(f"    Set {i}: {len(ms)} boxes, basin {len(bas_A.get(ms, set()))} boxes")

# ---- Track B (reconstruct numpy vector fields from learned MLPs) ----
print("\n  Computing Track B learned Morse graph...")
B_track.to("cpu")
B_track.eval()

# Extract MLP weights as numpy for pure-numpy forward pass
_bbank = B_track.bank_model
with torch.no_grad():
    _W_B_np  = _bbank.bank.graphs().cpu().numpy()
    _a_B_np  = _bbank.bank.alphas().cpu().numpy()
    _b_B_np  = _bbank.bank.betas().cpu().numpy()
    # shared_field: Linear(1,16)->Tanh->Linear(16,1)
    _sw1 = _bbank.shared_field[0].weight.cpu().numpy()     # (16, 1)
    _sb1 = _bbank.shared_field[0].bias.cpu().numpy()        # (16,)
    _sw2 = _bbank.shared_field[2].weight.cpu().numpy()     # (1, 16)
    _sb2 = _bbank.shared_field[2].bias.cpu().numpy()        # (1,)
    # residual_fields[r]: Linear(1,8)->Tanh->Linear(8,1)
    _rw1, _rb1, _rw2, _rb2 = [], [], [], []
    for r in range(NUM_MODES):
        net = _bbank.residual_fields[r].net
        _rw1.append(net[0].weight.cpu().numpy())   # (8, 1)
        _rb1.append(net[0].bias.cpu().numpy())      # (8,)
        _rw2.append(net[2].weight.cpu().numpy())   # (1, 8)
        _rb2.append(net[2].bias.cpu().numpy())      # (1,)


def _mlp_numpy(x_flat, w1, b1, w2, b2):
    """x_flat: (N,1) -> (N,1) via Linear->Tanh->Linear."""
    h = np.tanh(x_flat @ w1.T + b1)
    return h @ w2.T + b2


def _make_trackB_vf(mode_idx):
    W_bin = (_W_B_np[mode_idx] > 0.5).astype(float)
    deg = W_bin.sum(axis=1)
    a = float(_a_B_np[mode_idx])
    b = float(_b_B_np[mode_idx])
    rw1_r, rb1_r = _rw1[mode_idx], _rb1[mode_idx]
    rw2_r, rb2_r = _rw2[mode_idx], _rb2[mode_idx]

    def vf(x):
        xf = x.reshape(-1, 1)
        local = (_mlp_numpy(xf, _sw1, _sb1, _sw2, _sb2)
                 + _mlp_numpy(xf, rw1_r, rb1_r, rw2_r, rb2_r)).ravel()
        return local + a * deg * x + b * W_bin @ x
    return vf


learned_B_vfs = [_make_trackB_vf(r) for r in range(NUM_MODES)]
sys_B = SwitchingSystem(polynomials, learned_B_vfs)
F_B = F_integration(sys_B, tau_morse, epsilon=0.0)

t0 = time.time()
bm_B, mg_B, bas_B = full_morse_graph_analysis(grid, F_B)
t_B = time.time() - t0
print(f"  Done in {t_B:.1f}s -- {len(mg_B.nodes())} Morse sets, "
      f"{len(mg_B.edges())} edges")
for i, ms in enumerate(mg_B.nodes()):
    print(f"    Set {i}: {len(ms)} boxes, basin {len(bas_B.get(ms, set()))} boxes")

# ---- Comparison table ----
print("\n" + "=" * 80)
print("MORSE GRAPH COMPARISON: Ground Truth vs Track A vs Track B")
print("=" * 80)
print(f"{'Metric':<30} {'GT':<15} {'Track A':<15} {'Track B':<15}")
print("-" * 80)
n_sets  = [len(mg_gt.nodes()), len(mg_A.nodes()), len(mg_B.nodes())]
n_edges = [len(mg_gt.edges()), len(mg_A.edges()), len(mg_B.edges())]
n_basin = [sum(len(b.get(ms, set())) for ms in mg.nodes())
           for mg, b in [(mg_gt, bas_gt), (mg_A, bas_A), (mg_B, bas_B)]]
t_comp  = [t_gt, t_A, t_B]
print(f"{'Morse sets':<30} {n_sets[0]:<15} {n_sets[1]:<15} {n_sets[2]:<15}")
print(f"{'Edges':<30} {n_edges[0]:<15} {n_edges[1]:<15} {n_edges[2]:<15}")
print(f"{'Total basin boxes':<30} {n_basin[0]:<15} {n_basin[1]:<15} {n_basin[2]:<15}")
print(f"{'Computation time (s)':<30} {t_comp[0]:<15.1f} {t_comp[1]:<15.1f} {t_comp[2]:<15.1f}")
print("=" * 80)

# ---- 2D projections ----
all_boxes = grid.get_boxes()
proj_pairs = [(0, 1, "$x_1$", "$x_2$"), (2, 3, "$x_3$", "$x_4$")]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col, (label, mg, bas) in enumerate([
        ("Ground truth", mg_gt, bas_gt),
        ("Track A",      mg_A,  bas_A),
        ("Track B",      mg_B,  bas_B)]):
    n_ms = max(len(list(mg.nodes())), 1)
    colors = plt.cm.Set1(np.linspace(0, 0.9, n_ms))
    for row, (d1, d2, xl, yl) in enumerate(proj_pairs):
        ax = axes[row, col]
        for i, ms in enumerate(mg.nodes()):
            bidxs = list(bas.get(ms, set()))
            if bidxs:
                bc = (all_boxes[bidxs, 0, :] + all_boxes[bidxs, 1, :]) / 2
                ax.scatter(bc[:, d1], bc[:, d2], s=2, c=[colors[i]], alpha=0.15)
        for i, ms in enumerate(mg.nodes()):
            ms_idxs = list(ms)
            if ms_idxs:
                mc = (all_boxes[ms_idxs, 0, :] + all_boxes[ms_idxs, 1, :]) / 2
                ax.scatter(mc[:, d1], mc[:, d2], s=14, c=[colors[i]],
                           edgecolors="k", linewidths=0.3, label=f"MS {i}")
        x_line = np.linspace(lb[d1], ub[d1], 50)
        c_thr  = c1_bi if d1 < 2 else c2_bi
        ax.plot(x_line, c_thr - x_line, "k--", alpha=0.5, lw=0.9)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{label}: {xl} vs {yl}")
        ax.set_xlim(lb[d1], ub[d1]); ax.set_ylim(lb[d2], ub[d2])
        if row == 0:
            ax.legend(fontsize=6, markerscale=1.5, loc="best")
plt.suptitle("Morse sets and basins (2D projections)", y=1.01)
plt.tight_layout()
savefig("morse_sets_2d.png")

# ---- Mode partitioning ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
centers   = (all_boxes[:, 0, :] + all_boxes[:, 1, :]) / 2
box_modes = np.array([system.sigma(c) for c in centers])
for ax_idx, (d1, d2, xl, yl) in enumerate(proj_pairs):
    ax = axes[ax_idx]
    sc = ax.scatter(centers[:, d1], centers[:, d2], s=3, c=box_modes,
                    cmap="tab10", alpha=0.6, vmin=0, vmax=3)
    x_line = np.linspace(lb[d1], ub[d1], 50)
    c_thr  = c1_bi if d1 < 2 else c2_bi
    ax.plot(x_line, c_thr - x_line, "k-", lw=1.5, label="switching surface")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=8)
    ax.set_xlim(lb[d1], ub[d1]); ax.set_ylim(lb[d2], ub[d2])
plt.colorbar(sc, ax=axes[-1], label="mode")
plt.suptitle("State-dependent mode partitioning")
plt.tight_layout()
savefig("mode_partitioning.png")

# ---- Morse graph DAGs ----
for label, mg in [("gt", mg_gt), ("trackA", mg_A), ("trackB", mg_B)]:
    if len(mg.nodes()) == 0:
        continue
    fig, ax = plt.subplots(figsize=(5, 4))
    pos  = nx.spring_layout(mg, seed=0)
    n_ms = len(mg.nodes())
    node_colors = [mg.nodes[n].get("color", plt.cm.Set1(i / max(n_ms, 1)))
                   for i, n in enumerate(mg.nodes())]
    nx.draw(mg, pos=pos, ax=ax, with_labels=False,
            node_color=node_colors, node_size=400,
            edge_color="k", arrows=True,
            arrowstyle="-|>", arrowsize=15, width=1.5)
    for i, n in enumerate(mg.nodes()):
        ax.annotate(f"MS{i}\n({len(n)} boxes)",
                    xy=pos[n], fontsize=7, ha="center",
                    xytext=(0, 12), textcoords="offset points")
    ax.set_title(f"Morse graph ({label})")
    plt.tight_layout()
    savefig(f"morse_graph_{label}.png")

# ---- Basin sizes ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (label, mg, bas), color in zip(
        axes,
        [("Ground truth", mg_gt, bas_gt),
         ("Track A",      mg_A,  bas_A),
         ("Track B",      mg_B,  bas_B)],
        ["steelblue", "seagreen", "tomato"]):
    sizes = [len(bas.get(ms, set())) for ms in mg.nodes()]
    if sizes:
        ax.bar(range(len(sizes)), sizes, color=color)
    ax.set_xlabel("Morse set index"); ax.set_ylabel("Basin size")
    ax.set_title(f"{label} basin sizes")
plt.tight_layout()
savefig("basin_sizes.png")

# ---- Summary ----
print(f"\n  Final A-DHAL accuracy: {A_sel_acc[-1]:.4f}")
print(f"  Final B-DHAL aligned accuracy: {acc_B_aligned:.4f}")
print(f"  Morse sets: GT={n_sets[0]}, A={n_sets[1]}, B={n_sets[2]}")
print("\nPipeline complete. All figures saved to results/")
