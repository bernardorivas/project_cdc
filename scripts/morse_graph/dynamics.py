"""Oscillator factories and switched network simulator.

Extracted from autonomous_switched_network_oscillatory_examples_executed.ipynb.
"""

import numpy as np
from itertools import combinations


# ---------- Autonomous nonlinear node families ----------

def stuart_landau(mu=1.0, omega=1.0):
    def f(z):
        x, y = z
        r2 = x * x + y * y
        return np.array([mu * x - omega * y - r2 * x,
                         omega * x + mu * y - r2 * y])
    return f


def radial_poly(omega=1.0, roots=(0.7, 1.3), stable_pattern="one_cycle"):
    roots = tuple(roots)
    if stable_pattern == "one_cycle":
        r1 = roots[0]
        def alpha(s):
            return (r1**2 - s)
    elif stable_pattern == "two_cycles":
        r1, r2 = roots
        rm = 0.5 * (r1 + r2)
        sm = rm**2
        def alpha(s):
            return -(s - r1**2) * (s - r2**2) * (s - sm)
    elif stable_pattern == "three_cycles":
        r1, r2, r3 = roots
        rm1 = 0.5 * (r1 + r2)
        rm2 = 0.5 * (r2 + r3)
        sm1 = rm1**2
        sm2 = rm2**2
        def alpha(s):
            return (r1**2 - s) * (s - rm1**2) * (r2**2 - s) * (s - rm2**2) * (r3**2 - s)
    else:
        raise ValueError("Unknown stable_pattern")

    def f(z):
        x, y = z
        s = x * x + y * y
        a = alpha(s)
        return np.array([a * x - omega * y, omega * x + a * y])
    return f


def subcritical_hopf(mu=-0.05, omega=1.0, beta=1.4):
    def f(z):
        x, y = z
        r2 = x * x + y * y
        a = mu + r2 - beta * (r2**2)
        return np.array([a * x - omega * y, omega * x + a * y])
    return f


def van_der_pol(mu=1.5):
    def f(z):
        x, y = z
        return np.array([y, mu * (1 - x * x) * y - x])
    return f


def fitzhugh_nagumo(a=0.7, b=0.8, tau=12.5, I=0.5):
    def f(z):
        v, w = z
        return np.array([v - v**3 / 3 - w + I,
                         (v + a - b * w) / tau])
    return f


def selkov(alpha=0.08, beta=0.6):
    def f(z):
        x, y = z
        dx = -x + alpha * y + x * x * y
        dy = beta - alpha * y - x * x * y
        return np.array([dx, dy])
    return f


def toggle_osc_surrogate(a=2.6, b=2.2, k=1.0):
    def f(z):
        x, y = z
        return np.array([a * x - x**3 - k * y,
                         b * y - y**3 + k * x])
    return f


# ---------- Core simulator ----------

def pair_indices(N):
    return list(combinations(range(N), 2))


def adjacency_from_state(X, eps):
    N = X.shape[0]
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(X[i] - X[j]) <= eps:
                A[i, j] = 1
                A[j, i] = 1
    return A


def diffusive_coupling(X, A, gamma):
    N, d = X.shape
    out = np.zeros_like(X, dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j and A[i, j] == 1:
                out[i] += gamma * (X[j] - X[i])
    return out


def rk4_final_state(f_list, X0, T, dt, eps, gamma):
    """Lightweight RK4 integrator that only returns the final state.

    No trajectory or adjacency history is stored -- just steps forward in-place.
    Designed for repeated tau-map evaluations inside CMGDB.BoxMap.
    """
    s = np.array(X0, dtype=float)
    steps = int(round(T / dt))

    def F(state):
        A = adjacency_from_state(state, eps)
        val = np.zeros_like(state, dtype=float)
        for i, f in enumerate(f_list):
            val[i] = f(state[i])
        val += diffusive_coupling(state, A, gamma)
        return val

    for _ in range(steps):
        k1 = F(s)
        k2 = F(s + 0.5 * dt * k1)
        k3 = F(s + 0.5 * dt * k2)
        k4 = F(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return s


def simulate_switched_network(f_list, X0, T=60.0, dt=0.02, eps=1.2, gamma=0.15):
    X0 = np.array(X0, dtype=float)
    N, d = X0.shape
    steps = int(round(T / dt))
    t = np.linspace(0.0, T, steps + 1)

    X = np.zeros((steps + 1, N, d))
    A_hist = np.zeros((steps + 1, N, N), dtype=int)
    X[0] = X0
    A_hist[0] = adjacency_from_state(X[0], eps)

    def F(state):
        A = adjacency_from_state(state, eps)
        val = np.zeros_like(state, dtype=float)
        for i, f in enumerate(f_list):
            val[i] = f(state[i])
        val += diffusive_coupling(state, A, gamma)
        return val

    for k in range(steps):
        s = X[k]
        k1 = F(s)
        k2 = F(s + 0.5 * dt * k1)
        k3 = F(s + 0.5 * dt * k2)
        k4 = F(s + dt * k3)
        X[k + 1] = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        A_hist[k + 1] = adjacency_from_state(X[k + 1], eps)

    return t, X, A_hist
