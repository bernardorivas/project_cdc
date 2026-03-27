#!/usr/bin/env python3
"""Generate trajectory data from the 9 switched network examples in MORALS format.

Each trajectory is saved as a text file (rows = timesteps, cols = flat state dimensions).
Trajectories are subsampled at tau intervals so that consecutive saved states
correspond to one application of the time-tau map.

Usage (from code/):
    python scripts/generate_morals_data.py --example 1
    python scripts/generate_morals_data.py --example 1 --num-trajs 500
    python scripts/generate_morals_data.py --all
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — same pattern as run_example.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR))
_CODE_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..'))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from morse_graph.examples_config import get_examples, CMGDB_PARAMS
from morse_graph.dynamics import simulate_switched_network


def estimate_bounds(ex, pad_frac=0.25, pad_min=0.1):
    """Run a reference simulation from the default IC to estimate domain bounds."""
    _, X, _ = simulate_switched_network(
        ex['f_list'], ex['X0'], ex['T'], ex['dt'], ex['eps'], ex['gamma'])
    X_flat = X.reshape(X.shape[0], -1)
    lo = X_flat.min(axis=0)
    hi = X_flat.max(axis=0)
    span = hi - lo
    pad = np.maximum(pad_frac * span, pad_min)
    return lo - pad, hi + pad


def generate_trajectories(ex, num_trajs, tau, lb, ub, num_steps=20, seed=42):
    """Simulate from random ICs within *lb, ub* and subsample at *tau*.

    Each trajectory is *num_steps* tau-steps long (T_sim = num_steps * tau).
    Trajectories that diverge (NaN/Inf) are discarded and replaced by
    resampling a fresh IC until we reach *num_trajs* valid trajectories.
    """
    rng = np.random.RandomState(seed)
    N = ex['N']
    subsample_step = max(1, int(round(tau / ex['dt'])))
    T_sim = num_steps * tau

    trajectories = []
    discarded = 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        while len(trajectories) < num_trajs:
            x0_flat = rng.uniform(lb, ub)
            x0 = x0_flat.reshape(N, 2)

            _, X, _ = simulate_switched_network(
                ex['f_list'], x0, T_sim, ex['dt'], ex['eps'], ex['gamma'])

            X_flat = X.reshape(X.shape[0], -1)          # (steps+1, 2N)

            if not np.isfinite(X_flat).all():
                discarded += 1
                continue

            X_sub = X_flat[::subsample_step]             # subsample at tau
            trajectories.append(X_sub)

    return trajectories, discarded


def save_trajectories(trajectories, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Remove old trajectory files so stale data doesn't linger
    for old in os.listdir(save_dir):
        if old.endswith('.txt'):
            os.remove(os.path.join(save_dir, old))
    for i, traj in enumerate(trajectories):
        np.savetxt(os.path.join(save_dir, f'{i}.txt'), traj, delimiter=',')


def generate_for_example(ex, num_trajs, tau, num_steps, seed, data_root):
    """Full pipeline for one example: bounds -> simulate -> save."""
    name = ex['name']
    dim = 2 * ex['N']
    save_dir = os.path.join(data_root, name)

    print(f'[{name}] {dim}D, estimating domain bounds ...')
    lb, ub = estimate_bounds(ex)

    print(f'[{name}] Generating {num_trajs} trajectories '
          f'(tau={tau}, {num_steps} steps, T_sim={num_steps * tau}s) ...')
    t0 = time.time()
    trajs, discarded = generate_trajectories(
        ex, num_trajs, tau, lb, ub, num_steps=num_steps, seed=seed)
    elapsed = time.time() - t0

    save_trajectories(trajs, save_dir)

    discard_rate = discarded / (num_trajs + discarded) if discarded else 0.0

    # Save metadata alongside data
    T_sim = num_steps * tau
    meta = dict(
        example=name, dim=dim, N=ex['N'],
        num_trajs=num_trajs, tau=tau, num_steps=num_steps,
        T_sim=T_sim, dt=ex['dt'],
        eps=ex['eps'], gamma=ex['gamma'], seed=seed,
        pts_per_traj=int(trajs[0].shape[0]),
        total_pairs=sum(t.shape[0] - 1 for t in trajs),
        lower_bounds=lb.tolist(), upper_bounds=ub.tolist(),
        discarded=discarded, discard_rate=round(discard_rate, 4),
        elapsed_seconds=round(elapsed, 1),
    )
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[{name}] Done — {meta["pts_per_traj"]} pts/traj, '
          f'{meta["total_pairs"]} total pairs, '
          f'{discarded} discarded ({discard_rate:.1%}), {elapsed:.1f}s')
    print(f'  Saved to {save_dir}')
    return meta


def main():
    parser = argparse.ArgumentParser(
        description='Generate MORALS training data from switched network examples')
    parser.add_argument('--example', type=int, required=True, choices=range(1, 10),
                        help='Example number (1-9)')
    parser.add_argument('--num-trajs', type=int, default=200,
                        help='Number of trajectories per example (default: 200)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='Tau-steps per trajectory (default: 20)')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Subsampling interval in seconds (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory for data (default: MORALS/examples/data/)')
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(
        _CODE_DIR, 'MORALS', 'examples', 'data')

    examples = get_examples()
    generate_for_example(
        examples[args.example - 1], args.num_trajs, args.tau,
        args.num_steps, args.seed, data_root)


if __name__ == '__main__':
    main()
