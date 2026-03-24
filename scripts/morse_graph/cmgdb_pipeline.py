"""CMGDB Morse graph computation pipeline for switched network systems."""

import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add CMGDB to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CMGDB_SRC = os.path.join(_SCRIPT_DIR, '..', '..', 'MorseGraph', 'cmgdb', 'src')
if _CMGDB_SRC not in sys.path:
    sys.path.insert(0, _CMGDB_SRC)

import CMGDB

from .dynamics import simulate_switched_network, rk4_final_state


def estimate_domain_bounds(f_list, X0, T, dt, eps, gamma, padding_factor=0.25):
    """Run simulation to determine domain bounds, then pad."""
    t, X, A_hist = simulate_switched_network(f_list, X0, T=T, dt=dt, eps=eps, gamma=gamma)
    X_flat = X.reshape(len(t), -1)  # (steps+1, 2*N)
    lo = X_flat.min(axis=0)
    hi = X_flat.max(axis=0)
    span = hi - lo
    margin = padding_factor * span
    margin = np.maximum(margin, 0.1)
    return (lo - margin).tolist(), (hi + margin).tolist()


def make_tau_map(f_list, eps, gamma, tau, dt):
    """Build time-tau point map using the lightweight RK4 stepper."""
    N = len(f_list)

    def f(x):
        X0 = np.array(x).reshape(N, 2)
        Xf = rk4_final_state(f_list, X0, T=tau, dt=dt, eps=eps, gamma=gamma)
        return Xf.flatten().tolist()

    return f


def estimate_computation_time(f_list, eps, gamma, lower_bounds, upper_bounds,
                              tau, dt, subdiv_min, subdiv_max, subdiv_init,
                              subdiv_limit, padding=False,
                              box_mode='center', num_pts=10, n_samples=5):
    """Estimate CMGDB computation time by benchmarking individual BoxMap calls.

    Returns a dict with timing estimates and per-box cost breakdown.
    """
    dim = len(lower_bounds)
    N = len(f_list)
    tau_map = make_tau_map(f_list, eps, gamma, tau, dt)

    # Build a sample rectangle (the full domain)
    sample_rect = list(lower_bounds) + list(upper_bounds)

    # Time n_samples BoxMap evaluations with the chosen mode
    times = []
    for _ in range(n_samples):
        t0 = time.time()
        CMGDB.BoxMap(tau_map, sample_rect, mode=box_mode, padding=padding, num_pts=num_pts)
        times.append(time.time() - t0)

    t_box = np.median(times)
    if box_mode == 'corners':
        evals_per_box = 2 ** dim
    elif box_mode == 'center':
        evals_per_box = 1
    else:
        evals_per_box = num_pts

    # Estimate box counts at various subdivision levels.
    # At level k, the total grid has 2^k boxes.
    # CMGDB uniformly refines ALL boxes from subdiv_init to subdiv_min,
    # then adaptively refines only SCCs from subdiv_min to subdiv_max.
    boxes_init = 2 ** subdiv_init
    boxes_min = 2 ** subdiv_min
    boxes_max = 2 ** subdiv_max

    # Uniform phase: every box from init to min is evaluated.
    # Total = sum(2^k for k in range(subdiv_init, subdiv_min)) = 2^subdiv_min - 2^subdiv_init
    uniform_boxes = boxes_min - boxes_init

    # Adaptive phase: hard to predict, but bounded by subdiv_limit * num_SCCs
    # per refinement level, over (subdiv_max - subdiv_min) levels.
    adaptive_levels = subdiv_max - subdiv_min
    est_adaptive_boxes = subdiv_limit * adaptive_levels  # rough upper bound

    est_boxes_evaluated = uniform_boxes + est_adaptive_boxes

    print(f"\n  === Time estimate ===")
    print(f"  Dimension: {dim}D ({N} nodes x 2D)")
    print(f"  BoxMap mode: {box_mode} ({evals_per_box} tau-map evals/box)")
    print(f"  RK4 steps per tau-map: {int(round(tau / dt))}")
    print(f"  Median time per BoxMap call: {t_box:.4f}s")
    print(f"  Median time per tau-map eval: {t_box / evals_per_box:.6f}s")
    print(f"  Grid boxes at subdiv_init={subdiv_init}: {boxes_init:,}")
    print(f"  Grid boxes at subdiv_min={subdiv_min}: {boxes_min:,}")
    print(f"  Grid boxes at subdiv_max={subdiv_max}: {boxes_max:,}")
    print(f"  subdiv_limit: {subdiv_limit:,}")
    print(f"  Estimated boxes evaluated (rough): ~{est_boxes_evaluated:,}")
    print(f"  Estimated time (rough): ~{est_boxes_evaluated * t_box:.0f}s "
          f"({est_boxes_evaluated * t_box / 60:.1f} min)")
    print(f"  === end estimate ===\n")

    return {
        'dim': dim,
        'evals_per_box': evals_per_box,
        'box_mode': box_mode,
        'median_time_per_box': t_box,
        'boxes_at_init': boxes_init,
        'boxes_at_min': boxes_min,
        'boxes_at_max': boxes_max,
        'est_boxes_evaluated': est_boxes_evaluated,
        'est_time_seconds': est_boxes_evaluated * t_box,
    }


def compute_morse_graph(f_list, eps, gamma, lower_bounds, upper_bounds,
                        tau, dt, subdiv_min, subdiv_max, subdiv_init=0,
                        subdiv_limit=5000, padding=False,
                        box_mode='center', num_pts=10, conley=False):
    """Full CMGDB Morse graph computation. Returns (morse_graph, map_graph, elapsed_seconds).

    box_mode: 'corners' (2^d evals/box), 'center' (1 eval, forces padding),
              'random' (num_pts evals/box).
    conley: if True, compute Conley-Morse graph (with Conley indices). Requires CHomP.
    For high-dimensional systems, 'center' or 'random' is strongly recommended.
    """
    tau_map = make_tau_map(f_list, eps, gamma, tau, dt)

    def F(rect):
        return CMGDB.BoxMap(tau_map, rect, mode=box_mode, padding=padding, num_pts=num_pts)

    model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
                         lower_bounds, upper_bounds, F)

    t0 = time.time()
    if conley:
        mg, map_g = CMGDB.ComputeConleyMorseGraph(model)
    else:
        mg, map_g = CMGDB.ComputeMorseGraph(model)
    elapsed = time.time() - t0
    return mg, map_g, elapsed


def save_morse_graph_hasse(mg, output_path):
    """Save Hasse diagram as PNG via graphviz."""
    gv_source = CMGDB.PlotMorseGraph(mg, cmap=matplotlib.cm.cool)
    png_bytes = gv_source.pipe(format='png')
    with open(output_path, 'wb') as f:
        f.write(png_bytes)


def save_morse_set_projections(mg, N, output_dir):
    """Save 2D Morse set projections for each node's phase plane."""
    for i in range(N):
        proj_dims = [2 * i, 2 * i + 1]
        fname = os.path.join(output_dir, f'morse_sets_node{i}.png')
        CMGDB.PlotMorseSets(mg, proj_dims=proj_dims, cmap=matplotlib.cm.cool,
                            fig_w=6, fig_h=6,
                            xlabel=f'$z_{{{i},1}}$', ylabel=f'$z_{{{i},2}}$',
                            fig_fname=fname)
        plt.close('all')


def save_results(mg, map_g, output_dir, example_config, cmgdb_params,
                 lower_bounds, upper_bounds, elapsed):
    """Save all outputs: figures, data, timing, metadata."""
    os.makedirs(output_dir, exist_ok=True)

    N = example_config['N']
    name = example_config['name']

    # Hasse diagram
    print(f"  Saving Hasse diagram...")
    save_morse_graph_hasse(mg, os.path.join(output_dir, 'morse_graph.png'))

    # Morse set projections per node
    print(f"  Saving Morse set projections...")
    save_morse_set_projections(mg, N, output_dir)

    # Serialized Morse graph data
    print(f"  Saving Morse graph data...")
    CMGDB.SaveMorseGraphData(
        mg, map_g, os.path.join(output_dir, 'data.mgdb'),
        metadata={
            'example': name,
            'dim': 2 * N,
            'tau': cmgdb_params['tau'],
            'subdiv_min': cmgdb_params['subdiv_min'],
            'subdiv_max': cmgdb_params['subdiv_max'],
            'subdiv_limit': cmgdb_params['subdiv_limit'],
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'elapsed_seconds': elapsed,
        })

    # Timing
    timing = {
        'elapsed_seconds': elapsed,
        'num_morse_sets': mg.num_vertices(),
        'num_phase_space_boxes': map_g.num_vertices(),
    }
    with open(os.path.join(output_dir, 'timing.json'), 'w') as f:
        json.dump(timing, f, indent=2)

    # Metadata
    metadata = {
        'example_name': name,
        'N': N,
        'dim': 2 * N,
        'eps': example_config['eps'],
        'gamma': example_config['gamma'],
        'T_sim': example_config['T'],
        'dt': example_config['dt'],
        'tau': cmgdb_params['tau'],
        'subdiv_min': cmgdb_params['subdiv_min'],
        'subdiv_max': cmgdb_params['subdiv_max'],
        'subdiv_init': cmgdb_params['subdiv_init'],
        'subdiv_limit': cmgdb_params['subdiv_limit'],
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'elapsed_seconds': elapsed,
        'num_morse_sets': mg.num_vertices(),
        'num_phase_space_boxes': map_g.num_vertices(),
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  All outputs saved to {output_dir}")
