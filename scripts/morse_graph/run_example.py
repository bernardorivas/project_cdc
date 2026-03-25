#!/usr/bin/env python3
"""CMGDB Morse graph computation for autonomous switched network examples.

Usage:
    python run_example.py --example 1                    # run EX01
    python run_example.py --example 1 --estimate         # estimate time only
    python run_example.py --example 6 --tau 1.0 --subdiv-min 30
    python run_example.py --example 2 --padding          # BoxMap padding for 8D
"""

import argparse
import os
import sys
import traceback

# Ensure package imports work when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..'))
_CODE_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from morse_graph.examples_config import get_examples, CMGDB_PARAMS
from morse_graph.cmgdb_pipeline import (estimate_domain_bounds, compute_morse_graph,
                                         estimate_computation_time, save_results)


def parse_args():
    parser = argparse.ArgumentParser(description='CMGDB Morse graph for switched networks')
    parser.add_argument('--example', type=int, required=True, choices=range(1, 10),
                        help='Example number (1-9)')
    parser.add_argument('--estimate', action='store_true',
                        help='Only estimate computation time (no Morse graph)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test run with minimal subdivision (verifies full pipeline)')
    parser.add_argument('--tau', type=float, default=None,
                        help='Integration time for tau-map (default: dim-dependent)')
    parser.add_argument('--subdiv-min', type=int, default=None,
                        help='Min subdivision level (default: dim-dependent)')
    parser.add_argument('--subdiv-max', type=int, default=None,
                        help='Max subdivision level (default: dim-dependent)')
    parser.add_argument('--subdiv-init', type=int, default=None,
                        help='Initial subdivision level (default: dim-dependent)')
    parser.add_argument('--subdiv-limit', type=int, default=None,
                        help='Max boxes per SCC (default: dim-dependent)')
    parser.add_argument('--padding', action='store_true',
                        help='Use BoxMap padding (coarser but safer outer approximation)')
    parser.add_argument('--box-mode', type=str, default='center',
                        choices=['corners', 'center', 'random'],
                        help='BoxMap eval mode (default: center). corners=2^d evals/box, '
                             'center=1 eval (forces padding), random=num_pts evals')
    parser.add_argument('--num-pts', type=int, default=10,
                        help='Number of sample points for random mode (default: 10)')
    parser.add_argument('--conley', action='store_true',
                        help='Compute Conley-Morse graph (with Conley indices, requires CHomP)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory')
    return parser.parse_args()


def main():
    args = parse_args()

    examples = get_examples()
    ex = examples[args.example - 1]
    name = ex['name']
    N = ex['N']
    dim = 2 * N

    # Resolve CMGDB parameters with CLI overrides
    params = dict(CMGDB_PARAMS[dim])
    if args.test:
        # Minimal subdivision for quick pipeline verification
        params.update(subdiv_min=dim, subdiv_max=2 * dim, subdiv_init=0, subdiv_limit=500)
        print(f"[{name}] *** TEST MODE: minimal subdivision ***")
    if args.tau is not None:
        params['tau'] = args.tau
    if args.subdiv_min is not None:
        params['subdiv_min'] = args.subdiv_min
    if args.subdiv_max is not None:
        params['subdiv_max'] = args.subdiv_max
    if args.subdiv_init is not None:
        params['subdiv_init'] = args.subdiv_init
    if args.subdiv_limit is not None:
        params['subdiv_limit'] = args.subdiv_limit

    # Build a run tag from key parameters to avoid overwriting
    run_tag = (f"subdiv{params['subdiv_min']}-{params['subdiv_max']}"
               f"_tau{params['tau']}_{args.box_mode}")
    if args.conley:
        run_tag += "_conley"
    if args.test:
        run_tag = "test"

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(_CODE_DIR, 'results', 'morse_graphs', name, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{name}] {dim}D system ({N} nodes x 2D)")
    print(f"  tau={params['tau']}, subdiv={params['subdiv_min']}-{params['subdiv_max']}, "
          f"init={params['subdiv_init']}, limit={params['subdiv_limit']}, "
          f"padding={args.padding}, box_mode={args.box_mode}")

    try:
        # Step 1: Estimate domain bounds
        print(f"[{name}] Estimating domain bounds from simulation...")
        lb, ub = estimate_domain_bounds(
            ex['f_list'], ex['X0'], ex['T'], ex['dt'], ex['eps'], ex['gamma'])
        print(f"  bounds: [{', '.join(f'{l:.2f}' for l in lb)}] to "
              f"[{', '.join(f'{u:.2f}' for u in ub)}]")

        # Step 2 (optional): Estimate computation time
        if args.estimate:
            estimate_computation_time(
                ex['f_list'], ex['eps'], ex['gamma'], lb, ub,
                tau=params['tau'], dt=ex['dt'],
                subdiv_min=params['subdiv_min'], subdiv_max=params['subdiv_max'],
                subdiv_init=params['subdiv_init'], subdiv_limit=params['subdiv_limit'],
                padding=args.padding, box_mode=args.box_mode, num_pts=args.num_pts)
            return

        # Step 3: Compute Morse graph
        print(f"[{name}] Computing Morse graph...")
        mg, map_g, elapsed = compute_morse_graph(
            ex['f_list'], ex['eps'], ex['gamma'], lb, ub,
            tau=params['tau'], dt=ex['dt'],
            subdiv_min=params['subdiv_min'], subdiv_max=params['subdiv_max'],
            subdiv_init=params['subdiv_init'], subdiv_limit=params['subdiv_limit'],
            padding=args.padding, box_mode=args.box_mode, num_pts=args.num_pts,
            conley=args.conley)

        print(f"[{name}] Done: {mg.num_vertices()} Morse sets, "
              f"{map_g.num_vertices()} phase space boxes, {elapsed:.1f}s")

        # Step 4: Save all outputs
        print(f"[{name}] Saving results...")
        save_results(mg, map_g, out_dir, ex, params, lb, ub, elapsed)

        print(f"[{name}] Complete.")

    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        traceback.print_exc()
        error_path = os.path.join(out_dir, 'error.txt')
        with open(error_path, 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(1)


if __name__ == '__main__':
    main()
