#!/usr/bin/env python3
"""Run the full MORALS pipeline (train + Morse graph) for all 9 switched network examples.

Assumes trajectory data has already been generated via generate_morals_data.py.

Usage (from code/):
    python scripts/run_morals_pipeline.py                  # all 9 examples
    python scripts/run_morals_pipeline.py --example 1      # single example
    python scripts/run_morals_pipeline.py --skip-train      # only Morse graphs
    python scripts/run_morals_pipeline.py --skip-mg         # only training
"""

import argparse
import os
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..'))
_EXAMPLES_DIR = os.path.join(_CODE_DIR, 'MORALS', 'examples')

CONFIGS = [
    'ex01_stuart_landau_3nodes.txt',
    'ex02_stuart_landau_4nodes.txt',
    'ex03_radial_two_cycles_3nodes.txt',
    'ex04_radial_two_cycles_4nodes.txt',
    'ex05_subcritical_hopf_4nodes.txt',
    'ex06_vanderpol_3nodes.txt',
    'ex07_fitzhugh_nagumo_4nodes.txt',
    'ex08_selkov_3nodes.txt',
    'ex09_toggle_rotational_3nodes.txt',
]


def run_cmd(cmd, cwd, label):
    print(f'\n{"="*60}')
    print(f'[{label}] {" ".join(cmd)}')
    print(f'{"="*60}')
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    elapsed = time.time() - t0
    status = 'OK' if result.returncode == 0 else 'FAILED'
    print(f'[{label}] {status} ({elapsed:.1f}s)')
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run MORALS pipeline for switched networks')
    parser.add_argument('--example', type=int, choices=range(1, 10),
                        help='Single example (1-9). Default: all.')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, only compute Morse graphs')
    parser.add_argument('--skip-mg', action='store_true',
                        help='Skip Morse graph, only train')
    parser.add_argument('--sub', type=int, default=12,
                        help='Subdivision level for Morse graph (default: 12)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Max training epochs (default: 1500)')
    args = parser.parse_args()

    python = sys.executable

    if args.example:
        configs = [CONFIGS[args.example - 1]]
    else:
        configs = CONFIGS

    results = {}

    for cfg in configs:
        name = cfg.replace('.txt', '')

        # --- Train ---
        if not args.skip_train:
            ok = run_cmd(
                [python, 'train.py', '--config', cfg, '--verbose'],
                cwd=_EXAMPLES_DIR,
                label=f'{name} train',
            )
            results[f'{name} train'] = ok
            if not ok:
                print(f'[{name}] Training failed, skipping Morse graph.')
                continue

        # --- Morse graph ---
        if not args.skip_mg:
            ok = run_cmd(
                [python, 'get_MG_RoA.py',
                 '--config', cfg,
                 '--name_out', name,
                 '--sub', str(args.sub),
                 '--validation_type', 'trajectories',
                 '--RoA'],
                cwd=_EXAMPLES_DIR,
                label=f'{name} MG',
            )
            results[f'{name} MG'] = ok

    # --- Summary ---
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    for label, ok in results.items():
        status = 'OK' if ok else 'FAILED'
        print(f'  {label}: {status}')


if __name__ == '__main__':
    main()
