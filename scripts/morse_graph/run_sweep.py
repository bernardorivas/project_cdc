#!/usr/bin/env python3
"""Parameter sweep for CMGDB Morse graph computation.

Runs run_example.py with multiple (subdiv_init, subdiv_min, subdiv_max)
configurations, with and without --conley, in parallel subprocesses.

Usage:
    python run_sweep.py                        # full sweep, 4 workers
    python run_sweep.py --test                 # quick sanity check (tiny grid)
    python run_sweep.py --max-workers 8        # more parallelism
    python run_sweep.py --example 2            # sweep on example 2 instead
    python run_sweep.py --dry-run              # print jobs without running
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# (subdiv_init, subdiv_min, subdiv_max) -- sorted roughly by expected cost
SWEEP_CONFIGS = [
    # easy / sanity checks
    (0,  6,  6),
    (0,  6, 12),
    (0,  6, 18),
    (0,  6, 24),
    # serious runs
    (0, 12, 18),
    (0, 12, 24),
    (0, 18, 24),
    (0, 18, 32),
]

TEST_CONFIGS = [
    (0, 6, 6),
    (0, 6, 12),
]


def build_jobs(example, configs, conley_flags, box_mode='corners'):
    """Build list of (label, cmd) tuples."""
    jobs = []
    run_script = os.path.join(SCRIPT_DIR, 'run_example.py')
    python = sys.executable

    for init, smin, smax in configs:
        for conley in conley_flags:
            label = f"subdiv({init},{smin},{smax})"
            if conley:
                label += "_conley"

            cmd = [
                python, run_script,
                '--example', str(example),
                '--subdiv-min', str(smin),
                '--subdiv-max', str(smax),
            ]
            # subdiv_init is not a CLI arg in run_example.py --
            # it's set via CMGDB_PARAMS default. We pass it via env
            # or we need to add it. For now, the default params have
            # subdiv_init=6 for dim 6. We want init=0 for all sweep runs.
            # run_example.py doesn't expose --subdiv-init, so we'll add it.
            cmd += ['--subdiv-init', str(init)]
            cmd += ['--box-mode', box_mode]

            if conley:
                cmd.append('--conley')

            jobs.append((label, cmd))

    # sort by estimated cost: 2^subdiv_max is the dominant factor
    jobs.sort(key=lambda j: _job_cost_key(j[0]))
    return jobs


def _job_cost_key(label):
    """Extract a rough cost proxy from the label for sorting."""
    # parse subdiv_max from label like "subdiv(0,12,24)_conley"
    parts = label.split('(')[1].split(')')[0].split(',')
    smax = int(parts[2])
    conley = 1 if 'conley' in label else 0
    return (smax, conley)


def run_job(label, cmd):
    """Run a single job as subprocess, capturing output. Returns (label, returncode, elapsed, output)."""
    t0 = time.time()
    started = datetime.now().strftime('%H:%M:%S')
    header = f"[{started}] STARTED: {label}"
    print(header, flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=48 * 3600,  # 48h hard timeout per job
        )
        elapsed = time.time() - t0
        finished = datetime.now().strftime('%H:%M:%S')
        status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
        print(f"[{finished}] {status}: {label} ({format_duration(elapsed)})", flush=True)
        return label, result.returncode, elapsed, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        finished = datetime.now().strftime('%H:%M:%S')
        print(f"[{finished}] TIMEOUT: {label} ({format_duration(elapsed)})", flush=True)
        return label, -1, elapsed, "", "TIMEOUT"


def format_duration(seconds):
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for CMGDB Morse graph computation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--example', type=int, default=1, choices=range(1, 10),
                        help='Example number (default: 1)')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Max parallel subprocesses (default: 4)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test with only 2 small configs, no conley')
    parser.add_argument('--no-conley', action='store_true',
                        help='Skip conley runs (only run without --conley)')
    parser.add_argument('--conley-only', action='store_true',
                        help='Only run conley configurations')
    parser.add_argument('--box-mode', type=str, default='corners',
                        choices=['corners', 'center', 'random'],
                        help='BoxMap eval mode (default: corners)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print jobs without running them')
    args = parser.parse_args()

    if args.test:
        configs = TEST_CONFIGS
        conley_flags = [False]
    else:
        configs = SWEEP_CONFIGS
        if args.no_conley:
            conley_flags = [False]
        elif args.conley_only:
            conley_flags = [True]
        else:
            conley_flags = [False, True]

    jobs = build_jobs(args.example, configs, conley_flags, box_mode=args.box_mode)

    print(f"=== CMGDB Parameter Sweep ===")
    print(f"  Example: {args.example}")
    print(f"  Configs: {len(configs)} subdivision settings x {len(conley_flags)} conley = {len(jobs)} jobs")
    print(f"  Workers: {args.max_workers}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for i, (label, cmd) in enumerate(jobs, 1):
        marker = "[DRY]" if args.dry_run else f"[{i}/{len(jobs)}]"
        print(f"  {marker} {label}")
        if args.dry_run:
            print(f"         {' '.join(cmd)}")
    print()

    if args.dry_run:
        return

    results = []
    t_total = time.time()

    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_job, label, cmd): label for label, cmd in jobs}
        for future in as_completed(futures):
            label, rc, elapsed, stdout, stderr = future.result()
            results.append((label, rc, elapsed))
            if rc != 0 and stderr:
                # Print last few lines of stderr for failed jobs
                err_lines = stderr.strip().split('\n')[-5:]
                for line in err_lines:
                    print(f"  [stderr] {line}", flush=True)

    total_elapsed = time.time() - t_total

    # Summary
    print()
    print(f"=== Sweep Summary ===")
    print(f"  Total wall time: {format_duration(total_elapsed)}")
    print(f"  {'Job':<40s} {'Status':<10s} {'Time':>10s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    ok = 0
    for label, rc, elapsed in sorted(results, key=lambda r: _job_cost_key(r[0])):
        status = "OK" if rc == 0 else ("TIMEOUT" if rc == -1 else f"FAIL({rc})")
        if rc == 0:
            ok += 1
        print(f"  {label:<40s} {status:<10s} {format_duration(elapsed):>10s}")
    print(f"\n  {ok}/{len(results)} succeeded")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
