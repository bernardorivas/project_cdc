#!/bin/bash
# Run all 9 CMGDB Morse graph computations in parallel.
# Usage: bash run_all.sh          (4 parallel jobs, default)
#        bash run_all.sh 6        (6 parallel jobs)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NJOBS="${1:-4}"

echo "Running 9 examples with $NJOBS parallel jobs..."
for i in $(seq 1 9); do
    echo "python3 $SCRIPT_DIR/run_example.py --example $i 2>&1 | tee $SCRIPT_DIR/../../results/morse_graphs/log_ex$(printf '%02d' $i).txt"
done | xargs -P "$NJOBS" -I {} bash -c '{}'

echo "All done. Results in results/morse_graphs/"
