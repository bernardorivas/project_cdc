from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dhal_toy import ExperimentConfig, run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DHAL_TOY experiment")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--output", type=str, default=None, help="Output figure path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(epochs=args.epochs)
    if args.output is not None:
        output_path = Path(args.output)
        output_dir = output_path.parent.as_posix() if output_path.parent.as_posix() != "" else "."
        config = ExperimentConfig(
            epochs=args.epochs,
            output_dir=output_dir,
            figure_name=output_path.name,
        )
    run_experiment(config)


if __name__ == "__main__":
    main()
