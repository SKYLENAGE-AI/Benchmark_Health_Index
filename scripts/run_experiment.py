"""Portable command-line launcher for BHI supplementary experiments.

The scientific implementations read their configuration from
``config.settings``.  This launcher sets those paths before Python imports the
implementation, copies the baseline tables needed by an experiment into a new
result directory, and then invokes the implementation.
It does not alter any scientific parameter.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILES = (
    "final_bhi_rankings.csv",
    "capability_discrimination_final.csv",
    "anti_saturation_results.csv",
    "impact_results.csv",
    "benchmark_inclusion_report.csv",
)

MAIN_FLAGS = {
    "edr": "--threshold",
    "anti-saturation-weight": "--sat-weight-sensitivity",
    "dynamic-window": "--dynamic-window-sensitivity",
    "noise-critic": "--noise",
    "noise-equal": "--noise-equal",
    "dropout-critic": "--robustness",
    "dropout-equal": "--robustness-equal",
    "ablation": "--ablation",
    "correlation": "--indicator-correlation",
    "coverage": "--coverage-analysis",
    "dynamic-trends": "--plot-dynamic-trends",
}
STANDALONE = {
    "rcv": "rcv_percentile_sensitivity.py",
    "headroom": "safety_factor_sensitivity.py",
    "targeted-zero": "weak_model_low_score_stress_test.py",
    "targeted-fixed": "weak_model_fixed_decrement_sensitivity.py",
}
NO_BASELINE_COPY = {"dynamic-trends", "rcv", "headroom", "targeted-zero", "targeted-fixed"}


def resolved_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{label} is not a file: {path}")
    return path


def resolved_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"{label} is not a directory: {path}")
    return path


def verify_baseline(path: Path) -> None:
    missing = [name for name in BASELINE_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Baseline directory {path} is missing: {', '.join(missing)}"
        )


def prepare_output(path: Path, baseline: Path | None, copy_baseline: bool) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if copy_baseline:
        if baseline is None:
            raise ValueError("This experiment requires --baseline-dir")
        path.mkdir()
        for name in BASELINE_FILES:
            shutil.copy2(baseline / name, path / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(MAIN_FLAGS | STANDALONE))
    parser.add_argument("--input", required=True, help="Wide benchmark input CSV.")
    parser.add_argument(
        "--baseline-dir",
        help="Completed baseline snapshot containing the five core result CSVs.",
    )
    parser.add_argument("--output-dir", required=True, help="New experiment directory.")
    parser.add_argument(
        "--prior-output-dir",
        help="Completed targeted-zero directory; required by targeted-fixed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolved_file(args.input, "Input")
    output = Path(args.output_dir).expanduser().resolve()
    baseline = (
        resolved_directory(args.baseline_dir, "Baseline")
        if args.baseline_dir
        else None
    )
    if args.experiment != "dynamic-trends":
        if baseline is None:
            raise ValueError(f"{args.experiment} requires --baseline-dir")
        verify_baseline(baseline)
    if args.experiment == "targeted-fixed" and not args.prior_output_dir:
        raise ValueError("targeted-fixed requires --prior-output-dir")

    prepare_output(
        output,
        baseline,
        copy_baseline=args.experiment not in NO_BASELINE_COPY,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "BHI_IN_MASTER": str(input_path),
            "BHI_OUT_DIR": str(output),
            "BHI_EXPERIMENT_OUTPUT_DIR": str(output),
        }
    )
    if baseline is not None:
        environment["BHI_BASELINE_DIR"] = str(baseline)
    if args.prior_output_dir:
        prior = resolved_directory(args.prior_output_dir, "Prior output")
        environment["BHI_PRIOR_OUTLIER_OUTPUT_DIR"] = str(prior)

    if args.experiment in MAIN_FLAGS:
        command = [sys.executable, str(ROOT / "main.py"), MAIN_FLAGS[args.experiment]]
    else:
        command = [
            sys.executable,
            str(ROOT / "scripts" / STANDALONE[args.experiment]),
        ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
