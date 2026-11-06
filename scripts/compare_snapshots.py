"""Compare two completed BHI snapshot output directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import compare_temporal_snapshots as core  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--updated-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-label", default="Baseline snapshot")
    parser.add_argument("--updated-label", default="Updated snapshot")
    return parser.parse_args()


def load(root: Path, key: str, label: str) -> dict:
    root = root.expanduser().resolve()
    metadata = root / "run_metadata.json"
    spec = core.SnapshotSpec(key, label, root, metadata if metadata.is_file() else None)
    return core.validate_and_load_snapshot(spec)


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    baseline = load(args.baseline_dir, "baseline", args.baseline_label)
    updated = load(args.updated_dir, "updated", args.updated_label)
    comparison = core.ComparisonSpec(
        "baseline_vs_updated",
        f"{args.baseline_label} vs {args.updated_label}",
        "baseline",
        "updated",
    )
    matches, mapping_rows = core.build_matches(
        comparison, baseline["frame"], updated["frame"]
    )
    rng = np.random.default_rng(core.BOOTSTRAP_SEED)
    changes_all: list[pd.DataFrame] = []
    ranking_rows: list[dict] = []
    score_rows: list[dict] = []
    mover_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    for strategy, pairs in matches.items():
        changes = core.build_change_table(
            comparison, strategy, pairs, baseline["frame"], updated["frame"]
        )
        changes_all.append(changes)
        ranking_rows.append(
            core.ranking_metrics(
                comparison,
                strategy,
                pairs,
                baseline["frame"],
                updated["frame"],
                changes,
            )
        )
        score_rows.extend(core.score_metrics(comparison, strategy, changes))
        mover_rows.extend(core.top_movers(changes))
        bootstrap_rows.extend(
            core.bootstrap_intervals(comparison, strategy, changes, rng)
        )

    output.mkdir(parents=True)
    frames = {
        "benchmark_name_mapping.csv": pd.DataFrame(mapping_rows),
        "ranking_stability_metrics.csv": pd.DataFrame(ranking_rows),
        "score_stability_metrics.csv": pd.DataFrame(score_rows),
        "benchmark_level_changes.csv": pd.concat(changes_all, ignore_index=True),
        "top_movers.csv": pd.DataFrame(mover_rows),
        "bootstrap_confidence_intervals.csv": pd.DataFrame(bootstrap_rows),
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    summary = {
        "baseline": {
            "label": args.baseline_label,
            "model_count": baseline["model_count"],
            "ranked_benchmark_count": len(baseline["frame"]),
        },
        "updated": {
            "label": args.updated_label,
            "model_count": updated["model_count"],
            "ranked_benchmark_count": len(updated["frame"]),
        },
        "matching": {
            key: len(value) for key, value in matches.items()
        },
        "bootstrap_iterations": core.BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": core.BOOTSTRAP_SEED,
        "note": "Scores are snapshot-conditioned; rank-based metrics are primary.",
    }
    (output / "temporal_comparison_summary.json").write_text(
        json.dumps(core.json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote temporal comparison outputs to {output}")


if __name__ == "__main__":
    main()
