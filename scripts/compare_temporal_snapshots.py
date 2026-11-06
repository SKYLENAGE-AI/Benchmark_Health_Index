"""Deterministic, output-only temporal stability analysis for BHI snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "temporal_comparison"
REQUIRED_COLUMNS = ["Benchmark", "S_Disc", "S_Sat", "S_Imp", "BHI"]
SCORES = ["S_Disc", "S_Sat", "S_Imp", "BHI"]
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20260726
TOLERANCE = 1e-12


def display_path(path: Path) -> str:
    """Return a portable repository-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class SnapshotSpec:
    key: str
    label: str
    root: Path
    metadata_path: Path | None


@dataclass(frozen=True)
class ComparisonSpec:
    key: str
    label: str
    baseline: str
    updated: str


SNAPSHOTS: dict[str, SnapshotSpec] = {}

COMPARISONS = [
    ComparisonSpec(
        "comparison_a_original_vs_june",
        "Original submission vs June 30 updated snapshot",
        "original",
        "june",
    ),
    ComparisonSpec(
        "comparison_b_original_vs_april_control",
        "Original submission vs April control snapshot",
        "original",
        "april_control",
    ),
    ComparisonSpec(
        "comparison_c_april_control_vs_june",
        "April control snapshot vs June 30 updated snapshot",
        "april_control",
        "june",
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths() -> list[Path]:
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "main.py",
    ]
    paths.extend(sorted((PROJECT_ROOT / "config").glob("*.py")))
    paths.extend(sorted((PROJECT_ROOT / "src").glob("*.py")))
    for snapshot in SNAPSHOTS.values():
        paths.extend(
            snapshot.root / filename
            for filename in [
                "final_bhi_rankings.csv",
                "capability_discrimination_final.csv",
                "anti_saturation_results.csv",
                "impact_results.csv",
                "benchmark_inclusion_report.csv",
            ]
        )
        if snapshot.metadata_path is not None:
            paths.append(snapshot.metadata_path)
    return list(dict.fromkeys(paths))


def validate_and_load_snapshot(spec: SnapshotSpec) -> dict:
    final_path = spec.root / "final_bhi_rankings.csv"
    if not final_path.is_file():
        raise FileNotFoundError(f"Missing final ranking file: {final_path}")

    raw = pd.read_csv(final_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(
            f"{final_path} is missing required columns: {missing_columns}; "
            f"available columns are {list(raw.columns)}"
        )

    benchmark_missing = int(raw["Benchmark"].isna().sum())
    stripped = raw["Benchmark"].astype("string").str.strip()
    blank_benchmarks = int(stripped.eq("").fillna(False).sum())
    duplicate_count = int(stripped.duplicated(keep=False).sum())
    missing_values = {
        score: int(pd.to_numeric(raw[score], errors="coerce").isna().sum())
        for score in SCORES
    }
    if benchmark_missing or blank_benchmarks or duplicate_count:
        raise ValueError(
            f"Unsafe benchmark identifiers in {final_path}: missing={benchmark_missing}, "
            f"blank={blank_benchmarks}, duplicate_rows={duplicate_count}"
        )
    if any(missing_values.values()):
        raise ValueError(f"Missing/non-numeric component values in {final_path}: {missing_values}")

    frame = raw[REQUIRED_COLUMNS].copy()
    frame["Benchmark"] = stripped.astype(str)
    for score in SCORES:
        frame[score] = pd.to_numeric(frame[score], errors="raise")
    frame = frame.set_index("Benchmark", drop=False)

    metadata = None
    if spec.metadata_path is not None:
        if not spec.metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {spec.metadata_path}")
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))

    components = frame[["S_Disc", "S_Sat", "S_Imp"]]
    std = components.std()
    correlation = components.corr()
    conflict = (1 - correlation).sum()
    information = std * conflict
    weights = information / information.sum()
    reconstructed_bhi = components.mul(weights, axis=1).sum(axis=1)
    reconstruction_error = float((reconstructed_bhi - frame["BHI"]).abs().max())
    if reconstruction_error > TOLERANCE:
        raise ValueError(
            f"BHI reconstruction failed for {spec.label}: {reconstruction_error}"
        )
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=TOLERANCE):
        raise ValueError(f"CRITIC weights do not sum to one for {spec.label}")

    metadata_weight_differences = None
    if metadata is not None:
        metadata_weights = metadata["critic_weights"]
        metadata_weight_differences = {
            "s_disc": abs(float(weights["S_Disc"]) - metadata_weights["s_disc"]),
            "s_sat": abs(float(weights["S_Sat"]) - metadata_weights["s_sat"]),
            "s_imp": abs(float(weights["S_Imp"]) - metadata_weights["s_imp"]),
        }
        if max(metadata_weight_differences.values()) > TOLERANCE:
            raise ValueError(
                f"Metadata CRITIC weights disagree for {spec.label}: "
                f"{metadata_weight_differences}"
            )

    model_count = None
    if metadata is not None:
        model_count = int(metadata["total_model_count"])
    else:
        impact_path = spec.root / "impact_results.csv"
        impact = pd.read_csv(impact_path)
        if "Total_Models" in impact.columns and not impact.empty:
            unique_counts = impact["Total_Models"].dropna().unique()
            if len(unique_counts) == 1:
                model_count = int(unique_counts[0])

    validation = {
        "final_path": display_path(final_path),
        "benchmark_column": "Benchmark",
        "component_columns": SCORES,
        "row_count": int(len(frame)),
        "duplicate_benchmark_row_count": duplicate_count,
        "missing_benchmark_count": benchmark_missing,
        "blank_benchmark_count": blank_benchmarks,
        "missing_component_values": missing_values,
        "bhi_descending": bool(frame["BHI"].is_monotonic_decreasing),
        "benchmark_names_unique": bool(frame.index.is_unique),
        "bhi_reconstruction_max_abs_difference": reconstruction_error,
        "metadata_weight_max_abs_difference": (
            max(metadata_weight_differences.values())
            if metadata_weight_differences is not None
            else None
        ),
    }
    return {
        "spec": spec,
        "frame": frame,
        "metadata": metadata,
        "model_count": model_count,
        "weights": weights,
        "validation": validation,
    }


def normalize_cosmetic(value: str, expand_year: bool = False) -> tuple[str, list[str]]:
    reasons: list[str] = []
    normalized = unicodedata.normalize("NFKC", value).strip()
    if normalized != value:
        reasons.append("unicode_nfkc_or_trim")

    dash_normalized = re.sub(r"[\u2010-\u2015\u2212]", "-", normalized)
    if dash_normalized != normalized:
        reasons.append("unicode_dash")
    normalized = dash_normalized

    parenthesis_normalized = re.sub(r"\s*\(\s*", " (", normalized)
    parenthesis_normalized = re.sub(r"\s*\)\s*", ") ", parenthesis_normalized)
    if parenthesis_normalized != normalized:
        reasons.append("parenthesis_spacing")
    normalized = parenthesis_normalized

    separator_normalized = re.sub(r"[-_\s]+", " ", normalized)
    if separator_normalized != normalized:
        reasons.append("hyphen_underscore_space")
    normalized = re.sub(r"\s+", " ", separator_normalized).strip()

    case_normalized = normalized.casefold()
    if case_normalized != normalized:
        reasons.append("case")
    normalized = case_normalized

    if expand_year:
        year_normalized = re.sub(
            r"(?<![\d])(\d{2})(?![\d])",
            lambda match: f"20{match.group(1)}",
            normalized,
        )
        if year_normalized != normalized:
            reasons.append("two_digit_year_expansion")
        normalized = year_normalized

    return normalized, sorted(set(reasons))


def build_matches(
    comparison: ComparisonSpec,
    baseline: pd.DataFrame,
    updated: pd.DataFrame,
) -> tuple[dict[str, list[dict]], list[dict]]:
    exact_pairs = [
        {
            "baseline_benchmark": name,
            "updated_benchmark": name,
            "match_type": "exact",
            "normalization_reason": "exact_after_trim",
            "is_exact": True,
            "included_in_conservative_analysis": True,
            "ambiguity_flag": False,
        }
        for name in baseline.index
        if name in updated.index
    ]
    matched_baseline = {row["baseline_benchmark"] for row in exact_pairs}
    matched_updated = {row["updated_benchmark"] for row in exact_pairs}

    mapping_rows = [
        {"comparison_name": comparison.key, **row} for row in exact_pairs
    ]
    aliases: list[dict] = []

    def match_stage(expand_year: bool, match_type: str) -> None:
        remaining_baseline = [
            name for name in baseline.index if name not in matched_baseline
        ]
        remaining_updated = [
            name for name in updated.index if name not in matched_updated
        ]
        baseline_groups: dict[str, list[str]] = {}
        updated_groups: dict[str, list[str]] = {}
        for name in remaining_baseline:
            key, _ = normalize_cosmetic(name, expand_year=expand_year)
            baseline_groups.setdefault(key, []).append(name)
        for name in remaining_updated:
            key, _ = normalize_cosmetic(name, expand_year=expand_year)
            updated_groups.setdefault(key, []).append(name)

        for key in sorted(set(baseline_groups) & set(updated_groups)):
            baseline_candidates = baseline_groups[key]
            updated_candidates = updated_groups[key]
            ambiguous = len(baseline_candidates) != 1 or len(updated_candidates) != 1
            for baseline_name in baseline_candidates:
                for updated_name in updated_candidates:
                    _, baseline_reasons = normalize_cosmetic(
                        baseline_name, expand_year=expand_year
                    )
                    _, updated_reasons = normalize_cosmetic(
                        updated_name, expand_year=expand_year
                    )
                    reasons = sorted(set(baseline_reasons + updated_reasons))
                    row = {
                        "comparison_name": comparison.key,
                        "baseline_benchmark": baseline_name,
                        "updated_benchmark": updated_name,
                        "match_type": match_type,
                        "normalization_reason": ";".join(reasons) or match_type,
                        "is_exact": False,
                        "included_in_conservative_analysis": not ambiguous,
                        "ambiguity_flag": ambiguous,
                    }
                    mapping_rows.append(row)
                    if not ambiguous:
                        aliases.append({key: value for key, value in row.items() if key != "comparison_name"})
            if not ambiguous:
                matched_baseline.add(baseline_candidates[0])
                matched_updated.add(updated_candidates[0])

    match_stage(False, "cosmetic_normalization")
    match_stage(True, "unique_two_digit_year_expansion")

    conservative_pairs = exact_pairs + aliases
    for pairs in [exact_pairs, conservative_pairs]:
        baseline_names = [row["baseline_benchmark"] for row in pairs]
        updated_names = [row["updated_benchmark"] for row in pairs]
        if len(baseline_names) != len(set(baseline_names)):
            raise ValueError(f"Duplicate baseline mapping in {comparison.key}")
        if len(updated_names) != len(set(updated_names)):
            raise ValueError(f"Duplicate updated mapping in {comparison.key}")

    return {
        "exact_name": exact_pairs,
        "conservative_alias_adjusted": conservative_pairs,
    }, mapping_rows


def fixed_top_set(frame: pd.DataFrame, k: int) -> set[str]:
    ordered = frame.reset_index(drop=True).sort_values(
        ["BHI", "Benchmark"], ascending=[False, True], kind="mergesort"
    )
    return set(ordered.head(k)["Benchmark"])


def expanded_top_set(frame: pd.DataFrame, k: int) -> tuple[set[str], bool]:
    ordered = frame.reset_index(drop=True).sort_values(
        ["BHI", "Benchmark"], ascending=[False, True], kind="mergesort"
    )
    if len(ordered) <= k:
        return set(ordered["Benchmark"]), False
    cutoff = float(ordered.iloc[k - 1]["BHI"])
    expanded = set(ordered.loc[ordered["BHI"] >= cutoff, "Benchmark"])
    tie_at_cutoff = int((ordered["BHI"] == cutoff).sum()) > 1
    return expanded, tie_at_cutoff


def matched_overlap(
    pairs: list[dict],
    baseline_set: set[str],
    updated_set: set[str],
) -> int:
    return sum(
        row["baseline_benchmark"] in baseline_set
        and row["updated_benchmark"] in updated_set
        for row in pairs
    )


def build_change_table(
    comparison: ComparisonSpec,
    strategy: str,
    pairs: list[dict],
    baseline: pd.DataFrame,
    updated: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    baseline_full_rank = baseline["BHI"].rank(method="average", ascending=False)
    updated_full_rank = updated["BHI"].rank(method="average", ascending=False)
    baseline_n = len(baseline)
    updated_n = len(updated)

    pair_frame = pd.DataFrame(
        {
            "baseline_benchmark": [row["baseline_benchmark"] for row in pairs],
            "updated_benchmark": [row["updated_benchmark"] for row in pairs],
            "match_type": [row["match_type"] for row in pairs],
        }
    )
    pair_frame["baseline_bhi"] = [
        baseline.loc[name, "BHI"] for name in pair_frame["baseline_benchmark"]
    ]
    pair_frame["updated_bhi"] = [
        updated.loc[name, "BHI"] for name in pair_frame["updated_benchmark"]
    ]
    baseline_common_rank = pair_frame["baseline_bhi"].rank(
        method="average", ascending=False
    )
    updated_common_rank = pair_frame["updated_bhi"].rank(
        method="average", ascending=False
    )

    baseline_top10 = fixed_top_set(baseline, 10)
    updated_top10 = fixed_top_set(updated, 10)
    baseline_top20 = fixed_top_set(baseline, 20)
    updated_top20 = fixed_top_set(updated, 20)

    for position, pair in pair_frame.iterrows():
        baseline_name = pair["baseline_benchmark"]
        updated_name = pair["updated_benchmark"]
        row = {
            "comparison_name": comparison.key,
            "comparison_label": comparison.label,
            "matching_strategy": strategy,
            "baseline_benchmark_name": baseline_name,
            "updated_benchmark_name": updated_name,
            "alias_match_type": pair["match_type"],
        }
        for score in SCORES:
            baseline_value = float(baseline.loc[baseline_name, score])
            updated_value = float(updated.loc[updated_name, score])
            row[f"baseline_{score}"] = baseline_value
            row[f"updated_{score}"] = updated_value
            row[f"{score}_change"] = updated_value - baseline_value

        baseline_rank = float(baseline_common_rank.loc[position])
        updated_rank = float(updated_common_rank.loc[position])
        baseline_full = float(baseline_full_rank.loc[baseline_name])
        updated_full = float(updated_full_rank.loc[updated_name])
        baseline_percentile = (
            1 - (baseline_full - 1) / (baseline_n - 1) if baseline_n > 1 else 1.0
        )
        updated_percentile = (
            1 - (updated_full - 1) / (updated_n - 1) if updated_n > 1 else 1.0
        )
        percentile_change = updated_percentile - baseline_percentile
        row.update(
            {
                "baseline_common_only_rank": baseline_rank,
                "updated_common_only_rank": updated_rank,
                "common_only_rank_change": updated_rank - baseline_rank,
                "absolute_common_only_rank_change": abs(updated_rank - baseline_rank),
                "baseline_full_rank": baseline_full,
                "updated_full_rank": updated_full,
                "baseline_rank_percentile": baseline_percentile,
                "updated_rank_percentile": updated_percentile,
                "rank_percentile_change": percentile_change,
                "absolute_rank_percentile_change": abs(percentile_change),
                "baseline_top_10": baseline_name in baseline_top10,
                "updated_top_10": updated_name in updated_top10,
                "baseline_top_20": baseline_name in baseline_top20,
                "updated_top_20": updated_name in updated_top20,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_metrics(
    comparison: ComparisonSpec,
    strategy: str,
    pairs: list[dict],
    baseline: pd.DataFrame,
    updated: pd.DataFrame,
    changes: pd.DataFrame,
) -> dict:
    baseline_ranks = changes["baseline_common_only_rank"]
    updated_ranks = changes["updated_common_only_rank"]
    rho = float(spearmanr(baseline_ranks, updated_ranks).statistic)
    tau = float(kendalltau(baseline_ranks, updated_ranks, variant="b").statistic)

    result = {
        "comparison_name": comparison.key,
        "comparison_label": comparison.label,
        "matching_strategy": strategy,
        "common_benchmark_count": int(len(changes)),
        "spearman_rho_common_only": rho,
        "kendall_tau_b_common_only": tau,
    }
    for k in [10, 20]:
        baseline_fixed = fixed_top_set(baseline, k)
        updated_fixed = fixed_top_set(updated, k)
        fixed_shared = matched_overlap(pairs, baseline_fixed, updated_fixed)
        fixed_union = len(baseline_fixed) + len(updated_fixed) - fixed_shared
        baseline_expanded, baseline_tie = expanded_top_set(baseline, k)
        updated_expanded, updated_tie = expanded_top_set(updated, k)
        expanded_shared = matched_overlap(pairs, baseline_expanded, updated_expanded)
        expanded_union = (
            len(baseline_expanded) + len(updated_expanded) - expanded_shared
        )
        result.update(
            {
                f"top_{k}_shared_count": int(fixed_shared),
                f"top_{k}_overlap_rate": fixed_shared / k,
                f"top_{k}_jaccard": fixed_shared / fixed_union if fixed_union else 1.0,
                f"top_{k}_baseline_cutoff_tie": baseline_tie,
                f"top_{k}_updated_cutoff_tie": updated_tie,
                f"top_{k}_tie_aware_baseline_size": len(baseline_expanded),
                f"top_{k}_tie_aware_updated_size": len(updated_expanded),
                f"top_{k}_tie_aware_shared_count": int(expanded_shared),
                f"top_{k}_tie_aware_jaccard": (
                    expanded_shared / expanded_union if expanded_union else 1.0
                ),
            }
        )
        if fixed_shared > k:
            raise ValueError(f"Top-{k} overlap exceeds k for {comparison.key}")

    abs_rank = changes["absolute_common_only_rank_change"]
    abs_percentile = changes["absolute_rank_percentile_change"]
    signed_percentile = changes["rank_percentile_change"]
    direction_tolerance = 1e-12
    result.update(
        {
            "common_rank_change_mean_abs": float(abs_rank.mean()),
            "common_rank_change_median_abs": float(abs_rank.median()),
            "common_rank_change_p90_abs": float(abs_rank.quantile(0.9)),
            "common_rank_change_max_abs": float(abs_rank.max()),
            "rank_percentile_change_mean_abs": float(abs_percentile.mean()),
            "rank_percentile_change_median_abs": float(abs_percentile.median()),
            "rank_percentile_change_p90_abs": float(abs_percentile.quantile(0.9)),
            "rank_percentile_change_max_abs": float(abs_percentile.max()),
            "rank_percentile_change_mean_signed": float(signed_percentile.mean()),
            "rank_percentile_moving_up_count": int(
                (signed_percentile > direction_tolerance).sum()
            ),
            "rank_percentile_moving_down_count": int(
                (signed_percentile < -direction_tolerance).sum()
            ),
            "rank_percentile_unchanged_count": int(
                (signed_percentile.abs() <= direction_tolerance).sum()
            ),
        }
    )

    n = len(changes)
    quartile_cutoff = math.ceil(n * 0.25)
    half_cutoff = math.ceil(n * 0.5)
    baseline_quartile = changes["baseline_common_only_rank"] <= quartile_cutoff
    baseline_half = changes["baseline_common_only_rank"] <= half_cutoff
    result["top_quartile_retention"] = float(
        (
            changes.loc[baseline_quartile, "updated_common_only_rank"]
            <= quartile_cutoff
        ).mean()
    )
    result["top_half_retention"] = float(
        (
            changes.loc[baseline_half, "updated_common_only_rank"] <= half_cutoff
        ).mean()
    )
    if not (-1 <= rho <= 1 and -1 <= tau <= 1):
        raise ValueError(f"Invalid rank correlation for {comparison.key}")
    return result


def score_metrics(
    comparison: ComparisonSpec,
    strategy: str,
    changes: pd.DataFrame,
) -> list[dict]:
    rows = []
    for score in SCORES:
        baseline = changes[f"baseline_{score}"]
        updated = changes[f"updated_{score}"]
        delta = changes[f"{score}_change"]
        absolute = delta.abs()
        pearson = float(baseline.corr(updated, method="pearson"))
        spearman = float(baseline.corr(updated, method="spearman"))
        if not (-1 <= pearson <= 1 and -1 <= spearman <= 1):
            raise ValueError(f"Invalid score correlation for {comparison.key}/{score}")
        row = {
            "comparison_name": comparison.key,
            "comparison_label": comparison.label,
            "matching_strategy": strategy,
            "score": score,
            "common_benchmark_count": int(len(changes)),
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "mean_signed_change": float(delta.mean()),
            "mean_absolute_change": float(absolute.mean()),
            "median_absolute_change": float(absolute.median()),
            "root_mean_squared_change": float(np.sqrt(np.mean(delta**2))),
            "p90_absolute_change": float(absolute.quantile(0.9)),
            "max_absolute_change": float(absolute.max()),
            "baseline_mean": float(baseline.mean()),
            "baseline_std": float(baseline.std()),
            "updated_mean": float(updated.mean()),
            "updated_std": float(updated.std()),
        }
        for threshold in [0.01, 0.025, 0.05]:
            key = str(threshold).replace(".", "_")
            count = int((absolute < threshold).sum())
            row[f"absolute_change_below_{key}_count"] = count
            row[f"absolute_change_below_{key}_proportion"] = count / len(changes)
        rows.append(row)
    return rows


def top_movers(changes: pd.DataFrame) -> list[dict]:
    categories = [
        ("rank_percentile_upward", "rank_percentile_change", False, False),
        ("rank_percentile_downward", "rank_percentile_change", True, False),
        ("absolute_BHI_change", "BHI_change", False, True),
        ("absolute_S_Disc_change", "S_Disc_change", False, True),
        ("absolute_S_Sat_change", "S_Sat_change", False, True),
        ("absolute_S_Imp_change", "S_Imp_change", False, True),
    ]
    rows = []
    for category, column, ascending, by_absolute in categories:
        selection = changes[column].abs() if by_absolute else changes[column]
        order = selection.sort_values(ascending=ascending, kind="mergesort").index[:10]
        for rank, index in enumerate(order, start=1):
            change = changes.loc[index]
            rows.append(
                {
                    "comparison_name": change["comparison_name"],
                    "comparison_label": change["comparison_label"],
                    "matching_strategy": change["matching_strategy"],
                    "mover_category": category,
                    "mover_rank": rank,
                    "baseline_benchmark_name": change["baseline_benchmark_name"],
                    "updated_benchmark_name": change["updated_benchmark_name"],
                    "metric_value": float(change[column]),
                    "absolute_metric_value": float(abs(change[column])),
                    "rank_percentile_change": float(change["rank_percentile_change"]),
                    "BHI_change": float(change["BHI_change"]),
                    "S_Disc_change": float(change["S_Disc_change"]),
                    "S_Sat_change": float(change["S_Sat_change"]),
                    "S_Imp_change": float(change["S_Imp_change"]),
                }
            )
    return rows


def bootstrap_intervals(
    comparison: ComparisonSpec,
    strategy: str,
    changes: pd.DataFrame,
    rng: np.random.Generator,
) -> list[dict]:
    n = len(changes)
    baseline_rank = changes["baseline_common_only_rank"].to_numpy()
    updated_rank = changes["updated_common_only_rank"].to_numpy()
    percentile_abs = changes["absolute_rank_percentile_change"].to_numpy()
    bhi_abs = changes["BHI_change"].abs().to_numpy()
    definitions = {
        "spearman_rho": (
            float(spearmanr(baseline_rank, updated_rank).statistic),
            lambda index: float(
                spearmanr(baseline_rank[index], updated_rank[index]).statistic
            ),
        ),
        "kendall_tau_b": (
            float(kendalltau(baseline_rank, updated_rank, variant="b").statistic),
            lambda index: float(
                kendalltau(
                    baseline_rank[index], updated_rank[index], variant="b"
                ).statistic
            ),
        ),
        "median_absolute_rank_percentile_change": (
            float(np.median(percentile_abs)),
            lambda index: float(np.median(percentile_abs[index])),
        ),
        "median_absolute_BHI_change": (
            float(np.median(bhi_abs)),
            lambda index: float(np.median(bhi_abs[index])),
        ),
    }
    samples = {metric: [] for metric in definitions}
    skipped = {metric: 0 for metric in definitions}
    for _ in range(BOOTSTRAP_ITERATIONS):
        index = rng.integers(0, n, size=n)
        for metric, (_, function) in definitions.items():
            value = function(index)
            if np.isfinite(value):
                samples[metric].append(value)
            else:
                skipped[metric] += 1

    rows = []
    for metric, (estimate, _) in definitions.items():
        values = np.asarray(samples[metric])
        rows.append(
            {
                "comparison_name": comparison.key,
                "comparison_label": comparison.label,
                "matching_strategy": strategy,
                "metric": metric,
                "estimate": estimate,
                "ci_95_lower": float(np.quantile(values, 0.025)),
                "ci_95_upper": float(np.quantile(values, 0.975)),
                "requested_resamples": BOOTSTRAP_ITERATIONS,
                "valid_resamples": int(len(values)),
                "skipped_resamples": int(skipped[metric]),
                "seed": BOOTSTRAP_SEED,
            }
        )
    return rows


def make_weight_drift(snapshot_data: dict) -> pd.DataFrame:
    relationships = [
        ("original_submission", "original", "original"),
        ("comparison_b_april_control_vs_original", "original", "april_control"),
        ("comparison_a_june_vs_original", "original", "june"),
        ("comparison_c_june_vs_april_control", "april_control", "june"),
    ]
    rows = []
    for name, baseline_key, snapshot_key in relationships:
        baseline = snapshot_data[baseline_key]
        snapshot = snapshot_data[snapshot_key]
        baseline_weights = baseline["weights"]
        weights = snapshot["weights"]
        changes = weights - baseline_weights
        absolute = changes.abs()
        row = {
            "comparison_name": name,
            "baseline_snapshot": baseline["spec"].label,
            "snapshot_name": snapshot["spec"].label,
            "model_count": snapshot["model_count"],
            "ranked_benchmark_count": len(snapshot["frame"]),
        }
        for score, label in [
            ("S_Disc", "s_disc"),
            ("S_Sat", "s_sat"),
            ("S_Imp", "s_imp"),
        ]:
            row[f"{label}_weight"] = float(weights[score])
            row[f"{label}_baseline_weight"] = float(baseline_weights[score])
            row[f"{label}_absolute_change"] = float(absolute[score])
            row[f"{label}_percentage_point_change"] = float(changes[score] * 100)
            row[f"{label}_relative_percentage_change"] = (
                float(changes[score] / baseline_weights[score] * 100)
                if baseline_weights[score] != 0
                else np.nan
            )
        row["maximum_absolute_percentage_point_drift"] = float(
            absolute.max() * 100
        )
        row["sum_absolute_percentage_point_changes"] = float(
            absolute.sum() * 100
        )
        row["total_variation_distance"] = float(0.5 * absolute.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def parser_control_table(
    ranking: pd.DataFrame,
    scores: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    comparison_name = "comparison_b_original_vs_april_control"
    rows = []
    for strategy in ["exact_name", "conservative_alias_adjusted"]:
        rank = ranking[
            (ranking["comparison_name"] == comparison_name)
            & (ranking["matching_strategy"] == strategy)
        ].iloc[0]
        score_subset = scores[
            (scores["comparison_name"] == comparison_name)
            & (scores["matching_strategy"] == strategy)
        ].set_index("score")
        change_subset = changes[
            (changes["comparison_name"] == comparison_name)
            & (changes["matching_strategy"] == strategy)
        ]
        rows.append(
            {
                "matching_strategy": strategy,
                "common_benchmark_count": int(rank["common_benchmark_count"]),
                "benchmark_membership_unchanged": (
                    int(rank["common_benchmark_count"]) == 158
                ),
                "spearman_rho": rank["spearman_rho_common_only"],
                "kendall_tau_b": rank["kendall_tau_b_common_only"],
                "top_10_shared_count": int(rank["top_10_shared_count"]),
                "top_20_shared_count": int(rank["top_20_shared_count"]),
                "common_rank_change_mean_abs": rank["common_rank_change_mean_abs"],
                "common_rank_change_median_abs": rank[
                    "common_rank_change_median_abs"
                ],
                "rank_percentile_change_mean_signed": rank[
                    "rank_percentile_change_mean_signed"
                ],
                "rank_percentile_change_median_abs": rank[
                    "rank_percentile_change_median_abs"
                ],
                "s_disc_max_abs_difference": float(
                    change_subset["S_Disc_change"].abs().max()
                ),
                "s_sat_max_abs_difference": float(
                    change_subset["S_Sat_change"].abs().max()
                ),
                "s_imp_max_abs_difference": float(
                    change_subset["S_Imp_change"].abs().max()
                ),
                "bhi_max_abs_difference": float(
                    change_subset["BHI_change"].abs().max()
                ),
                "s_disc_mean_abs_difference": score_subset.loc[
                    "S_Disc", "mean_absolute_change"
                ],
                "s_sat_mean_abs_difference": score_subset.loc[
                    "S_Sat", "mean_absolute_change"
                ],
                "s_imp_mean_abs_difference": score_subset.loc[
                    "S_Imp", "mean_absolute_change"
                ],
                "bhi_mean_abs_difference": score_subset.loc[
                    "BHI", "mean_absolute_change"
                ],
                "s_disc_numerically_unchanged": bool(
                    change_subset["S_Disc_change"].abs().max() <= TOLERANCE
                ),
                "s_imp_numerically_unchanged": bool(
                    change_subset["S_Imp_change"].abs().max() <= TOLERANCE
                ),
                "only_s_sat_and_bhi_changed": bool(
                    change_subset["S_Disc_change"].abs().max() <= TOLERANCE
                    and change_subset["S_Imp_change"].abs().max() <= TOLERANCE
                    and change_subset["S_Sat_change"].abs().max() > TOLERANCE
                    and change_subset["BHI_change"].abs().max() > TOLERANCE
                ),
            }
        )
    return pd.DataFrame(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bool, int)) else False:
        return None
    return value


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include="number").columns:
        selected[column] = selected[column].map(lambda value: f"{value:.6f}")
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in selected.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_report(
    weight_drift: pd.DataFrame,
    ranking: pd.DataFrame,
    scores: pd.DataFrame,
    parser_control: pd.DataFrame,
    movers: pd.DataFrame,
    matching_summary: dict,
) -> None:
    sections = [
        (
            "1. Original submission vs June — primary comparison",
            "comparison_a_original_vs_june",
        ),
        (
            "2. Original submission vs April control — parser-only control",
            "comparison_b_original_vs_april_control",
        ),
        (
            "3. April control vs June — method-consistent temporal comparison",
            "comparison_c_april_control_vs_june",
        ),
    ]
    lines = [
        "# BHI temporal stability analysis",
        "",
        "Scores are snapshot-conditioned: component distributions, min-max normalization, "
        "CRITIC weights, and model/benchmark populations can change. Ranking correlations "
        "and rank-percentile movement are therefore primary evidence; raw-score changes "
        "are supporting evidence.",
        "",
        "## CRITIC weight drift",
        "",
        markdown_table(
            weight_drift,
            [
                "comparison_name",
                "s_disc_weight",
                "s_sat_weight",
                "s_imp_weight",
                "maximum_absolute_percentage_point_drift",
                "total_variation_distance",
            ],
        ),
        "",
    ]
    for title, comparison_name in sections:
        lines.extend([f"## {title}", ""])
        rank_subset = ranking[
            ranking["comparison_name"] == comparison_name
        ]
        lines.extend(
            [
                "### Ranking stability",
                "",
                markdown_table(
                    rank_subset,
                    [
                        "matching_strategy",
                        "common_benchmark_count",
                        "spearman_rho_common_only",
                        "kendall_tau_b_common_only",
                        "top_10_shared_count",
                        "top_20_shared_count",
                        "common_rank_change_median_abs",
                        "rank_percentile_change_median_abs",
                        "top_quartile_retention",
                        "top_half_retention",
                    ],
                ),
                "",
                "### Score/component stability",
                "",
                markdown_table(
                    scores[
                        (scores["comparison_name"] == comparison_name)
                        & (scores["matching_strategy"] == "exact_name")
                    ],
                    [
                        "score",
                        "pearson_correlation",
                        "spearman_correlation",
                        "mean_signed_change",
                        "mean_absolute_change",
                        "median_absolute_change",
                        "max_absolute_change",
                    ],
                ),
                "",
            ]
        )
        if comparison_name == "comparison_b_original_vs_april_control":
            lines.extend(
                [
                    "### Parser-only control",
                    "",
                    markdown_table(
                        parser_control,
                        [
                            "matching_strategy",
                            "common_benchmark_count",
                            "spearman_rho",
                            "kendall_tau_b",
                            "top_10_shared_count",
                            "top_20_shared_count",
                            "s_disc_max_abs_difference",
                            "s_sat_max_abs_difference",
                            "s_imp_max_abs_difference",
                            "bhi_max_abs_difference",
                        ],
                    ),
                    "",
                ]
            )
        top = movers[
            (movers["comparison_name"] == comparison_name)
            & (movers["matching_strategy"] == "exact_name")
            & movers["mover_category"].isin(
                ["rank_percentile_upward", "rank_percentile_downward"]
            )
            & (movers["mover_rank"] <= 5)
        ]
        lines.extend(
            [
                "### Selected top movers",
                "",
                markdown_table(
                    top,
                    [
                        "mover_category",
                        "mover_rank",
                        "baseline_benchmark_name",
                        "updated_benchmark_name",
                        "rank_percentile_change",
                        "BHI_change",
                        "S_Disc_change",
                        "S_Sat_change",
                        "S_Imp_change",
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Matching sensitivity",
            "",
            "Exact-name matching is primary. Conservative matching adds only unique cosmetic "
            "normalizations; ambiguous candidates are excluded.",
            "",
            "```json",
            json.dumps(json_safe(matching_summary), ensure_ascii=False, indent=2),
            "```",
            "",
            "No causal interpretation of individual movements is made here.",
        ]
    )
    (OUTPUT_DIR / "temporal_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the three-snapshot temporal stability analysis using completed "
            "BHI output directories."
        )
    )
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--april-control-dir", type=Path, required=True)
    parser.add_argument("--june-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-metadata", type=Path)
    parser.add_argument("--april-control-metadata", type=Path)
    parser.add_argument("--june-metadata", type=Path)
    return parser.parse_args()


def main() -> None:
    global OUTPUT_DIR, SNAPSHOTS
    args = parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    SNAPSHOTS = {
        "original": SnapshotSpec(
            "original", "Original submission", args.original_dir.resolve(),
            args.original_metadata.resolve() if args.original_metadata else None,
        ),
        "april_control": SnapshotSpec(
            "april_control", "April control snapshot",
            args.april_control_dir.resolve(),
            (
                args.april_control_metadata.resolve()
                if args.april_control_metadata
                else None
            ),
        ),
        "june": SnapshotSpec(
            "june", "June 30 snapshot", args.june_dir.resolve(),
            args.june_metadata.resolve() if args.june_metadata else None,
        ),
    }
    before_hashes = {path.as_posix(): sha256_file(path) for path in source_paths()}
    snapshot_data = {
        key: validate_and_load_snapshot(spec) for key, spec in SNAPSHOTS.items()
    }
    weight_drift = make_weight_drift(snapshot_data)

    mapping_rows: list[dict] = []
    all_changes: list[pd.DataFrame] = []
    ranking_rows: list[dict] = []
    score_rows: list[dict] = []
    mover_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    matching_summary = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for comparison in COMPARISONS:
        baseline = snapshot_data[comparison.baseline]["frame"]
        updated = snapshot_data[comparison.updated]["frame"]
        matches, comparison_mapping = build_matches(comparison, baseline, updated)
        mapping_rows.extend(comparison_mapping)
        matching_summary[comparison.key] = {
            "exact_count": len(matches["exact_name"]),
            "conservative_count": len(matches["conservative_alias_adjusted"]),
            "added_alias_count": (
                len(matches["conservative_alias_adjusted"])
                - len(matches["exact_name"])
            ),
            "ambiguous_candidate_count": sum(
                bool(row["ambiguity_flag"]) for row in comparison_mapping
            ),
        }
        for strategy, pairs in matches.items():
            changes = build_change_table(
                comparison, strategy, pairs, baseline, updated
            )
            if (
                changes["baseline_benchmark_name"].duplicated().any()
                or changes["updated_benchmark_name"].duplicated().any()
            ):
                raise ValueError(
                    f"Non-unique benchmark-level mapping for {comparison.key}/{strategy}"
                )
            all_changes.append(changes)
            rank = ranking_metrics(
                comparison, strategy, pairs, baseline, updated, changes
            )
            ranking_rows.append(rank)
            score_rows.extend(score_metrics(comparison, strategy, changes))
            mover_rows.extend(top_movers(changes))
            bootstrap_rows.extend(
                bootstrap_intervals(comparison, strategy, changes, rng)
            )

    changes_frame = pd.concat(all_changes, ignore_index=True)
    ranking_frame = pd.DataFrame(ranking_rows)
    score_frame = pd.DataFrame(score_rows)
    mapping_frame = pd.DataFrame(mapping_rows)
    movers_frame = pd.DataFrame(mover_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    parser_frame = parser_control_table(
        ranking_frame, score_frame, changes_frame
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "critic_weight_drift.csv": weight_drift,
        "benchmark_name_mapping.csv": mapping_frame,
        "ranking_stability_metrics.csv": ranking_frame,
        "score_stability_metrics.csv": score_frame,
        "benchmark_level_changes.csv": changes_frame,
        "top_movers.csv": movers_frame,
        "parser_control_metrics.csv": parser_frame,
        "bootstrap_confidence_intervals.csv": bootstrap_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    after_hashes = {path.as_posix(): sha256_file(path) for path in source_paths()}
    unchanged = before_hashes == after_hashes
    if not unchanged:
        changed = [
            path
            for path in before_hashes
            if before_hashes[path] != after_hashes.get(path)
        ]
        raise RuntimeError(f"Read-only source invariant failed; changed files: {changed}")

    for filename in outputs:
        pd.read_csv(OUTPUT_DIR / filename)

    paper_weights = {"S_Disc": 0.3263, "S_Sat": 0.3574, "S_Imp": 0.3163}
    original_weights = snapshot_data["original"]["weights"]
    paper_differences_pp = {
        score: float((original_weights[score] - paper_weights[score]) * 100)
        for score in paper_weights
    }
    ambiguous = mapping_frame[mapping_frame["ambiguity_flag"]].to_dict("records")
    warnings = [
        "Scores are snapshot-conditioned; rank-based metrics are primary.",
        "Comparison A includes temporal-data and parser-control differences.",
        "Comparison C is the method-consistent temporal control.",
    ]
    if ambiguous:
        warnings.append("Ambiguous alias candidates were excluded.")

    summary = {
        "validation": {
            key: value["validation"] for key, value in snapshot_data.items()
        },
        "snapshot_counts": {
            key: {
                "model_count": value["model_count"],
                "ranked_benchmark_count": len(value["frame"]),
            }
            for key, value in snapshot_data.items()
        },
        "reconstructed_critic_weights": {
            key: {
                "s_disc": float(value["weights"]["S_Disc"]),
                "s_sat": float(value["weights"]["S_Sat"]),
                "s_imp": float(value["weights"]["S_Imp"]),
            }
            for key, value in snapshot_data.items()
        },
        "original_paper_weight_difference_percentage_points": paper_differences_pp,
        "weight_drift": weight_drift.to_dict("records"),
        "exact_name_metrics": ranking_frame[
            ranking_frame["matching_strategy"] == "exact_name"
        ].to_dict("records"),
        "conservative_alias_adjusted_metrics": ranking_frame[
            ranking_frame["matching_strategy"] == "conservative_alias_adjusted"
        ].to_dict("records"),
        "parser_only_control": parser_frame.to_dict("records"),
        "method_consistent_temporal_control": ranking_frame[
            ranking_frame["comparison_name"]
            == "comparison_c_april_control_vs_june"
        ].to_dict("records"),
        "key_score_stability_metrics": score_frame.to_dict("records"),
        "benchmark_matching": matching_summary,
        "ambiguous_benchmark_aliases": ambiguous,
        "bootstrap_confidence_intervals": bootstrap_frame.to_dict("records"),
        "warnings": warnings,
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "all_source_files_unchanged": unchanged,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }
    (OUTPUT_DIR / "temporal_comparison_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_report(
        weight_drift,
        ranking_frame,
        score_frame,
        parser_frame,
        movers_frame,
        matching_summary,
    )

    print(f"Wrote temporal comparison outputs to {OUTPUT_DIR}")
    print("All source and snapshot hashes unchanged:", unchanged)
    print(
        ranking_frame[
            [
                "comparison_name",
                "matching_strategy",
                "common_benchmark_count",
                "spearman_rho_common_only",
                "kendall_tau_b_common_only",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
