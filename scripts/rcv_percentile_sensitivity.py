"""RCV percentile-range sensitivity analysis against a supplied BHI baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CONFIG  # noqa: E402
from src.preprocessor import MasterDataPreprocessor  # noqa: E402


OUTPUT_DIR = Path(os.environ.get(
    "BHI_EXPERIMENT_OUTPUT_DIR",
    PROJECT_ROOT / "data" / "output" / "rcv_percentile_sensitivity",
))
INPUT_PATH = Path(os.environ.get(
    "BHI_IN_MASTER",
    PROJECT_ROOT / "data" / "input" / "2026_0412_cleaned.csv",
))
ORIGINAL_OUTPUT_DIR = Path(os.environ.get(
    "BHI_BASELINE_DIR",
    PROJECT_ROOT / "data" / "output",
))
DEFAULT_CONDITION = "P10_P90"
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20260726
TOLERANCE = 1e-12
ANALYSIS_TIMESTAMP_UTC = datetime.now(timezone.utc).isoformat()

CONDITIONS = [
    ("P0_P100", 0, 100),
    ("P5_P95", 5, 95),
    ("P10_P90", 10, 90),
    ("P15_P85", 15, 85),
    ("P20_P80", 20, 80),
    ("P25_P75", 25, 75),
]


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrity_paths() -> list[Path]:
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "scripts" / "rcv_percentile_sensitivity.py",
        PROJECT_ROOT / "data" / "input" / "2026_0412_cleaned.csv",
        PROJECT_ROOT / "data" / "input" / "2026_0630_cleaned.csv",
    ]
    paths.extend(sorted((PROJECT_ROOT / "config").glob("*.py")))
    paths.extend(sorted((PROJECT_ROOT / "src").glob("*.py")))
    paths.extend(
        ORIGINAL_OUTPUT_DIR / name
        for name in [
            "final_bhi_rankings.csv",
            "capability_discrimination_final.csv",
            "anti_saturation_results.csv",
            "impact_results.csv",
            "benchmark_inclusion_report.csv",
        ]
    )
    for directory in [
        ORIGINAL_OUTPUT_DIR / "snapshot_20260412",
        ORIGINAL_OUTPUT_DIR / "snapshot_20260630",
        ORIGINAL_OUTPUT_DIR / "temporal_comparison",
    ]:
        paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return list(
        dict.fromkeys(path.resolve() for path in paths if path.is_file())
    )


def reconstruct_weights(components: pd.DataFrame) -> pd.Series:
    std = components.std()
    conflict = (1 - components.corr()).sum()
    information = std * conflict
    weights = information / information.sum()
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=TOLERANCE):
        raise ValueError("CRITIC weights do not sum to one")
    return weights


def fixed_top_set(frame: pd.DataFrame, k: int) -> set[str]:
    ordered = (
        frame.reset_index()
        .sort_values(["BHI", "Benchmark"], ascending=[False, True], kind="mergesort")
        .head(k)
    )
    return set(ordered["Benchmark"])


def expanded_top_set(frame: pd.DataFrame, k: int) -> tuple[set[str], bool]:
    ordered = frame.reset_index().sort_values(
        ["BHI", "Benchmark"], ascending=[False, True], kind="mergesort"
    )
    if len(ordered) <= k:
        return set(ordered["Benchmark"]), False
    cutoff = float(ordered.iloc[k - 1]["BHI"])
    expanded = set(ordered.loc[ordered["BHI"] >= cutoff, "Benchmark"])
    return expanded, int((ordered["BHI"] == cutoff).sum()) > 1


def calculate_condition(
    disc_scores: pd.DataFrame,
    original_sat: pd.DataFrame,
    original_imp: pd.DataFrame,
    label: str,
    lower: int,
    upper: int,
) -> dict:
    records = []
    for benchmark, row in disc_scores.iterrows():
        values = row.dropna().to_numpy()
        n = len(values)
        robust_cv = (
            np.percentile(values, upper) - np.percentile(values, lower)
        ) / 100.0
        differences = np.abs(values[:, None] - values)
        edr = (
            (np.sum(differences > CONFIG["edr_threshold"]) / 2)
            / (n * (n - 1) / 2)
            if n > 1
            else 0
        )
        records.append(
            {
                "Benchmark": benchmark,
                "N_Models": n,
                "Robust_CV": robust_cv,
                "EDR": edr,
            }
        )

    detail = pd.DataFrame(records).set_index("Benchmark")
    normalized = (
        detail[["Robust_CV", "EDR"]] - detail[["Robust_CV", "EDR"]].min()
    ) / (
        detail[["Robust_CV", "EDR"]].max()
        - detail[["Robust_CV", "EDR"]].min()
        + 1e-9
    )
    internal_weights = normalized.std() / normalized.std().sum()
    if not math.isclose(
        float(internal_weights.sum()), 1.0, abs_tol=TOLERANCE
    ):
        raise ValueError(f"Internal weights do not sum to one for {label}")
    detail["Normalized_Robust_CV"] = normalized["Robust_CV"]
    detail["Normalized_EDR"] = normalized["EDR"]
    detail["S_Disc"] = (
        normalized["Robust_CV"] * internal_weights["Robust_CV"]
        + normalized["EDR"] * internal_weights["EDR"]
    )

    final = (
        detail[["S_Disc"]]
        .merge(original_sat[["S_Sat"]], left_index=True, right_index=True)
        .merge(original_imp[["S_Imp"]], left_index=True, right_index=True)
    )
    critic_weights = reconstruct_weights(final)
    final["BHI"] = (
        final["S_Disc"] * critic_weights["S_Disc"]
        + final["S_Sat"] * critic_weights["S_Sat"]
        + final["S_Imp"] * critic_weights["S_Imp"]
    )
    final["Rank"] = final["BHI"].rank(method="average", ascending=False)
    n_ranked = len(final)
    final["Rank_Percentile"] = (
        1 - (final["Rank"] - 1) / (n_ranked - 1) if n_ranked > 1 else 1.0
    )
    sorted_final = final.sort_values(["BHI", "S_Imp"], ascending=False)
    return {
        "condition": label,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "detail": detail,
        "final": final,
        "sorted_final": sorted_final,
        "internal_weights": internal_weights,
        "critic_weights": critic_weights,
    }


def default_reproduction_check(
    default: dict,
    submitted_disc: pd.DataFrame,
    submitted_final: pd.DataFrame,
) -> pd.DataFrame:
    detail = default["detail"]
    final = default["final"]
    submitted_weights = reconstruct_weights(
        submitted_final[["S_Disc", "S_Sat", "S_Imp"]]
    )
    checks = [
        (
            "benchmark_set_matches",
            set(detail.index) == set(submitted_disc.index)
            and set(final.index) == set(submitted_final.index),
            0.0,
        ),
        (
            "max_abs_robust_cv_difference",
            float(
                (
                    detail.loc[submitted_disc.index, "Robust_CV"]
                    - submitted_disc["Robust_CV"]
                )
                .abs()
                .max()
            )
            <= TOLERANCE,
            float(
                (
                    detail.loc[submitted_disc.index, "Robust_CV"]
                    - submitted_disc["Robust_CV"]
                )
                .abs()
                .max()
            ),
        ),
        (
            "max_abs_edr_difference",
            float(
                (detail.loc[submitted_disc.index, "EDR"] - submitted_disc["EDR"])
                .abs()
                .max()
            )
            <= TOLERANCE,
            float(
                (detail.loc[submitted_disc.index, "EDR"] - submitted_disc["EDR"])
                .abs()
                .max()
            ),
        ),
        (
            "max_abs_s_disc_difference",
            float(
                (
                    detail.loc[submitted_disc.index, "S_Disc"]
                    - submitted_disc["S_Disc"]
                )
                .abs()
                .max()
            )
            <= TOLERANCE,
            float(
                (
                    detail.loc[submitted_disc.index, "S_Disc"]
                    - submitted_disc["S_Disc"]
                )
                .abs()
                .max()
            ),
        ),
        (
            "max_abs_s_disc_critic_weight_difference",
            abs(
                float(default["critic_weights"]["S_Disc"])
                - float(submitted_weights["S_Disc"])
            )
            <= TOLERANCE,
            abs(
                float(default["critic_weights"]["S_Disc"])
                - float(submitted_weights["S_Disc"])
            ),
        ),
        (
            "max_abs_s_sat_critic_weight_difference",
            abs(
                float(default["critic_weights"]["S_Sat"])
                - float(submitted_weights["S_Sat"])
            )
            <= TOLERANCE,
            abs(
                float(default["critic_weights"]["S_Sat"])
                - float(submitted_weights["S_Sat"])
            ),
        ),
        (
            "max_abs_s_imp_critic_weight_difference",
            abs(
                float(default["critic_weights"]["S_Imp"])
                - float(submitted_weights["S_Imp"])
            )
            <= TOLERANCE,
            abs(
                float(default["critic_weights"]["S_Imp"])
                - float(submitted_weights["S_Imp"])
            ),
        ),
        (
            "max_abs_bhi_difference",
            float(
                (
                    final.loc[submitted_final.index, "BHI"]
                    - submitted_final["BHI"]
                )
                .abs()
                .max()
            )
            <= TOLERANCE,
            float(
                (
                    final.loc[submitted_final.index, "BHI"]
                    - submitted_final["BHI"]
                )
                .abs()
                .max()
            ),
        ),
        (
            "ranking_order_matches",
            default["sorted_final"].index.tolist() == submitted_final.index.tolist(),
            0.0,
        ),
    ]
    result = pd.DataFrame(
        [
            {
                "check": name,
                "passed": bool(passed),
                "observed_max_abs_difference": value,
                "tolerance": TOLERANCE,
            }
            for name, passed, value in checks
        ]
    )
    if not result["passed"].all():
        failures = result.loc[~result["passed"]]
        raise ValueError(
            "P10_P90 did not reproduce the submission:\n"
            + failures.to_string(index=False)
        )
    return result


def ranking_metrics(condition: dict, default: dict) -> dict:
    current = condition["final"]
    reference = default["final"]
    ranks_current = current.loc[reference.index, "Rank"]
    ranks_default = reference["Rank"]
    rho = float(spearmanr(ranks_default, ranks_current).statistic)
    tau = float(kendalltau(ranks_default, ranks_current, variant="b").statistic)
    if not (-1 <= rho <= 1 and -1 <= tau <= 1):
        raise ValueError(f"Invalid rank correlation for {condition['condition']}")

    row = {
        "condition": condition["condition"],
        "reference_condition": DEFAULT_CONDITION,
        "lower_percentile": condition["lower"],
        "upper_percentile": condition["upper"],
        "percentile_width": condition["width"],
        "ranked_benchmark_count": len(current),
        "spearman_rho": rho,
        "kendall_tau_b": tau,
    }
    for k in [10, 20]:
        default_fixed = fixed_top_set(reference, k)
        current_fixed = fixed_top_set(current, k)
        shared = len(default_fixed & current_fixed)
        union = len(default_fixed | current_fixed)
        default_expanded, default_tie = expanded_top_set(reference, k)
        current_expanded, current_tie = expanded_top_set(current, k)
        expanded_shared = len(default_expanded & current_expanded)
        expanded_union = len(default_expanded | current_expanded)
        if shared > k:
            raise ValueError(f"Top-{k} overlap exceeds k")
        row.update(
            {
                f"top_{k}_shared_count": shared,
                f"top_{k}_overlap_rate": shared / k,
                f"top_{k}_jaccard": shared / union if union else 1.0,
                f"top_{k}_reference_cutoff_tie": default_tie,
                f"top_{k}_condition_cutoff_tie": current_tie,
                f"top_{k}_tie_aware_reference_size": len(default_expanded),
                f"top_{k}_tie_aware_condition_size": len(current_expanded),
                f"top_{k}_tie_aware_shared_count": expanded_shared,
                f"top_{k}_tie_aware_jaccard": (
                    expanded_shared / expanded_union if expanded_union else 1.0
                ),
            }
        )

    absolute_rank = (ranks_current - ranks_default).abs()
    percentile_change = (
        current.loc[reference.index, "Rank_Percentile"]
        - reference["Rank_Percentile"]
    )
    absolute_percentile = percentile_change.abs()
    row.update(
        {
            "rank_change_mean_abs": float(absolute_rank.mean()),
            "rank_change_median_abs": float(absolute_rank.median()),
            "rank_change_p90_abs": float(absolute_rank.quantile(0.9)),
            "rank_change_max_abs": float(absolute_rank.max()),
            "rank_percentile_change_mean_abs": float(absolute_percentile.mean()),
            "rank_percentile_change_median_abs": float(
                absolute_percentile.median()
            ),
            "rank_percentile_change_p90_abs": float(
                absolute_percentile.quantile(0.9)
            ),
            "rank_percentile_change_max_abs": float(absolute_percentile.max()),
        }
    )
    return row


def score_metrics(condition: dict, default: dict) -> list[dict]:
    rows = []
    pairs = {
        "Robust_CV": (
            default["detail"]["Robust_CV"],
            condition["detail"]["Robust_CV"],
        ),
        "S_Disc": (
            default["detail"]["S_Disc"],
            condition["detail"]["S_Disc"],
        ),
        "BHI": (
            default["final"]["BHI"],
            condition["final"]["BHI"],
        ),
    }
    for score, (reference, current) in pairs.items():
        current = current.reindex(reference.index)
        delta = current - reference
        absolute = delta.abs()
        pearson = float(reference.corr(current, method="pearson"))
        spearman = float(reference.corr(current, method="spearman"))
        if not (-1 <= pearson <= 1 and -1 <= spearman <= 1):
            raise ValueError(
                f"Invalid score correlation for {condition['condition']}/{score}"
            )
        row = {
            "condition": condition["condition"],
            "reference_condition": DEFAULT_CONDITION,
            "score": score,
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "mean_signed_change": float(delta.mean()),
            "mean_absolute_change": float(absolute.mean()),
            "median_absolute_change": float(absolute.median()),
            "root_mean_squared_change": float(np.sqrt(np.mean(delta**2))),
            "p90_absolute_change": float(absolute.quantile(0.9)),
            "max_absolute_change": float(absolute.max()),
        }
        for threshold in [0.01, 0.025, 0.05]:
            key = str(threshold).replace(".", "_")
            count = int((absolute < threshold).sum())
            row[f"absolute_change_below_{key}_count"] = count
            row[f"absolute_change_below_{key}_percentage"] = (
                count / len(absolute) * 100
            )
        rows.append(row)
    return rows


def condition_weight_row(condition: dict, default: dict) -> dict:
    internal = condition["internal_weights"]
    default_internal = default["internal_weights"]
    critic = condition["critic_weights"]
    default_critic = default["critic_weights"]
    internal_delta = internal - default_internal
    critic_delta = critic - default_critic
    return {
        "condition": condition["condition"],
        "lower_percentile": condition["lower"],
        "upper_percentile": condition["upper"],
        "percentile_width": condition["width"],
        "ranked_benchmark_count": len(condition["final"]),
        "internal_robust_cv_weight": float(internal["Robust_CV"]),
        "internal_edr_weight": float(internal["EDR"]),
        "internal_robust_cv_weight_change": float(internal_delta["Robust_CV"]),
        "internal_edr_weight_change": float(internal_delta["EDR"]),
        "maximum_absolute_internal_weight_drift": float(
            internal_delta.abs().max()
        ),
        "internal_weight_total_variation_distance": float(
            0.5 * internal_delta.abs().sum()
        ),
        "critic_s_disc_weight": float(critic["S_Disc"]),
        "critic_s_sat_weight": float(critic["S_Sat"]),
        "critic_s_imp_weight": float(critic["S_Imp"]),
        "critic_s_disc_weight_change": float(critic_delta["S_Disc"]),
        "critic_s_sat_weight_change": float(critic_delta["S_Sat"]),
        "critic_s_imp_weight_change": float(critic_delta["S_Imp"]),
        "maximum_absolute_critic_percentage_point_drift": float(
            critic_delta.abs().max() * 100
        ),
        "sum_absolute_critic_percentage_point_changes": float(
            critic_delta.abs().sum() * 100
        ),
        "critic_weight_total_variation_distance": float(
            0.5 * critic_delta.abs().sum()
        ),
    }


def benchmark_rows(condition: dict, default: dict) -> list[dict]:
    rows = []
    for benchmark in default["final"].index:
        detail = condition["detail"].loc[benchmark]
        final = condition["final"].loc[benchmark]
        default_final = default["final"].loc[benchmark]
        default_detail = default["detail"].loc[benchmark]
        rows.append(
            {
                "condition": condition["condition"],
                "benchmark": benchmark,
                "lower_percentile": condition["lower"],
                "upper_percentile": condition["upper"],
                "raw_robust_cv": float(detail["Robust_CV"]),
                "normalized_robust_cv": float(detail["Normalized_Robust_CV"]),
                "edr": float(detail["EDR"]),
                "normalized_edr": float(detail["Normalized_EDR"]),
                "s_disc": float(detail["S_Disc"]),
                "original_submitted_s_sat": float(final["S_Sat"]),
                "original_submitted_s_imp": float(final["S_Imp"]),
                "bhi": float(final["BHI"]),
                "rank": float(final["Rank"]),
                "rank_percentile": float(final["Rank_Percentile"]),
                "default_p10_p90_rank": float(default_final["Rank"]),
                "rank_change": float(final["Rank"] - default_final["Rank"]),
                "absolute_rank_change": float(
                    abs(final["Rank"] - default_final["Rank"])
                ),
                "default_p10_p90_rank_percentile": float(
                    default_final["Rank_Percentile"]
                ),
                "rank_percentile_change": float(
                    final["Rank_Percentile"]
                    - default_final["Rank_Percentile"]
                ),
                "absolute_rank_percentile_change": float(
                    abs(
                        final["Rank_Percentile"]
                        - default_final["Rank_Percentile"]
                    )
                ),
                "bhi_change_from_p10_p90": float(
                    final["BHI"] - default_final["BHI"]
                ),
                "s_disc_change_from_p10_p90": float(
                    detail["S_Disc"] - default_detail["S_Disc"]
                ),
                "robust_cv_change_from_p10_p90": float(
                    detail["Robust_CV"] - default_detail["Robust_CV"]
                ),
            }
        )
    return rows


def top_movers(condition: dict, default: dict) -> list[dict]:
    if condition["condition"] == DEFAULT_CONDITION:
        return []
    changes = pd.DataFrame(benchmark_rows(condition, default)).set_index(
        "benchmark"
    )
    categories = [
        ("rank_percentile_upward", "rank_percentile_change", False, False),
        ("rank_percentile_downward", "rank_percentile_change", True, False),
        ("absolute_bhi_change", "bhi_change_from_p10_p90", False, True),
        ("absolute_s_disc_change", "s_disc_change_from_p10_p90", False, True),
        (
            "absolute_robust_cv_change",
            "robust_cv_change_from_p10_p90",
            False,
            True,
        ),
    ]
    rows = []
    for category, column, ascending, use_absolute in categories:
        selection = changes[column].abs() if use_absolute else changes[column]
        selected = selection.sort_values(
            ascending=ascending, kind="mergesort"
        ).index[:10]
        for mover_rank, benchmark in enumerate(selected, start=1):
            item = changes.loc[benchmark]
            rows.append(
                {
                    "condition": condition["condition"],
                    "mover_category": category,
                    "mover_rank": mover_rank,
                    "benchmark": benchmark,
                    "metric_value": float(item[column]),
                    "absolute_metric_value": float(abs(item[column])),
                    "rank_percentile_change": float(
                        item["rank_percentile_change"]
                    ),
                    "bhi_change": float(item["bhi_change_from_p10_p90"]),
                    "s_disc_change": float(
                        item["s_disc_change_from_p10_p90"]
                    ),
                    "robust_cv_change": float(
                        item["robust_cv_change_from_p10_p90"]
                    ),
                }
            )
    return rows


def bootstrap_rows(
    condition: dict,
    default: dict,
    rng: np.random.Generator,
) -> list[dict]:
    if condition["condition"] == DEFAULT_CONDITION:
        return []
    reference = default["final"]
    current = condition["final"].reindex(reference.index)
    reference_rank = reference["Rank"].to_numpy()
    current_rank = current["Rank"].to_numpy()
    percentile_absolute = (
        current["Rank_Percentile"] - reference["Rank_Percentile"]
    ).abs().to_numpy()
    bhi_absolute = (current["BHI"] - reference["BHI"]).abs().to_numpy()
    definitions = {
        "spearman_rho": (
            float(spearmanr(reference_rank, current_rank).statistic),
            lambda index: float(
                spearmanr(reference_rank[index], current_rank[index]).statistic
            ),
        ),
        "kendall_tau_b": (
            float(
                kendalltau(reference_rank, current_rank, variant="b").statistic
            ),
            lambda index: float(
                kendalltau(
                    reference_rank[index], current_rank[index], variant="b"
                ).statistic
            ),
        ),
        "median_absolute_rank_percentile_change": (
            float(np.median(percentile_absolute)),
            lambda index: float(np.median(percentile_absolute[index])),
        ),
        "median_absolute_bhi_change": (
            float(np.median(bhi_absolute)),
            lambda index: float(np.median(bhi_absolute[index])),
        ),
    }
    samples = {metric: [] for metric in definitions}
    skipped = {metric: 0 for metric in definitions}
    n = len(reference)
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
                "condition": condition["condition"],
                "reference_condition": DEFAULT_CONDITION,
                "metric": metric,
                "estimate": estimate,
                "ci_95_lower": float(np.quantile(values, 0.025)),
                "ci_95_upper": float(np.quantile(values, 0.975)),
                "requested_resamples": BOOTSTRAP_ITERATIONS,
                "valid_resamples": len(values),
                "skipped_resamples": skipped[metric],
                "seed": BOOTSTRAP_SEED,
            }
        )
    return rows


def trend_summary(
    conditions: dict[str, dict], benchmark_frame: pd.DataFrame
) -> dict:
    order = [label for label, _, _ in CONDITIONS]
    monotonic_rcv = []
    nonmonotonic_s_disc = []
    nonmonotonic_bhi = []
    sensitivity = []
    for benchmark, group in benchmark_frame.groupby("benchmark"):
        ordered = group.set_index("condition").loc[order]
        robust = ordered["raw_robust_cv"].to_numpy()
        s_disc = ordered["s_disc"].to_numpy()
        bhi = ordered["bhi"].to_numpy()

        def monotonic(values):
            differences = np.diff(values)
            return bool(
                np.all(differences >= -TOLERANCE)
                or np.all(differences <= TOLERANCE)
            )

        if monotonic(robust):
            monotonic_rcv.append(benchmark)
        if not monotonic(s_disc):
            nonmonotonic_s_disc.append(benchmark)
        if not monotonic(bhi):
            nonmonotonic_bhi.append(benchmark)
        sensitivity.append(
            {
                "benchmark": benchmark,
                "maximum_absolute_rank_percentile_change": float(
                    ordered["absolute_rank_percentile_change"].max()
                ),
                "rank_range": float(ordered["rank"].max() - ordered["rank"].min()),
            }
        )
    sensitivity.sort(
        key=lambda row: (
            -row["maximum_absolute_rank_percentile_change"],
            row["benchmark"],
        )
    )
    return {
        "benchmark_count": int(benchmark_frame["benchmark"].nunique()),
        "monotonic_robust_cv_count": len(monotonic_rcv),
        "monotonic_robust_cv_benchmarks": monotonic_rcv,
        "nonmonotonic_s_disc_count": len(nonmonotonic_s_disc),
        "nonmonotonic_s_disc_benchmarks": nonmonotonic_s_disc,
        "nonmonotonic_bhi_count": len(nonmonotonic_bhi),
        "nonmonotonic_bhi_benchmarks": nonmonotonic_bhi,
        "most_rank_sensitive_benchmarks": sensitivity[:20],
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include="number").columns:
        selected[column] = selected[column].map(lambda value: f"{value:.6f}")
    lines = [
        "| " + " | ".join(selected.columns) + " |",
        "| " + " | ".join(["---"] * len(selected.columns)) + " |",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[column]) for column in selected.columns)
            + " |"
        )
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    weights: pd.DataFrame,
    ranking: pd.DataFrame,
    scores: pd.DataFrame,
    reproduction: pd.DataFrame,
    movers: pd.DataFrame,
    bootstrap: pd.DataFrame,
    trends: dict,
) -> None:
    alternatives = ranking[ranking["condition"] != DEFAULT_CONDITION]
    largest = alternatives.loc[
        alternatives["rank_percentile_change_mean_abs"].idxmax()
    ]
    top = movers[
        (movers["condition"] == largest["condition"])
        & movers["mover_category"].isin(
            ["rank_percentile_upward", "rank_percentile_downward"]
        )
        & (movers["mover_rank"] <= 5)
    ]
    lines = [
        "# RCV Percentile-Range Sensitivity Analysis",
        "",
        "## Experimental purpose",
        "",
        "This experiment evaluates whether BHI rankings and weights depend materially "
        "on the percentile endpoints used for Robust_CV. P10-P90 is the "
        "default, not an assumed unique optimum. Only the RCV percentile range changes.",
        "",
        "The supplied baseline discrimination matrix and its S_Sat and S_Imp outputs "
        "are used; no other snapshot is an analytical input.",
        "",
        "## RCV definitions tested",
        "",
        "| Condition | Definition |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| {label} | (P{upper} - P{lower}) / 100 |"
        for label, lower, upper in CONDITIONS
    )
    lines.extend(
        [
            "",
            "## Default reproduction",
            "",
            markdown_table(
                reproduction,
                ["check", "passed", "observed_max_abs_difference", "tolerance"],
            ),
            "",
            "P10_P90 reproduces the supplied baseline benchmark set, RCV, EDR, S_Disc, "
            "CRITIC weights, BHI values, and ranking order within floating-point tolerance.",
            "",
            "## Internal and final weights",
            "",
            markdown_table(
                weights,
                [
                    "condition",
                    "internal_robust_cv_weight",
                    "internal_edr_weight",
                    "critic_s_disc_weight",
                    "critic_s_sat_weight",
                    "critic_s_imp_weight",
                    "maximum_absolute_critic_percentage_point_drift",
                ],
            ),
            "",
            "## Ranking stability relative to P10_P90",
            "",
            markdown_table(
                ranking,
                [
                    "condition",
                    "spearman_rho",
                    "kendall_tau_b",
                    "top_10_shared_count",
                    "top_20_shared_count",
                    "rank_change_median_abs",
                    "rank_percentile_change_median_abs",
                    "rank_percentile_change_max_abs",
                ],
            ),
            "",
            "## Score stability",
            "",
            markdown_table(
                scores[scores["condition"] != DEFAULT_CONDITION],
                [
                    "condition",
                    "score",
                    "pearson_correlation",
                    "spearman_correlation",
                    "mean_absolute_change",
                    "median_absolute_change",
                    "max_absolute_change",
                ],
            ),
            "",
            f"## Top movers for the largest-deviation condition ({largest['condition']})",
            "",
            markdown_table(
                top,
                [
                    "mover_category",
                    "mover_rank",
                    "benchmark",
                    "rank_percentile_change",
                    "bhi_change",
                    "s_disc_change",
                    "robust_cv_change",
                ],
            ),
            "",
            "## Bootstrap confidence intervals",
            "",
            markdown_table(
                bootstrap,
                [
                    "condition",
                    "metric",
                    "estimate",
                    "ci_95_lower",
                    "ci_95_upper",
                    "valid_resamples",
                    "skipped_resamples",
                ],
            ),
            "",
            "## Monotonicity and trend checks",
            "",
            f"- Benchmarks with monotonic Robust_CV across progressive trimming: "
            f"{trends['monotonic_robust_cv_count']}/{trends['benchmark_count']}.",
            f"- Benchmarks with non-monotonic S_Disc: "
            f"{trends['nonmonotonic_s_disc_count']}.",
            f"- Benchmarks with non-monotonic BHI: "
            f"{trends['nonmonotonic_bhi_count']}.",
            "",
            "## Warnings and interpretation constraints",
            "",
            "- P0_P100 may reflect extreme observations more strongly, but its behavior is "
            "reported rather than assumed.",
            "- Scores and CRITIC weights are recalibrated for every condition; raw BHI is "
            "not assumed invariant.",
            "- Broad ranking, Top-k, rank-percentile, score, and weight evidence should be "
            "considered together.",
            "- This experiment does not establish that P10_P90 is theoretically unique or optimal.",
            "- Results from snapshots other than the supplied baseline are outside scope.",
            "",
            "## Concise evidence-based summary",
            "",
            f"The largest mean absolute rank-percentile deviation from P10_P90 occurs "
            f"for {largest['condition']} at "
            f"{largest['rank_percentile_change_mean_abs']:.6f}; its Spearman rho is "
            f"{largest['spearman_rho']:.6f} and Kendall tau-b is "
            f"{largest['kendall_tau_b']:.6f}. The tables above report all conditions, "
            "including the full P0-P100 range, without selecting only favorable results.",
        ]
    )
    (output_dir / "rcv_sensitivity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_analysis_record(
    output_dir: Path,
    weights: pd.DataFrame,
    ranking: pd.DataFrame,
    reproduction: pd.DataFrame,
    source_hashes: dict[str, str],
) -> None:
    primary = ranking[ranking["condition"] != DEFAULT_CONDITION]
    lines = [
        "# Permanent Record: RCV Percentile-Range Sensitivity Analysis",
        "",
        f"- Analysis date: 2026-07-26",
        f"- Analysis timestamp UTC: {ANALYSIS_TIMESTAMP_UTC}",
        "- Script: `scripts/rcv_percentile_sensitivity.py`",
        "- Command: `python scripts/rcv_percentile_sensitivity.py`",
        "- Default reference: `P10_P90`",
        "- Bootstrap: 2,000 resamples, seed 20260726",
        "",
        "## Scientific scope",
        "",
        f"- Input: `{display_path(INPUT_PATH)}`",
        f"- Baseline directory: `{display_path(ORIGINAL_OUTPUT_DIR)}`",
        "- No other snapshot was used analytically.",
        "",
        "## Conditions",
        "",
    ]
    lines.extend(
        f"- `{label}`: Robust_CV = (P{upper} - P{lower}) / 100"
        for label, lower, upper in CONDITIONS
    )
    lines.extend(
        [
            "",
            "Only the Robust_CV percentile endpoints vary. EDR, threshold, eligibility, "
            "min-max normalization, internal weighting, baseline S_Sat/S_Imp, "
            "CRITIC, BHI, benchmark intersection, and tie handling remain fixed.",
            "",
            "## P10_P90 reproduction",
            "",
            markdown_table(
                reproduction,
                ["check", "passed", "observed_max_abs_difference"],
            ),
            "",
            "## Condition weights",
            "",
            markdown_table(
                weights,
                [
                    "condition",
                    "internal_robust_cv_weight",
                    "internal_edr_weight",
                    "critic_s_disc_weight",
                    "critic_s_sat_weight",
                    "critic_s_imp_weight",
                    "maximum_absolute_critic_percentage_point_drift",
                    "critic_weight_total_variation_distance",
                ],
            ),
            "",
            "## Primary ranking metrics",
            "",
            markdown_table(
                primary,
                [
                    "condition",
                    "spearman_rho",
                    "kendall_tau_b",
                    "top_10_shared_count",
                    "top_20_shared_count",
                    "rank_change_median_abs",
                    "rank_percentile_change_median_abs",
                    "rank_percentile_change_max_abs",
                ],
            ),
            "",
            "## Direct source SHA-256 values",
            "",
        ]
    )
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in source_hashes.items())
    lines.extend(
        [
            "",
            "## Warnings and limitations",
            "",
            "- P10_P90 is the default, not a proven unique optimum.",
            "- P0_P100 is retained even if it is less stable.",
            "- Recalculated normalization and CRITIC weights mean raw BHI need not be invariant.",
            "- This is a single-baseline sensitivity experiment, not a temporal analysis.",
            "- Other snapshots are excluded from scientific calculations.",
        ]
    )
    (output_dir / "ANALYSIS_RECORD.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_artifact_manifest(output_dir: Path, generated_files: list[Path]) -> None:
    source_roles = [
        (
            PROJECT_ROOT / "scripts" / "rcv_percentile_sensitivity.py",
            "analysis_script",
            "generated",
        ),
        (INPUT_PATH, "original_submission_input", "source"),
        (
            ORIGINAL_OUTPUT_DIR / "final_bhi_rankings.csv",
            "original_final_ranking",
            "source",
        ),
        (
            ORIGINAL_OUTPUT_DIR / "capability_discrimination_final.csv",
            "original_discrimination_output",
            "source",
        ),
        (
            ORIGINAL_OUTPUT_DIR / "anti_saturation_results.csv",
            "original_anti_saturation_output",
            "source",
        ),
        (
            ORIGINAL_OUTPUT_DIR / "impact_results.csv",
            "original_impact_output",
            "source",
        ),
        (
            ORIGINAL_OUTPUT_DIR / "benchmark_inclusion_report.csv",
            "original_inclusion_report",
            "source",
        ),
        (PROJECT_ROOT / "main.py", "pipeline_entry_source", "source"),
        (
            PROJECT_ROOT / "config" / "settings.py",
            "configuration_source",
            "source",
        ),
        (
            PROJECT_ROOT / "src" / "preprocessor.py",
            "preprocessing_source",
            "source",
        ),
        (
            PROJECT_ROOT / "README.md",
            "project_method_guide",
            "source",
        ),
    ]
    entries = []
    for path in generated_files:
        source_roles.append((path, "rcv_sensitivity_artifact", "generated"))
    for path, role, kind in source_roles:
        stat = path.stat()
        entries.append(
            {
                "relative_path": display_path(path),
                "file_size_bytes": stat.st_size,
                "sha256": sha256_file(path),
                "last_modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
                "artifact_role": role,
                "generated_or_source": kind,
                "analysis_timestamp_utc": ANALYSIS_TIMESTAMP_UTC,
            }
        )
    entries.sort(key=lambda row: row["relative_path"])
    manifest_path = output_dir / "artifact_manifest_sha256.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "file_size_bytes",
                "sha256",
                "last_modified_utc",
                "artifact_role",
                "generated_or_source",
                "analysis_timestamp_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(entries)
    manifest_hash = sha256_file(manifest_path)
    (output_dir / "MANIFEST_SHA256.txt").write_text(
        f"artifact_manifest_sha256.csv  {manifest_hash}\n", encoding="utf-8"
    )


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {OUTPUT_DIR}"
        )
    before_hashes = {
        display_path(path): sha256_file(path)
        for path in integrity_paths()
    }

    master = MasterDataPreprocessor(CONFIG).load_and_process_master(INPUT_PATH)
    valid_counts = master.disc_scores.notna().sum(axis=1)
    disc_scores = master.disc_scores.loc[
        valid_counts >= CONFIG["min_model_count"]
    ]
    submitted_disc = pd.read_csv(
        ORIGINAL_OUTPUT_DIR / "capability_discrimination_final.csv"
    ).set_index("Benchmark")
    original_sat = pd.read_csv(
        ORIGINAL_OUTPUT_DIR / "anti_saturation_results.csv"
    ).set_index("Benchmark")
    original_imp = pd.read_csv(
        ORIGINAL_OUTPUT_DIR / "impact_results.csv"
    ).set_index("Benchmark")
    submitted_final = pd.read_csv(
        ORIGINAL_OUTPUT_DIR / "final_bhi_rankings.csv"
    ).set_index("Benchmark")

    conditions = {
        label: calculate_condition(
            disc_scores, original_sat, original_imp, label, lower, upper
        )
        for label, lower, upper in CONDITIONS
    }
    default = conditions[DEFAULT_CONDITION]
    reproduction = default_reproduction_check(
        default, submitted_disc, submitted_final
    )

    weights = pd.DataFrame(
        [
            condition_weight_row(condition, default)
            for condition in conditions.values()
        ]
    )
    ranking = pd.DataFrame(
        [ranking_metrics(condition, default) for condition in conditions.values()]
    )
    scores = pd.DataFrame(
        [
            row
            for condition in conditions.values()
            for row in score_metrics(condition, default)
        ]
    )
    benchmark_frame = pd.DataFrame(
        [
            row
            for condition in conditions.values()
            for row in benchmark_rows(condition, default)
        ]
    )
    movers = pd.DataFrame(
        [
            row
            for condition in conditions.values()
            for row in top_movers(condition, default)
        ]
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = pd.DataFrame(
        [
            row
            for condition in conditions.values()
            for row in bootstrap_rows(condition, default, rng)
        ]
    )
    trends = trend_summary(conditions, benchmark_frame)

    expected_rows = len(default["final"]) * len(CONDITIONS)
    if len(benchmark_frame) != expected_rows:
        raise ValueError("Unexpected benchmark-level row count")
    counts = benchmark_frame.groupby(["condition", "benchmark"]).size()
    if not (counts == 1).all():
        raise ValueError("A benchmark does not appear exactly once per condition")
    if set(weights["condition"]) != {label for label, _, _ in CONDITIONS}:
        raise ValueError("Not all six conditions are present")

    OUTPUT_DIR.mkdir(parents=True)
    output_frames = {
        "rcv_condition_weights.csv": weights,
        "rcv_ranking_stability_metrics.csv": ranking,
        "rcv_score_stability_metrics.csv": scores,
        "rcv_benchmark_level_results.csv": benchmark_frame,
        "rcv_top_movers.csv": movers,
        "rcv_bootstrap_confidence_intervals.csv": bootstrap,
        "default_reproduction_check.csv": reproduction,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    direct_source_hashes = {
        display_path(path): sha256_file(path)
        for path in [
            INPUT_PATH,
            ORIGINAL_OUTPUT_DIR / "final_bhi_rankings.csv",
            ORIGINAL_OUTPUT_DIR / "capability_discrimination_final.csv",
            ORIGINAL_OUTPUT_DIR / "anti_saturation_results.csv",
            ORIGINAL_OUTPUT_DIR / "impact_results.csv",
            ORIGINAL_OUTPUT_DIR / "benchmark_inclusion_report.csv",
            PROJECT_ROOT / "main.py",
            PROJECT_ROOT / "config" / "settings.py",
            PROJECT_ROOT / "src" / "preprocessor.py",
            PROJECT_ROOT / "README.md",
        ]
    }
    summary = {
        "analysis_date": "2026-07-26",
        "analysis_timestamp_utc": ANALYSIS_TIMESTAMP_UTC,
        "objective": "RCV percentile-range sensitivity on the supplied baseline",
        "input_path": display_path(INPUT_PATH),
        "analytical_output_sources": [
            "data/output/capability_discrimination_final.csv",
            "data/output/anti_saturation_results.csv",
            "data/output/impact_results.csv",
            "data/output/final_bhi_rankings.csv",
            "data/output/benchmark_inclusion_report.csv",
        ],
        "conditions": [
            {"condition": label, "lower": lower, "upper": upper}
            for label, lower, upper in CONDITIONS
        ],
        "default_condition": DEFAULT_CONDITION,
        "default_reproduction": reproduction.to_dict("records"),
        "condition_weights": weights.to_dict("records"),
        "ranking_stability": ranking.to_dict("records"),
        "score_stability": scores.to_dict("records"),
        "bootstrap_confidence_intervals": bootstrap.to_dict("records"),
        "trend_analysis": trends,
        "direct_source_hashes": direct_source_hashes,
        "warnings": [
            "P10_P90 is the default, not an assumed optimum.",
            "P0_P100 is reported even if less stable.",
            "BHI and weights are recalibrated for every percentile condition.",
            "Other snapshots are outside the analytical scope.",
        ],
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (OUTPUT_DIR / "rcv_sensitivity_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_report(
        OUTPUT_DIR,
        weights,
        ranking,
        scores,
        reproduction,
        movers,
        bootstrap,
        trends,
    )
    write_analysis_record(
        OUTPUT_DIR,
        weights,
        ranking,
        reproduction,
        direct_source_hashes,
    )

    generated_for_manifest = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
        and path.name
        not in {"artifact_manifest_sha256.csv", "MANIFEST_SHA256.txt"}
    )
    write_artifact_manifest(OUTPUT_DIR, generated_for_manifest)

    after_hashes = {
        display_path(path): sha256_file(path)
        for path in integrity_paths()
    }
    if before_hashes != after_hashes:
        changed = [
            path
            for path in before_hashes
            if before_hashes[path] != after_hashes.get(path)
        ]
        raise RuntimeError(f"Existing-file integrity failure: {changed}")

    for path in OUTPUT_DIR.iterdir():
        if path.suffix == ".csv":
            pd.read_csv(path)
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))

    expected_names = {
        "rcv_sensitivity_report.md",
        "rcv_sensitivity_summary.json",
        "rcv_condition_weights.csv",
        "rcv_ranking_stability_metrics.csv",
        "rcv_score_stability_metrics.csv",
        "rcv_benchmark_level_results.csv",
        "rcv_top_movers.csv",
        "rcv_bootstrap_confidence_intervals.csv",
        "default_reproduction_check.csv",
        "ANALYSIS_RECORD.md",
        "artifact_manifest_sha256.csv",
        "MANIFEST_SHA256.txt",
    }
    actual_names = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError(
            f"Unexpected output directory contents: {sorted(actual_names)}"
        )

    print(f"Wrote RCV sensitivity outputs to {OUTPUT_DIR}")
    print("Existing source/input/output hashes unchanged:", before_hashes == after_hashes)
    print(
        ranking[
            [
                "condition",
                "spearman_rho",
                "kendall_tau_b",
                "top_10_shared_count",
                "top_20_shared_count",
                "rank_percentile_change_median_abs",
                "rank_percentile_change_max_abs",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
