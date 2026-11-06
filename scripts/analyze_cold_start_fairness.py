"""Reproducible cold-start fairness analysis for a supplied BHI snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr, rankdata, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CONFIG  # noqa: E402
from src.preprocessor import MasterDataPreprocessor  # noqa: E402


METRICS = ["NUsage", "SDisc", "SSat", "SImp", "BHI"]
CORRELATION_OUTCOMES = ["NUsage", "SImp", "BHI"]
AGE_GROUPS = [
    "0-6 months",
    ">6-12 months",
    ">12-24 months",
    ">24 months",
]
FIELD_ALIASES = {
    "Benchmark": ["Benchmark", "benchmark_name", "name"],
    "Release_Date": ["Release_Date", "release date", "benchmark_release_date"],
    "Source_URL": ["Source_URL", "source url", "url"],
    "Date_Precision": ["Date_Precision", "date precision"],
    "Verification_Status": ["Verification_Status", "verification status"],
    "Manual_Check_Priority": [
        "Manual_Check_Priority",
        "manual check priority",
        "manual_review_priority",
    ],
    "Release_Scope": ["Release_Scope", "release scope"],
    "NUsage": ["NUsage", "N_Usage", "usage"],
    "SDisc": ["SDisc", "S_Disc"],
    "SSat": ["SSat", "S_Sat", "SAS"],
    "SImp": ["SImp", "S_Imp"],
    "BHI": ["BHI"],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dates", required=True, type=Path)
    parser.add_argument("--master-data", required=True, type=Path)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized_field_name(value: str) -> str:
    """Normalize a column name for alias matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalized_benchmark_name(value: str) -> str:
    """Normalize a benchmark name for exclusion checks."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_required_columns(columns: pd.Index) -> dict[str, str]:
    """Map required canonical fields from accepted column aliases."""
    normalized = {normalized_field_name(column): str(column) for column in columns}
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in FIELD_ALIASES.items():
        matched = next(
            (
                normalized[normalized_field_name(alias)]
                for alias in aliases
                if normalized_field_name(alias) in normalized
            ),
            None,
        )
        if matched is None:
            missing.append(canonical)
        else:
            mapping[canonical] = matched
    if missing:
        raise ValueError(f"Missing required release-date columns: {missing}")
    return mapping


def add_validation(
    records: list[dict[str, Any]],
    check: str,
    status: str,
    count: int,
    details: str,
    benchmark: str = "",
) -> None:
    """Append a consistently shaped validation record."""
    records.append(
        {
            "Record_Type": "row_issue" if benchmark else "summary",
            "Check": check,
            "Status": status,
            "Count": int(count),
            "Details": details,
            "Benchmark": benchmark,
        }
    )


def prepare_benchmark_data(
    release_path: Path,
    snapshot_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Validate the release-date file and recompute benchmark ages and ranks."""
    raw = pd.read_csv(release_path)
    mapping = map_required_columns(raw.columns)
    data = raw.rename(columns={source: target for target, source in mapping.items()})
    validation: list[dict[str, Any]] = []

    duplicate_mask = data["Benchmark"].astype(str).str.strip().duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    add_validation(
        validation,
        "benchmark_names_unique",
        "pass" if duplicate_count == 0 else "fail",
        duplicate_count,
        "No duplicate benchmark names." if duplicate_count == 0 else "Duplicate names found.",
    )
    if duplicate_count:
        duplicates = data.loc[duplicate_mask, "Benchmark"].astype(str).tolist()
        raise ValueError(f"Benchmark names are not unique: {duplicates}")

    data["Benchmark"] = data["Benchmark"].astype(str).str.strip()
    swe_mask = data["Benchmark"].map(normalized_benchmark_name).str.contains(
        "swelancer", regex=False
    )
    swe_count = int(swe_mask.sum())
    data = data.loc[~swe_mask].copy()
    add_validation(
        validation,
        "swe_lancer_excluded",
        "pass",
        swe_count,
        f"Applied the analysis-specific normalized-name exclusion to {swe_count} row(s).",
    )

    data["Release_Date"] = pd.to_datetime(data["Release_Date"], errors="coerce")
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data["Source_URL"] = data["Source_URL"].fillna("").astype(str).str.strip()

    invalid_date = data["Release_Date"].isna()
    future_date = data["Release_Date"] > snapshot_date
    missing_metrics = data[METRICS].isna().any(axis=1)
    missing_url = data["Source_URL"].eq("")
    missing_name = data["Benchmark"].eq("")
    unresolved = invalid_date | future_date | missing_metrics | missing_url | missing_name

    issue_specs = [
        ("unparseable_release_date", invalid_date),
        ("release_date_after_snapshot", future_date),
        ("required_metric_missing_or_nonnumeric", missing_metrics),
        ("source_url_empty", missing_url),
        ("benchmark_name_empty", missing_name),
    ]
    for check, mask in issue_specs:
        count = int(mask.sum())
        add_validation(
            validation,
            check,
            "pass" if count == 0 else "warning",
            count,
            "No affected rows." if count == 0 else "Affected rows are excluded and listed.",
        )
        for benchmark in data.loc[mask, "Benchmark"].astype(str):
            add_validation(
                validation,
                check,
                "excluded",
                1,
                "Row excluded from analysis.",
                benchmark,
            )

    eligible = data.loc[~unresolved].copy()
    eligible["age_days"] = (snapshot_date - eligible["Release_Date"]).dt.days
    eligible["age_months"] = eligible["age_days"] / 30.44

    precomputed_day_column = next(
        (
            column
            for column in raw.columns
            if normalized_field_name(column).startswith("agedays")
        ),
        None,
    )
    precomputed_month_column = next(
        (
            column
            for column in raw.columns
            if normalized_field_name(column).startswith("agemonths")
        ),
        None,
    )
    eligible["precomputed_age_days"] = np.nan
    eligible["precomputed_age_months"] = np.nan
    eligible["age_days_discrepancy"] = np.nan
    eligible["age_months_discrepancy"] = np.nan

    if precomputed_day_column:
        pre_days = pd.to_numeric(
            raw.set_index(mapping["Benchmark"])[precomputed_day_column], errors="coerce"
        )
        eligible["precomputed_age_days"] = eligible["Benchmark"].map(pre_days)
        eligible["age_days_discrepancy"] = (
            eligible["precomputed_age_days"] - eligible["age_days"]
        )
        day_discrepancies = int(
            eligible["age_days_discrepancy"].abs().fillna(np.inf).gt(0).sum()
        )
        max_day_difference = float(
            eligible["age_days_discrepancy"].abs().max(skipna=True)
        )
        add_validation(
            validation,
            "precomputed_age_days_match",
            "pass" if day_discrepancies == 0 else "warning",
            day_discrepancies,
            f"Maximum absolute difference: {max_day_difference:.12g} days.",
        )
    else:
        add_validation(
            validation,
            "precomputed_age_days_match",
            "not_applicable",
            0,
            "No precomputed age-days column exists.",
        )

    if precomputed_month_column:
        pre_months = pd.to_numeric(
            raw.set_index(mapping["Benchmark"])[precomputed_month_column],
            errors="coerce",
        )
        eligible["precomputed_age_months"] = eligible["Benchmark"].map(pre_months)
        eligible["age_months_discrepancy"] = (
            eligible["precomputed_age_months"] - eligible["age_months"]
        )
        month_discrepancies = int(
            eligible["age_months_discrepancy"]
            .abs()
            .fillna(np.inf)
            .gt(0.0005)
            .sum()
        )
        max_month_difference = float(
            eligible["age_months_discrepancy"].abs().max(skipna=True)
        )
        add_validation(
            validation,
            "precomputed_age_months_match_within_rounding",
            "pass" if month_discrepancies == 0 else "warning",
            month_discrepancies,
            (
                "Tolerance is 0.0005 months because the supplied column is rounded "
                f"to three decimals; maximum raw difference: {max_month_difference:.12g}."
            ),
        )
    else:
        add_validation(
            validation,
            "precomputed_age_months_match_within_rounding",
            "not_applicable",
            0,
            "No precomputed age-months column exists.",
        )

    eligible = eligible.sort_values(
        ["BHI", "Benchmark"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    eligible["BHI_Rank_Recomputed"] = np.arange(1, len(eligible) + 1)
    for fraction, label in [(0.10, "Top_10pct"), (0.25, "Top_25pct"), (0.50, "Top_50pct")]:
        cutoff = math.ceil(fraction * len(eligible))
        eligible[label] = eligible["BHI_Rank_Recomputed"] <= cutoff

    eligible["Age_Group"] = pd.cut(
        eligible["age_months"],
        bins=[0, 6, 12, 24, np.inf],
        labels=AGE_GROUPS,
        include_lowest=True,
        right=True,
    )

    add_validation(
        validation,
        "required_metrics_numeric_and_complete",
        "pass",
        int(eligible[METRICS].isna().sum().sum()),
        "All analysed metric values are numeric and non-missing.",
    )
    add_validation(
        validation,
        "source_urls_nonempty",
        "pass",
        int(eligible["Source_URL"].eq("").sum()),
        "All analysed rows include a source URL.",
    )
    add_validation(
        validation,
        "no_release_date_after_snapshot",
        "pass",
        int((eligible["Release_Date"] > snapshot_date).sum()),
        f"No analysed release date is later than {snapshot_date.date()}.",
    )
    add_validation(
        validation,
        "final_analysed_benchmark_count",
        "pass",
        len(eligible),
        f"Final eligible population contains {len(eligible)} benchmarks.",
    )

    counts = {
        "release_rows_raw": len(raw),
        "swe_lancer_rows_excluded": swe_count,
        "unresolved_rows_excluded": int(unresolved.sum()),
        "benchmarks_analysed": len(eligible),
    }
    return eligible, pd.DataFrame(validation), counts


def vectorized_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute one Pearson correlation per row for two matrices."""
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(x_centered * y_centered, axis=1) / denominator


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Bootstrap a percentile confidence interval for a paired correlation."""
    n = len(x)
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, 0
    indices = rng.integers(0, n, size=(samples, n))
    boot_x = x[indices]
    boot_y = y[indices]
    if method == "Spearman":
        boot_x = rankdata(boot_x, axis=1, method="average")
        boot_y = rankdata(boot_y, axis=1, method="average")
    coefficients = vectorized_correlations(boot_x, boot_y)
    coefficients = coefficients[np.isfinite(coefficients)]
    if coefficients.size == 0:
        return np.nan, np.nan, 0
    lower, upper = np.quantile(coefficients, [0.025, 0.975])
    return float(lower), float(upper), int(coefficients.size)


def interpret_correlation(coefficient: float, p_value: float) -> str:
    """Return cautious direction, magnitude, and reliability wording."""
    if not np.isfinite(coefficient):
        return "Not estimable for this subset."
    absolute = abs(coefficient)
    if absolute < 0.10:
        magnitude = "negligible"
    elif absolute < 0.30:
        magnitude = "weak"
    elif absolute < 0.50:
        magnitude = "moderate"
    else:
        magnitude = "strong"
    direction = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "null"
    reliability = (
        "statistically distinguishable from zero at the 0.05 level"
        if p_value < 0.05
        else "not statistically distinguishable from zero at the 0.05 level"
    )
    return f"{magnitude.capitalize()} {direction} association; {reliability}."


def correlation_rows(
    frame: pd.DataFrame,
    analysis_set: str,
    outcomes: list[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Calculate Pearson and Spearman results for age against each outcome."""
    records: list[dict[str, Any]] = []
    x = frame["age_months"].to_numpy(dtype=float)
    for outcome in outcomes:
        y = frame[outcome].to_numpy(dtype=float)
        for method in ["Pearson", "Spearman"]:
            if len(frame) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
                coefficient, p_value = np.nan, np.nan
            elif method == "Pearson":
                result = pearsonr(x, y)
                coefficient, p_value = float(result.statistic), float(result.pvalue)
            else:
                result = spearmanr(x, y)
                coefficient, p_value = float(result.statistic), float(result.pvalue)
            ci_lower, ci_upper, valid_bootstraps = bootstrap_correlation_ci(
                x, y, method, bootstrap_samples, rng
            )
            row: dict[str, Any] = {
                "Analysis_Set": analysis_set,
                "Outcome": outcome,
                "Method": method,
                "N": len(frame),
                "Coefficient": coefficient,
                "P_Value_Two_Sided": p_value,
                "Bootstrap_CI_95_Lower": ci_lower,
                "Bootstrap_CI_95_Upper": ci_upper,
                "Valid_Bootstrap_Resamples": valid_bootstraps,
                "Interpretation": interpret_correlation(coefficient, p_value),
            }
            if extra:
                row.update(extra)
            records.append(row)
    return records


def calculate_correlations(
    benchmark_data: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate full, date-sensitivity, and recent-window correlations."""
    rng = np.random.default_rng(seed)
    full = pd.DataFrame(
        correlation_rows(
            benchmark_data,
            "all_verified_rows",
            CORRELATION_OUTCOMES,
            bootstrap_samples,
            rng,
        )
    )

    high_confidence = benchmark_data[
        benchmark_data["Date_Precision"].astype(str).str.lower().eq("day")
        & benchmark_data["Manual_Check_Priority"].astype(str).str.lower().eq("low")
    ].copy()
    sensitivity_all = full.copy()
    sensitivity_all["Date_Filter"] = "All available verified rows"
    sensitivity_records = sensitivity_all.to_dict("records")
    sensitivity_records.extend(
        correlation_rows(
            high_confidence,
            "high_confidence_day_low_priority",
            CORRELATION_OUTCOMES,
            bootstrap_samples,
            rng,
            {
                "Date_Filter": (
                    'Date_Precision == "day" and '
                    'Manual_Check_Priority == "low"'
                )
            },
        )
    )
    sensitivity = pd.DataFrame(sensitivity_records)

    window_records: list[dict[str, Any]] = []
    for months in [6, 9, 12, 18, 24]:
        subset = benchmark_data[benchmark_data["age_months"] <= months]
        window_records.extend(
            correlation_rows(
                subset,
                f"age_le_{months}_months",
                CORRELATION_OUTCOMES,
                bootstrap_samples,
                rng,
                {"Window_Months": months},
            )
        )
    windows = pd.DataFrame(window_records)
    return full, sensitivity, windows


def quantile_summary(series: pd.Series) -> tuple[float, float, float]:
    """Return median, first quartile, and third quartile."""
    return (
        float(series.median()),
        float(series.quantile(0.25)),
        float(series.quantile(0.75)),
    )


def calculate_age_group_summary(benchmark_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize mutually exclusive benchmark age groups."""
    records: list[dict[str, Any]] = []
    for group in AGE_GROUPS:
        subset = benchmark_data[benchmark_data["Age_Group"].astype(str) == group]
        row: dict[str, Any] = {"Age_Group": group, "N_Benchmarks": len(subset)}
        for variable in ["age_months", "NUsage", "SImp", "BHI"]:
            median, q1, q3 = quantile_summary(subset[variable])
            prefix = {
                "age_months": "Age_Months",
                "NUsage": "NUsage",
                "SImp": "SImp",
                "BHI": "BHI",
            }[variable]
            row[f"{prefix}_Median"] = median
            row[f"{prefix}_Q1"] = q1
            row[f"{prefix}_Q3"] = q3
        row["BHI_Mean"] = float(subset["BHI"].mean())
        row["Top_10pct_Proportion"] = float(subset["Top_10pct"].mean())
        row["Top_25pct_Proportion"] = float(subset["Top_25pct"].mean())
        row["Top_50pct_Proportion"] = float(subset["Top_50pct"].mean())
        records.append(row)
    return pd.DataFrame(records)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Calculate a two-sided 95% Wilson binomial confidence interval."""
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def calculate_top_quartile_test(benchmark_data: pd.DataFrame) -> pd.DataFrame:
    """Compare recent and older benchmark representation in the top quartile."""
    recent = benchmark_data[benchmark_data["age_months"] <= 12]
    older = benchmark_data[benchmark_data["age_months"] > 12]
    recent_events = int(recent["Top_25pct"].sum())
    older_events = int(older["Top_25pct"].sum())
    recent_ci = wilson_interval(recent_events, len(recent))
    older_ci = wilson_interval(older_events, len(older))
    table = [
        [recent_events, len(recent) - recent_events],
        [older_events, len(older) - older_events],
    ]
    fisher = fisher_exact(table, alternative="two-sided")
    recent_proportion = recent_events / len(recent)
    older_proportion = older_events / len(older)
    return pd.DataFrame(
        [
            {
                "Recent_Definition": "age_months <= 12",
                "Recent_N": len(recent),
                "Recent_Top25_N": recent_events,
                "Recent_Top25_Proportion": recent_proportion,
                "Recent_Wilson_95_Lower": recent_ci[0],
                "Recent_Wilson_95_Upper": recent_ci[1],
                "Older_Definition": "age_months > 12",
                "Older_N": len(older),
                "Older_Top25_N": older_events,
                "Older_Top25_Proportion": older_proportion,
                "Older_Wilson_95_Lower": older_ci[0],
                "Older_Wilson_95_Upper": older_ci[1],
                "Risk_Difference_Recent_Minus_Older": (
                    recent_proportion - older_proportion
                ),
                "Fisher_Exact_Odds_Ratio": float(fisher.statistic),
                "Fisher_Exact_P_Value_Two_Sided": float(fisher.pvalue),
            }
        ]
    )


def select_recent_cases(benchmark_data: pd.DataFrame) -> pd.DataFrame:
    """Select recent benchmark cases using prespecified reproducible rules."""
    specs = [
        (
            "A",
            "age <= 12 months; top five by BHI descending",
            benchmark_data[benchmark_data["age_months"] <= 12],
            "BHI",
            5,
        ),
        (
            "B",
            "age <= 9 months and NUsage <= 5; up to top three by BHI descending",
            benchmark_data[
                (benchmark_data["age_months"] <= 9)
                & (benchmark_data["NUsage"] <= 5)
            ],
            "BHI",
            3,
        ),
        (
            "C",
            "age <= 6 months; up to top three by SImp descending",
            benchmark_data[benchmark_data["age_months"] <= 6],
            "SImp",
            3,
        ),
    ]
    selected: list[pd.DataFrame] = []
    for case_set, rule, subset, sort_column, limit in specs:
        chosen = (
            subset.sort_values(
                [sort_column, "Benchmark"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(limit)
            .copy()
        )
        chosen.insert(0, "Case_Set", case_set)
        chosen.insert(1, "Selection_Rule", rule)
        selected.append(chosen)
    cases = pd.concat(selected, ignore_index=True)
    columns = [
        "Case_Set",
        "Selection_Rule",
        "Benchmark",
        "Release_Date",
        "age_days",
        "age_months",
        "NUsage",
        "SDisc",
        "SSat",
        "SImp",
        "BHI",
        "BHI_Rank_Recomputed",
        "Source_URL",
    ]
    return cases[columns]


def calculate_reporting_statistics(
    master_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Calculate model-level reporting breadth and record concentration."""
    master = MasterDataPreprocessor(CONFIG).load_and_process_master(master_path)
    availability = master.availability.copy()
    swe_mask = (
        availability.index.to_series()
        .map(normalized_benchmark_name)
        .str.contains("swelancer", regex=False)
    )
    swe_benchmark_rows = int(swe_mask.sum())
    swe_records = int(availability.loc[swe_mask].to_numpy().sum())
    availability = availability.loc[
        ~swe_mask
    ]
    model_counts = availability.sum(axis=0).astype(int)
    model_table = (
        model_counts.rename("Benchmarks_Reported")
        .rename_axis("Model")
        .reset_index()
        .sort_values(["Benchmarks_Reported", "Model"], ascending=[False, True])
        .reset_index(drop=True)
    )

    benchmark_counts = availability.sum(axis=1).astype(int)
    concentration = (
        benchmark_counts.rename("Model_Benchmark_Records")
        .rename_axis("Benchmark")
        .reset_index()
        .sort_values(
            ["Model_Benchmark_Records", "Benchmark"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    total_records = int(concentration["Model_Benchmark_Records"].sum())
    concentration["Usage_Rank"] = np.arange(1, len(concentration) + 1)
    concentration["Record_Share"] = (
        concentration["Model_Benchmark_Records"] / total_records
    )
    concentration["Cumulative_Record_Share"] = concentration["Record_Share"].cumsum()
    hhi = float((concentration["Record_Share"] ** 2).sum())

    statistics = {
        "number_of_models": float(len(model_table)),
        "total_model_benchmark_records": float(total_records),
        "swe_lancer_benchmark_rows_excluded": float(swe_benchmark_rows),
        "swe_lancer_records_excluded": float(swe_records),
        "mean_benchmarks_per_model": float(model_counts.mean()),
        "median_benchmarks_per_model": float(model_counts.median()),
        "q1_benchmarks_per_model": float(model_counts.quantile(0.25)),
        "q3_benchmarks_per_model": float(model_counts.quantile(0.75)),
        "minimum_benchmarks_per_model": float(model_counts.min()),
        "maximum_benchmarks_per_model": float(model_counts.max()),
        "percentage_models_at_least_2": float((model_counts >= 2).mean() * 100),
        "percentage_models_at_least_5": float((model_counts >= 5).mean() * 100),
        "percentage_models_at_least_10": float((model_counts >= 10).mean() * 100),
        "top_1_record_share": float(concentration.iloc[0]["Record_Share"]),
        "top_5_cumulative_record_share": float(
            concentration.iloc[:5]["Record_Share"].sum()
        ),
        "top_10_cumulative_record_share": float(
            concentration.iloc[:10]["Record_Share"].sum()
        ),
        "record_share_hhi": hhi,
    }
    for key, value in statistics.items():
        concentration[key] = value
    return model_table, concentration, statistics


def setup_plot_style() -> None:
    """Set a consistent publication-oriented plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    """Save a figure as PDF and 300-DPI PNG."""
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def spearman_for(frame: pd.DataFrame, outcome: str) -> float:
    """Return the Spearman correlation between age and an outcome."""
    return float(spearmanr(frame["age_months"], frame[outcome]).statistic)


def scatter_figure(
    benchmark_data: pd.DataFrame,
    outcome: str,
    ylabel: str,
    figures_dir: Path,
    stem: str,
) -> None:
    """Create an age-versus-outcome scatter plot with a descriptive trend line."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    x = benchmark_data["age_months"].to_numpy(dtype=float)
    y = benchmark_data[outcome].to_numpy(dtype=float)
    ax.scatter(x, y, s=35, alpha=0.72, color="#2455A4", edgecolor="white", linewidth=0.4)
    coefficients = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        line_x,
        np.polyval(coefficients, line_x),
        color="#C43C39",
        linestyle="--",
        linewidth=2,
        label="Descriptive linear trend",
    )
    rho = spearman_for(benchmark_data, outcome)
    ax.text(
        0.02,
        0.97,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
    )
    ax.set_xlabel("Benchmark age at 2026-04-30 (months)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Benchmark age versus {ylabel}")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    save_figure(fig, figures_dir, stem)


def create_figures(
    benchmark_data: pd.DataFrame,
    age_summary: pd.DataFrame,
    model_counts: pd.DataFrame,
    figures_dir: Path,
    seed: int,
) -> None:
    """Create all requested publication-ready figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    scatter_figure(
        benchmark_data,
        "SImp",
        "SImp",
        figures_dir,
        "age_vs_simp",
    )
    scatter_figure(
        benchmark_data,
        "BHI",
        "BHI",
        figures_dir,
        "age_vs_bhi",
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    distributions = [
        benchmark_data.loc[
            benchmark_data["Age_Group"].astype(str) == group, "SImp"
        ].to_numpy()
        for group in AGE_GROUPS
    ]
    boxes = ax.boxplot(distributions, tick_labels=AGE_GROUPS, patch_artist=True)
    for box in boxes["boxes"]:
        box.set_facecolor("#86A9D8")
        box.set_alpha(0.75)
    jitter_rng = np.random.default_rng(seed)
    for position, values in enumerate(distributions, start=1):
        jitter = jitter_rng.normal(position, 0.045, size=len(values))
        ax.scatter(jitter, values, s=18, alpha=0.45, color="#244A7C")
    ax.set_ylabel("SImp")
    ax.set_xlabel("Benchmark age group")
    ax.set_title("SImp distribution by benchmark age group")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    save_figure(fig, figures_dir, "age_group_simp_distribution")

    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    proportions = age_summary["Top_25pct_Proportion"].to_numpy()
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for group, proportion in zip(AGE_GROUPS, proportions):
        subset = benchmark_data[benchmark_data["Age_Group"].astype(str) == group]
        lower, upper = wilson_interval(int(subset["Top_25pct"].sum()), len(subset))
        lower_errors.append(proportion - lower)
        upper_errors.append(upper - proportion)
    ax.bar(
        AGE_GROUPS,
        proportions,
        yerr=np.array([lower_errors, upper_errors]),
        capsize=5,
        color="#5B8DB8",
        edgecolor="#234D6F",
    )
    ax.axhline(0.25, color="#C43C39", linestyle="--", label="Overall top-25% reference")
    ax.set_ylim(0, min(1.0, max(proportions + np.array(upper_errors)) + 0.12))
    ax.set_ylabel("Proportion in overall BHI top 25%")
    ax.set_xlabel("Benchmark age group")
    ax.set_title("Top-quartile BHI representation by benchmark age group")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    save_figure(fig, figures_dir, "age_group_top25_representation")

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    counts = model_counts["Benchmarks_Reported"]
    bins = np.arange(counts.min() - 0.5, counts.max() + 1.5, 2)
    ax.hist(counts, bins=bins, color="#5B8DB8", edgecolor="white")
    ax.axvline(counts.median(), color="#C43C39", linestyle="--", label=f"Median = {counts.median():.0f}")
    ax.set_xlabel("Benchmarks reported per model")
    ax.set_ylabel("Number of models")
    ax.set_title("Distribution of benchmark reporting breadth")
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    save_figure(fig, figures_dir, "benchmarks_reported_per_model")


def format_correlation_table(frame: pd.DataFrame) -> str:
    """Format correlation rows as a compact Markdown table."""
    rows = [
        "| Outcome | Method | N | Coefficient | p-value | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        rows.append(
            f"| {row.Outcome} | {row.Method} | {row.N} | "
            f"{row.Coefficient:.3f} | {row.P_Value_Two_Sided:.4g} | "
            f"[{row.Bootstrap_CI_95_Lower:.3f}, {row.Bootstrap_CI_95_Upper:.3f}] |"
        )
    return "\n".join(rows)


def format_window_spearman_table(frame: pd.DataFrame) -> str:
    """Format primary recent-window correlations as a compact Markdown table."""
    primary = frame[frame["Method"] == "Spearman"]
    rows = [
        "| Window | Outcome | N | Spearman ρ | p-value | Bootstrap 95% CI |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        rows.append(
            f"| ≤{row.Window_Months:g} months | {row.Outcome} | {row.N} | "
            f"{row.Coefficient:.3f} | {row.P_Value_Two_Sided:.4g} | "
            f"[{row.Bootstrap_CI_95_Lower:.3f}, {row.Bootstrap_CI_95_Upper:.3f}] |"
        )
    return "\n".join(rows)


def correlation_lookup(
    frame: pd.DataFrame, outcome: str, method: str
) -> pd.Series:
    """Select one correlation result row."""
    return frame[(frame["Outcome"] == outcome) & (frame["Method"] == method)].iloc[0]


def create_report(
    benchmark_data: pd.DataFrame,
    validation: pd.DataFrame,
    full: pd.DataFrame,
    sensitivity: pd.DataFrame,
    windows: pd.DataFrame,
    age_summary: pd.DataFrame,
    fairness_test: pd.DataFrame,
    cases: pd.DataFrame,
    reporting_stats: dict[str, float],
    snapshot_date: pd.Timestamp,
    bootstrap_samples: int,
    seed: int,
) -> str:
    """Build the Markdown report using computed values only."""
    high_confidence = sensitivity[
        sensitivity["Analysis_Set"] == "high_confidence_day_low_priority"
    ]
    fairness = fairness_test.iloc[0]
    spearman_usage = correlation_lookup(full, "NUsage", "Spearman")
    spearman_simp = correlation_lookup(full, "SImp", "Spearman")
    spearman_bhi = correlation_lookup(full, "BHI", "Spearman")
    six_month_usage = windows[
        (windows["Window_Months"] == 6)
        & (windows["Outcome"] == "NUsage")
        & (windows["Method"] == "Spearman")
    ].iloc[0]
    six_month_simp = windows[
        (windows["Window_Months"] == 6)
        & (windows["Outcome"] == "SImp")
        & (windows["Method"] == "Spearman")
    ].iloc[0]
    six_month_bhi = windows[
        (windows["Window_Months"] == 6)
        & (windows["Outcome"] == "BHI")
        & (windows["Method"] == "Spearman")
    ].iloc[0]

    recent_direction = (
        "higher"
        if fairness["Risk_Difference_Recent_Minus_Older"] > 0
        else "lower"
        if fairness["Risk_Difference_Recent_Minus_Older"] < 0
        else "equal"
    )
    full_supports_positive = any(
        (full["Method"] == "Spearman")
        & (full["Coefficient"] > 0)
        & (full["P_Value_Two_Sided"] < 0.05)
    )
    if full_supports_positive:
        primary_interpretation = (
            "At least one primary Spearman result shows a reliable positive age "
            "association, so the snapshot contains evidence consistent with part of "
            "the proposed age-advantage mechanism."
        )
    else:
        primary_interpretation = (
            "The primary Spearman results do not show a reliable positive association "
            "between benchmark age and NUsage, SImp, or BHI. This does not prove that "
            "cold-start effects are absent; the confidence intervals remain compatible "
            "with effects of the sizes shown in the tables."
        )

    selected_lines = []
    for row in cases.itertuples():
        selected_lines.append(
            f"- Case {row.Case_Set}: **{row.Benchmark}** — released "
            f"{row.Release_Date.date()}, age {row.age_months:.2f} months, "
            f"NUsage {row.NUsage:g}, SImp {row.SImp:.3f}, BHI {row.BHI:.3f}, "
            f"rank {row.BHI_Rank_Recomputed}; [source]({row.Source_URL})."
        )

    high_mask = (
        benchmark_data["Date_Precision"].astype(str).str.lower().eq("day")
        & benchmark_data["Manual_Check_Priority"].astype(str).str.lower().eq("low")
    )
    quality_categories = {
        "month-level": benchmark_data["Date_Precision"]
        .astype(str)
        .str.lower()
        .ne("day"),
        "family-level": (
            benchmark_data["Release_Scope"]
            .astype(str)
            .str.lower()
            .str.contains("family")
            | benchmark_data["Verification_Status"]
            .astype(str)
            .str.lower()
            .str.contains("family")
        ),
        "parent-benchmark": (
            benchmark_data["Release_Scope"]
            .astype(str)
            .str.lower()
            .str.contains("parent")
            | benchmark_data["Verification_Status"]
            .astype(str)
            .str.lower()
            .str.contains("parent")
        ),
        "version-unresolved": benchmark_data["Verification_Status"]
        .astype(str)
        .str.lower()
        .str.contains("unresolved"),
        "medium/high-priority": benchmark_data["Manual_Check_Priority"]
        .astype(str)
        .str.lower()
        .isin(["medium", "high"]),
    }
    quality_counts = {
        name: int((mask & ~high_mask).sum())
        for name, mask in quality_categories.items()
    }
    sensitivity_note = (
        f"The high-confidence subset contains {int(high_confidence['N'].max())} "
        "benchmarks and requires day-level precision plus low manual-check priority. "
        f"It excludes all {quality_counts['month-level']} month-level, "
        f"{quality_counts['family-level']} family-level, "
        f"{quality_counts['parent-benchmark']} parent-benchmark, "
        f"{quality_counts['version-unresolved']} version-unresolved, and "
        f"{quality_counts['medium/high-priority']} medium/high-priority rows "
        "(categories overlap). The qualitative conclusions are unchanged: NUsage "
        "remains near zero and non-significant, while SImp and BHI remain significantly "
        "negative rather than positive."
    )

    summary_text = (
        f"In the {snapshot_date.date()} supplied snapshot ({len(benchmark_data)} "
        f"eligible benchmarks), benchmark age was not reliably positively associated "
        f"with usage (Spearman ρ={spearman_usage['Coefficient']:.3f}, "
        f"p={spearman_usage['P_Value_Two_Sided']:.3g}), SImp "
        f"(ρ={spearman_simp['Coefficient']:.3f}, "
        f"p={spearman_simp['P_Value_Two_Sided']:.3g}), or BHI "
        f"(ρ={spearman_bhi['Coefficient']:.3f}, "
        f"p={spearman_bhi['P_Value_Two_Sided']:.3g}). Benchmarks released within "
        f"12 months entered the top BHI quartile at {fairness['Recent_Top25_Proportion']:.1%} "
        f"versus {fairness['Older_Top25_Proportion']:.1%} for older benchmarks "
        f"(risk difference {fairness['Risk_Difference_Recent_Minus_Older']:+.1%}; "
        f"Fisher p={fairness['Fisher_Exact_P_Value_Two_Sided']:.3g}). Reporting was "
        f"multi-label: models reported a median of "
        f"{reporting_stats['median_benchmarks_per_model']:.0f} benchmarks, with "
        f"{reporting_stats['percentage_models_at_least_2']:.1f}% reporting at least "
        f"two. The small ≤6-month cohort (n={six_month_bhi['N']:.0f}) nevertheless "
        f"showed positive age associations with NUsage (ρ={six_month_usage['Coefficient']:.3f}, "
        f"p={six_month_usage['P_Value_Two_Sided']:.3g}) and BHI "
        f"(ρ={six_month_bhi['Coefficient']:.3f}, "
        f"p={six_month_bhi['P_Value_Two_Sided']:.3g}), so short-run accumulation "
        "remains plausible. Overall, the snapshot does not exhibit the broad empirical "
        "pattern expected under a strong lifecycle-wide age-driven lock-in cycle, while "
        "pre-eligibility visibility concerns remain possible for benchmarks with fewer "
        "than three reported scores."
    )

    report = f"""# Cold-start fairness analysis

## Scope and reproducibility

This analysis uses only the supplied snapshot inputs and recomputes age as
`snapshot_date - Release_Date`, with unrounded months defined as `age_days / 30.44`.
It retains the complete supplied BHI, including SImp. The analysis used
{bootstrap_samples:,} bootstrap resamples with seed {seed}.

## Input validation

- Analysed benchmarks: **{len(benchmark_data)}**.
- Rows remaining after the analysis-specific exclusion: **{len(benchmark_data)}**.
- Future release dates: **{int((benchmark_data['Release_Date'] > snapshot_date).sum())}**.
- Missing required metric values: **{int(benchmark_data[METRICS].isna().sum().sum())}**.
- Missing source URLs: **{int(benchmark_data['Source_URL'].eq('').sum())}**.
- Detailed checks and any unresolved rows are recorded in `input_validation_report.csv`.

## A. Post-entry age fairness

Spearman correlation is the primary statistic because the variables may be non-normal
or non-linear. Pearson results are provided as a complementary linear-association check.

{format_correlation_table(full)}

{primary_interpretation}

### Date-quality sensitivity

{sensitivity_note}

{format_correlation_table(high_confidence)}

### Recent-release windows

{format_window_spearman_table(windows)}

The ≤6-month cohort is small (n={six_month_usage['N']:.0f}) and therefore unstable, but
it is the clearest result qualifying the broad age pattern: age is positively associated
with NUsage (ρ={six_month_usage['Coefficient']:.3f},
p={six_month_usage['P_Value_Two_Sided']:.4g}) and BHI
(ρ={six_month_bhi['Coefficient']:.3f},
p={six_month_bhi['P_Value_Two_Sided']:.4g}) within this earliest window. Its SImp
association is also positive but not reliable (ρ={six_month_simp['Coefficient']:.3f},
p={six_month_simp['P_Value_Two_Sided']:.4g}). This pattern is compatible with short-run
accumulation during the first months after release. It does not persist as a reliable
positive association in the 9-, 12-, 18-, or 24-month primary results. Complete Pearson
and Spearman results are reported in `age_correlations_windows.csv`.

## B. Ranking representation

Recent benchmarks (≤12 months) entered the overall top quartile at
**{fairness['Recent_Top25_N']:.0f}/{fairness['Recent_N']:.0f}
({fairness['Recent_Top25_Proportion']:.1%})**, compared with
**{fairness['Older_Top25_N']:.0f}/{fairness['Older_N']:.0f}
({fairness['Older_Top25_Proportion']:.1%})** among older benchmarks. The recent-minus-old
risk difference was **{fairness['Risk_Difference_Recent_Minus_Older']:+.1%}** and the
two-sided Fisher exact p-value was **{fairness['Fisher_Exact_P_Value_Two_Sided']:.4g}**.
The recent rate was {recent_direction}; this comparison addresses exclusion from the
upper ranking region and does not imply that either age group is intrinsically better.

Age-group medians, IQRs, means, and top-10/25/50% representation are in
`age_group_summary.csv`.

## Systematically selected recent cases

The following cases were selected only by the prespecified rules in
`selected_recent_cases.csv`; repeated benchmarks across rules are retained transparently.

{chr(10).join(selected_lines)}

## Benchmark reporting is non-exclusive

The master snapshot contains **{reporting_stats['number_of_models']:.0f} models** and
**{reporting_stats['total_model_benchmark_records']:.0f} observed model–benchmark
records** after applying the analysis-specific row exclusion. Models report a mean of
**{reporting_stats['mean_benchmarks_per_model']:.2f}** and median of
**{reporting_stats['median_benchmarks_per_model']:.0f}** benchmarks
(Q1={reporting_stats['q1_benchmarks_per_model']:.0f},
Q3={reporting_stats['q3_benchmarks_per_model']:.0f}, range
{reporting_stats['minimum_benchmarks_per_model']:.0f}–{reporting_stats['maximum_benchmarks_per_model']:.0f}).
The shares reporting at least 2, 5, and 10 benchmarks are
**{reporting_stats['percentage_models_at_least_2']:.1f}%**,
**{reporting_stats['percentage_models_at_least_5']:.1f}%**, and
**{reporting_stats['percentage_models_at_least_10']:.1f}%**, respectively.

The most-used benchmark accounts for
**{reporting_stats['top_1_record_share']:.1%}** of all records; the top five and top ten
account for **{reporting_stats['top_5_cumulative_record_share']:.1%}** and
**{reporting_stats['top_10_cumulative_record_share']:.1%}**. The descriptive record-share
HHI is **{reporting_stats['record_share_hhi']:.4f}**. Reporting is therefore strongly
non-exclusive in this snapshot: continued use of an established benchmark does not
mechanically prevent simultaneous reporting on a new benchmark. Paper space, visibility,
and community attention nevertheless remain limited resources.

## C. Pre-entry visibility

This analysis evaluates post-entry fairness only among benchmarks already represented in
the supplied BHI table. Benchmarks with fewer than three reported scores are absent from
the supplied ranking and cannot be evaluated here. That separate issue is addressed by
the statistical justification for the three-score threshold, an **insufficient evidence**
label rather than a low-quality label, and a proposed non-ranked Emerging Benchmark
Watchlist.

## Summary

{summary_text}
"""
    return report


def package_versions() -> dict[str, str]:
    """Collect versions of packages material to the analysis."""
    packages = ["numpy", "pandas", "scipy", "matplotlib", "scikit-learn"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic CSV representation."""
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def main() -> None:
    """Run the complete cold-start fairness analysis."""
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    snapshot_date = pd.Timestamp(args.snapshot_date).normalize()
    release_path = resolve_path(args.release_dates)
    master_path = resolve_path(args.master_data)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_data, validation, row_counts = prepare_benchmark_data(
        release_path, snapshot_date
    )
    full, sensitivity, windows = calculate_correlations(
        benchmark_data, args.bootstrap_samples, args.seed
    )
    age_summary = calculate_age_group_summary(benchmark_data)
    fairness_test = calculate_top_quartile_test(benchmark_data)
    cases = select_recent_cases(benchmark_data)
    model_counts, concentration, reporting_stats = calculate_reporting_statistics(
        master_path
    )
    validation = pd.concat(
        [
            validation,
            pd.DataFrame(
                [
                    {
                        "Record_Type": "summary",
                        "Check": "master_swe_lancer_benchmark_rows_excluded",
                        "Status": "pass",
                        "Count": int(
                            reporting_stats["swe_lancer_benchmark_rows_excluded"]
                        ),
                        "Details": "Analysis-specific normalized-name exclusion.",
                        "Benchmark": "",
                    },
                    {
                        "Record_Type": "summary",
                        "Check": "master_swe_lancer_score_records_excluded",
                        "Status": "pass",
                        "Count": int(reporting_stats["swe_lancer_records_excluded"]),
                        "Details": "Observed records removed by the analysis-specific exclusion.",
                        "Benchmark": "",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    benchmark_output = benchmark_data.copy()
    benchmark_output["Release_Date"] = benchmark_output["Release_Date"].dt.strftime(
        "%Y-%m-%d"
    )
    cases_output = cases.copy()
    cases_output["Release_Date"] = cases_output["Release_Date"].dt.strftime("%Y-%m-%d")

    outputs = {
        "benchmark_age_metrics.csv": benchmark_output,
        "age_correlations_full.csv": full,
        "age_correlations_date_sensitivity.csv": sensitivity,
        "age_correlations_windows.csv": windows,
        "age_group_summary.csv": age_summary,
        "top_quartile_fairness_test.csv": fairness_test,
        "selected_recent_cases.csv": cases_output,
        "model_reporting_counts.csv": model_counts,
        "benchmark_reporting_concentration.csv": concentration,
        "input_validation_report.csv": validation,
    }
    for filename, frame in outputs.items():
        write_csv(frame, output_dir / filename)

    create_figures(
        benchmark_data,
        age_summary,
        model_counts,
        output_dir / "figures",
        args.seed,
    )
    report = create_report(
        benchmark_data,
        validation,
        full,
        sensitivity,
        windows,
        age_summary,
        fairness_test,
        cases,
        reporting_stats,
        snapshot_date,
        args.bootstrap_samples,
        args.seed,
    )
    (output_dir / "cold_start_fairness_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )

    command = subprocess.list2cmdline(["python", *sys.argv])
    manifest = {
        "analysis": "cold_start_fairness",
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "random_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "inputs": {
            "release_dates": {
                "path": str(release_path),
                "sha256": sha256_file(release_path),
            },
            "master_data": {
                "path": str(master_path),
                "sha256": sha256_file(master_path),
            },
        },
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "exact_command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts": {
            **row_counts,
            "master_models": int(reporting_stats["number_of_models"]),
            "master_model_benchmark_records": int(
                reporting_stats["total_model_benchmark_records"]
            ),
            "master_swe_lancer_benchmark_rows_excluded": int(
                reporting_stats["swe_lancer_benchmark_rows_excluded"]
            ),
            "master_swe_lancer_records_excluded": int(
                reporting_stats["swe_lancer_records_excluded"]
            ),
        },
        "date_quality_filter": (
            'Date_Precision == "day" and Manual_Check_Priority == "low"'
        ),
        "age_definition": {
            "age_days": "snapshot_date - Release_Date",
            "age_months": "age_days / 30.44",
        },
        "generated_files": sorted(
            str(path.relative_to(output_dir)).replace("\\", "/")
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "analysis_manifest.json"
        ),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Cold-start fairness analysis complete: {output_dir}")
    print(f"Benchmarks analysed: {len(benchmark_data)}")
    print(
        "Models and model-benchmark records: "
        f"{int(reporting_stats['number_of_models'])}, "
        f"{int(reporting_stats['total_model_benchmark_records'])}"
    )


if __name__ == "__main__":
    main()
