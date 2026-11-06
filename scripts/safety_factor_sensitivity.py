"""Open-upper-bound safety-factor sensitivity on a supplied BHI baseline."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CONFIG, WEIGHT_DYNAMIC, WEIGHT_STATIC  # noqa: E402
from src.capability import calculate_lobo_capabilities  # noqa: E402
from src.preprocessor import MasterDataPreprocessor  # noqa: E402


OUTPUT_DIR = Path(os.environ.get(
    "BHI_EXPERIMENT_OUTPUT_DIR",
    ROOT / "data" / "output" / "safety_factor_sensitivity",
))
INPUT_PATH = Path(os.environ.get(
    "BHI_IN_MASTER",
    ROOT / "data" / "input" / "2026_0412_cleaned.csv",
))
BASE_DIR = Path(os.environ.get(
    "BHI_BASELINE_DIR",
    ROOT / "data" / "output",
))
ORIGINAL_PREPROCESSOR = Path(os.environ.get(
    "BHI_SUBMISSION_PREPROCESSOR",
    ROOT / "config" / "submission_preprocessor_reference.py",
))
DEFAULT = "SF_1_25"
CONDITIONS = [
    ("SF_1_10", 1.10),
    ("SF_1_20", 1.20),
    ("SF_1_25", 1.25),
    ("SF_1_33", 1.33),
    ("SF_1_50", 1.50),
]
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20260726
TOLERANCE = 1e-12
TIMESTAMP = datetime.now(timezone.utc).isoformat()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integrity_paths() -> list[Path]:
    """Every pre-existing workspace file, excluding this experiment's new files."""
    paths = []
    script = Path(__file__).resolve()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == script or OUTPUT_DIR.resolve() in resolved.parents:
            continue
        paths.append(resolved)
    return sorted(set(paths), key=display_path)


def submission_era_dates(model_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    """Reproduce the original pd.to_datetime(..., errors='coerce') call exactly."""
    raw = pd.read_csv(INPUT_PATH)
    first = raw.columns[0]
    names = raw[first].fillna("").astype(str).str.strip().str.lower()
    mask = names.isin({"release date", "release_date"})
    if not mask.any():
        return pd.Series(pd.NaT, index=model_columns), pd.Series("", index=model_columns)
    raw_dates = raw.loc[mask, model_columns].iloc[0]
    return pd.to_datetime(raw_dates, errors="coerce"), raw_dates


def reconstruct_weights(components: pd.DataFrame) -> pd.Series:
    std = components.std()
    conflict = (1 - components.corr()).sum()
    information = std * conflict
    weights = information / information.sum()
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=TOLERANCE):
        raise ValueError("CRITIC weights do not sum to one")
    return weights


def fixed_top(frame: pd.DataFrame, k: int) -> set[str]:
    ordered = (
        frame.reset_index()
        .sort_values(["BHI", "Benchmark"], ascending=[False, True], kind="mergesort")
        .head(k)
    )
    return set(ordered["Benchmark"])


def expanded_top(frame: pd.DataFrame, k: int) -> tuple[set[str], bool]:
    ordered = frame.reset_index().sort_values(
        ["BHI", "Benchmark"], ascending=[False, True], kind="mergesort"
    )
    k = min(k, len(ordered))
    if not k:
        return set(), False
    cutoff = float(ordered.iloc[k - 1]["BHI"])
    expanded = set(ordered.loc[ordered["BHI"] >= cutoff, "Benchmark"])
    return expanded, int((ordered["BHI"] == cutoff).sum()) > 1


def open_upper_rule(name: str, tags: dict) -> bool:
    """Mirror the submitted if/elif branch ordering and case handling."""
    if tags["lib_0_1"] in name:
        return False
    if tags["lib_standard"] in name:
        return False
    return tags["unbounded_hib"].lower() in name.lower()


def calculate_condition(label: str, factor: float) -> dict:
    config = copy.deepcopy(CONFIG)
    config["safety_factor"] = factor
    master = MasterDataPreprocessor(config).load_and_process_master(INPUT_PATH)
    model_dates, raw_model_dates = submission_era_dates(master.model_columns)
    master.model_dates = model_dates
    min_count = config["min_model_count"]

    disc_counts = master.disc_scores.notna().sum(axis=1)
    sat_counts = master.sat_scores.notna().sum(axis=1)
    impact_counts = master.availability.sum(axis=1)

    disc_matrix = master.disc_scores.loc[disc_counts >= min_count]
    disc_rows = []
    for benchmark, row in disc_matrix.iterrows():
        values = row.dropna().to_numpy()
        n = len(values)
        rcv = (np.percentile(values, 90) - np.percentile(values, 10)) / 100
        differences = np.abs(values[:, None] - values)
        edr = (
            (np.sum(differences > config["edr_threshold"]) / 2)
            / (n * (n - 1) / 2)
            if n > 1
            else 0
        )
        disc_rows.append(
            {
                "Benchmark": benchmark,
                "N_Models": n,
                "Robust_CV": rcv,
                "EDR": edr,
            }
        )
    disc = pd.DataFrame(disc_rows).set_index("Benchmark")
    normalized_disc = (disc[["Robust_CV", "EDR"]] - disc[["Robust_CV", "EDR"]].min()) / (
        disc[["Robust_CV", "EDR"]].max()
        - disc[["Robust_CV", "EDR"]].min()
        + 1e-9
    )
    internal_weights = normalized_disc.std() / normalized_disc.std().sum()
    disc["Normalized_Robust_CV"] = normalized_disc["Robust_CV"]
    disc["Normalized_EDR"] = normalized_disc["EDR"]
    disc["S_Disc"] = normalized_disc.mul(internal_weights).sum(axis=1)

    sat_matrix = master.sat_scores.loc[sat_counts >= min_count]
    bench_data = sat_matrix.reset_index()
    saturation_rows = []
    impact_rows = []
    capabilities = {}
    n_total = len(bench_data)
    for index in range(n_total):
        benchmark = bench_data.loc[index, "Benchmark"]
        caps = calculate_lobo_capabilities(bench_data, index, n_total)
        capabilities[benchmark] = caps["Capability_Score"].copy()
        scores = bench_data.iloc[index, 1:]
        valid = scores.dropna()
        common = valid.index.intersection(caps.index)
        s_static = 1 - (
            ((valid[common] / 100) * caps.loc[common, "Capability_Score"]).sum()
            / (caps.loc[common, "Capability_Score"].sum() + 1e-9)
        )
        regression_data = pd.DataFrame(
            {"d": (model_dates - model_dates.min()).dt.days, "s": scores}
        ).dropna()
        if len(regression_data) >= 2:
            regression = LinearRegression().fit(
                regression_data[["d"]], regression_data["s"] / 100
            )
            s_dynamic = max(
                0,
                1
                - (
                    regression_data["s"].mean() / 100
                    + regression.coef_[0] * 30
                ),
            )
        else:
            s_dynamic = s_static
        s_sat = WEIGHT_STATIC * s_static + WEIGHT_DYNAMIC * s_dynamic
        saturation_rows.append(
            {
                "Benchmark": benchmark,
                "S_Static": s_static,
                "S_Dynamic": s_dynamic,
                "S_Sat": s_sat,
            }
        )

        cap = caps["Capability_Score"].reindex(master.model_columns).fillna(0)
        denominator = cap.sum()
        evaluated = (
            master.availability.loc[benchmark]
            .reindex(master.model_columns)
            .fillna(False)
        )
        evaluated_models = evaluated[evaluated].index
        numerator = cap.loc[evaluated_models].sum()
        s_imp_raw = numerator / denominator if denominator > 0 else 0
        impact_rows.append(
            {
                "Benchmark": benchmark,
                "N_Models_With_Score": len(evaluated_models),
                "Total_Models": len(master.model_columns),
                "Numerator_Cap": numerator,
                "Denominator_Cap": denominator,
                "S_Imp_Raw": s_imp_raw,
                "S_Imp": np.sqrt(s_imp_raw),
            }
        )

    saturation = pd.DataFrame(saturation_rows).set_index("Benchmark")
    impact = pd.DataFrame(impact_rows).set_index("Benchmark")
    final = (
        disc[["S_Disc"]]
        .merge(saturation[["S_Static", "S_Dynamic", "S_Sat"]], left_index=True, right_index=True)
        .merge(impact[["S_Imp_Raw", "S_Imp"]], left_index=True, right_index=True)
    )
    critic_weights = reconstruct_weights(final[["S_Disc", "S_Sat", "S_Imp"]])
    final["BHI"] = final[["S_Disc", "S_Sat", "S_Imp"]].mul(critic_weights).sum(axis=1)
    final["Rank"] = final["BHI"].rank(method="average", ascending=False)
    n_ranked = len(final)
    final["Rank_Percentile"] = 1 - (final["Rank"] - 1) / (n_ranked - 1)

    raw_csv = pd.read_csv(INPUT_PATH)
    first = raw_csv.columns[0]
    names = raw_csv[first].fillna("").astype(str).str.strip()
    names_lower = names.str.lower()
    benchmark_raw = raw_csv.loc[
        ~names_lower.isin({"vendor", "release date", "release_date", "", "nan", "unknown"})
    ].copy()
    benchmark_raw.index = pd.Index(
        benchmark_raw[first].astype(str).str.strip(), name="Benchmark"
    )
    preprocessor = MasterDataPreprocessor(config)
    raw_values = benchmark_raw[master.model_columns].apply(
        lambda col: col.map(preprocessor._clean_sat_cell)
    )
    affected = [
        name
        for name in master.benchmark_names
        if open_upper_rule(str(name), config["tags"])
    ]

    return {
        "label": label,
        "factor": factor,
        "mapped_maximum": 100 / factor,
        "headroom": 100 - 100 / factor,
        "master": master,
        "raw_model_dates": raw_model_dates,
        "disc_counts": disc_counts,
        "sat_counts": sat_counts,
        "impact_counts": impact_counts,
        "disc_matrix": disc_matrix,
        "sat_matrix": sat_matrix,
        "raw_values": raw_values,
        "affected": affected,
        "disc": disc,
        "saturation": saturation,
        "impact": impact,
        "capabilities": capabilities,
        "internal_weights": internal_weights,
        "critic_weights": critic_weights,
        "final": final,
    }


def max_aligned_difference(
    calculated: pd.DataFrame, submitted: pd.DataFrame, column: str
) -> float:
    common = calculated.index.intersection(submitted.index)
    return float((calculated.loc[common, column] - submitted.loc[common, column]).abs().max())


def reproduction_check(condition: dict) -> pd.DataFrame:
    submitted_disc = pd.read_csv(
        BASE_DIR / "capability_discrimination_final.csv"
    ).set_index("Benchmark")
    submitted_sat = pd.read_csv(BASE_DIR / "anti_saturation_results.csv").set_index(
        "Benchmark"
    )
    submitted_imp = pd.read_csv(BASE_DIR / "impact_results.csv").set_index("Benchmark")
    submitted_final = pd.read_csv(BASE_DIR / "final_bhi_rankings.csv").set_index(
        "Benchmark"
    )
    calculated = condition
    checks = []

    def add(check: str, value: float, tolerance: float = TOLERANCE) -> None:
        checks.append(
            {
                "check": check,
                "passed": bool(value <= tolerance),
                "observed_max_abs_difference": value,
                "tolerance": tolerance,
            }
        )

    add("disc_benchmark_set_matches", float(set(calculated["disc"].index) != set(submitted_disc.index)))
    add("sat_benchmark_set_matches", float(set(calculated["saturation"].index) != set(submitted_sat.index)))
    add("impact_benchmark_set_matches", float(set(calculated["impact"].index) != set(submitted_imp.index)))
    add("final_benchmark_set_matches", float(set(calculated["final"].index) != set(submitted_final.index)))
    add(
        "disc_submitted_order_matches",
        float(
            list(calculated["disc"].sort_values("S_Disc", ascending=False).index)
            != list(submitted_disc.index)
        ),
    )
    add(
        "sat_submitted_order_matches",
        float(
            list(calculated["saturation"].sort_values("S_Sat", ascending=False).index)
            != list(submitted_sat.index)
        ),
    )
    add(
        "impact_submitted_order_matches",
        float(
            list(calculated["impact"].sort_values("S_Imp", ascending=False).index)
            != list(submitted_imp.index)
        ),
    )
    for column in ["Robust_CV", "EDR", "S_Disc"]:
        add(f"max_abs_{column}_difference", max_aligned_difference(calculated["disc"], submitted_disc, column))
    for column in ["S_Static", "S_Dynamic", "S_Sat"]:
        add(f"max_abs_{column}_difference", max_aligned_difference(calculated["saturation"], submitted_sat, column))
    for column in ["S_Imp_Raw", "S_Imp"]:
        add(f"max_abs_{column}_difference", max_aligned_difference(calculated["impact"], submitted_imp, column))
    for column in ["S_Disc", "S_Sat", "S_Imp", "BHI"]:
        add(f"max_abs_final_{column}_difference", max_aligned_difference(calculated["final"], submitted_final, column))

    submitted_internal = submitted_disc[["Robust_CV", "EDR"]]
    submitted_normalized = (submitted_internal - submitted_internal.min()) / (
        submitted_internal.max() - submitted_internal.min() + 1e-9
    )
    submitted_internal_weights = (
        submitted_normalized.std() / submitted_normalized.std().sum()
    )
    submitted_critic = reconstruct_weights(
        submitted_final[["S_Disc", "S_Sat", "S_Imp"]]
    )
    for key in ["Robust_CV", "EDR"]:
        add(
            f"max_abs_internal_{key}_weight_difference",
            abs(float(calculated["internal_weights"][key] - submitted_internal_weights[key])),
        )
    for key in ["S_Disc", "S_Sat", "S_Imp"]:
        add(
            f"max_abs_critic_{key}_weight_difference",
            abs(float(calculated["critic_weights"][key] - submitted_critic[key])),
        )
    submitted_order = list(
        submitted_final.sort_values(["BHI", "S_Imp"], ascending=False).index
    )
    calculated_order = list(
        calculated["final"].sort_values(["BHI", "S_Imp"], ascending=False).index
    )
    add("final_ranking_order_matches", float(submitted_order != calculated_order))
    frame = pd.DataFrame(checks)
    if not frame["passed"].all():
        raise RuntimeError(
            "SF_1_25 failed the submission reproduction gate:\n"
            + frame.loc[~frame["passed"]].to_string(index=False)
        )
    return frame


def condition_weight_rows(conditions: dict[str, dict]) -> pd.DataFrame:
    reference = conditions[DEFAULT]
    rows = []
    for label, factor in CONDITIONS:
        condition = conditions[label]
        iw = condition["internal_weights"]
        cw = condition["critic_weights"]
        diw = iw - reference["internal_weights"]
        dcw = cw - reference["critic_weights"]
        rows.append(
            {
                "condition": label,
                "safety_factor": factor,
                "mapped_observed_maximum_score": 100 / factor,
                "implied_remaining_headroom": 100 - 100 / factor,
                "number_of_models": len(condition["master"].model_columns),
                "number_of_raw_benchmark_rows": len(condition["master"].benchmark_names),
                "number_of_open_upper_bound_benchmarks": len(condition["affected"]),
                "number_of_final_ranked_benchmarks": len(condition["final"]),
                "internal_robust_cv_weight": iw["Robust_CV"],
                "internal_edr_weight": iw["EDR"],
                "internal_robust_cv_weight_change": diw["Robust_CV"],
                "internal_edr_weight_change": diw["EDR"],
                "maximum_absolute_internal_weight_change": np.abs(diw).max(),
                "internal_weight_total_variation_distance": 0.5 * np.abs(diw).sum(),
                "critic_s_disc_weight": cw["S_Disc"],
                "critic_s_sat_weight": cw["S_Sat"],
                "critic_s_imp_weight": cw["S_Imp"],
                "critic_s_disc_weight_change": dcw["S_Disc"],
                "critic_s_sat_weight_change": dcw["S_Sat"],
                "critic_s_imp_weight_change": dcw["S_Imp"],
                "maximum_absolute_critic_percentage_point_drift": np.abs(dcw).max() * 100,
                "sum_absolute_critic_percentage_point_changes": np.abs(dcw).sum() * 100,
                "critic_weight_total_variation_distance": 0.5 * np.abs(dcw).sum(),
            }
        )
    return pd.DataFrame(rows)


def rank_metrics(condition: dict, reference: dict) -> dict:
    current = condition["final"]
    baseline = reference["final"]
    names = baseline.index.intersection(current.index)
    br = baseline.loc[names, "Rank"]
    cr = current.loc[names, "Rank"]
    rank_change = cr - br
    percentile_change = (
        current.loc[names, "Rank_Percentile"]
        - baseline.loc[names, "Rank_Percentile"]
    )
    row = {
        "condition": condition["label"],
        "reference_condition": reference["label"],
        "ranked_benchmark_count": len(names),
        "spearman_rho": float(spearmanr(br, cr).statistic),
        "kendall_tau_b": float(kendalltau(br, cr, variant="b").statistic),
    }
    for k in [10, 20]:
        bt = fixed_top(baseline, k)
        ct = fixed_top(current, k)
        be, b_tie = expanded_top(baseline, k)
        ce, c_tie = expanded_top(current, k)
        shared = len(bt & ct)
        expanded_shared = len(be & ce)
        row.update(
            {
                f"top_{k}_shared_count": shared,
                f"top_{k}_overlap_rate": shared / k,
                f"top_{k}_jaccard": shared / len(bt | ct),
                f"top_{k}_reference_cutoff_tie": b_tie,
                f"top_{k}_condition_cutoff_tie": c_tie,
                f"top_{k}_tie_aware_reference_size": len(be),
                f"top_{k}_tie_aware_condition_size": len(ce),
                f"top_{k}_tie_aware_shared_count": expanded_shared,
                f"top_{k}_tie_aware_jaccard": expanded_shared / len(be | ce),
            }
        )
    for prefix, values in [
        ("rank_change", np.abs(rank_change)),
        ("rank_percentile_change", np.abs(percentile_change)),
    ]:
        row.update(
            {
                f"{prefix}_mean_abs": float(values.mean()),
                f"{prefix}_median_abs": float(values.median()),
                f"{prefix}_p90_abs": float(values.quantile(0.9)),
                f"{prefix}_max_abs": float(values.max()),
            }
        )
    return row


def subset_metrics(condition: dict, reference: dict) -> dict:
    affected = [
        name for name in reference["affected"] if name in reference["final"].index
    ]
    current = condition["final"].loc[affected]
    baseline = reference["final"].loc[affected]
    subset_base_rank = baseline["BHI"].rank(method="average", ascending=False)
    subset_current_rank = current["BHI"].rank(method="average", ascending=False)
    subset_delta = subset_current_rank - subset_base_rank
    full_delta = current["Rank"] - baseline["Rank"]
    percentile_delta = current["Rank_Percentile"] - baseline["Rank_Percentile"]
    tolerance = 1e-12
    row = {
        "condition": condition["label"],
        "reference_condition": reference["label"],
        "affected_final_benchmark_count": len(affected),
        "subset_spearman_rho": float(
            spearmanr(subset_base_rank, subset_current_rank).statistic
        ),
        "subset_kendall_tau_b": float(
            kendalltau(subset_base_rank, subset_current_rank, variant="b").statistic
        ),
        "subset_rank_change_mean_abs": float(np.abs(subset_delta).mean()),
        "subset_rank_change_median_abs": float(np.median(np.abs(subset_delta))),
        "subset_rank_change_p90_abs": float(np.quantile(np.abs(subset_delta), 0.9)),
        "subset_rank_change_max_abs": float(np.abs(subset_delta).max()),
        "full_rank_change_mean_abs": float(np.abs(full_delta).mean()),
        "full_rank_change_median_abs": float(np.median(np.abs(full_delta))),
        "full_rank_percentile_change_mean_abs": float(np.abs(percentile_delta).mean()),
        "full_rank_percentile_change_median_abs": float(np.median(np.abs(percentile_delta))),
        "full_rank_percentile_change_max_abs": float(np.abs(percentile_delta).max()),
        "upward_count": int((percentile_delta > tolerance).sum()),
        "downward_count": int((percentile_delta < -tolerance).sum()),
        "unchanged_count": int((np.abs(percentile_delta) <= tolerance).sum()),
        "upward_proportion": float((percentile_delta > tolerance).mean()),
        "downward_proportion": float((percentile_delta < -tolerance).mean()),
        "unchanged_proportion": float((np.abs(percentile_delta) <= tolerance).mean()),
        "affected_in_full_top_10": len(set(affected) & fixed_top(condition["final"], 10)),
        "affected_in_full_top_20": len(set(affected) & fixed_top(condition["final"], 20)),
    }
    for requested_k in [10, 20]:
        k = min(requested_k, len(affected))
        base_set = fixed_top(baseline, k)
        condition_set = fixed_top(current, k)
        shared = len(base_set & condition_set)
        row.update(
            {
                f"subset_top_{requested_k}_actual_k": k,
                f"subset_top_{requested_k}_shared_count": shared,
                f"subset_top_{requested_k}_overlap_rate": shared / k,
                f"subset_top_{requested_k}_jaccard": shared / len(base_set | condition_set),
            }
        )
    return row


def distribution_metrics(
    condition_label: str,
    group: str,
    score: str,
    baseline: pd.Series,
    current: pd.Series,
    observation_unit: str = "benchmark",
    direct_change_scope: str = "global_recalibration_possible",
) -> dict:
    names = baseline.index.intersection(current.index)
    base = baseline.loc[names].astype(float)
    comp = current.loc[names].astype(float)
    delta = comp - base
    absolute = np.abs(delta)
    pearson = base.corr(comp, method="pearson")
    spearman = base.corr(comp, method="spearman")
    return {
        "condition": condition_label,
        "reference_condition": DEFAULT,
        "group": group,
        "score": score,
        "observation_unit": observation_unit,
        "direct_or_global_change": direct_change_scope,
        "observation_count": len(base),
        "pearson_correlation": None if pd.isna(pearson) else float(pearson),
        "spearman_correlation": None if pd.isna(spearman) else float(spearman),
        "mean_signed_change": float(delta.mean()),
        "mean_absolute_change": float(absolute.mean()),
        "median_absolute_change": float(absolute.median()),
        "root_mean_squared_change": float(np.sqrt(np.mean(delta**2))),
        "p90_absolute_change": float(absolute.quantile(0.9)),
        "max_absolute_change": float(absolute.max()),
        **{
            f"absolute_change_below_{str(threshold).replace('.', '_')}_count": int(
                (absolute < threshold).sum()
            )
            for threshold in [0.01, 0.025, 0.05]
        },
        **{
            f"absolute_change_below_{str(threshold).replace('.', '_')}_percentage": float(
                (absolute < threshold).mean() * 100
            )
            for threshold in [0.01, 0.025, 0.05]
        },
    }


def score_metric_rows(condition: dict, reference: dict) -> list[dict]:
    affected = set(reference["affected"]) & set(reference["final"].index)
    groups = {
        "all_final": list(reference["final"].index),
        "open_upper_bound_final": sorted(affected),
        "unaffected_final": sorted(set(reference["final"].index) - affected),
    }
    rows = []
    frames = {
        "Robust_CV": ("disc", "Robust_CV", "global_recalibration_possible"),
        "EDR": ("disc", "EDR", "expected_invariant"),
        "S_Disc": ("disc", "S_Disc", "global_recalibration_possible"),
        "S_Static": ("saturation", "S_Static", "direct_for_affected_only"),
        "S_Dynamic": ("saturation", "S_Dynamic", "direct_for_affected_only"),
        "S_Sat": ("saturation", "S_Sat", "direct_for_affected_only"),
        "S_Imp_Raw": ("impact", "S_Imp_Raw", "expected_invariant"),
        "S_Imp": ("impact", "S_Imp", "expected_invariant"),
        "BHI": ("final", "BHI", "global_recalibration_possible"),
    }
    for group, names in groups.items():
        for score, (frame_name, column, scope) in frames.items():
            base = reference[frame_name].loc[names, column]
            current = condition[frame_name].loc[names, column]
            rows.append(
                distribution_metrics(
                    condition["label"], group, score, base, current, direct_change_scope=scope
                )
            )

        base_cells = reference["sat_matrix"].loc[names].stack()
        current_cells = condition["sat_matrix"].loc[names].stack()
        base_cells.index = base_cells.index.map(lambda item: f"{item[0]}::{item[1]}")
        current_cells.index = current_cells.index.map(lambda item: f"{item[0]}::{item[1]}")
        rows.append(
            distribution_metrics(
                condition["label"],
                group,
                "Normalized_Score_Cell",
                base_cells,
                current_cells,
                observation_unit="benchmark_model_cell",
                direct_change_scope="direct_for_affected_only",
            )
        )
    return rows


def benchmark_rows(condition: dict, reference: dict) -> list[dict]:
    rows = []
    affected = set(reference["affected"])
    for benchmark, values in condition["final"].iterrows():
        raw = condition["raw_values"].loc[benchmark].dropna()
        default = reference["final"].loc[benchmark]
        disc = condition["disc"].loc[benchmark]
        rows.append(
            {
                "condition": condition["label"],
                "safety_factor": condition["factor"],
                "benchmark": benchmark,
                "open_upper_bound_flag": benchmark in affected,
                "observed_raw_maximum": float(raw.max()),
                "mapped_maximum_score": (
                    100 / condition["factor"]
                    if benchmark in affected
                    else float(condition["sat_matrix"].loc[benchmark].max())
                ),
                "Robust_CV": disc["Robust_CV"],
                "Normalized_Robust_CV": disc["Normalized_Robust_CV"],
                "EDR": disc["EDR"],
                "Normalized_EDR": disc["Normalized_EDR"],
                "S_Disc": values["S_Disc"],
                "S_Static": values["S_Static"],
                "S_Dynamic": values["S_Dynamic"],
                "S_Sat": values["S_Sat"],
                "S_Imp_Raw": values["S_Imp_Raw"],
                "S_Imp": values["S_Imp"],
                "BHI": values["BHI"],
                "rank": values["Rank"],
                "rank_percentile": values["Rank_Percentile"],
                "default_rank": default["Rank"],
                "rank_change": values["Rank"] - default["Rank"],
                "absolute_rank_change": abs(values["Rank"] - default["Rank"]),
                "default_rank_percentile": default["Rank_Percentile"],
                "rank_percentile_change": values["Rank_Percentile"]
                - default["Rank_Percentile"],
                "absolute_rank_percentile_change": abs(
                    values["Rank_Percentile"] - default["Rank_Percentile"]
                ),
                "BHI_change_from_default": values["BHI"] - default["BHI"],
                "S_Disc_change_from_default": values["S_Disc"] - default["S_Disc"],
                "S_Sat_change_from_default": values["S_Sat"] - default["S_Sat"],
                "S_Imp_change_from_default": values["S_Imp"] - default["S_Imp"],
            }
        )
    return rows


def top_mover_rows(condition: dict, reference: dict) -> list[dict]:
    if condition["label"] == DEFAULT:
        return []
    frame = condition["final"].copy()
    frame["rank_percentile_change"] = (
        frame["Rank_Percentile"] - reference["final"]["Rank_Percentile"]
    )
    for score in ["BHI", "S_Disc", "S_Sat"]:
        source = (
            condition["disc"] if score == "S_Disc" else
            condition["saturation"] if score == "S_Sat" else condition["final"]
        )
        baseline = (
            reference["disc"] if score == "S_Disc" else
            reference["saturation"] if score == "S_Sat" else reference["final"]
        )
        frame[f"{score}_change"] = source[score] - baseline[score]
    rows = []
    affected = set(reference["affected"]) & set(frame.index)
    groups = {"all_final": frame, "open_upper_bound_final": frame.loc[sorted(affected)]}
    categories = [
        ("rank_percentile_upward", "rank_percentile_change", False),
        ("rank_percentile_downward", "rank_percentile_change", True),
        ("absolute_BHI_change", "BHI_change", None),
        ("absolute_S_Disc_change", "S_Disc_change", None),
        ("absolute_S_Sat_change", "S_Sat_change", None),
    ]
    for group, subset in groups.items():
        for category, metric, ascending in categories:
            if group == "open_upper_bound_final" and category == "absolute_S_Disc_change":
                continue
            if ascending is None:
                ordered = subset.assign(_order=np.abs(subset[metric])).sort_values(
                    ["_order"], ascending=False, kind="mergesort"
                )
            elif category == "rank_percentile_upward":
                ordered = subset.loc[subset[metric] > TOLERANCE].sort_values(
                    metric, ascending=False, kind="mergesort"
                )
            elif category == "rank_percentile_downward":
                ordered = subset.loc[subset[metric] < -TOLERANCE].sort_values(
                    metric, ascending=True, kind="mergesort"
                )
            else:
                ordered = subset.sort_values(metric, ascending=ascending, kind="mergesort")
            for mover_rank, (benchmark, row) in enumerate(ordered.head(10).iterrows(), 1):
                rows.append(
                    {
                        "condition": condition["label"],
                        "group": group,
                        "mover_category": category,
                        "mover_rank": mover_rank,
                        "benchmark": benchmark,
                        "metric_value": row[metric],
                        "absolute_metric_value": abs(row[metric]),
                        "rank_percentile_change": row["rank_percentile_change"],
                        "BHI_change": row["BHI_change"],
                        "S_Disc_change": row["S_Disc_change"],
                        "S_Sat_change": row["S_Sat_change"],
                    }
                )
    return rows


def bootstrap_rows(
    condition: dict, reference: dict, rng: np.random.Generator
) -> list[dict]:
    if condition["label"] == DEFAULT:
        return []
    rows = []
    affected = set(reference["affected"]) & set(reference["final"].index)
    groups = {
        "full_leaderboard": list(reference["final"].index),
        "open_upper_bound_subset": sorted(affected),
    }
    for group, names in groups.items():
        base = reference["final"].loc[names]
        current = condition["final"].loc[names]
        arrays = {
            "spearman_rho": (base["Rank"].to_numpy(), current["Rank"].to_numpy()),
            "kendall_tau_b": (base["Rank"].to_numpy(), current["Rank"].to_numpy()),
            "median_absolute_rank_percentile_change": (
                base["Rank_Percentile"].to_numpy(),
                current["Rank_Percentile"].to_numpy(),
            ),
            "median_absolute_bhi_change": (
                base["BHI"].to_numpy(),
                current["BHI"].to_numpy(),
            ),
        }
        for metric, (left, right) in arrays.items():
            if metric == "spearman_rho":
                estimate = float(spearmanr(left, right).statistic)
            elif metric == "kendall_tau_b":
                estimate = float(kendalltau(left, right, variant="b").statistic)
            else:
                estimate = float(np.median(np.abs(right - left)))
            values = []
            skipped = 0
            for _ in range(BOOTSTRAP_ITERATIONS):
                sample = rng.integers(0, len(left), len(left))
                if metric == "spearman_rho":
                    value = spearmanr(left[sample], right[sample]).statistic
                elif metric == "kendall_tau_b":
                    value = kendalltau(
                        left[sample], right[sample], variant="b"
                    ).statistic
                else:
                    value = np.median(np.abs(right[sample] - left[sample]))
                if np.isfinite(value):
                    values.append(float(value))
                else:
                    skipped += 1
            rows.append(
                {
                    "condition": condition["label"],
                    "reference_condition": DEFAULT,
                    "group": group,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_95_lower": float(np.quantile(values, 0.025)) if values else None,
                    "ci_95_upper": float(np.quantile(values, 0.975)) if values else None,
                    "requested_resamples": BOOTSTRAP_ITERATIONS,
                    "valid_resamples": len(values),
                    "skipped_resamples": skipped,
                    "seed": BOOTSTRAP_SEED,
                }
            )
    return rows


def invariant_rows(conditions: dict[str, dict]) -> pd.DataFrame:
    reference = conditions[DEFAULT]
    rows = []

    def add(condition: str, invariant: str, observed: float, expected: str, passed: bool, diagnosis: str) -> None:
        rows.append(
            {
                "condition": condition,
                "invariant": invariant,
                "observed_value": observed,
                "expected": expected,
                "passed": bool(passed),
                "diagnosis": diagnosis,
            }
        )

    for label, _ in CONDITIONS:
        condition = conditions[label]
        same_models = condition["master"].model_columns == reference["master"].model_columns
        same_benchmarks = list(condition["master"].benchmark_names) == list(reference["master"].benchmark_names)
        same_disc_eligibility = condition["disc_counts"].equals(reference["disc_counts"])
        same_sat_eligibility = condition["sat_counts"].equals(reference["sat_counts"])
        same_impact_eligibility = condition["impact_counts"].equals(reference["impact_counts"])
        same_availability = condition["master"].availability.equals(reference["master"].availability)
        add(label, "input_model_set_and_order_identical", float(not same_models), "0 mismatches", same_models, "")
        add(label, "input_benchmark_set_and_order_identical", float(not same_benchmarks), "0 mismatches", same_benchmarks, "")
        add(label, "disc_eligibility_identical", float(not same_disc_eligibility), "0 mismatches", same_disc_eligibility, "")
        add(label, "sat_eligibility_identical", float(not same_sat_eligibility), "0 mismatches", same_sat_eligibility, "")
        add(label, "impact_eligibility_identical", float(not same_impact_eligibility), "0 mismatches", same_impact_eligibility, "")
        add(label, "availability_matrix_identical", float(not same_availability), "0 mismatches", same_availability, "")

        cap_max = 0.0
        for benchmark in reference["capabilities"]:
            cap_max = max(
                cap_max,
                float(
                    (
                        condition["capabilities"][benchmark]
                        - reference["capabilities"][benchmark]
                    )
                    .abs()
                    .max()
                ),
            )
        add(label, "lobo_capability_scores_invariant", cap_max, "<= 1e-12", cap_max <= TOLERANCE, "Positive within-benchmark scaling preserves pairwise order.")
        for score, frame_name in [("S_Imp_Raw", "impact"), ("S_Imp", "impact")]:
            difference = float(
                (condition[frame_name][score] - reference[frame_name][score]).abs().max()
            )
            add(label, f"{score}_invariant", difference, "<= 1e-12", difference <= TOLERANCE, "LOBO and availability are unchanged.")

        unaffected = sorted(
            set(reference["master"].benchmark_names) - set(reference["affected"])
        )
        for matrix_name in ["disc_matrix", "sat_matrix"]:
            difference = float(
                (
                    condition[matrix_name].loc[
                        condition[matrix_name].index.intersection(unaffected)
                    ]
                    - reference[matrix_name].loc[
                        reference[matrix_name].index.intersection(unaffected)
                    ]
                )
                .abs()
                .max()
                .max()
            )
            add(label, f"unaffected_{matrix_name}_invariant", difference, "<= 1e-12", difference <= TOLERANCE, "")

        affected_final = sorted(set(reference["affected"]) & set(reference["final"].index))
        unaffected_final = sorted(set(reference["final"].index) - set(affected_final))
        for score in ["S_Static", "S_Dynamic", "S_Sat"]:
            direct = float(
                (
                    condition["saturation"].loc[unaffected_final, score]
                    - reference["saturation"].loc[unaffected_final, score]
                )
                .abs()
                .max()
            )
            add(label, f"unaffected_{score}_invariant", direct, "<= 1e-12", direct <= TOLERANCE, "Direct saturation changes are confined to the open-upper-bound subset.")
    frame = pd.DataFrame(rows)
    if not frame["passed"].all():
        raise RuntimeError(
            "Expected invariant failed:\n"
            + frame.loc[~frame["passed"]].to_string(index=False)
        )
    return frame


def open_upper_rows(reference: dict) -> pd.DataFrame:
    rows = []
    final_names = set(reference["final"].index)
    for benchmark in reference["affected"]:
        raw = reference["raw_values"].loc[benchmark].dropna()
        rows.append(
            {
                "benchmark": benchmark,
                "detected_tag_or_rule": (
                    "CONFIG['tags']['unbounded_hib'] == '(no limited)'; "
                    "case-insensitive match after lower-better branches"
                ),
                "observed_raw_maximum": float(raw.max()),
                "observed_raw_minimum": float(raw.min()),
                "valid_model_score_count": len(raw),
                "default_normalized_maximum_under_factor_1_25": 80.0,
                "reaches_final_bhi_table": benchmark in final_names,
            }
        )
    return pd.DataFrame(rows)


def trend_summary(conditions: dict[str, dict]) -> dict:
    labels = [label for label, _ in CONDITIONS]
    affected = sorted(set(conditions[DEFAULT]["affected"]) & set(conditions[DEFAULT]["final"].index))
    counts = {
        "affected_final_benchmark_count": len(affected),
        "monotonic_s_static_count": 0,
        "monotonic_s_sat_count": 0,
        "monotonic_bhi_count": 0,
        "nonmonotonic_rank_count": 0,
    }
    details = []

    def monotonic(values: list[float]) -> bool:
        differences = np.diff(values)
        return bool(np.all(differences >= -TOLERANCE) or np.all(differences <= TOLERANCE))

    for benchmark in affected:
        series = {
            "mapped_maximum_score": [conditions[label]["mapped_maximum"] for label in labels],
            "S_Static": [conditions[label]["final"].loc[benchmark, "S_Static"] for label in labels],
            "S_Dynamic": [conditions[label]["final"].loc[benchmark, "S_Dynamic"] for label in labels],
            "S_Sat": [conditions[label]["final"].loc[benchmark, "S_Sat"] for label in labels],
            "S_Disc": [conditions[label]["final"].loc[benchmark, "S_Disc"] for label in labels],
            "BHI": [conditions[label]["final"].loc[benchmark, "BHI"] for label in labels],
            "Rank": [conditions[label]["final"].loc[benchmark, "Rank"] for label in labels],
            "Rank_Percentile": [conditions[label]["final"].loc[benchmark, "Rank_Percentile"] for label in labels],
        }
        static_mono = monotonic(series["S_Static"])
        sat_mono = monotonic(series["S_Sat"])
        bhi_mono = monotonic(series["BHI"])
        rank_nonmono = not monotonic(series["Rank"])
        counts["monotonic_s_static_count"] += int(static_mono)
        counts["monotonic_s_sat_count"] += int(sat_mono)
        counts["monotonic_bhi_count"] += int(bhi_mono)
        counts["nonmonotonic_rank_count"] += int(rank_nonmono)
        details.append(
            {
                "benchmark": benchmark,
                "monotonic_s_static": static_mono,
                "monotonic_s_sat": sat_mono,
                "monotonic_bhi": bhi_mono,
                "nonmonotonic_rank": rank_nonmono,
                **series,
            }
        )
    return {**counts, "benchmark_details": details}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(lambda x: f"{x:.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def write_report(
    weights: pd.DataFrame,
    ranking: pd.DataFrame,
    subset: pd.DataFrame,
    scores: pd.DataFrame,
    reproduction: pd.DataFrame,
    invariants: pd.DataFrame,
    movers: pd.DataFrame,
    bootstrap: pd.DataFrame,
    open_upper: pd.DataFrame,
    trends: dict,
) -> None:
    score_focus = scores[
        (scores["condition"] != DEFAULT)
        & scores["score"].isin(["Normalized_Score_Cell", "S_Disc", "S_Sat", "S_Imp", "BHI"])
    ][
        [
            "condition",
            "group",
            "score",
            "mean_absolute_change",
            "median_absolute_change",
            "max_absolute_change",
        ]
    ]
    largest = ranking.loc[
        ranking["rank_percentile_change_mean_abs"].idxmax(), "condition"
    ]
    lead_movers = movers[
        (movers["condition"] == largest)
        & (movers["group"] == "open_upper_bound_final")
        & movers["mover_category"].isin(
            ["rank_percentile_upward", "rank_percentile_downward", "absolute_BHI_change"]
        )
        & (movers["mover_rank"] <= 5)
    ]
    text = f"""# Open-Upper-Bound Safety-Factor Sensitivity Analysis

## Experimental purpose

This experiment changes only `CONFIG["safety_factor"]` for the open-upper-bound normalization branch. Factor 1.25 is the default transparent buffer choice, not a claimed unique optimum.

## Formula and conditions

`normalized_score = raw_score / (observed_maximum * safety_factor) * 100`.

{markdown_table(weights, ["condition", "safety_factor", "mapped_observed_maximum_score", "implied_remaining_headroom"])}

## Date-parser behavior

The analysis applies `pd.to_datetime(raw_model_dates, errors="coerce")` as recorded in `{ORIGINAL_PREPROCESSOR}` to the supplied model dates.

## Default reproduction

{markdown_table(reproduction, ["check", "passed", "observed_max_abs_difference", "tolerance"], 12)}

## Open-upper-bound subset

The exact open-upper-bound branch identifies {len(open_upper)} rows; {int(open_upper["reaches_final_bhi_table"].sum())} reach the final BHI table.

{markdown_table(open_upper, ["benchmark", "observed_raw_maximum", "observed_raw_minimum", "valid_model_score_count", "reaches_final_bhi_table"])}

## Internal and final weights

{markdown_table(weights, ["condition", "internal_robust_cv_weight", "internal_edr_weight", "critic_s_disc_weight", "critic_s_sat_weight", "critic_s_imp_weight", "maximum_absolute_critic_percentage_point_drift"])}

## Full-leaderboard stability

{markdown_table(ranking, ["condition", "spearman_rho", "kendall_tau_b", "top_10_shared_count", "top_20_shared_count", "rank_change_median_abs", "rank_percentile_change_median_abs", "rank_percentile_change_max_abs"])}

## Affected-subset stability

{markdown_table(subset, ["condition", "affected_final_benchmark_count", "subset_spearman_rho", "subset_kendall_tau_b", "full_rank_change_mean_abs", "full_rank_percentile_change_median_abs", "full_rank_percentile_change_max_abs", "upward_count", "downward_count", "unchanged_count"])}

## Score stability

{markdown_table(score_focus, list(score_focus.columns))}

The normalized-score metric uses matched benchmark-model cells. All other rows use benchmark observations. Direct saturation-score changes should be confined to the affected subset; S_Disc and BHI can move globally through recalibration.

## Invariant checks

All {len(invariants)} invariant checks passed. Maximum observed LOBO capability and S_Imp differences are recorded in `invariant_checks.csv`.

## Top movers for largest mean rank-percentile deviation ({largest})

{markdown_table(lead_movers, ["group", "mover_category", "mover_rank", "benchmark", "metric_value", "BHI_change", "S_Sat_change"])}

## Bootstrap confidence intervals

{markdown_table(bootstrap, ["condition", "group", "metric", "estimate", "ci_95_lower", "ci_95_upper", "valid_resamples", "skipped_resamples"])}

## Trend analysis

- Affected final benchmarks with monotonic S_Static: {trends["monotonic_s_static_count"]}/{trends["affected_final_benchmark_count"]}.
- Affected final benchmarks with monotonic S_Sat: {trends["monotonic_s_sat_count"]}/{trends["affected_final_benchmark_count"]}.
- Affected final benchmarks with monotonic BHI: {trends["monotonic_bhi_count"]}/{trends["affected_final_benchmark_count"]}.
- Affected final benchmarks with non-monotonic rank movement: {trends["nonmonotonic_rank_count"]}/{trends["affected_final_benchmark_count"]}.

## Warnings and interpretation limits

- Factor 1.25 maps the observed maximum to 80 and leaves 20 nominal points of headroom; this is transparent, not uniquely optimal.
- The experiment applies to the supplied baseline and exact `(no limited)` branch only.
- Global normalization and CRITIC recalibration can move unaffected benchmarks in the final ranking.
- Raw BHI is not assumed invariant, and localized movers are reported without causal speculation.

## Summary

The full tables report all tested factors without selection. The broad-ranking conclusion should be based on the minimum correlations, Top-k retention, affected-subset movement, score changes, and weight drift reported above, while acknowledging the largest observed condition and localized changes.
"""
    (OUTPUT_DIR / "safety_factor_sensitivity_report.md").write_text(
        text, encoding="utf-8"
    )


def write_analysis_record(
    weights: pd.DataFrame,
    ranking: pd.DataFrame,
    subset: pd.DataFrame,
    reproduction: pd.DataFrame,
    open_upper: pd.DataFrame,
    source_hashes: dict,
) -> None:
    text = f"""# Permanent Record: Open-Upper-Bound Safety-Factor Sensitivity

- Analysis date: 2026-07-26
- Analysis timestamp UTC: {TIMESTAMP}
- Script: `scripts/safety_factor_sensitivity.py`
- Commands: `python -m compileall scripts/safety_factor_sensitivity.py`; `python scripts/safety_factor_sensitivity.py`
- Input: `{display_path(INPUT_PATH)}`
- Baseline directory: `{display_path(BASE_DIR)}`
- Conditions: {", ".join(f"`{label}`={factor:.2f}" for label, factor in CONDITIONS)}
- Default: `{DEFAULT}`
- Formula: `raw_score / (observed_maximum * safety_factor) * 100`
- Bootstrap: {BOOTSTRAP_ITERATIONS:,} matched-pair resamples, seed {BOOTSTRAP_SEED}

## Date-parser behavior

The standalone script uses the `pd.to_datetime(raw_model_dates, errors="coerce")` behavior documented in `{ORIGINAL_PREPROCESSOR}`.

## Open-upper-bound benchmarks

Count: {len(open_upper)} total, {int(open_upper["reaches_final_bhi_table"].sum())} final.

{", ".join(open_upper["benchmark"])}

## Default reproduction

{markdown_table(reproduction, ["check", "passed", "observed_max_abs_difference"], 12)}

## Weights and drift

{markdown_table(weights, ["condition", "internal_robust_cv_weight", "internal_edr_weight", "critic_s_disc_weight", "critic_s_sat_weight", "critic_s_imp_weight", "internal_weight_total_variation_distance", "critic_weight_total_variation_distance"], 12)}

## Primary full ranking metrics

{markdown_table(ranking, ["condition", "spearman_rho", "kendall_tau_b", "top_10_shared_count", "top_20_shared_count", "rank_percentile_change_median_abs", "rank_percentile_change_max_abs"], 12)}

## Primary subset metrics

{markdown_table(subset, ["condition", "subset_spearman_rho", "subset_kendall_tau_b", "full_rank_percentile_change_median_abs", "full_rank_percentile_change_max_abs"], 12)}

## Direct source SHA-256

""" + "\n".join(f"- `{path}`: `{digest}`" for path, digest in source_hashes.items()) + """

## Warnings

- The analysis tests sensitivity; it does not prove 1.25 uniquely correct.
- Factor 1.25 is interpretable as mapping the current maximum to 80.
- Global recalibration permits movements outside the directly affected subset.
- This is a single-baseline sensitivity analysis, not a temporal comparison.
"""
    (OUTPUT_DIR / "ANALYSIS_RECORD.md").write_text(text, encoding="utf-8")


def write_manifest(generated: list[Path]) -> None:
    sources = [
        INPUT_PATH,
        BASE_DIR / "final_bhi_rankings.csv",
        BASE_DIR / "capability_discrimination_final.csv",
        BASE_DIR / "anti_saturation_results.csv",
        BASE_DIR / "impact_results.csv",
        BASE_DIR / "benchmark_inclusion_report.csv",
        ROOT / "main.py",
        ROOT / "config" / "settings.py",
        ROOT / "src" / "preprocessor.py",
        ROOT / "src" / "capability.py",
        ROOT / "README.md",
    ]
    entries = []
    for path, kind in [(Path(__file__).resolve(), "generated"), *[(p, "source") for p in sources], *[(p, "generated") for p in generated]]:
        entries.append(
            {
                "relative_path": display_path(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "generated_or_source": kind,
                "analysis_timestamp_utc": TIMESTAMP,
            }
        )
    manifest = OUTPUT_DIR / "artifact_manifest_sha256.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(sorted(entries, key=lambda row: row["relative_path"]))
    digest = sha256_file(manifest)
    (OUTPUT_DIR / "MANIFEST_SHA256.txt").write_text(
        f"artifact_manifest_sha256.csv  {digest}\n", encoding="utf-8"
    )


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {OUTPUT_DIR}")
    if not ORIGINAL_PREPROCESSOR.is_file():
        raise FileNotFoundError(
            f"Parser reference file not found: {ORIGINAL_PREPROCESSOR}"
        )
    original_text = ORIGINAL_PREPROCESSOR.read_text(encoding="utf-8")
    if "pd.to_datetime(df.loc[release_mask, model_cols].iloc[0], errors='coerce')" not in original_text:
        raise RuntimeError("Parser reference does not contain the required date call")

    before_hashes = {
        display_path(path): sha256_file(path)
        for path in integrity_paths()
    }
    conditions = {
        label: calculate_condition(label, factor) for label, factor in CONDITIONS
    }
    reference = conditions[DEFAULT]
    reproduction = reproduction_check(reference)
    invariants = invariant_rows(conditions)
    weights = condition_weight_rows(conditions)
    ranking = pd.DataFrame(
        [rank_metrics(conditions[label], reference) for label, _ in CONDITIONS]
    )
    subset = pd.DataFrame(
        [subset_metrics(conditions[label], reference) for label, _ in CONDITIONS]
    )
    scores = pd.DataFrame(
        [
            row
            for label, _ in CONDITIONS
            for row in score_metric_rows(conditions[label], reference)
        ]
    )
    benchmark_level = pd.DataFrame(
        [
            row
            for label, _ in CONDITIONS
            for row in benchmark_rows(conditions[label], reference)
        ]
    )
    movers = pd.DataFrame(
        [
            row
            for label, _ in CONDITIONS
            for row in top_mover_rows(conditions[label], reference)
        ]
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = pd.DataFrame(
        [
            row
            for label, _ in CONDITIONS
            for row in bootstrap_rows(conditions[label], reference, rng)
        ]
    )
    open_upper = open_upper_rows(reference)
    trends = trend_summary(conditions)

    OUTPUT_DIR.mkdir(parents=True)
    frames = {
        "safety_factor_condition_weights.csv": weights,
        "safety_factor_ranking_stability_metrics.csv": ranking,
        "safety_factor_subset_stability_metrics.csv": subset,
        "safety_factor_score_stability_metrics.csv": scores,
        "safety_factor_benchmark_level_results.csv": benchmark_level,
        "safety_factor_top_movers.csv": movers,
        "safety_factor_bootstrap_confidence_intervals.csv": bootstrap,
        "open_upper_bound_benchmarks.csv": open_upper,
        "default_reproduction_check.csv": reproduction,
        "invariant_checks.csv": invariants,
    }
    for filename, frame in frames.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    source_paths = [
        INPUT_PATH,
        BASE_DIR / "final_bhi_rankings.csv",
        BASE_DIR / "capability_discrimination_final.csv",
        BASE_DIR / "anti_saturation_results.csv",
        BASE_DIR / "impact_results.csv",
        BASE_DIR / "benchmark_inclusion_report.csv",
        ROOT / "main.py",
        ROOT / "config" / "settings.py",
        ROOT / "src" / "preprocessor.py",
        ROOT / "src" / "capability.py",
        ROOT / "README.md",
        ORIGINAL_PREPROCESSOR,
    ]
    source_hashes = {
        (
            display_path(path)
        ): sha256_file(path)
        for path in source_paths
    }
    summary = {
        "analysis_date": "2026-07-26",
        "analysis_timestamp_utc": TIMESTAMP,
        "input": display_path(INPUT_PATH),
        "baseline_outputs": [
            f"data/output/{name}"
            for name in [
                "final_bhi_rankings.csv",
                "capability_discrimination_final.csv",
                "anti_saturation_results.csv",
                "impact_results.csv",
                "benchmark_inclusion_report.csv",
            ]
        ],
        "date_parser_method": (
            "pd.to_datetime(raw_model_dates, errors='coerce') verified from "
            + str(ORIGINAL_PREPROCESSOR)
        ),
        "parsed_date_count": int(
            reference["master"].model_dates.notna().sum()
        ),
        "conditions": weights.to_dict("records"),
        "open_upper_bound_benchmarks": open_upper.to_dict("records"),
        "default_reproduction": reproduction.to_dict("records"),
        "invariant_checks": invariants.to_dict("records"),
        "full_ranking_stability": ranking.to_dict("records"),
        "subset_stability": subset.to_dict("records"),
        "score_stability": scores.to_dict("records"),
        "bootstrap_confidence_intervals": bootstrap.to_dict("records"),
        "trend_analysis": trends,
        "source_hashes": source_hashes,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (OUTPUT_DIR / "safety_factor_sensitivity_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_report(
        weights,
        ranking,
        subset,
        scores,
        reproduction,
        invariants,
        movers,
        bootstrap,
        open_upper,
        trends,
    )
    write_analysis_record(
        weights, ranking, subset, reproduction, open_upper, source_hashes
    )

    generated = sorted(
        p
        for p in OUTPUT_DIR.iterdir()
        if p.is_file()
        and p.name not in {"artifact_manifest_sha256.csv", "MANIFEST_SHA256.txt"}
    )
    write_manifest(generated)

    after_hashes = {
        display_path(path): sha256_file(path)
        for path in integrity_paths()
    }
    if before_hashes != after_hashes:
        changed = [
            name
            for name in before_hashes
            if before_hashes[name] != after_hashes.get(name)
        ]
        raise RuntimeError(f"Pre-existing file integrity failure: {changed}")

    expected_names = set(frames) | {
        "safety_factor_sensitivity_report.md",
        "safety_factor_sensitivity_summary.json",
        "ANALYSIS_RECORD.md",
        "artifact_manifest_sha256.csv",
        "MANIFEST_SHA256.txt",
    }
    actual_names = {p.name for p in OUTPUT_DIR.iterdir() if p.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(f"Unexpected output contents: {sorted(actual_names)}")
    for path in OUTPUT_DIR.iterdir():
        if path.suffix == ".csv":
            pd.read_csv(path)
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))

    if len(benchmark_level) != len(CONDITIONS) * len(reference["final"]):
        raise RuntimeError("Unexpected benchmark-level row count")
    if not (
        benchmark_level.groupby(["condition", "benchmark"]).size() == 1
    ).all():
        raise RuntimeError("Benchmark-condition rows are not unique")
    if not np.allclose(
        weights["internal_robust_cv_weight"] + weights["internal_edr_weight"],
        1,
        atol=TOLERANCE,
    ):
        raise RuntimeError("Internal weights fail unit-sum validation")
    if not np.allclose(
        weights[
            ["critic_s_disc_weight", "critic_s_sat_weight", "critic_s_imp_weight"]
        ].sum(axis=1),
        1,
        atol=TOLERANCE,
    ):
        raise RuntimeError("Final weights fail unit-sum validation")

    print(f"Wrote safety-factor sensitivity outputs to {OUTPUT_DIR}")
    print("Pre-existing files unchanged:", before_hashes == after_hashes)
    print("Open-upper-bound rows/final:", len(open_upper), int(open_upper["reaches_final_bhi_table"].sum()))
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
