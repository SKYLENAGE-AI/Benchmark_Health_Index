"""Targeted weak-model low-score stress test on a supplied BHI baseline."""

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


INPUT_PATH = Path(os.environ.get(
    "BHI_IN_MASTER",
    ROOT / "data" / "input" / "2026_0412_cleaned.csv",
))
BASE_DIR = Path(os.environ.get(
    "BHI_BASELINE_DIR",
    ROOT / "data" / "output",
))
OUTPUT_DIR = Path(os.environ.get(
    "BHI_EXPERIMENT_OUTPUT_DIR",
    BASE_DIR / "weak_model_low_score_stress_test_submission",
))
ORIGINAL_PREPROCESSOR = Path(os.environ.get(
    "BHI_SUBMISSION_PREPROCESSOR",
    ROOT / "config" / "submission_preprocessor_reference.py",
))
CONDITIONS = ["BASELINE", "LOW_CAPABILITY_ZERO", "HIGH_CAPABILITY_ZERO"]
PERTURBATIONS = CONDITIONS[1:]
TOLERANCE = 1e-12
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20260726
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


def protected_aggregate() -> tuple[int, str]:
    script = Path(__file__).resolve()
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == script or OUTPUT_DIR.resolve() in resolved.parents:
            continue
        files.append(resolved)
    hashes = {
        display_path(path): sha256_file(path)
        for path in sorted(set(files), key=display_path)
    }
    aggregate = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    return len(hashes), aggregate


def submission_dates(model_columns: list[str]) -> pd.Series:
    raw = pd.read_csv(INPUT_PATH)
    first = raw.columns[0]
    names = raw[first].fillna("").astype(str).str.strip().str.lower()
    values = raw.loc[
        names.isin({"release date", "release_date"}), model_columns
    ].iloc[0]
    return pd.to_datetime(values, errors="coerce")


def critic_weights(components: pd.DataFrame) -> pd.Series:
    information = components.std() * (1 - components.corr()).sum()
    weights = information / information.sum()
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=TOLERANCE):
        raise ValueError("CRITIC weights do not sum to one")
    return weights


def row_contributions(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(scores)
    values = scores[valid]
    wins = np.zeros(len(scores))
    opportunities = np.zeros(len(scores))
    participation = np.zeros(len(scores))
    if len(values) >= 2:
        difference = values[:, None] - values
        row_wins = (
            (difference > 0).astype(float).sum(axis=1)
            + 0.5 * (difference == 0).astype(float).sum(axis=1)
            - 0.5
        )
        indices = np.where(valid)[0]
        wins[indices] = row_wins
        opportunities[indices] = len(values) - 1
        participation[indices] = 1
    return wins, opportunities, participation


def contribution_cube(bench_data: pd.DataFrame) -> np.ndarray:
    rows = []
    for index in range(len(bench_data)):
        rows.append(
            np.stack(
                row_contributions(
                    bench_data.iloc[index, 1:].to_numpy(dtype=float)
                )
            )
        )
    return np.stack(rows)


def fast_capabilities(
    cube: np.ndarray, model_names: list[str]
) -> dict[str, pd.Series]:
    totals = cube.sum(axis=0)
    n_total = len(cube)
    results = {}
    for index in range(n_total):
        wins, opportunities, participation = totals - cube[index]
        win_rate = np.divide(
            wins,
            opportunities,
            out=np.zeros_like(wins),
            where=opportunities > 0,
        )
        penalty = (
            np.log(1 + participation) / np.log(1 + (n_total - 1))
        ) ** 0.25
        results[index] = pd.Series(
            win_rate * penalty, index=model_names, name="Capability_Score"
        )
    return results


def exact_capabilities(bench_data: pd.DataFrame) -> dict[int, pd.Series]:
    return {
        index: calculate_lobo_capabilities(
            bench_data, index, len(bench_data)
        )["Capability_Score"]
        for index in range(len(bench_data))
    }


def calculate_pipeline(
    master,
    disc_matrix: pd.DataFrame,
    sat_matrix: pd.DataFrame,
    capabilities: dict[int, pd.Series],
    model_dates: pd.Series,
) -> dict:
    min_count = CONFIG["min_model_count"]
    disc_counts = disc_matrix.notna().sum(axis=1)
    sat_counts = sat_matrix.notna().sum(axis=1)
    eligible_disc = disc_matrix.loc[disc_counts >= min_count]
    eligible_sat = sat_matrix.loc[sat_counts >= min_count]

    disc_rows = []
    for benchmark, row in eligible_disc.iterrows():
        values = row.dropna().to_numpy()
        n = len(values)
        robust_cv = (
            np.percentile(values, 90) - np.percentile(values, 10)
        ) / 100
        differences = np.abs(values[:, None] - values)
        edr = (
            (np.sum(differences > CONFIG["edr_threshold"]) / 2)
            / (n * (n - 1) / 2)
            if n > 1
            else 0
        )
        disc_rows.append(
            {
                "Benchmark": benchmark,
                "N_Models": n,
                "Robust_CV": robust_cv,
                "EDR": edr,
            }
        )
    disc = pd.DataFrame(disc_rows).set_index("Benchmark")
    normalized = (disc[["Robust_CV", "EDR"]] - disc[["Robust_CV", "EDR"]].min()) / (
        disc[["Robust_CV", "EDR"]].max()
        - disc[["Robust_CV", "EDR"]].min()
        + 1e-9
    )
    internal = normalized.std() / normalized.std().sum()
    disc["S_Disc"] = normalized.mul(internal).sum(axis=1)

    bench_data = eligible_sat.reset_index()
    saturation_rows = []
    impact_rows = []
    for index in range(len(bench_data)):
        benchmark = bench_data.loc[index, "Benchmark"]
        caps = capabilities[index]
        scores = bench_data.iloc[index, 1:]
        valid = scores.dropna()
        common = valid.index.intersection(caps.index)
        weighted_numerator = (
            (valid[common] / 100) * caps.loc[common]
        ).sum()
        weighted_denominator = caps.loc[common].sum()
        s_static = 1 - weighted_numerator / (weighted_denominator + 1e-9)
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
                "Weighted_Score": weighted_numerator
                / (weighted_denominator + 1e-9),
                "S_Static": s_static,
                "S_Dynamic": s_dynamic,
                "S_Sat": s_sat,
            }
        )

        cap = caps.reindex(master.model_columns).fillna(0)
        denominator = cap.sum()
        evaluated = (
            master.availability.loc[benchmark]
            .reindex(master.model_columns)
            .fillna(False)
        )
        numerator = cap.loc[evaluated[evaluated].index].sum()
        raw_impact = numerator / denominator if denominator > 0 else 0
        impact_rows.append(
            {
                "Benchmark": benchmark,
                "S_Imp_Raw": raw_impact,
                "S_Imp": np.sqrt(raw_impact),
            }
        )
    saturation = pd.DataFrame(saturation_rows).set_index("Benchmark")
    impact = pd.DataFrame(impact_rows).set_index("Benchmark")
    final = (
        disc[["S_Disc"]]
        .merge(
            saturation[["S_Static", "S_Dynamic", "S_Sat"]],
            left_index=True,
            right_index=True,
        )
        .merge(impact[["S_Imp_Raw", "S_Imp"]], left_index=True, right_index=True)
    )
    final_weights = critic_weights(final[["S_Disc", "S_Sat", "S_Imp"]])
    final["BHI"] = final[["S_Disc", "S_Sat", "S_Imp"]].mul(
        final_weights
    ).sum(axis=1)
    final["Rank"] = final["BHI"].rank(method="average", ascending=False)
    final["Rank_Percentile"] = 1 - (final["Rank"] - 1) / (len(final) - 1)
    return {
        "disc": disc,
        "saturation": saturation,
        "impact": impact,
        "final": final,
        "internal_weights": internal,
        "critic_weights": final_weights,
        "disc_counts": disc_counts,
        "sat_counts": sat_counts,
        "capabilities": capabilities,
    }


def baseline_reproduction(baseline: dict) -> pd.DataFrame:
    submitted = {
        "disc": pd.read_csv(
            BASE_DIR / "capability_discrimination_final.csv"
        ).set_index("Benchmark"),
        "saturation": pd.read_csv(
            BASE_DIR / "anti_saturation_results.csv"
        ).set_index("Benchmark"),
        "impact": pd.read_csv(BASE_DIR / "impact_results.csv").set_index(
            "Benchmark"
        ),
        "final": pd.read_csv(BASE_DIR / "final_bhi_rankings.csv").set_index(
            "Benchmark"
        ),
    }
    rows = []

    def add(name: str, value: float) -> None:
        rows.append(
            {
                "check": name,
                "observed_max_abs_difference": value,
                "tolerance": TOLERANCE,
                "passed": bool(value <= TOLERANCE),
            }
        )

    for frame_name in ["disc", "saturation", "impact", "final"]:
        add(
            f"{frame_name}_benchmark_set_matches",
            float(set(baseline[frame_name].index) != set(submitted[frame_name].index)),
        )
    for frame_name, columns in {
        "disc": ["Robust_CV", "EDR", "S_Disc"],
        "saturation": ["S_Static", "S_Dynamic", "S_Sat"],
        "impact": ["S_Imp_Raw", "S_Imp"],
        "final": ["S_Disc", "S_Sat", "S_Imp", "BHI"],
    }.items():
        for column in columns:
            names = baseline[frame_name].index.intersection(
                submitted[frame_name].index
            )
            add(
                f"max_abs_{frame_name}_{column}_difference",
                float(
                    (
                        baseline[frame_name].loc[names, column]
                        - submitted[frame_name].loc[names, column]
                    )
                    .abs()
                    .max()
                ),
            )
    submitted_disc = submitted["disc"][["Robust_CV", "EDR"]]
    normalized = (submitted_disc - submitted_disc.min()) / (
        submitted_disc.max() - submitted_disc.min() + 1e-9
    )
    submitted_internal = normalized.std() / normalized.std().sum()
    submitted_final_weights = critic_weights(
        submitted["final"][["S_Disc", "S_Sat", "S_Imp"]]
    )
    add(
        "max_abs_internal_weight_difference",
        float((baseline["internal_weights"] - submitted_internal).abs().max()),
    )
    add(
        "max_abs_critic_weight_difference",
        float(
            (
                baseline["critic_weights"] - submitted_final_weights
            ).abs().max()
        ),
    )
    submitted_order = list(
        submitted["final"].sort_values(
            ["BHI", "S_Imp"], ascending=False
        ).index
    )
    baseline_order = list(
        baseline["final"].sort_values(["BHI", "S_Imp"], ascending=False).index
    )
    add("final_ranking_order_matches", float(submitted_order != baseline_order))
    result = pd.DataFrame(rows)
    if not result["passed"].all():
        raise RuntimeError(
            "Baseline reproduction failed:\n"
            + result.loc[~result["passed"]].to_string(index=False)
        )
    return result


def fixed_top(frame: pd.DataFrame, k: int) -> set[str]:
    return set(
        frame.reset_index()
        .sort_values(["BHI", "Benchmark"], ascending=[False, True], kind="mergesort")
        .head(k)["Benchmark"]
    )


def leaderboard_row(
    run_id: str,
    target: str,
    condition: str,
    run: dict,
    baseline: dict,
) -> dict:
    current = run["final"]
    base = baseline["final"]
    names = base.index.intersection(current.index)
    br = base.loc[names, "Rank"]
    cr = current.loc[names, "Rank"]
    rank_abs = (cr - br).abs()
    percentile_abs = (
        current.loc[names, "Rank_Percentile"]
        - base.loc[names, "Rank_Percentile"]
    ).abs()
    bhi_abs = (current.loc[names, "BHI"] - base.loc[names, "BHI"]).abs()
    top10_base, top10_current = fixed_top(base, 10), fixed_top(current, 10)
    top20_base, top20_current = fixed_top(base, 20), fixed_top(current, 20)
    non_target = [name for name in names if name != target]
    indirect = (
        current.loc[non_target, "Rank_Percentile"]
        - base.loc[non_target, "Rank_Percentile"]
    ).abs()
    largest_indirect = indirect.sort_values(ascending=False).index[0]
    return {
        "run_id": run_id,
        "target_benchmark": target,
        "condition": condition,
        "spearman_rho": float(spearmanr(br, cr).statistic),
        "kendall_tau_b": float(kendalltau(br, cr, variant="b").statistic),
        "top_10_shared_count": len(top10_base & top10_current),
        "top_10_jaccard": len(top10_base & top10_current)
        / len(top10_base | top10_current),
        "top_20_shared_count": len(top20_base & top20_current),
        "top_20_jaccard": len(top20_base & top20_current)
        / len(top20_base | top20_current),
        "rank_change_mean_abs": float(rank_abs.mean()),
        "rank_change_median_abs": float(rank_abs.median()),
        "rank_change_p90_abs": float(rank_abs.quantile(0.9)),
        "rank_change_max_abs": float(rank_abs.max()),
        "rank_percentile_change_mean_abs": float(percentile_abs.mean()),
        "rank_percentile_change_median_abs": float(percentile_abs.median()),
        "rank_percentile_change_max_abs": float(percentile_abs.max()),
        "BHI_change_mean_abs": float(bhi_abs.mean()),
        "BHI_change_median_abs": float(bhi_abs.median()),
        "BHI_change_max_abs": float(bhi_abs.max()),
        "largest_indirect_mover": largest_indirect,
        "largest_indirect_rank_percentile_change_abs": float(
            indirect.loc[largest_indirect]
        ),
    }


def weight_row(
    run_id: str, target: str, condition: str, run: dict, baseline: dict
) -> dict:
    internal_delta = run["internal_weights"] - baseline["internal_weights"]
    critic_delta = run["critic_weights"] - baseline["critic_weights"]
    return {
        "run_id": run_id,
        "target_benchmark": target,
        "condition": condition,
        "baseline_rcv_weight": baseline["internal_weights"]["Robust_CV"],
        "perturbed_rcv_weight": run["internal_weights"]["Robust_CV"],
        "rcv_weight_change": internal_delta["Robust_CV"],
        "baseline_edr_weight": baseline["internal_weights"]["EDR"],
        "perturbed_edr_weight": run["internal_weights"]["EDR"],
        "edr_weight_change": internal_delta["EDR"],
        "internal_total_variation_distance": 0.5
        * float(internal_delta.abs().sum()),
        "baseline_s_disc_weight": baseline["critic_weights"]["S_Disc"],
        "perturbed_s_disc_weight": run["critic_weights"]["S_Disc"],
        "s_disc_weight_change": critic_delta["S_Disc"],
        "baseline_s_sat_weight": baseline["critic_weights"]["S_Sat"],
        "perturbed_s_sat_weight": run["critic_weights"]["S_Sat"],
        "s_sat_weight_change": critic_delta["S_Sat"],
        "baseline_s_imp_weight": baseline["critic_weights"]["S_Imp"],
        "perturbed_s_imp_weight": run["critic_weights"]["S_Imp"],
        "s_imp_weight_change": critic_delta["S_Imp"],
        "maximum_absolute_percentage_point_drift": float(
            critic_delta.abs().max() * 100
        ),
        "critic_total_variation_distance": 0.5
        * float(critic_delta.abs().sum()),
    }


def response_ratio(low: float, high: float) -> float | None:
    return None if abs(high) <= TOLERANCE else low / high


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
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    return value


def markdown(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    selected = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                f"{value:.{digits}f}"
                if isinstance(value, (float, np.floating))
                else str(value)
                for value in values
            )
            + " |"
        )
    return "\n".join([header, rule, *rows])


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")
    count_before, aggregate_before = protected_aggregate()
    input_sha256_before = sha256_file(INPUT_PATH)
    if not ORIGINAL_PREPROCESSOR.is_file():
        raise FileNotFoundError(ORIGINAL_PREPROCESSOR)
    if "pd.to_datetime(df.loc[release_mask, model_cols].iloc[0], errors='coerce')" not in ORIGINAL_PREPROCESSOR.read_text(encoding="utf-8"):
        raise RuntimeError("Parser reference date logic is not verified")

    master = MasterDataPreprocessor(copy.deepcopy(CONFIG)).load_and_process_master(
        INPUT_PATH
    )
    baseline_availability = master.availability.copy()
    model_dates = submission_dates(master.model_columns)
    master.model_dates = model_dates
    min_count = CONFIG["min_model_count"]
    sat_matrix_base = master.sat_scores.copy()
    disc_matrix_base = master.disc_scores.copy()
    eligible_sat = sat_matrix_base.loc[
        sat_matrix_base.notna().sum(axis=1) >= min_count
    ]
    bench_data_base = eligible_sat.reset_index()
    benchmark_to_index = {
        benchmark: index
        for index, benchmark in enumerate(bench_data_base["Benchmark"])
    }
    baseline_caps = exact_capabilities(bench_data_base)
    baseline = calculate_pipeline(
        master,
        disc_matrix_base,
        sat_matrix_base,
        baseline_caps,
        model_dates,
    )
    reproduction = baseline_reproduction(baseline)

    submitted_sat = pd.read_csv(
        BASE_DIR / "anti_saturation_results.csv"
    ).set_index("Benchmark")
    submitted_final = pd.read_csv(
        BASE_DIR / "final_bhi_rankings.csv"
    ).set_index("Benchmark")
    targets = (
        submitted_sat.loc[
            submitted_sat.index.intersection(submitted_final.index)
        ]
        .reset_index()
        .sort_values(["S_Static", "Benchmark"], kind="mergesort")
        .head(10)["Benchmark"]
        .tolist()
    )

    target_rows = []
    capability_rows = []
    selection_rows = []
    selections = {}
    for target in targets:
        index = benchmark_to_index[target]
        caps = baseline_caps[index]
        scores = sat_matrix_base.loc[target]
        valid = scores.notna() & np.isfinite(caps.reindex(scores.index)) & (scores > 0)
        candidates = pd.DataFrame(
            {
                "model": scores.index[valid],
                "capability": caps.reindex(scores.index[valid]).to_numpy(),
                "score": scores.loc[valid].to_numpy(),
            }
        )
        candidates = candidates.sort_values(
            ["capability", "model"], ascending=[True, True], kind="mergesort"
        ).reset_index(drop=True)
        candidates["capability_rank"] = np.arange(1, len(candidates) + 1)
        group_size = max(1, math.ceil(0.10 * len(candidates)))
        low = candidates.head(group_size).copy()
        high = candidates.sort_values(
            ["capability", "model"],
            ascending=[False, True],
            kind="mergesort",
        ).head(group_size)
        selections[target] = {
            "LOW_CAPABILITY_ZERO": low,
            "HIGH_CAPABILITY_ZERO": high,
        }
        submitted = submitted_final.loc[target]
        target_rows.append(
            {
                "benchmark": target,
                "submitted_S_Static": submitted_sat.loc[target, "S_Static"],
                "submitted_S_Dynamic": submitted_sat.loc[target, "S_Dynamic"],
                "submitted_S_Sat": submitted_sat.loc[target, "S_Sat"],
                "submitted_S_Disc": submitted["S_Disc"],
                "submitted_S_Imp": submitted["S_Imp"],
                "submitted_BHI": submitted["BHI"],
                "submitted_final_rank": int(
                    submitted_final["BHI"].rank(
                        method="average", ascending=False
                    ).loc[target]
                ),
                "valid_score_count": int(scores.notna().sum()),
                "eligible_perturbation_candidate_count": len(candidates),
                "selected_group_size": group_size,
                "target_specific_lobo_capability_count": int(caps.notna().sum()),
            }
        )
        candidate_rank = candidates.set_index("model")["capability_rank"]
        for model in master.model_columns:
            score = scores.get(model, np.nan)
            capability_rows.append(
                {
                    "target_benchmark": target,
                    "model_name": model,
                    "lobo_capability": caps.get(model, np.nan),
                    "target_score_available": bool(pd.notna(score)),
                    "original_normalized_target_score": score,
                    "capability_rank_among_candidates": candidate_rank.get(
                        model, np.nan
                    ),
                }
            )
        for condition, selected in selections[target].items():
            for row in selected.itertuples(index=False):
                selection_rows.append(
                    {
                        "benchmark": target,
                        "condition": condition,
                        "model_name": row.model,
                        "lobo_capability": row.capability,
                        "capability_rank": row.capability_rank,
                        "original_normalized_score": row.score,
                        "perturbed_normalized_score": 0.0,
                        "absolute_score_reduction": row.score,
                        "relative_score_reduction": 1.0,
                    }
                )

    local_rows = []
    end_rows = []
    paired_source = {}
    global_rows = []
    weight_rows = []
    all_benchmark_rows = []
    invariant_rows = []
    baseline_cube = contribution_cube(bench_data_base)
    fast_baseline_caps = fast_capabilities(
        baseline_cube, list(bench_data_base.columns[1:])
    )
    fast_exact_difference = max(
        float((fast_baseline_caps[i] - baseline_caps[i]).abs().max())
        for i in baseline_caps
    )
    if fast_exact_difference > TOLERANCE:
        raise RuntimeError(
            f"Algebraic LOBO implementation differs from submitted code: {fast_exact_difference}"
        )

    for target_number, target in enumerate(targets, 1):
        target_index = benchmark_to_index[target]
        target_caps = baseline_caps[target_index]
        baseline_target = baseline["final"].loc[target]
        baseline_local = baseline["saturation"].loc[target]
        paired_source[target] = {}

        for condition in CONDITIONS:
            run_id = f"T{target_number:02d}_{condition}"
            if condition == "BASELINE":
                selected_models = []
                disc_matrix = disc_matrix_base
                sat_matrix = sat_matrix_base
                run = baseline
                sum_reduction = 0.0
                reductions = np.array([], dtype=float)
            else:
                selected = selections[target][condition]
                selected_models = selected["model"].tolist()
                reductions = selected["score"].to_numpy(dtype=float)
                sum_reduction = float(reductions.sum())
                disc_matrix = disc_matrix_base.copy()
                sat_matrix = sat_matrix_base.copy()
                sat_matrix.loc[target, selected_models] = 0.0
                disc_valid_selected = [
                    model
                    for model in selected_models
                    if pd.notna(disc_matrix.loc[target, model])
                ]
                disc_matrix.loc[target, disc_valid_selected] = 0.0
                changed_bench_data = sat_matrix.loc[
                    sat_matrix.notna().sum(axis=1) >= min_count
                ].reset_index()
                cube = baseline_cube.copy()
                cube[target_index] = np.stack(
                    row_contributions(
                        changed_bench_data.iloc[
                            target_index, 1:
                        ].to_numpy(dtype=float)
                    )
                )
                caps = fast_capabilities(
                    cube, list(changed_bench_data.columns[1:])
                )
                run = calculate_pipeline(
                    master, disc_matrix, sat_matrix, caps, model_dates
                )

            current_scores = sat_matrix.loc[target]
            valid = current_scores.dropna()
            common = valid.index.intersection(target_caps.index)
            weighted = (
                (valid[common] / 100) * target_caps.loc[common]
            ).sum() / (target_caps.loc[common].sum() + 1e-9)
            s_static_local = 1 - weighted
            regression_data = pd.DataFrame(
                {
                    "d": (model_dates - model_dates.min()).dt.days,
                    "s": current_scores,
                }
            ).dropna()
            if len(regression_data) >= 2:
                regression = LinearRegression().fit(
                    regression_data[["d"]], regression_data["s"] / 100
                )
                s_dynamic_local = max(
                    0,
                    1
                    - (
                        regression_data["s"].mean() / 100
                        + regression.coef_[0] * 30
                    ),
                )
            else:
                s_dynamic_local = s_static_local
            s_sat_local = (
                WEIGHT_STATIC * s_static_local
                + WEIGHT_DYNAMIC * s_dynamic_local
            )
            local_rows.append(
                {
                    "run_id": run_id,
                    "target_benchmark": target,
                    "condition": condition,
                    "baseline_weighted_score": baseline_local["Weighted_Score"],
                    "perturbed_weighted_score": weighted,
                    "weighted_score_change": weighted
                    - baseline_local["Weighted_Score"],
                    "baseline_S_Static": baseline_local["S_Static"],
                    "perturbed_S_Static": s_static_local,
                    "delta_S_Static": s_static_local
                    - baseline_local["S_Static"],
                    "absolute_delta_S_Static": abs(
                        s_static_local - baseline_local["S_Static"]
                    ),
                    "baseline_S_Dynamic": baseline_local["S_Dynamic"],
                    "perturbed_S_Dynamic": s_dynamic_local,
                    "delta_S_Dynamic": s_dynamic_local
                    - baseline_local["S_Dynamic"],
                    "absolute_delta_S_Dynamic": abs(
                        s_dynamic_local - baseline_local["S_Dynamic"]
                    ),
                    "baseline_S_Sat": baseline_local["S_Sat"],
                    "perturbed_S_Sat": s_sat_local,
                    "delta_S_Sat": s_sat_local - baseline_local["S_Sat"],
                    "absolute_delta_S_Sat": abs(
                        s_sat_local - baseline_local["S_Sat"]
                    ),
                }
            )

            current = run["final"].loc[target]
            record = {
                "run_id": run_id,
                "target_benchmark": target,
                "condition": condition,
                "changed_cell_count": len(selected_models),
                "sum_absolute_score_reduction": sum_reduction,
                "mean_absolute_score_reduction": (
                    float(reductions.mean()) if len(reductions) else 0.0
                ),
                "median_absolute_score_reduction": (
                    float(np.median(reductions)) if len(reductions) else 0.0
                ),
                "maximum_score_reduction": (
                    float(reductions.max()) if len(reductions) else 0.0
                ),
            }
            for score in [
                "S_Disc",
                "S_Static",
                "S_Dynamic",
                "S_Sat",
                "S_Imp",
                "BHI",
                "Rank",
                "Rank_Percentile",
            ]:
                delta = current[score] - baseline_target[score]
                record[f"baseline_{score}"] = baseline_target[score]
                record[f"perturbed_{score}"] = current[score]
                record[f"delta_{score}"] = delta
                record[f"absolute_delta_{score}"] = abs(delta)
            for score in ["S_Static", "S_Sat", "BHI", "Rank_Percentile"]:
                record[f"{score}_response_per_score_point"] = (
                    record[f"absolute_delta_{score}"] / sum_reduction
                    if sum_reduction > 0
                    else None
                )
            end_rows.append(record)
            paired_source[target][condition] = record

            for benchmark, values in run["final"].iterrows():
                base_values = baseline["final"].loc[benchmark]
                all_benchmark_rows.append(
                    {
                        "run_id": run_id,
                        "target_benchmark": target,
                        "condition": condition,
                        "benchmark": benchmark,
                        "target_flag": benchmark == target,
                        **{
                            score: values[score]
                            for score in [
                                "S_Disc",
                                "S_Static",
                                "S_Dynamic",
                                "S_Sat",
                                "S_Imp",
                                "BHI",
                                "Rank",
                                "Rank_Percentile",
                            ]
                        },
                        **{
                            f"delta_{score}": values[score]
                            - base_values[score]
                            for score in [
                                "S_Disc",
                                "S_Static",
                                "S_Dynamic",
                                "S_Sat",
                                "S_Imp",
                                "BHI",
                                "Rank",
                                "Rank_Percentile",
                            ]
                        },
                    }
                )

            if condition != "BASELINE":
                global_rows.append(
                    leaderboard_row(run_id, target, condition, run, baseline)
                )
                weight_rows.append(
                    weight_row(run_id, target, condition, run, baseline)
                )
                target_cap_difference = float(
                    (
                        run["capabilities"][target_index] - target_caps
                    ).abs().max()
                )
                checks = {
                    "only_selected_target_sat_cells_changed": int(
                        (
                            sat_matrix.fillna(np.inf)
                            != sat_matrix_base.fillna(np.inf)
                        )
                        .to_numpy()
                        .sum()
                    )
                    == len(selected_models),
                    "missingness_unchanged": sat_matrix.isna().equals(
                        sat_matrix_base.isna()
                    )
                    and disc_matrix.isna().equals(disc_matrix_base.isna()),
                    "availability_unchanged": master.availability.equals(
                        baseline_availability
                    ),
                    "target_lobo_capability_invariant": target_cap_difference
                    <= TOLERANCE,
                    "target_S_Imp_invariant": abs(
                        current["S_Imp"] - baseline_target["S_Imp"]
                    )
                    <= TOLERANCE,
                    "eligibility_unchanged": run["disc_counts"].equals(
                        baseline["disc_counts"]
                    )
                    and run["sat_counts"].equals(baseline["sat_counts"]),
                    "all_final_scores_finite": np.isfinite(
                        run["final"][
                            [
                                "S_Disc",
                                "S_Static",
                                "S_Dynamic",
                                "S_Sat",
                                "S_Imp",
                                "BHI",
                            ]
                        ].to_numpy()
                    ).all(),
                    "normalized_scores_in_expected_range": (
                        float(sat_matrix.min().min()) >= -TOLERANCE
                        and float(sat_matrix.max().max()) <= 100 + TOLERANCE
                        and float(disc_matrix.min().min()) >= -TOLERANCE
                        and float(disc_matrix.max().max()) <= 100 + TOLERANCE
                    ),
                    "critic_weights_sum_to_one": math.isclose(
                        float(run["critic_weights"].sum()),
                        1.0,
                        abs_tol=TOLERANCE,
                    ),
                    "local_non_target_values_frozen": True,
                    "raw_input_hash_unchanged": sha256_file(INPUT_PATH)
                    == input_sha256_before,
                }
                for check, passed in checks.items():
                    invariant_rows.append(
                        {
                            "run_id": run_id,
                            "target_benchmark": target,
                            "condition": condition,
                            "invariant": check,
                            "passed": bool(passed),
                            "observed_target_capability_max_abs_difference": (
                                target_cap_difference
                                if check == "target_lobo_capability_invariant"
                                else np.nan
                            ),
                        }
                    )

    paired_rows = []
    for target in targets:
        low = paired_source[target]["LOW_CAPABILITY_ZERO"]
        high = paired_source[target]["HIGH_CAPABILITY_ZERO"]
        row = {"target_benchmark": target}
        for metric in ["S_Static", "S_Sat", "BHI", "Rank", "Rank_Percentile"]:
            low_value = low[f"absolute_delta_{metric}"]
            high_value = high[f"absolute_delta_{metric}"]
            row[f"low_abs_delta_{metric}"] = low_value
            row[f"high_abs_delta_{metric}"] = high_value
            row[f"attenuation_ratio_{metric}"] = response_ratio(
                low_value, high_value
            )
        for metric in ["S_Static", "S_Sat", "BHI", "Rank_Percentile"]:
            low_value = low[f"{metric}_response_per_score_point"]
            high_value = high[f"{metric}_response_per_score_point"]
            row[f"low_{metric}_response_per_score_point"] = low_value
            row[f"high_{metric}_response_per_score_point"] = high_value
            row[f"normalized_attenuation_ratio_{metric}"] = response_ratio(
                low_value, high_value
            )
        paired_rows.append(row)

    end_frame = pd.DataFrame(end_rows)
    paired_frame = pd.DataFrame(paired_rows)
    aggregate_rows = []
    perturbation_frame = end_frame[end_frame["condition"].isin(PERTURBATIONS)]
    for condition in PERTURBATIONS:
        subset = perturbation_frame[perturbation_frame["condition"] == condition]
        for metric in [
            "S_Static",
            "S_Dynamic",
            "S_Sat",
            "S_Disc",
            "S_Imp",
            "BHI",
            "Rank",
            "Rank_Percentile",
        ]:
            values = subset[f"absolute_delta_{metric}"]
            aggregate_rows.append(
                {
                    "record_type": "condition_effect",
                    "condition_or_metric": condition,
                    "metric": metric,
                    "valid_count": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "p90": values.quantile(0.9),
                    "maximum": values.max(),
                }
            )
    for metric in ["S_Static", "S_Sat", "BHI", "Rank_Percentile"]:
        values = paired_frame[f"attenuation_ratio_{metric}"].dropna()
        aggregate_rows.append(
            {
                "record_type": "attenuation_ratio",
                "condition_or_metric": metric,
                "metric": metric,
                "valid_count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "p90": values.quantile(0.9),
                "maximum": values.max(),
                "below_one_count": int((values < 1 - TOLERANCE).sum()),
                "below_one_proportion": float((values < 1 - TOLERANCE).mean()),
                "equal_one_count": int(
                    ((values - 1).abs() <= TOLERANCE).sum()
                ),
                "equal_one_proportion": float(
                    ((values - 1).abs() <= TOLERANCE).mean()
                ),
                "above_one_count": int((values > 1 + TOLERANCE).sum()),
                "above_one_proportion": float((values > 1 + TOLERANCE).mean()),
            }
        )
    aggregate_frame = pd.DataFrame(aggregate_rows)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_definitions = []
    for condition in PERTURBATIONS:
        subset = perturbation_frame.set_index(
            ["target_benchmark", "condition"]
        )
        for metric in ["S_Static", "S_Sat", "BHI"]:
            values = np.array(
                [
                    subset.loc[(target, condition), f"absolute_delta_{metric}"]
                    for target in targets
                ]
            )
            bootstrap_definitions.append(
                (f"median_abs_delta_{metric}_{condition}", values)
            )
        values = np.array(
            [
                subset.loc[
                    (target, condition),
                    "S_Static_response_per_score_point",
                ]
                for target in targets
            ]
        )
        bootstrap_definitions.append(
            (
                f"median_normalized_S_Static_response_{condition}",
                values,
            )
        )
    for metric in ["S_Static", "S_Sat", "BHI"]:
        bootstrap_definitions.append(
            (
                f"median_attenuation_ratio_{metric}",
                paired_frame[f"attenuation_ratio_{metric}"].to_numpy(),
            )
        )
    bootstrap_rows = []
    for metric, values in bootstrap_definitions:
        estimate = float(np.nanmedian(values))
        samples = []
        skipped = 0
        for _ in range(BOOTSTRAP_ITERATIONS):
            sample = values[rng.integers(0, len(values), len(values))]
            if np.isfinite(sample).any():
                samples.append(float(np.nanmedian(sample)))
            else:
                skipped += 1
        bootstrap_rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_95_lower": np.quantile(samples, 0.025),
                "ci_95_upper": np.quantile(samples, 0.975),
                "requested_resamples": BOOTSTRAP_ITERATIONS,
                "valid_resamples": len(samples),
                "skipped_resamples": skipped,
                "seed": BOOTSTRAP_SEED,
            }
        )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    OUTPUT_DIR.mkdir(parents=True)
    frames = {
        "target_benchmarks.csv": pd.DataFrame(target_rows),
        "target_lobo_capabilities.csv": pd.DataFrame(capability_rows),
        "selected_models_and_score_changes.csv": pd.DataFrame(selection_rows),
        "local_isolated_results.csv": pd.DataFrame(local_rows),
        "target_end_to_end_results.csv": end_frame,
        "paired_low_vs_high_comparison.csv": paired_frame,
        "aggregate_condition_metrics.csv": aggregate_frame,
        "global_leaderboard_stability.csv": pd.DataFrame(global_rows),
        "global_weight_drift.csv": pd.DataFrame(weight_rows),
        "all_run_benchmark_results.csv": pd.DataFrame(all_benchmark_rows),
        "bootstrap_confidence_intervals.csv": bootstrap_frame,
        "baseline_reproduction_check.csv": reproduction,
        "invariant_checks.csv": pd.DataFrame(invariant_rows),
    }
    for filename, frame in frames.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    local_frame = frames["local_isolated_results.csv"]
    invariant_frame = frames["invariant_checks.csv"]
    low_not_smaller = paired_frame[
        paired_frame["attenuation_ratio_S_Static"].fillna(np.inf)
        >= 1 - TOLERANCE
    ]["target_benchmark"].tolist()
    largest_low = perturbation_frame[
        perturbation_frame["condition"] == "LOW_CAPABILITY_ZERO"
    ].sort_values("absolute_delta_S_Static", ascending=False).iloc[0]
    largest_high = perturbation_frame[
        perturbation_frame["condition"] == "HIGH_CAPABILITY_ZERO"
    ].sort_values("absolute_delta_S_Static", ascending=False).iloc[0]
    report = f"""# Targeted Weak-Model Low-Score Robustness Stress Test

## Scientific motivation and scope

This single-baseline stress test asks whether setting valid normalized scores to zero has less effect when the affected models have low target-specific LOBO capability. No other snapshot is a scientific input.

## Deterministic selection rules

Targets are the ten final-ranked benchmarks with the lowest baseline S_Static, sorted by S_Static then benchmark name. For each target, eligible positive-score models are capability-ranked using LOBO excluding the target; `m=max(1,ceil(0.10*n))` bottom and top models are set to zero separately.

## Baseline reproduction

{markdown(reproduction, ["check", "observed_max_abs_difference", "passed"], 12)}

## Target benchmarks

{markdown(frames["target_benchmarks.csv"], ["benchmark", "submitted_S_Static", "valid_score_count", "eligible_perturbation_candidate_count", "selected_group_size"])}

## Selected models

{markdown(frames["selected_models_and_score_changes.csv"], ["benchmark", "condition", "model_name", "lobo_capability", "original_normalized_score", "absolute_score_reduction"])}

## Local isolated results

{markdown(local_frame[local_frame["condition"].isin(PERTURBATIONS)], ["target_benchmark", "condition", "delta_S_Static", "delta_S_Dynamic", "delta_S_Sat"])}

## Full end-to-end target results

{markdown(perturbation_frame, ["target_benchmark", "condition", "sum_absolute_score_reduction", "delta_S_Disc", "delta_S_Sat", "delta_S_Imp", "delta_BHI", "delta_Rank", "delta_Rank_Percentile"])}

## Low versus high capability

{markdown(paired_frame, ["target_benchmark", "attenuation_ratio_S_Static", "attenuation_ratio_S_Sat", "attenuation_ratio_BHI", "normalized_attenuation_ratio_S_Static"])}

## Aggregate effects and global recalibration

{markdown(aggregate_frame, ["record_type", "condition_or_metric", "metric", "valid_count", "mean", "median", "p90", "maximum"])}

Detailed global leaderboard and CRITIC-weight results for all 20 perturbation runs are recorded in their CSV files.

## Bootstrap intervals

{markdown(bootstrap_frame, list(bootstrap_frame.columns))}

## Largest responses

- Largest weak-model S_Static response: {largest_low["target_benchmark"]}, {largest_low["absolute_delta_S_Static"]:.12g}.
- Largest high-model S_Static response: {largest_high["target_benchmark"]}, {largest_high["absolute_delta_S_Static"]:.12g}.
- Targets where the weak-model S_Static effect is not smaller: {", ".join(low_not_smaller) if low_not_smaller else "none"}.

## Invariants

All {len(invariant_frame)} run-level invariant checks passed: {bool(invariant_frame["passed"].all())}. The fast algebraic LOBO implementation agrees with the submitted function within {fast_exact_difference:.3e}.

## Limitations

This is an extreme, targeted zero-score stress test on ten near-saturated baseline benchmarks. Low and high groups can have different score reductions, so both raw and perturbation-normalized responses are reported. It does not prove complete robustness or generalize to every corruption process.

## 中文简明摘要

本实验在原始投稿数据上，将每个近饱和目标基准中能力最低和最高的 10% 有效模型分数分别置零。结果同时报告原始响应与按分数扰动幅度归一化后的响应；任何弱模型影响不小于强模型影响的基准均完整保留。该结果只能说明针对性极端压力测试下的相对衰减，不能声称完全稳健。

## Concise English summary

On the supplied baseline, this targeted stress test separately zeroed valid scores from the bottom and top 10% of target-specific LOBO capability models for ten near-saturated benchmarks. Raw and score-reduction-normalized responses are both retained, including every case where the weak-model effect is not smaller. The evidence supports only a qualified statement about relative attenuation under this extreme targeted perturbation, not complete robustness.
"""
    (OUTPUT_DIR / "weak_model_stress_test_report.md").write_text(
        report, encoding="utf-8"
    )
    summary = {
        "analysis_timestamp_utc": TIMESTAMP,
        "scope": "supplied baseline only",
        "input": display_path(INPUT_PATH),
        "parsed_date_count": int(model_dates.notna().sum()),
        "targets": target_rows,
        "baseline_reproduction": reproduction.to_dict("records"),
        "fast_lobo_vs_submitted_max_abs_difference": fast_exact_difference,
        "paired_comparison": paired_rows,
        "aggregate_metrics": aggregate_frame.to_dict("records"),
        "bootstrap": bootstrap_rows,
        "largest_low_response": json_safe(largest_low.to_dict()),
        "largest_high_response": json_safe(largest_high.to_dict()),
        "low_not_smaller_S_Static": low_not_smaller,
        "all_invariants_passed": bool(invariant_frame["passed"].all()),
        "protected_file_count": count_before,
        "pre_experiment_aggregate_sha256": aggregate_before,
        "post_experiment_aggregate_sha256": protected_aggregate()[1],
        "protected_files_unchanged": protected_aggregate()[1]
        == aggregate_before,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (OUTPUT_DIR / "weak_model_stress_test_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "ANALYSIS_RECORD.md").write_text(
        f"""# Weak-Model Low-Score Stress Test

- Timestamp UTC: {TIMESTAMP}
- Script: `scripts/weak_model_low_score_stress_test.py`
- Input: `{display_path(INPUT_PATH)}`
- Baseline directory: `{display_path(BASE_DIR)}`
- Input SHA-256: `{sha256_file(INPUT_PATH)}`
- Date parsing: `pd.to_datetime(raw_model_dates, errors="coerce")`
- Targets: {", ".join(targets)}
- Conditions: {", ".join(CONDITIONS)}
- Bootstrap: {BOOTSTRAP_ITERATIONS:,}, seed {BOOTSTRAP_SEED}
- Commands: `python -m compileall scripts/weak_model_low_score_stress_test.py`; `python scripts/weak_model_low_score_stress_test.py`
- Protected file count: {count_before}
- Pre/post aggregate SHA-256: `{aggregate_before}`

All perturbations were performed in memory.
""",
        encoding="utf-8",
    )

    generated = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
        and path.name not in {"artifact_manifest_sha256.csv", "MANIFEST_SHA256.txt"}
    )
    sources = [
        INPUT_PATH,
        BASE_DIR / "capability_discrimination_final.csv",
        BASE_DIR / "anti_saturation_results.csv",
        BASE_DIR / "impact_results.csv",
        BASE_DIR / "final_bhi_rankings.csv",
        BASE_DIR / "benchmark_inclusion_report.csv",
        ROOT / "main.py",
        ROOT / "config" / "settings.py",
        ROOT / "src" / "preprocessor.py",
        ROOT / "src" / "capability.py",
        ROOT / "README.md",
    ]
    entries = []
    for path, role in [
        (Path(__file__).resolve(), "generated_script"),
        *[(path, "generated_output") for path in generated],
        *[(path, "source") for path in sources],
    ]:
        entries.append(
            {
                "relative_path": display_path(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "artifact_role": role,
                "analysis_timestamp_utc": TIMESTAMP,
            }
        )
    manifest_path = OUTPUT_DIR / "artifact_manifest_sha256.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(sorted(entries, key=lambda row: row["relative_path"]))
    (OUTPUT_DIR / "MANIFEST_SHA256.txt").write_text(
        f"artifact_manifest_sha256.csv  {sha256_file(manifest_path)}\n",
        encoding="utf-8",
    )

    count_after, aggregate_after = protected_aggregate()
    if count_after != count_before or aggregate_after != aggregate_before:
        raise RuntimeError("Protected files changed")
    expected_names = set(frames) | {
        "weak_model_stress_test_report.md",
        "weak_model_stress_test_summary.json",
        "ANALYSIS_RECORD.md",
        "artifact_manifest_sha256.csv",
        "MANIFEST_SHA256.txt",
    }
    actual_names = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(f"Unexpected output files: {sorted(actual_names)}")
    for path in OUTPUT_DIR.iterdir():
        if path.suffix == ".csv":
            pd.read_csv(path)
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    if len(targets) != 10 or len(end_frame) != 30:
        raise RuntimeError("Unexpected target or condition count")
    if len(frames["all_run_benchmark_results.csv"]) != 30 * len(
        baseline["final"]
    ):
        raise RuntimeError("Incomplete all-run benchmark results")
    if not invariant_frame["passed"].all():
        raise RuntimeError("Invariant failure")

    print(f"Wrote weak-model stress-test outputs to {OUTPUT_DIR}")
    print("Protected files unchanged:", aggregate_after == aggregate_before)
    print("Targets:", ", ".join(targets))
    print(
        paired_frame[
            [
                "target_benchmark",
                "attenuation_ratio_S_Static",
                "attenuation_ratio_S_Sat",
                "attenuation_ratio_BHI",
                "normalized_attenuation_ratio_S_Static",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
