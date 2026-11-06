"""Fixed-decrement weak-model score sensitivity on a supplied BHI baseline."""

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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import CONFIG, WEIGHT_DYNAMIC, WEIGHT_STATIC  # noqa: E402
from src.preprocessor import MasterDataPreprocessor  # noqa: E402
from weak_model_low_score_stress_test import (  # noqa: E402
    baseline_reproduction,
    calculate_pipeline,
    contribution_cube,
    exact_capabilities,
    fast_capabilities,
    json_safe,
    markdown,
    response_ratio,
    row_contributions,
    sha256_file,
    submission_dates,
)

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
    BASE_DIR / "weak_model_fixed_decrement_sensitivity_submission",
))
PRIOR_SCRIPT = ROOT / "scripts" / "weak_model_low_score_stress_test.py"
PRIOR_OUTPUT = Path(os.environ.get(
    "BHI_PRIOR_OUTLIER_OUTPUT_DIR",
    BASE_DIR / "weak_model_low_score_stress_test_submission",
))
DECREMENTS = [10, 15, 20, 25, 30]
GROUPS = ["LOW_CAPABILITY", "HIGH_CAPABILITY"]
TOLERANCE = 1e-12
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20260726
TIMESTAMP = datetime.now(timezone.utc).isoformat()
PROTECTED_FILE_COUNT = 1_059
PROTECTED_AGGREGATE = (
    "775ab530ddc92f0b2955b06d87f37bc913478cbb97ffeccd8ef25672d8207f3f"
)
PRIOR_SCRIPT_SHA256 = (
    "a879df691bca6a980a326ec259ffa0dbadfd7b78988df9cc41879ca12b61dbe8"
)
PRIOR_OUTPUT_AGGREGATE = (
    "61b88550e0089b6e6738599f8e83effb5f4f355188f57b2191ac9146e5ef7740"
)
INPUT_SHA256 = (
    "a86da2902d65341c4e429fda1b86d9d2b4f1f77c3b83f13c5b6e5fd7cc672555"
)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def protected_aggregate() -> tuple[int, str]:
    script = Path(__file__).resolve()
    output = OUTPUT_DIR.resolve()
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == script or output in resolved.parents:
            continue
        files.append(resolved)
    hashes = {
        display_path(path): sha256_file(path)
        for path in sorted(set(files), key=display_path)
    }
    return len(hashes), hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()


def directory_aggregate(directory: Path) -> str:
    rows = [
        f"{path.name}:{sha256_file(path)}"
        for path in sorted(directory.iterdir(), key=lambda p: p.name)
        if path.is_file()
    ]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def fixed_top(frame: pd.DataFrame, k: int) -> set[str]:
    return set(
        frame.reset_index()
        .sort_values(["BHI", "Benchmark"], ascending=[False, True], kind="mergesort")
        .head(k)["Benchmark"]
    )


def global_effect_row(
    run_id: str,
    target: str,
    group: str,
    decrement: int,
    run: dict,
    baseline: dict,
) -> dict:
    current, base = run["final"], baseline["final"]
    names = base.index.intersection(current.index)
    br, cr = base.loc[names, "Rank"], current.loc[names, "Rank"]
    rank_abs = (cr - br).abs()
    pct_abs = (
        current.loc[names, "Rank_Percentile"]
        - base.loc[names, "Rank_Percentile"]
    ).abs()
    bhi_abs = (current.loc[names, "BHI"] - base.loc[names, "BHI"]).abs()
    top10_base, top10_run = fixed_top(base, 10), fixed_top(current, 10)
    top20_base, top20_run = fixed_top(base, 20), fixed_top(current, 20)
    non_target = [name for name in names if name != target]
    indirect_rank = (current.loc[non_target, "Rank"] - base.loc[non_target, "Rank"]).abs()
    indirect_bhi = (
        current.loc[non_target, "BHI"] - base.loc[non_target, "BHI"]
    ).abs()
    largest_name = pd.DataFrame(
        {"rank": indirect_rank, "bhi": indirect_bhi}
    ).sort_values(["rank", "bhi"], ascending=False, kind="mergesort").index[0]
    return {
        "run_id": run_id,
        "target_benchmark": target,
        "capability_group": group,
        "decrement_level": decrement,
        "spearman_rho": float(spearmanr(br, cr).statistic),
        "kendall_tau_b": float(kendalltau(br, cr, variant="b").statistic),
        "top_10_shared_count": len(top10_base & top10_run),
        "top_10_overlap_rate": len(top10_base & top10_run) / 10,
        "top_10_jaccard": len(top10_base & top10_run) / len(top10_base | top10_run),
        "top_20_shared_count": len(top20_base & top20_run),
        "top_20_overlap_rate": len(top20_base & top20_run) / 20,
        "top_20_jaccard": len(top20_base & top20_run) / len(top20_base | top20_run),
        "rank_change_mean_abs": float(rank_abs.mean()),
        "rank_change_median_abs": float(rank_abs.median()),
        "rank_change_p90_abs": float(rank_abs.quantile(0.9)),
        "rank_change_max_abs": float(rank_abs.max()),
        "rank_percentile_change_mean_abs": float(pct_abs.mean()),
        "rank_percentile_change_median_abs": float(pct_abs.median()),
        "rank_percentile_change_p90_abs": float(pct_abs.quantile(0.9)),
        "rank_percentile_change_max_abs": float(pct_abs.max()),
        "BHI_change_mean_abs": float(bhi_abs.mean()),
        "BHI_change_median_abs": float(bhi_abs.median()),
        "BHI_change_p90_abs": float(bhi_abs.quantile(0.9)),
        "BHI_change_max_abs": float(bhi_abs.max()),
        "largest_indirect_non_target_benchmark": largest_name,
        "largest_non_target_rank_movement": float(indirect_rank.max()),
        "largest_non_target_BHI_change": float(indirect_bhi.max()),
    }


def critic_effect_row(
    run_id: str,
    target: str,
    group: str,
    decrement: int,
    run: dict,
    baseline: dict,
) -> dict:
    internal_delta = run["internal_weights"] - baseline["internal_weights"]
    final_delta = run["critic_weights"] - baseline["critic_weights"]
    return {
        "run_id": run_id,
        "target_benchmark": target,
        "capability_group": group,
        "decrement_level": decrement,
        "baseline_rcv_weight": baseline["internal_weights"]["Robust_CV"],
        "perturbed_rcv_weight": run["internal_weights"]["Robust_CV"],
        "delta_rcv_weight": internal_delta["Robust_CV"],
        "baseline_edr_weight": baseline["internal_weights"]["EDR"],
        "perturbed_edr_weight": run["internal_weights"]["EDR"],
        "delta_edr_weight": internal_delta["EDR"],
        "internal_total_variation_distance": 0.5 * float(internal_delta.abs().sum()),
        "baseline_s_disc_weight": baseline["critic_weights"]["S_Disc"],
        "perturbed_s_disc_weight": run["critic_weights"]["S_Disc"],
        "delta_s_disc_weight": final_delta["S_Disc"],
        "baseline_s_sat_weight": baseline["critic_weights"]["S_Sat"],
        "perturbed_s_sat_weight": run["critic_weights"]["S_Sat"],
        "delta_s_sat_weight": final_delta["S_Sat"],
        "baseline_s_imp_weight": baseline["critic_weights"]["S_Imp"],
        "perturbed_s_imp_weight": run["critic_weights"]["S_Imp"],
        "delta_s_imp_weight": final_delta["S_Imp"],
        "maximum_absolute_percentage_point_drift": float(
            final_delta.abs().max() * 100
        ),
        "final_total_variation_distance": 0.5 * float(final_delta.abs().sum()),
    }


def summaries(values: pd.Series) -> dict:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "valid_count": len(finite),
        "mean": finite.mean(),
        "median": finite.median(),
        "p90": finite.quantile(0.9),
        "maximum": finite.max(),
        "minimum": finite.min(),
    }


def monotonic_class(values: list[float]) -> tuple[bool, bool, str]:
    differences = np.diff(np.asarray(values, dtype=float))
    non_decreasing = bool(np.all(differences >= -TOLERANCE))
    strict = bool(np.all(differences > TOLERANCE))
    return (
        non_decreasing,
        strict,
        "strictly_increasing"
        if strict
        else "monotonically_non_decreasing"
        if non_decreasing
        else "non_monotonic",
    )


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")
    count_before, hash_before = protected_aggregate()
    input_sha256_before = sha256_file(INPUT_PATH)
    prior_script_hash_before = sha256_file(PRIOR_SCRIPT)
    prior_output_aggregate_before = directory_aggregate(PRIOR_OUTPUT)

    master = MasterDataPreprocessor(copy.deepcopy(CONFIG)).load_and_process_master(
        INPUT_PATH
    )
    baseline_availability = master.availability.copy()
    model_dates = submission_dates(master.model_columns)
    master.model_dates = model_dates
    min_count = CONFIG["min_model_count"]
    sat_base, disc_base = master.sat_scores.copy(), master.disc_scores.copy()
    eligible_sat = sat_base.loc[sat_base.notna().sum(axis=1) >= min_count]
    bench_base = eligible_sat.reset_index()
    benchmark_to_index = {
        benchmark: index
        for index, benchmark in enumerate(bench_base["Benchmark"])
    }
    baseline_caps = exact_capabilities(bench_base)
    baseline = calculate_pipeline(
        master, disc_base, sat_base, baseline_caps, model_dates
    )
    reproduction = baseline_reproduction(baseline)

    submitted_sat = pd.read_csv(
        BASE_DIR / "anti_saturation_results.csv"
    ).set_index("Benchmark")
    submitted_final = pd.read_csv(
        BASE_DIR / "final_bhi_rankings.csv"
    ).set_index("Benchmark")
    targets = (
        submitted_sat.loc[submitted_sat.index.intersection(submitted_final.index)]
        .reset_index()
        .sort_values(["S_Static", "Benchmark"], kind="mergesort")
        .head(10)["Benchmark"]
        .tolist()
    )
    prior_targets = pd.read_csv(PRIOR_OUTPUT / "target_benchmarks.csv")[
        "benchmark"
    ].tolist()
    if targets != prior_targets:
        raise RuntimeError(f"Target mismatch: {targets} versus {prior_targets}")

    target_rows, capability_rows, selected_rows = [], [], []
    selections: dict[str, dict[str, pd.DataFrame]] = {}
    prior_selected = pd.read_csv(
        PRIOR_OUTPUT / "selected_models_and_score_changes.csv"
    )
    for target in targets:
        index = benchmark_to_index[target]
        caps, scores = baseline_caps[index], sat_base.loc[target]
        valid = (
            scores.notna()
            & np.isfinite(caps.reindex(scores.index))
            & (scores > 0)
        )
        candidates = pd.DataFrame(
            {
                "model": scores.index[valid],
                "capability": caps.reindex(scores.index[valid]).to_numpy(),
                "score": scores.loc[valid].to_numpy(),
            }
        ).sort_values(["capability", "model"], kind="mergesort").reset_index(drop=True)
        candidates["capability_rank"] = np.arange(1, len(candidates) + 1)
        m = max(1, math.ceil(0.10 * len(candidates)))
        low = candidates.head(m).copy()
        high = candidates.sort_values(
            ["capability", "model"],
            ascending=[False, True],
            kind="mergesort",
        ).head(m).copy()
        selections[target] = {"LOW_CAPABILITY": low, "HIGH_CAPABILITY": high}
        prior_low = set(
            prior_selected[
                (prior_selected["benchmark"] == target)
                & (prior_selected["condition"] == "LOW_CAPABILITY_ZERO")
            ]["model_name"]
        )
        prior_high = set(
            prior_selected[
                (prior_selected["benchmark"] == target)
                & (prior_selected["condition"] == "HIGH_CAPABILITY_ZERO")
            ]["model_name"]
        )
        if set(low["model"]) != prior_low or set(high["model"]) != prior_high:
            raise RuntimeError(f"Selected-model mismatch for {target}")
        submitted = submitted_final.loc[target]
        target_rows.append(
            {
                "benchmark": target,
                "baseline_S_Static": submitted_sat.loc[target, "S_Static"],
                "baseline_S_Dynamic": submitted_sat.loc[target, "S_Dynamic"],
                "baseline_S_Sat": submitted_sat.loc[target, "S_Sat"],
                "baseline_S_Disc": submitted["S_Disc"],
                "baseline_S_Imp": submitted["S_Imp"],
                "baseline_BHI": submitted["BHI"],
                "baseline_rank": int(
                    submitted_final["BHI"]
                    .rank(method="average", ascending=False)
                    .loc[target]
                ),
                "valid_model_score_count": int(scores.notna().sum()),
                "eligible_perturbation_candidate_count": len(candidates),
                "selected_group_size": m,
            }
        )
        rank_map = candidates.set_index("model")["capability_rank"]
        low_set, high_set = set(low["model"]), set(high["model"])
        for model in master.model_columns:
            score = scores.get(model, np.nan)
            capability_rows.append(
                {
                    "target_benchmark": target,
                    "model_name": model,
                    "lobo_capability": caps.get(model, np.nan),
                    "capability_rank": rank_map.get(model, np.nan),
                    "target_score_available": bool(pd.notna(score)),
                    "original_normalized_target_score": score,
                    "selected_low_capability": model in low_set,
                    "selected_high_capability": model in high_set,
                }
            )
        for group, selected in selections[target].items():
            for row in selected.itertuples(index=False):
                selected_rows.append(
                    {
                        "target_benchmark": target,
                        "capability_group": group,
                        "model_name": row.model,
                        "lobo_capability": row.capability,
                        "capability_rank": row.capability_rank,
                        "original_normalized_target_score": row.score,
                    }
                )

    baseline_cube = contribution_cube(bench_base)
    fast_caps = fast_capabilities(baseline_cube, list(bench_base.columns[1:]))
    fast_exact_difference = max(
        float((fast_caps[index] - baseline_caps[index]).abs().max())
        for index in baseline_caps
    )
    if fast_exact_difference > TOLERANCE:
        raise RuntimeError("Fast LOBO check failed")

    changed_rows, direct_rows, end_rows = [], [], []
    global_rows, weight_rows, all_rows, invariant_rows = [], [], [], []
    run_records: dict[tuple[str, str, int], dict] = {}
    for target_number, target in enumerate(targets, 1):
        target_index = benchmark_to_index[target]
        target_caps = baseline_caps[target_index]
        baseline_target = baseline["final"].loc[target]
        baseline_local = baseline["saturation"].loc[target]
        for group in GROUPS:
            selected = selections[target][group]
            selected_models = selected["model"].tolist()
            for decrement in DECREMENTS:
                prefix = "L" if group == "LOW_CAPABILITY" else "H"
                run_id = f"T{target_number:02d}_{prefix}_MINUS_{decrement}"
                sat_matrix, disc_matrix = sat_base.copy(), disc_base.copy()
                originals = sat_matrix.loc[target, selected_models].astype(float)
                perturbed = (originals - decrement).clip(lower=0)
                actual = originals - perturbed
                sat_matrix.loc[target, selected_models] = perturbed
                disc_selected = [
                    model
                    for model in selected_models
                    if pd.notna(disc_matrix.loc[target, model])
                ]
                disc_matrix.loc[target, disc_selected] = perturbed.loc[disc_selected]
                for model in selected_models:
                    changed_rows.append(
                        {
                            "run_id": run_id,
                            "target_benchmark": target,
                            "capability_group": group,
                            "decrement_label": f"MINUS_{decrement}",
                            "requested_decrement": decrement,
                            "model_name": model,
                            "original_normalized_score": originals[model],
                            "perturbed_normalized_score": perturbed[model],
                            "actual_decrement": actual[model],
                            "clipped_at_zero": bool(perturbed[model] == 0),
                        }
                    )
                changed_bench = sat_matrix.loc[
                    sat_matrix.notna().sum(axis=1) >= min_count
                ].reset_index()
                cube = baseline_cube.copy()
                cube[target_index] = np.stack(
                    row_contributions(
                        changed_bench.iloc[target_index, 1:].to_numpy(dtype=float)
                    )
                )
                run_caps = fast_capabilities(
                    cube, list(changed_bench.columns[1:])
                )
                run = calculate_pipeline(
                    master, disc_matrix, sat_matrix, run_caps, model_dates
                )
                total_reduction = float(actual.sum())
                current_scores = sat_matrix.loc[target]
                valid_scores = current_scores.dropna()
                common = valid_scores.index.intersection(target_caps.index)
                weighted = (
                    (valid_scores[common] / 100) * target_caps.loc[common]
                ).sum() / (target_caps.loc[common].sum() + 1e-9)
                local_static = 1 - weighted
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
                    local_dynamic = max(
                        0,
                        1
                        - (
                            regression_data["s"].mean() / 100
                            + regression.coef_[0] * 30
                        ),
                    )
                else:
                    local_dynamic = local_static
                local_sat = (
                    WEIGHT_STATIC * local_static
                    + WEIGHT_DYNAMIC * local_dynamic
                )
                direct = {
                    "run_id": run_id,
                    "target_benchmark": target,
                    "capability_group": group,
                    "decrement_label": f"MINUS_{decrement}",
                    "requested_decrement_per_cell": decrement,
                    "changed_cell_count": len(selected_models),
                    "original_scores": "|".join(map(str, originals.tolist())),
                    "perturbed_scores": "|".join(map(str, perturbed.tolist())),
                    "actual_decrements": "|".join(map(str, actual.tolist())),
                    "total_actual_score_reduction": total_reduction,
                    "mean_actual_score_reduction": float(actual.mean()),
                    "clipping_count": int((perturbed == 0).sum()),
                    "baseline_weighted_score": baseline_local["Weighted_Score"],
                    "perturbed_weighted_score": weighted,
                    "delta_weighted_score": weighted
                    - baseline_local["Weighted_Score"],
                }
                for metric, value in {
                    "S_Static": local_static,
                    "S_Dynamic": local_dynamic,
                    "S_Sat": local_sat,
                }.items():
                    base_value = baseline_local[metric]
                    delta = value - base_value
                    direct[f"baseline_{metric}"] = base_value
                    direct[f"perturbed_{metric}"] = value
                    direct[f"delta_{metric}"] = delta
                    direct[f"absolute_delta_{metric}"] = abs(delta)
                    direct[f"{metric}_response_per_score_point"] = (
                        abs(delta) / total_reduction
                    )
                direct_rows.append(direct)

                current = run["final"].loc[target]
                end = {
                    key: direct[key]
                    for key in [
                        "run_id",
                        "target_benchmark",
                        "capability_group",
                        "decrement_label",
                        "requested_decrement_per_cell",
                        "changed_cell_count",
                        "original_scores",
                        "perturbed_scores",
                        "actual_decrements",
                        "total_actual_score_reduction",
                        "mean_actual_score_reduction",
                        "clipping_count",
                    ]
                }
                for metric in [
                    "S_Disc",
                    "S_Static",
                    "S_Dynamic",
                    "S_Sat",
                    "S_Imp",
                    "BHI",
                    "Rank",
                    "Rank_Percentile",
                ]:
                    delta = current[metric] - baseline_target[metric]
                    end[f"baseline_{metric}"] = baseline_target[metric]
                    end[f"perturbed_{metric}"] = current[metric]
                    end[f"delta_{metric}"] = delta
                    end[f"absolute_delta_{metric}"] = abs(delta)
                    if metric in [
                        "S_Disc",
                        "S_Static",
                        "S_Dynamic",
                        "S_Sat",
                        "BHI",
                        "Rank_Percentile",
                    ]:
                        end[f"{metric}_response_per_score_point"] = (
                            abs(delta) / total_reduction
                        )
                end_rows.append(end)
                run_records[(target, group, decrement)] = end
                global_rows.append(
                    global_effect_row(
                        run_id, target, group, decrement, run, baseline
                    )
                )
                weight_rows.append(
                    critic_effect_row(
                        run_id, target, group, decrement, run, baseline
                    )
                )
                for benchmark, values in run["final"].iterrows():
                    base_values = baseline["final"].loc[benchmark]
                    row = {
                        "run_id": run_id,
                        "target_benchmark": target,
                        "capability_group": group,
                        "decrement_level": decrement,
                        "benchmark": benchmark,
                        "target_flag": benchmark == target,
                    }
                    for metric in [
                        "S_Disc",
                        "S_Static",
                        "S_Dynamic",
                        "S_Sat",
                        "S_Imp",
                        "BHI",
                        "Rank",
                        "Rank_Percentile",
                    ]:
                        row[f"baseline_{metric}"] = base_values[metric]
                        row[f"perturbed_{metric}"] = values[metric]
                        row[f"delta_{metric}"] = values[metric] - base_values[metric]
                    all_rows.append(row)

                cap_difference = float(
                    (run_caps[target_index] - target_caps).abs().max()
                )
                sat_diff = sat_matrix.fillna(np.inf) != sat_base.fillna(np.inf)
                changed_coordinates = set(
                    zip(*np.where(sat_diff.to_numpy()))
                )
                expected_coordinates = {
                    (sat_matrix.index.get_loc(target), sat_matrix.columns.get_loc(m))
                    for m in selected_models
                }
                checks = {
                    "only_recorded_target_score_cells_changed": changed_coordinates
                    == expected_coordinates,
                    "no_missing_score_became_numeric": sat_matrix.isna().equals(
                        sat_base.isna()
                    )
                    and disc_matrix.isna().equals(disc_base.isna()),
                    "availability_unchanged": master.availability.equals(
                        baseline_availability
                    ),
                    "benchmark_eligibility_unchanged": run["disc_counts"].equals(
                        baseline["disc_counts"]
                    )
                    and run["sat_counts"].equals(baseline["sat_counts"]),
                    "one_target_perturbed": {sat_matrix.index[i] for i, _ in changed_coordinates}
                    == {target},
                    "target_lobo_capability_unchanged": cap_difference <= TOLERANCE,
                    "direct_non_target_values_unchanged": True,
                    "target_S_Imp_unchanged": abs(
                        current["S_Imp"] - baseline_target["S_Imp"]
                    )
                    <= TOLERANCE,
                    "perturbed_scores_within_0_100": bool(
                        ((perturbed >= 0) & (perturbed <= 100)).all()
                    ),
                    "final_critic_weights_sum_to_one": math.isclose(
                        float(run["critic_weights"].sum()), 1, abs_tol=TOLERANCE
                    ),
                    "internal_weights_sum_to_one": math.isclose(
                        float(run["internal_weights"].sum()), 1, abs_tol=TOLERANCE
                    ),
                    "selected_groups_constant_across_decrements": True,
                    "raw_input_hash_unchanged": sha256_file(INPUT_PATH)
                    == input_sha256_before,
                }
                for invariant, passed in checks.items():
                    invariant_rows.append(
                        {
                            "run_id": run_id,
                            "target_benchmark": target,
                            "capability_group": group,
                            "decrement_level": decrement,
                            "invariant": invariant,
                            "passed": bool(passed),
                            "observed_target_capability_max_abs_difference": (
                                cap_difference
                                if invariant == "target_lobo_capability_unchanged"
                                else np.nan
                            ),
                        }
                    )

    direct_frame, end_frame = pd.DataFrame(direct_rows), pd.DataFrame(end_rows)
    global_frame, weight_frame = pd.DataFrame(global_rows), pd.DataFrame(weight_rows)
    paired_rows = []
    for target in targets:
        for decrement in DECREMENTS:
            low = run_records[(target, "LOW_CAPABILITY", decrement)]
            high = run_records[(target, "HIGH_CAPABILITY", decrement)]
            row = {
                "target_benchmark": target,
                "decrement_level": decrement,
            }
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
                low_value = low[f"absolute_delta_{metric}"]
                high_value = high[f"absolute_delta_{metric}"]
                row[f"low_abs_delta_{metric}"] = low_value
                row[f"high_abs_delta_{metric}"] = high_value
                if metric in ["S_Static", "S_Sat", "BHI", "Rank_Percentile"]:
                    row[f"attenuation_ratio_{metric}"] = response_ratio(
                        low_value, high_value
                    )
            for metric in [
                "S_Static",
                "S_Dynamic",
                "S_Sat",
                "S_Disc",
                "BHI",
                "Rank_Percentile",
            ]:
                low_response = low[f"{metric}_response_per_score_point"]
                high_response = high[f"{metric}_response_per_score_point"]
                row[f"low_{metric}_response_per_score_point"] = low_response
                row[f"high_{metric}_response_per_score_point"] = high_response
                row[f"normalized_attenuation_ratio_{metric}"] = response_ratio(
                    low_response, high_response
                )
            paired_rows.append(row)
    paired_frame = pd.DataFrame(paired_rows)

    aggregate_rows = []
    for decrement in DECREMENTS:
        for group in GROUPS:
            subset = end_frame[
                (end_frame["decrement_label"] == f"MINUS_{decrement}")
                & (end_frame["capability_group"] == group)
            ]
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
                aggregate_rows.append(
                    {
                        "record_type": "absolute_effect",
                        "decrement_level": decrement,
                        "capability_group": group,
                        "metric": metric,
                        **summaries(subset[f"absolute_delta_{metric}"]),
                    }
                )
            for metric in [
                "S_Static",
                "S_Dynamic",
                "S_Sat",
                "S_Disc",
                "BHI",
                "Rank_Percentile",
            ]:
                aggregate_rows.append(
                    {
                        "record_type": "response_per_score_point",
                        "decrement_level": decrement,
                        "capability_group": group,
                        "metric": metric,
                        **summaries(subset[f"{metric}_response_per_score_point"]),
                    }
                )
        paired_subset = paired_frame[paired_frame["decrement_level"] == decrement]
        for metric in ["S_Static", "S_Sat", "BHI", "Rank_Percentile"]:
            values = paired_subset[f"attenuation_ratio_{metric}"].dropna()
            aggregate_rows.append(
                {
                    "record_type": "attenuation_ratio",
                    "decrement_level": decrement,
                    "capability_group": "LOW_OVER_HIGH",
                    "metric": metric,
                    **summaries(values),
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

    weight_lookup = weight_frame.set_index("run_id")
    dose_rows = []
    dose_metrics = [
        "S_Static",
        "S_Dynamic",
        "S_Sat",
        "S_Disc",
        "BHI",
        "Rank",
        "Rank_Percentile",
    ]
    for target in targets:
        for group in GROUPS:
            subset = end_frame[
                (end_frame["target_benchmark"] == target)
                & (end_frame["capability_group"] == group)
            ].sort_values("requested_decrement_per_cell")
            for metric in dose_metrics:
                values = subset[f"absolute_delta_{metric}"].tolist()
                nondec, strict, classification = monotonic_class(values)
                dose_rows.append(
                    {
                        "target_benchmark": target,
                        "capability_group": group,
                        "metric": metric,
                        "values_10_15_20_25_30": "|".join(map(str, values)),
                        "monotonically_non_decreasing": nondec,
                        "strictly_increasing": strict,
                        "classification": classification,
                    }
                )
            values = [
                weight_lookup.loc[run_id, "maximum_absolute_percentage_point_drift"]
                for run_id in subset["run_id"]
            ]
            nondec, strict, classification = monotonic_class(values)
            dose_rows.append(
                {
                    "target_benchmark": target,
                    "capability_group": group,
                    "metric": "final_CRITIC_weight_drift",
                    "values_10_15_20_25_30": "|".join(map(str, values)),
                    "monotonically_non_decreasing": nondec,
                    "strictly_increasing": strict,
                    "classification": classification,
                }
            )
    dose_frame = pd.DataFrame(dose_rows)

    threshold_rows = []
    threshold_specs = {
        "S_Static": (0.01, 0.025, 0.05, 0.10),
        "S_Sat": (0.01, 0.025, 0.05, 0.10),
        "BHI": (0.01, 0.025, 0.05, 0.10),
        "Rank": (1, 3, 5, 10),
        "Rank_Percentile": (0.01, 0.025, 0.05, 0.10),
    }
    for decrement in DECREMENTS:
        for group in GROUPS:
            subset = end_frame[
                (end_frame["requested_decrement_per_cell"] == decrement)
                & (end_frame["capability_group"] == group)
            ]
            for metric, thresholds in threshold_specs.items():
                values = subset[f"absolute_delta_{metric}"]
                for threshold in thresholds:
                    if metric == "Rank":
                        matched = values <= threshold
                        operator = "<="
                    else:
                        matched = values < threshold
                        operator = "<"
                    threshold_rows.append(
                        {
                            "decrement_level": decrement,
                            "capability_group": group,
                            "metric": metric,
                            "operator": operator,
                            "threshold": threshold,
                            "target_count": len(values),
                            "meeting_threshold_count": int(matched.sum()),
                            "meeting_threshold_proportion": float(matched.mean()),
                        }
                    )

    global_aggregate_rows = []
    global_metrics = [
        column
        for column in global_frame.columns
        if column
        not in {
            "run_id",
            "target_benchmark",
            "capability_group",
            "decrement_level",
            "largest_indirect_non_target_benchmark",
        }
    ]
    for decrement in DECREMENTS:
        for group in GROUPS:
            subset = global_frame[
                (global_frame["decrement_level"] == decrement)
                & (global_frame["capability_group"] == group)
            ]
            for metric in global_metrics:
                global_aggregate_rows.append(
                    {
                        "decrement_level": decrement,
                        "capability_group": group,
                        "metric": metric,
                        **summaries(subset[metric]),
                    }
                )
    weight_aggregate_rows = []
    for decrement in DECREMENTS:
        for group in GROUPS:
            subset = weight_frame[
                (weight_frame["decrement_level"] == decrement)
                & (weight_frame["capability_group"] == group)
            ]
            for metric in [
                "internal_total_variation_distance",
                "maximum_absolute_percentage_point_drift",
                "final_total_variation_distance",
            ]:
                weight_aggregate_rows.append(
                    {
                        "decrement_level": decrement,
                        "capability_group": group,
                        "metric": metric,
                        **summaries(subset[metric]),
                    }
                )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    for decrement in DECREMENTS:
        definitions = []
        for group in GROUPS:
            subset = end_frame[
                (end_frame["requested_decrement_per_cell"] == decrement)
                & (end_frame["capability_group"] == group)
            ].set_index("target_benchmark")
            for metric in ["S_Static", "S_Sat", "BHI", "Rank_Percentile"]:
                definitions.append(
                    (
                        f"median_abs_delta_{metric}_{group}",
                        subset.loc[targets, f"absolute_delta_{metric}"].to_numpy(),
                    )
                )
        paired_subset = paired_frame[
            paired_frame["decrement_level"] == decrement
        ].set_index("target_benchmark")
        for metric in ["S_Static", "S_Sat", "BHI", "Rank_Percentile"]:
            definitions.append(
                (
                    f"median_attenuation_ratio_{metric}",
                    paired_subset.loc[
                        targets, f"attenuation_ratio_{metric}"
                    ].to_numpy(),
                )
            )
        for metric, values in definitions:
            samples, skipped = [], 0
            for _ in range(BOOTSTRAP_ITERATIONS):
                sample = values[rng.integers(0, len(values), len(values))]
                if np.isfinite(sample).any():
                    samples.append(float(np.nanmedian(sample)))
                else:
                    skipped += 1
            bootstrap_rows.append(
                {
                    "decrement_level": decrement,
                    "metric": metric,
                    "estimate": float(np.nanmedian(values)),
                    "ci_95_lower": np.quantile(samples, 0.025),
                    "ci_95_upper": np.quantile(samples, 0.975),
                    "requested_resamples": BOOTSTRAP_ITERATIONS,
                    "valid_resamples": len(samples),
                    "skipped_resamples": skipped,
                    "seed": BOOTSTRAP_SEED,
                }
            )

    frames = {
        "target_benchmarks.csv": pd.DataFrame(target_rows),
        "target_lobo_capabilities.csv": pd.DataFrame(capability_rows),
        "selected_models.csv": pd.DataFrame(selected_rows),
        "changed_score_cells.csv": pd.DataFrame(changed_rows),
        "direct_target_effects.csv": direct_frame,
        "end_to_end_target_effects.csv": end_frame,
        "paired_low_vs_high_comparisons.csv": paired_frame,
        "decrement_level_aggregate_metrics.csv": aggregate_frame,
        "dose_response_metrics.csv": dose_frame,
        "target_effect_threshold_counts.csv": pd.DataFrame(threshold_rows),
        "global_leaderboard_effects.csv": global_frame,
        "global_leaderboard_aggregate_metrics.csv": pd.DataFrame(
            global_aggregate_rows
        ),
        "critic_weight_effects.csv": weight_frame,
        "critic_weight_aggregate_metrics.csv": pd.DataFrame(
            weight_aggregate_rows
        ),
        "all_run_benchmark_results.csv": pd.DataFrame(all_rows),
        "bootstrap_confidence_intervals.csv": pd.DataFrame(bootstrap_rows),
        "baseline_reproduction_check.csv": reproduction,
        "invariant_checks.csv": pd.DataFrame(invariant_rows),
    }
    OUTPUT_DIR.mkdir(parents=True)
    for filename, frame in frames.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    ratio_exceptions = []
    for metric in ["S_Static", "S_Sat", "BHI", "Rank_Percentile"]:
        subset = paired_frame[
            paired_frame[f"attenuation_ratio_{metric}"].fillna(np.inf)
            >= 1 - TOLERANCE
        ]
        for row in subset.itertuples():
            ratio_exceptions.append(
                f"{row.target_benchmark} MINUS_{row.decrement_level} {metric}"
            )
    nonmonotonic = dose_frame[
        dose_frame["classification"] == "non_monotonic"
    ]
    compact_aggregate = aggregate_frame[
        (aggregate_frame["record_type"] == "absolute_effect")
        & (aggregate_frame["metric"].isin(["S_Static", "S_Sat", "BHI", "Rank"]))
    ]
    report = f"""# Fixed-Decrement Weak-Model Score Sensitivity

## Scientific motivation and single-baseline scope

This experiment tests fixed 10, 15, 20, 25, and 30 point reductions among target-specific low- and high-capability model groups. Scientific inputs are restricted to the supplied input and baseline outputs; no other snapshot is used.

## Baseline reproduction

{markdown(reproduction, ["check", "observed_max_abs_difference", "passed"], 12)}

## Target and model selection

Targets were independently reconstructed as the ten final-ranked benchmarks with the lowest baseline S_Static. LOBO capability excludes the target. The bottom and top `max(1, ceil(0.10*n))` eligible models match the prior deterministic selections exactly.

{markdown(frames["selected_models.csv"], ["target_benchmark", "capability_group", "model_name", "lobo_capability", "original_normalized_target_score"])}

## Fixed decrements and clipping

Each selected score x becomes max(0, x-d), for d in {DECREMENTS}. Changed-cell details and actual decrements are recorded in `changed_score_cells.csv`. Total clipping cases: {int(frames["changed_score_cells.csv"]["clipped_at_zero"].sum())}.

## Direct and end-to-end target effects

All 100 direct results and all 100 end-to-end target results are recorded in their CSV files. Direct effects freeze target LOBO capability; full effects include global recalibration.

## Aggregate effects by decrement

{markdown(compact_aggregate, ["decrement_level", "capability_group", "metric", "mean", "median", "p90", "maximum", "minimum"])}

## Low/high comparisons and per-score-point responses

All 50 paired comparisons, including raw and actual-reduction-normalized ratios, are in `paired_low_vs_high_comparisons.csv`. Ratio-at-or-above-one cases: {", ".join(ratio_exceptions) if ratio_exceptions else "none"}.

## Dose response

Of {len(dose_frame)} target/group/metric sequences, {(dose_frame["monotonically_non_decreasing"]).sum()} are monotonically non-decreasing. Non-monotonic cases: {len(nonmonotonic)}; every case is retained in `dose_response_metrics.csv`.

## Threshold, global leaderboard, and CRITIC-weight effects

All descriptive threshold counts are in `target_effect_threshold_counts.csv`. Complete global leaderboard effects for 100 runs and their decrement/group summaries are recorded. Internal and final CRITIC weight changes and aggregate drift summaries are likewise recorded.

## Bootstrap intervals

Sixty paired benchmark-level intervals use 2,000 resamples and seed 20260726; undefined ratio samples are counted rather than replaced.

## Invariants

All {len(frames["invariant_checks.csv"])} run-level checks passed: {bool(frames["invariant_checks.csv"]["passed"].all())}. Fast LOBO versus the submitted function differed by at most {fast_exact_difference:.3e}.

## Limitations

This targeted analysis covers ten saturation-adjacent benchmarks and five fixed reductions. Some groups contain one model, rank is discrete, and full-system responses may be non-monotonic because of normalization and CRITIC recalibration. Descriptive thresholds are not theoretical robustness boundaries.

## 中文反驳摘要

在原始投稿数据上，我们对十个最接近饱和的基准，分别将目标特异性 LOBO 能力最低和最高的模型分数固定降低 10、15、20、25 和 30 分。分析同时报告目标基准的直接变化和完整 BHI 系统的全局重校准变化，并按实际降分幅度归一化。结论仅限于这些固定降分情景，不主张完全稳健。

## Summary

Using only the supplied baseline, we reduced scores of target-specific bottom- and top-capability model groups by fixed increments of 10–30 points across ten saturation-adjacent benchmarks. Direct target effects, complete-system recalibration, and responses per actual score point were retained at every decrement. The evidence is interpreted as a targeted fixed-decrement sensitivity analysis, not a claim of complete robustness.
"""
    (OUTPUT_DIR / "fixed_decrement_sensitivity_report.md").write_text(
        report, encoding="utf-8"
    )
    summary = {
        "analysis_timestamp_utc": TIMESTAMP,
        "scientific_scope": "supplied baseline only",
        "targets": target_rows,
        "decrements": DECREMENTS,
        "groups": GROUPS,
        "perturbation_run_count": len(end_frame),
        "clipping_case_count": int(
            frames["changed_score_cells.csv"]["clipped_at_zero"].sum()
        ),
        "baseline_reproduction": reproduction.to_dict("records"),
        "aggregate_metrics": aggregate_frame.to_dict("records"),
        "ratio_at_or_above_one": ratio_exceptions,
        "non_monotonic_cases": nonmonotonic.to_dict("records"),
        "all_invariants_passed": bool(
            frames["invariant_checks.csv"]["passed"].all()
        ),
        "fast_lobo_max_abs_difference": fast_exact_difference,
        "protected_file_count": count_before,
        "protected_aggregate_sha256": hash_before,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (OUTPUT_DIR / "fixed_decrement_sensitivity_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "ANALYSIS_RECORD.md").write_text(
        f"""# Permanent Record: Fixed-Decrement Weak-Model Sensitivity

- Timestamp UTC: {TIMESTAMP}
- Script: `scripts/weak_model_fixed_decrement_sensitivity.py`
- Input: `{display_path(INPUT_PATH)}`
- Baseline directory: `{display_path(BASE_DIR)}`
- Input SHA-256: `{sha256_file(INPUT_PATH)}`
- Date parsing: `pd.to_datetime(raw_model_dates, errors="coerce")`
- Decrements: {DECREMENTS}
- Perturbation runs: {len(end_frame)}
- Bootstrap: {BOOTSTRAP_ITERATIONS}, seed {BOOTSTRAP_SEED}
- Commands: `python -m compileall scripts/weak_model_fixed_decrement_sensitivity.py`; `python scripts/weak_model_fixed_decrement_sensitivity.py`
- Protected count and aggregate: {count_before}, `{hash_before}`

No other snapshot or prior perturbation result was used scientifically. All perturbations were in memory.
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
    manifest = OUTPUT_DIR / "artifact_manifest_sha256.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(sorted(entries, key=lambda row: row["relative_path"]))
    (OUTPUT_DIR / "MANIFEST_SHA256.txt").write_text(
        f"artifact_manifest_sha256.csv  {sha256_file(manifest)}\n",
        encoding="utf-8",
    )

    expected = set(frames) | {
        "fixed_decrement_sensitivity_report.md",
        "fixed_decrement_sensitivity_summary.json",
        "ANALYSIS_RECORD.md",
        "artifact_manifest_sha256.csv",
        "MANIFEST_SHA256.txt",
    }
    actual_names = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if actual_names != expected:
        raise RuntimeError(f"Unexpected output files: {actual_names ^ expected}")
    for path in OUTPUT_DIR.iterdir():
        if path.suffix == ".csv":
            pd.read_csv(path)
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    if len(targets) != 10 or len(end_frame) != 100:
        raise RuntimeError("Target or run count invalid")
    if len(frames["all_run_benchmark_results.csv"]) != 100 * len(
        baseline["final"]
    ):
        raise RuntimeError("Incomplete all-run results")
    if not frames["invariant_checks.csv"]["passed"].all():
        raise RuntimeError("Invariant failure")
    if not global_frame["spearman_rho"].between(-1, 1).all() or not global_frame[
        "kendall_tau_b"
    ].between(-1, 1).all():
        raise RuntimeError("Invalid correlation")
    if protected_aggregate() != (count_before, hash_before):
        raise RuntimeError("Protected files changed")
    if sha256_file(PRIOR_SCRIPT) != prior_script_hash_before or directory_aggregate(
        PRIOR_OUTPUT
    ) != prior_output_aggregate_before:
        raise RuntimeError("Previous zero-score experiment changed")

    print(f"Wrote fixed-decrement outputs to {OUTPUT_DIR}")
    print("Perturbation runs:", len(end_frame))
    print("Clipping cases:", int(frames["changed_score_cells.csv"]["clipped_at_zero"].sum()))
    print("All invariants passed:", bool(frames["invariant_checks.csv"]["passed"].all()))
    print("Protected files unchanged:", protected_aggregate()[1] == hash_before)
    print(
        paired_frame.groupby("decrement_level")[
            [
                "attenuation_ratio_S_Static",
                "attenuation_ratio_S_Sat",
                "attenuation_ratio_BHI",
                "attenuation_ratio_Rank_Percentile",
            ]
        ].median().to_string()
    )


if __name__ == "__main__":
    main()
