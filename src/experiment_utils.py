from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.capability import calculate_lobo_capabilities


@dataclass
class BHIComputationResult:
    bhi: pd.Series
    indicators: pd.DataFrame
    weights: dict
    n_benchmarks_used: int
    n_models_used: int


def get_standard_benchmark_rows(master, min_model_count):
    disc_counts = master.disc_scores.notna().sum(axis=1)
    sat_counts = master.sat_scores.notna().sum(axis=1)
    return [
        bench for bench in master.benchmark_names
        if disc_counts.loc[bench] >= min_model_count
        and sat_counts.loc[bench] >= min_model_count
    ]


def calculate_rank_tiers(ranks, tier_size=10):
    valid_ranks = ranks.dropna().astype(float)
    return (((valid_ranks - 1) // tier_size) + 1).astype(int)


def calculate_ordinal_ranks(scores):
    ordered = scores.dropna().sort_values(ascending=False, kind='mergesort')
    return pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)


def compare_tier_stability(baseline_tiers, iter_bhi, tier_size=10):
    iter_tiers = calculate_rank_tiers(calculate_ordinal_ranks(iter_bhi), tier_size)
    common = baseline_tiers.index.intersection(iter_tiers.index)
    if len(common) == 0:
        return {
            'same_tier_retention': np.nan,
            'adjacent_tier_retention': np.nan,
            'mean_abs_tier_shift': np.nan,
            'max_tier_shift': np.nan,
        }

    tier_shift = (baseline_tiers.loc[common] - iter_tiers.loc[common]).abs()
    return {
        'same_tier_retention': (tier_shift == 0).mean(),
        'adjacent_tier_retention': (tier_shift <= 1).mean(),
        'mean_abs_tier_shift': tier_shift.mean(),
        'max_tier_shift': tier_shift.max(),
    }


def compute_bhi_from_matrices(
    disc_scores,
    sat_scores,
    availability,
    model_dates,
    model_columns,
    benchmark_rows,
    active_models=None,
    weighting="critic",
    min_model_count=3,
    edr_threshold=2.0,
    weight_static=0.8,
    weight_dynamic=0.2,
):
    active_models = list(active_models) if active_models is not None else list(model_columns)
    benchmark_rows = [
        bench for bench in benchmark_rows
        if bench in disc_scores.index
        and bench in sat_scores.index
        and bench in availability.index
    ]

    if not benchmark_rows or not active_models:
        return _empty_result(weighting, len(active_models))

    disc_sub = disc_scores.loc[benchmark_rows, active_models]
    sat_sub = sat_scores.loc[benchmark_rows, active_models]
    avail_sub = availability.loc[benchmark_rows, active_models]

    disc_counts = disc_sub.notna().sum(axis=1)
    sat_counts = sat_sub.notna().sum(axis=1)
    eligible_benchmarks = [
        bench for bench in benchmark_rows
        if disc_counts.loc[bench] >= min_model_count
        and sat_counts.loc[bench] >= min_model_count
    ]

    if not eligible_benchmarks:
        return _empty_result(weighting, len(active_models))

    s_disc = _calculate_disc_scores(disc_sub.loc[eligible_benchmarks], edr_threshold)
    sat_imp = _calculate_sat_and_impact(
        sat_sub.loc[eligible_benchmarks],
        avail_sub.loc[eligible_benchmarks],
        model_dates.reindex(active_models),
        active_models,
        weight_static,
        weight_dynamic,
    )

    indicators = pd.DataFrame({'S_Disc': s_disc}).merge(
        sat_imp[['S_Sat', 'S_Imp']],
        left_index=True,
        right_index=True
    )

    weights = _calculate_final_weights(indicators, weighting)
    indicators['BHI'] = (
        indicators['S_Disc'] * weights['S_Disc']
        + indicators['S_Sat'] * weights['S_Sat']
        + indicators['S_Imp'] * weights['S_Imp']
    )

    return BHIComputationResult(
        bhi=indicators['BHI'],
        indicators=indicators,
        weights=weights,
        n_benchmarks_used=len(indicators),
        n_models_used=len(active_models),
    )


def _calculate_disc_scores(df_disc_norm, edr_threshold):
    disc_list = []
    for b_name, row in df_disc_norm.iterrows():
        v = row.dropna().values
        n = len(v)
        rcv = (np.percentile(v, 90) - np.percentile(v, 10)) / 100.0
        diffs = np.abs(v[:, None] - v)
        edr = (np.sum(diffs > edr_threshold) / 2) / (n * (n - 1) / 2) if n > 1 else 0
        disc_list.append({'Benchmark': b_name, 'Robust_CV': rcv, 'EDR': edr})

    df_disc = pd.DataFrame(disc_list).set_index('Benchmark')
    m_n = (df_disc - df_disc.min()) / (df_disc.max() - df_disc.min() + 1e-9)
    w_d = m_n.std() / (m_n.std().sum() + 1e-9)
    return m_n['Robust_CV'] * w_d['Robust_CV'] + m_n['EDR'] * w_d['EDR']


def _calculate_sat_and_impact(
    df_sat_norm,
    availability,
    model_dates,
    active_models,
    weight_static,
    weight_dynamic,
):
    bench_data = df_sat_norm.reset_index()
    n_total = len(bench_data)
    results = []

    for i in range(n_total):
        b_name = bench_data.loc[i, 'Benchmark']
        caps = calculate_lobo_capabilities(bench_data, i, n_total)

        scores = bench_data.iloc[i, 1:]
        valid = scores.dropna()
        common = valid.index.intersection(caps.index)

        s_static = 1 - (
            ((valid[common] / 100.0) * caps.loc[common, 'Capability_Score']).sum()
            / (caps.loc[common, 'Capability_Score'].sum() + 1e-9)
        )

        v_reg = pd.DataFrame({
            'd': (model_dates - model_dates.min()).dt.days,
            's': scores
        }).dropna()
        if len(v_reg) >= 2:
            reg = LinearRegression().fit(v_reg[['d']], v_reg['s'] / 100.0)
            s_dynamic = max(0, 1 - (v_reg['s'].mean() / 100.0 + reg.coef_[0] * 30))
        else:
            s_dynamic = s_static

        cap = caps['Capability_Score'].reindex(active_models).fillna(0.0)
        denominator = cap.sum()
        evaluated = availability.loc[b_name].reindex(active_models).fillna(False)
        evaluated_models = evaluated[evaluated].index
        numerator = cap.loc[evaluated_models].sum()
        s_imp_raw = numerator / denominator if denominator > 0 else 0.0

        results.append({
            'Benchmark': b_name,
            'S_Static': s_static,
            'S_Dynamic': s_dynamic,
            'S_Sat': weight_static * s_static + weight_dynamic * s_dynamic,
            'S_Imp_Raw': s_imp_raw,
            'S_Imp': np.sqrt(s_imp_raw),
        })

    return pd.DataFrame(results).set_index('Benchmark')


def _calculate_final_weights(indicators, weighting):
    if weighting == "equal":
        return {'S_Disc': 1 / 3, 'S_Sat': 1 / 3, 'S_Imp': 1 / 3}

    std = indicators[['S_Disc', 'S_Sat', 'S_Imp']].std()
    conflict = (1 - indicators[['S_Disc', 'S_Sat', 'S_Imp']].corr()).sum()
    c_metric = std * conflict
    weights = c_metric / (c_metric.sum() + 1e-9)
    return weights.to_dict()


def _empty_result(weighting, n_models_used):
    weights = _calculate_final_weights(
        pd.DataFrame({'S_Disc': [], 'S_Sat': [], 'S_Imp': []}),
        weighting
    ) if weighting == "equal" else {'S_Disc': np.nan, 'S_Sat': np.nan, 'S_Imp': np.nan}
    return BHIComputationResult(
        bhi=pd.Series(dtype=float),
        indicators=pd.DataFrame(columns=['S_Disc', 'S_Sat', 'S_Imp', 'BHI']),
        weights=weights,
        n_benchmarks_used=0,
        n_models_used=n_models_used,
    )
