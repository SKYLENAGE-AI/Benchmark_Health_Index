# src/robustness_test_equal.py

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from config.settings import (BOOTSTRAP_ITERATIONS, CONFIG, DROP_RATIOS,
                             IN_MASTER, OUT_DIR, RANDOM_SEED, WEIGHT_DYNAMIC,
                             WEIGHT_STATIC)
from src.experiment_utils import compute_bhi_from_matrices, get_standard_benchmark_rows
from src.preprocessor import MasterDataPreprocessor
from src.utils import configure_matplotlib_fonts


class RobustnessTesterEqual:
    def __init__(self):
        self.pre = MasterDataPreprocessor(CONFIG)
        self.rng = random.Random(RANDOM_SEED)

    def load_and_align(self):
        self.master = self.pre.load_and_process_master(IN_MASTER)
        self.target_benchmarks = get_standard_benchmark_rows(self.master, CONFIG['min_model_count'])
        self.common_models = list(self.master.model_columns)

    def run_bhi_pipeline_equal(self, active_models):
        return compute_bhi_from_matrices(
            self.master.disc_scores,
            self.master.sat_scores,
            self.master.availability,
            self.master.model_dates,
            self.master.model_columns,
            self.target_benchmarks,
            active_models=active_models,
            weighting="equal",
            min_model_count=CONFIG['min_model_count'],
            edr_threshold=CONFIG['edr_threshold'],
            weight_static=WEIGHT_STATIC,
            weight_dynamic=WEIGHT_DYNAMIC,
        )

    def execute_decay_experiment(self):
        self.load_and_align()
        print("Loading standard baseline for equal-weight robustness comparison...")
        baseline_result = compute_bhi_from_matrices(
            self.master.disc_scores,
            self.master.sat_scores,
            self.master.availability,
            self.master.model_dates,
            self.master.model_columns,
            self.target_benchmarks,
            active_models=self.common_models,
            weighting="critic",
            min_model_count=CONFIG['min_model_count'],
            edr_threshold=CONFIG['edr_threshold'],
            weight_static=WEIGHT_STATIC,
            weight_dynamic=WEIGHT_DYNAMIC,
        )
        baseline_bhi = self._load_standard_baseline(baseline_result.bhi)
        baseline_ranks = baseline_bhi.rank(ascending=False)

        results = []
        for ratio in DROP_RATIOS:
            print(f"Testing Equal-Weight Drop Ratio: {ratio}...")
            num_to_drop = int(len(self.common_models) * ratio)
            iter_spearman, iter_kendall = [], []
            iter_top10, iter_shift = [], []
            iter_n_benchmarks = []
            ratio_01_bhis = []

            for _ in range(BOOTSTRAP_ITERATIONS):
                sampled = self.rng.sample(self.common_models, len(self.common_models) - num_to_drop)
                result = self.run_bhi_pipeline_equal(sampled)
                stats = self._compare_with_baseline(baseline_bhi, baseline_ranks, result.bhi)
                iter_spearman.append(stats['spearman'])
                iter_kendall.append(stats['kendall'])
                iter_top10.append(stats['top10_overlap'])
                iter_shift.append(stats['mean_rank_shift'])
                iter_n_benchmarks.append(result.n_benchmarks_used)

                if abs(ratio - 0.1) < 1e-6:
                    ratio_01_bhis.append(result.bhi)

            if abs(ratio - 0.1) < 1e-6:
                self._save_detail_table(ratio_01_bhis, baseline_ranks, f"{OUT_DIR}/robustness_at_0.1_detail_EQUAL.csv")

            results.append({
                'ratio': ratio,
                'drop_ratio': ratio,
                'spearman_mean': np.nanmean(iter_spearman),
                'spearman_std': np.nanstd(iter_spearman),
                'kendall_mean': np.nanmean(iter_kendall),
                'kendall_std': np.nanstd(iter_kendall),
                'top10_overlap_mean': np.nanmean(iter_top10),
                'mean_rank_shift': np.nanmean(iter_shift),
                'weight_disc_mean': 1 / 3,
                'weight_sat_mean': 1 / 3,
                'weight_imp_mean': 1 / 3,
                'n_benchmarks_used_mean': np.nanmean(iter_n_benchmarks),
                'n_benchmarks_missing_mean': len(self.target_benchmarks) - np.nanmean(iter_n_benchmarks),
                'n_models_used': len(self.common_models) - num_to_drop
            })

        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{OUT_DIR}/stability_decay_results_EQUAL.csv", index=False)
        self.plot_curves_equal(df_res)

    def _load_standard_baseline(self, fallback_bhi):
        try:
            baseline = pd.read_csv(f"{OUT_DIR}/final_bhi_rankings.csv").set_index('Benchmark')['BHI']
            common = baseline.index.intersection(fallback_bhi.index)
            return baseline.loc[common]
        except FileNotFoundError:
            return fallback_bhi

    def _compare_with_baseline(self, baseline_bhi, baseline_ranks, iter_bhi):
        common = baseline_bhi.index.intersection(iter_bhi.index)
        if len(common) < 2:
            return {'spearman': np.nan, 'kendall': np.nan, 'top10_overlap': np.nan, 'mean_rank_shift': np.nan}

        iter_ranks = iter_bhi.loc[common].rank(ascending=False)
        return {
            'spearman': spearmanr(baseline_bhi.loc[common], iter_bhi.loc[common])[0],
            'kendall': kendalltau(baseline_bhi.loc[common], iter_bhi.loc[common])[0],
            'top10_overlap': len(set(baseline_bhi.loc[common].nlargest(10).index) & set(iter_bhi.loc[common].nlargest(10).index)),
            'mean_rank_shift': (baseline_ranks.loc[common] - iter_ranks).abs().mean(),
        }

    def _save_detail_table(self, bhi_samples, baseline_ranks, output_path):
        if not bhi_samples:
            return

        df_boots = pd.concat(bhi_samples, axis=1)
        rank_boots = df_boots.rank(ascending=False, axis=0)
        summary = pd.DataFrame({
            'Benchmark': self.target_benchmarks,
            'Baseline_Rank': baseline_ranks.reindex(self.target_benchmarks).values,
            'Mean_Rank': rank_boots.reindex(self.target_benchmarks).mean(axis=1).values,
            'CI_Lower': rank_boots.reindex(self.target_benchmarks).quantile(0.025, axis=1).values,
            'CI_Upper': rank_boots.reindex(self.target_benchmarks).quantile(0.975, axis=1).values,
        }).sort_values('Baseline_Rank')
        summary.to_csv(output_path, index=False)
        print(f"The detailed ranking statistics at ratio 0.1 have been saved: {output_path}")

    def plot_curves_equal(self, df):
        import matplotlib.ticker as ticker

        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df['ratio'], df['spearman_mean'],
                label=r"Equal Spearman's $\rho$",
                marker='o', color='#87ceeb', linewidth=2)
        ax.fill_between(df['ratio'],
                        df['spearman_mean'] - df['spearman_std'],
                        df['spearman_mean'] + df['spearman_std'],
                        color='#87ceeb', alpha=0.2)

        ax.plot(df['ratio'], df['kendall_mean'],
                label=r"Equal Kendall's $\tau$",
                marker='s', color='#ffcc99', linewidth=2)
        ax.fill_between(df['ratio'],
                        df['kendall_mean'] - df['kendall_std'],
                        df['kendall_mean'] + df['kendall_std'],
                        color='#ffcc99', alpha=0.2)

        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.6, label='Stability Threshold (0.9)')
        ax.set_xlim(0.00, 0.60)
        ax.set_ylim(0.45, 1.00)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        ax.legend(frameon=True, loc='lower left')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        output_path = f"{OUT_DIR}/stability_decay_curve_EQUAL.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"The equal-weight vector attenuation curve has been saved to: {output_path}")
