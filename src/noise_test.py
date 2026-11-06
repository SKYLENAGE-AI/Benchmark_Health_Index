# src/noise_test.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from config.settings import (CONFIG, IN_MASTER, NOISE_ITERATIONS, NOISE_SIGMAS,
                             OUT_DIR, RANDOM_SEED, WEIGHT_DYNAMIC, WEIGHT_STATIC)
from src.experiment_utils import (calculate_ordinal_ranks,
                                  calculate_rank_tiers,
                                  compare_tier_stability,
                                  compute_bhi_from_matrices,
                                  get_standard_benchmark_rows)
from src.preprocessor import MasterDataPreprocessor
from src.utils import configure_matplotlib_fonts


class NoiseTester:
    def __init__(self):
        self.pre = MasterDataPreprocessor(CONFIG)
        self.rng = np.random.default_rng(RANDOM_SEED)

    def load_and_align(self):
        self.master = self.pre.load_and_process_master(IN_MASTER)
        self.target_benchmarks = get_standard_benchmark_rows(self.master, CONFIG['min_model_count'])
        self.common_models = list(self.master.model_columns)

    def run_bhi_pipeline(self, noise_map=None):
        disc_scores = self.master.disc_scores.loc[self.target_benchmarks, self.common_models].copy()
        sat_scores = self.master.sat_scores.loc[self.target_benchmarks, self.common_models].copy()

        if noise_map is not None:
            disc_scores = (disc_scores * (1 + noise_map)).clip(0, 100)
            sat_scores = (sat_scores * (1 + noise_map)).clip(0, 100)

        return compute_bhi_from_matrices(
            disc_scores,
            sat_scores,
            self.master.availability.loc[self.target_benchmarks, self.common_models],
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

    def execute_noise_experiment(self):
        self.load_and_align()
        baseline_result = self.run_bhi_pipeline(noise_map=None)
        baseline_bhi = self._load_standard_baseline(baseline_result.bhi)
        baseline_ranks = baseline_bhi.rank(ascending=False)
        baseline_tiers = calculate_rank_tiers(calculate_ordinal_ranks(baseline_bhi), tier_size=10)
        top20_denominator = max(1, min(20, len(baseline_bhi)))

        decay_stats = []
        for sigma in NOISE_SIGMAS:
            print(f"Testing Noise Sigma: {sigma}...")
            iter_rho, iter_tau, iter_pearson, iter_bhis = [], [], [], []
            iter_mae, iter_score_stability = [], []
            iter_top10, iter_top20, iter_shift = [], [], []
            iter_same_tier, iter_adjacent_tier = [], []
            iter_mean_tier_shift, iter_max_tier_shift = [], []
            iter_w_disc, iter_w_sat, iter_w_imp = [], [], []
            iter_n_benchmarks = []

            for _ in range(NOISE_ITERATIONS):
                noise_map = pd.DataFrame(
                    self.rng.normal(0, sigma, size=(len(self.target_benchmarks), len(self.common_models))),
                    index=self.target_benchmarks,
                    columns=self.common_models,
                )
                result = self.run_bhi_pipeline(noise_map)
                stats = self._compare_with_baseline(baseline_bhi, baseline_ranks, result.bhi)
                tier_stats = compare_tier_stability(baseline_tiers, result.bhi, tier_size=10)
                iter_rho.append(stats['spearman'])
                iter_tau.append(stats['kendall'])
                iter_pearson.append(stats['bhi_score_pearson'])
                iter_mae.append(stats['bhi_score_mae'])
                iter_score_stability.append(stats['bhi_score_stability'])
                iter_top10.append(stats['top10_overlap'])
                iter_top20.append(stats['top20_overlap'])
                iter_shift.append(stats['mean_rank_shift'])
                iter_same_tier.append(tier_stats['same_tier_retention'])
                iter_adjacent_tier.append(tier_stats['adjacent_tier_retention'])
                iter_mean_tier_shift.append(tier_stats['mean_abs_tier_shift'])
                iter_max_tier_shift.append(tier_stats['max_tier_shift'])
                iter_w_disc.append(result.weights.get('S_Disc', np.nan))
                iter_w_sat.append(result.weights.get('S_Sat', np.nan))
                iter_w_imp.append(result.weights.get('S_Imp', np.nan))
                iter_n_benchmarks.append(result.n_benchmarks_used)

                if abs(sigma - 0.05) < 1e-6:
                    iter_bhis.append(result.bhi)

            if abs(sigma - 0.05) < 1e-6:
                self._save_detail_table(iter_bhis, baseline_ranks, f"{OUT_DIR}/noise_at_0.05_detail.csv")

            decay_stats.append({
                'sigma': sigma,
                'noise_sigma': sigma,
                'rho_mean': np.nanmean(iter_rho),
                'rho_std': np.nanstd(iter_rho),
                'tau_mean': np.nanmean(iter_tau),
                'tau_std': np.nanstd(iter_tau),
                'bhi_score_pearson_mean': np.nanmean(iter_pearson),
                'bhi_score_pearson_std': np.nanstd(iter_pearson),
                'bhi_score_mae_mean': np.nanmean(iter_mae),
                'bhi_score_mae_std': np.nanstd(iter_mae),
                'bhi_score_stability_mean': np.nanmean(iter_score_stability),
                'bhi_score_stability_std': np.nanstd(iter_score_stability),
                'top10_overlap_mean': np.nanmean(iter_top10),
                'top20_overlap_mean': np.nanmean(iter_top20),
                'top20_overlap_rate_mean': np.nanmean(iter_top20) / top20_denominator,
                'mean_rank_shift': np.nanmean(iter_shift),
                'same_tier_retention_10': np.nanmean(iter_same_tier),
                'same_tier_retention_10_std': np.nanstd(iter_same_tier),
                'adjacent_tier_retention_10': np.nanmean(iter_adjacent_tier),
                'adjacent_tier_retention_10_std': np.nanstd(iter_adjacent_tier),
                'mean_abs_tier_shift_10': np.nanmean(iter_mean_tier_shift),
                'mean_abs_tier_shift_10_std': np.nanstd(iter_mean_tier_shift),
                'max_tier_shift_10': np.nanmean(iter_max_tier_shift),
                'max_tier_shift_10_std': np.nanstd(iter_max_tier_shift),
                'weight_disc_mean': np.nanmean(iter_w_disc),
                'weight_sat_mean': np.nanmean(iter_w_sat),
                'weight_imp_mean': np.nanmean(iter_w_imp),
                'n_benchmarks_used_mean': np.nanmean(iter_n_benchmarks),
                'n_benchmarks_missing_mean': len(self.target_benchmarks) - np.nanmean(iter_n_benchmarks),
                'n_models_used': len(self.common_models),
            })

        df_res = pd.DataFrame(decay_stats)
        df_res.to_csv(f"{OUT_DIR}/noise_decay_results.csv", index=False)
        self.plot_noise_curves(df_res)
        self.plot_noise_curves_mae(df_res)

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
            return {
                'spearman': np.nan,
                'kendall': np.nan,
                'bhi_score_pearson': np.nan,
                'bhi_score_mae': np.nan,
                'bhi_score_stability': np.nan,
                'top10_overlap': np.nan,
                'top20_overlap': np.nan,
                'mean_rank_shift': np.nan,
            }

        iter_ranks = iter_bhi.loc[common].rank(ascending=False)
        top10_n = min(10, len(common))
        top20_n = min(20, len(common))
        score_mae = (iter_bhi.loc[common] - baseline_bhi.loc[common]).abs().mean()
        return {
            'spearman': spearmanr(baseline_bhi.loc[common], iter_bhi.loc[common])[0],
            'kendall': kendalltau(baseline_bhi.loc[common], iter_bhi.loc[common])[0],
            'bhi_score_pearson': baseline_bhi.loc[common].corr(iter_bhi.loc[common], method='pearson'),
            'bhi_score_mae': score_mae,
            'bhi_score_stability': 1 - score_mae,
            'top10_overlap': len(set(baseline_bhi.loc[common].nlargest(top10_n).index) & set(iter_bhi.loc[common].nlargest(top10_n).index)),
            'top20_overlap': len(set(baseline_bhi.loc[common].nlargest(top20_n).index) & set(iter_bhi.loc[common].nlargest(top20_n).index)),
            'mean_rank_shift': (baseline_ranks.loc[common] - iter_ranks).abs().mean(),
        }

    def _save_detail_table(self, bhi_samples, baseline_ranks, output_path):
        if not bhi_samples:
            return

        df_05 = pd.concat(bhi_samples, axis=1)
        rank_05 = df_05.rank(ascending=False, axis=0)
        detail = pd.DataFrame({
            'Benchmark': self.target_benchmarks,
            'Baseline_Rank': baseline_ranks.reindex(self.target_benchmarks).values,
            'Mean_Rank': rank_05.reindex(self.target_benchmarks).mean(axis=1).values,
            'CI_Lower': rank_05.reindex(self.target_benchmarks).quantile(0.025, axis=1).values,
            'CI_Upper': rank_05.reindex(self.target_benchmarks).quantile(0.975, axis=1).values,
        }).sort_values('Baseline_Rank')
        detail.to_csv(output_path, index=False)
        print(f"0.05 detailed table has been saved to {output_path}.")

    def plot_noise_curves(self, df):
        import matplotlib.ticker as ticker

        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df['sigma'],
                df['rho_mean'],
                label='Spearman rank correlation',
                marker='o', color='#2E75B6', linewidth=2.2)
        ax.fill_between(df['sigma'],
                        df['rho_mean'] - df['rho_std'],
                        df['rho_mean'] + df['rho_std'],
                        color='#2E75B6', alpha=0.16)

        ax.plot(df['sigma'],
                df['bhi_score_pearson_mean'],
                label='Pearson correlation of BHI scores',
                marker='s', color='#70AD47', linewidth=2.2)
        ax.fill_between(df['sigma'],
                        df['bhi_score_pearson_mean'] - df['bhi_score_pearson_std'],
                        df['bhi_score_pearson_mean'] + df['bhi_score_pearson_std'],
                        color='#70AD47', alpha=0.16)

        ax.set_title('Noise Robustness of BHI under CRITIC Weighting')
        ax.set_xlabel('Noise sigma')
        ax.set_ylabel('Stability metric')
        ax.set_xlim(0.00, 0.20)
        ax.set_ylim(0.00, 1.00)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        ax.legend(frameon=True, loc='lower left')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        output_path = f"{OUT_DIR}/noise_decay_curve.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"The noise robustness vector curve has been saved to: {output_path}")

    def plot_noise_curves_mae(self, df):
        import matplotlib.ticker as ticker

        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, ax = plt.subplots(figsize=(10, 6))

        spearman_color = '#6A1B9A'
        stability_color = '#D6A700'

        ax.plot(df['sigma'],
                df['rho_mean'],
                label='Spearman rank correlation',
                marker='o', color=spearman_color, linewidth=2.2)
        ax.fill_between(df['sigma'],
                        df['rho_mean'] - df['rho_std'],
                        df['rho_mean'] + df['rho_std'],
                        color=spearman_color, alpha=0.16)

        ax.plot(df['sigma'],
                df['bhi_score_stability_mean'],
                label='BHI score stability (1 - MAE)',
                marker='s', color=stability_color, linewidth=2.2)
        ax.fill_between(df['sigma'],
                        df['bhi_score_stability_mean'] - df['bhi_score_stability_std'],
                        df['bhi_score_stability_mean'] + df['bhi_score_stability_std'],
                        color=stability_color, alpha=0.20)

        lower_bound = np.nanmin([
            (df['rho_mean'] - df['rho_std']).min(),
            (df['bhi_score_stability_mean'] - df['bhi_score_stability_std']).min(),
        ])
        upper_bound = np.nanmax([
            (df['rho_mean'] + df['rho_std']).max(),
            (df['bhi_score_stability_mean'] + df['bhi_score_stability_std']).max(),
        ])

        ax.axhline(y=0.9,
                   color='red',
                   linestyle='--',
                   linewidth=1.6,
                   alpha=0.75,
                   label='0.9 reference')
        ax.set_title('Noise Robustness of BHI under CRITIC Weighting (MAE Variant)')
        ax.set_xlabel('Noise sigma')
        ax.set_ylabel('Stability metric')
        ax.set_xlim(0.00, 0.20)
        ax.set_ylim(min(0.60, lower_bound), max(1.00, upper_bound))
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        ax.legend(frameon=True, loc='lower left')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        output_path = f"{OUT_DIR}/noise_decay_curve_MAE.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"The MAE-variant noise robustness curve has been saved to: {output_path}")
