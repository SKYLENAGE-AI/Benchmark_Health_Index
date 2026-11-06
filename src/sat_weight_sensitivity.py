import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from config.settings import OUT_DIR
from src.utils import configure_matplotlib_fonts


class SatWeightSensitivity:
    WEIGHT_SETTINGS = [
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2),
        (0.9, 0.1),
    ]

    def __init__(self):
        self.df_disc = self._read_required_csv(
            os.path.join(OUT_DIR, 'capability_discrimination_final.csv'),
            ['Benchmark', 'S_Disc']
        )
        self.df_sat = self._read_required_csv(
            os.path.join(OUT_DIR, 'anti_saturation_results.csv'),
            ['Benchmark', 'S_Static', 'S_Dynamic']
        )
        self.df_imp = self._read_required_csv(
            os.path.join(OUT_DIR, 'impact_results.csv'),
            ['Benchmark', 'S_Imp']
        )
        self.baseline_bhi = self._read_required_csv(
            os.path.join(OUT_DIR, 'final_bhi_rankings.csv'),
            ['Benchmark', 'BHI']
        )
        self.baseline_ranks = pd.Series(
            np.arange(1, len(self.baseline_bhi) + 1),
            index=self.baseline_bhi['Benchmark']
        )

    def run(self):
        print("Starting static/dynamic weight sensitivity experiment...")

        rows = []
        for static_weight, dynamic_weight in self.WEIGHT_SETTINGS:
            print(f"Processing S_Static/S_Dynamic = {static_weight:.1f}/{dynamic_weight:.1f}...")
            bhi = self._calculate_bhi(static_weight, dynamic_weight)
            stats = self._compare_to_baseline(bhi)
            rows.append({
                'Static_Weight': static_weight,
                'Dynamic_Weight': dynamic_weight,
                'Spearman': stats['spearman'],
                'Kendall': stats['kendall'],
                'Top10_Overlap': stats['top10_overlap'],
                'Top20_Overlap': stats['top20_overlap'],
                'Avg_Rank_Shift': stats['avg_rank_shift'],
                'Max_Rank_Shift': stats['max_rank_shift'],
            })

        df_results = pd.DataFrame(rows)
        csv_path = os.path.join(OUT_DIR, 'static_dynamic_weight_sensitivity.csv')
        df_results.to_csv(csv_path, index=False)
        self._plot(df_results)
        print(f"Static/dynamic weight sensitivity results saved to: {csv_path}")

    def _calculate_bhi(self, static_weight, dynamic_weight):
        df_sat_variant = self.df_sat.copy()
        df_sat_variant['S_Sat'] = (
            static_weight * df_sat_variant['S_Static']
            + dynamic_weight * df_sat_variant['S_Dynamic']
        )

        final = (
            self.df_disc[['Benchmark', 'S_Disc']]
            .merge(df_sat_variant[['Benchmark', 'S_Sat']], on='Benchmark')
            .merge(self.df_imp[['Benchmark', 'S_Imp']], on='Benchmark')
            .dropna(subset=['S_Disc', 'S_Sat', 'S_Imp'])
            .set_index('Benchmark')
        )

        weights = self._critic_weights(final[['S_Disc', 'S_Sat', 'S_Imp']])
        return (
            final['S_Disc'] * weights['S_Disc']
            + final['S_Sat'] * weights['S_Sat']
            + final['S_Imp'] * weights['S_Imp']
        )

    def _critic_weights(self, indicators):
        std = indicators.std()
        conflict = (1 - indicators.corr()).sum()
        c_metric = std * conflict
        return c_metric / (c_metric.sum() + 1e-9)

    def _compare_to_baseline(self, bhi):
        common = self.baseline_ranks.index.intersection(bhi.index)
        if len(common) < 2:
            return {
                'spearman': np.nan,
                'kendall': np.nan,
                'top10_overlap': np.nan,
                'top20_overlap': np.nan,
                'avg_rank_shift': np.nan,
                'max_rank_shift': np.nan,
            }

        baseline_ranks = self.baseline_ranks.loc[common]
        current_ranks = self._ordinal_ranks(bhi.loc[common]).loc[common]
        rank_shift = (baseline_ranks - current_ranks).abs()

        baseline_ordered = baseline_ranks.sort_values().index
        current_ordered = current_ranks.sort_values().index
        top10_n = min(10, len(common))
        top20_n = min(20, len(common))

        return {
            'spearman': spearmanr(baseline_ranks, current_ranks)[0],
            'kendall': kendalltau(baseline_ranks, current_ranks)[0],
            'top10_overlap': len(set(baseline_ordered[:top10_n]) & set(current_ordered[:top10_n])),
            'top20_overlap': len(set(baseline_ordered[:top20_n]) & set(current_ordered[:top20_n])),
            'avg_rank_shift': rank_shift.mean(),
            'max_rank_shift': rank_shift.max(),
        }

    def _ordinal_ranks(self, scores):
        ordered = scores.dropna().sort_values(ascending=False, kind='mergesort')
        return pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)

    def _plot(self, df_results):
        import matplotlib.ticker as ticker

        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df_results['Static_Weight'],
                df_results['Spearman'],
                label="Spearman rank correlation",
                marker='o',
                color='#6A1B9A',
                linewidth=2.2)
        ax.plot(df_results['Static_Weight'],
                df_results['Kendall'],
                label="Kendall rank correlation",
                marker='s',
                color='#D6A700',
                linewidth=2.2)

        metric_min = df_results[['Spearman', 'Kendall']].min().min()
        metric_max = df_results[['Spearman', 'Kendall']].max().max()
        ax.axhline(y=0.9,
                   color='red',
                   linestyle='--',
                   linewidth=1.6,
                   alpha=0.75,
                   label='0.9 reference')
        ax.set_title('Sensitivity to Static/Dynamic Weights in Anti-Saturation')
        ax.set_xlabel('Static weight')
        ax.set_ylabel('Rank similarity metric')
        ax.set_xlim(0.55, 0.95)
        ax.set_ylim(min(0.6, metric_min), max(1.0, metric_max))
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.legend(frameon=True, loc='lower left')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        pdf_path = os.path.join(OUT_DIR, 'static_dynamic_weight_sensitivity_curve.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Static/dynamic weight sensitivity curve saved to: {pdf_path}")

    def _read_required_csv(self, path, columns):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}. Please run python main.py first."
            )

        df = pd.read_csv(path)
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        return df[columns].copy()
