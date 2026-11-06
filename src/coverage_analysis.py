import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import OUT_DIR
from src.utils import configure_matplotlib_fonts


class CoverageBiasAnalysis:
    TARGET_METRICS = ['S_Disc', 'S_Sat', 'S_Imp', 'BHI']

    def run(self):
        print("Starting coverage bias analysis...")
        data = self._load_data()

        corr = self._calculate_correlations(data)
        corr_path = os.path.join(OUT_DIR, 'coverage_correlation.csv')
        corr.to_csv(corr_path, index=False)

        top20 = self._top20_table(data)
        top20_path = os.path.join(OUT_DIR, 'top20_coverage_analysis.csv')
        top20.to_csv(top20_path, index=False)

        self._plot_coverage_vs_bhi(data)

        print(f"Coverage correlation table saved to: {corr_path}")
        print(f"Top-20 coverage analysis saved to: {top20_path}")

    def _load_data(self):
        final = self._read_required_csv(
            os.path.join(OUT_DIR, 'final_bhi_rankings.csv'),
            ['Benchmark', 'S_Disc', 'S_Sat', 'S_Imp', 'BHI']
        )
        impact = self._read_required_csv(
            os.path.join(OUT_DIR, 'impact_results.csv'),
            ['Benchmark', 'N_Models_With_Score']
        ).rename(columns={'N_Models_With_Score': 'Coverage'})

        merged = final.merge(impact, on='Benchmark', how='inner')
        return merged.dropna(subset=['Coverage'] + self.TARGET_METRICS).copy()

    def _calculate_correlations(self, data):
        rows = []
        for metric in self.TARGET_METRICS:
            rows.append({
                'Metric': metric,
                'Pearson': data['Coverage'].corr(data[metric], method='pearson'),
                'Spearman': data['Coverage'].corr(data[metric], method='spearman'),
            })
        return pd.DataFrame(rows)

    def _top20_table(self, data):
        top20 = data.sort_values('BHI', ascending=False).head(20).copy()
        top20.insert(0, 'Rank', np.arange(1, len(top20) + 1))
        return top20[['Rank', 'Benchmark', 'Coverage', 'S_Disc', 'S_Sat', 'S_Imp', 'BHI']]

    def _plot_coverage_vs_bhi(self, data):
        import matplotlib.ticker as ticker

        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, ax = plt.subplots(figsize=(10, 6))

        x = data['Coverage'].astype(float)
        y = data['BHI'].astype(float)

        ax.scatter(x, y, color='#6A1B9A', alpha=0.72, edgecolor='white', linewidth=0.6)

        if len(data) >= 2 and x.nunique() > 1:
            coef = np.polyfit(x, y, deg=1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color='#D6A700', linewidth=2.2, label='Linear trend')
            ax.legend(frameon=True, loc='best')

        ax.set_title('Coverage Bias Analysis')
        ax.set_xlabel('Coverage (N_Models_With_Score)')
        ax.set_ylabel('BHI')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        output_path = os.path.join(OUT_DIR, 'coverage_vs_bhi.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Coverage vs BHI scatter plot saved to: {output_path}")

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
