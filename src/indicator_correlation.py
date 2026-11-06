import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import OUT_DIR
from src.utils import configure_matplotlib_fonts


class IndicatorCorrelationAnalysis:
    INDICATORS = ['S_Disc', 'S_Sat', 'S_Imp']

    def run(self):
        print("Starting indicator orthogonality analysis...")
        indicators = self._load_indicator_scores()

        pearson = indicators.corr(method='pearson')
        spearman = indicators.corr(method='spearman')

        pearson_path = os.path.join(OUT_DIR, 'indicator_pearson_correlation.csv')
        spearman_path = os.path.join(OUT_DIR, 'indicator_spearman_correlation.csv')
        pearson.to_csv(pearson_path)
        spearman.to_csv(spearman_path)

        self._plot_heatmap(pearson, spearman)

        print(f"Pearson indicator correlation saved to: {pearson_path}")
        print(f"Spearman indicator correlation saved to: {spearman_path}")

    def plot_precomputed(self, pearson_path, spearman_path, output_path):
        """Plot preserved correlation matrices without recomputing them."""
        pearson = pd.read_csv(pearson_path, index_col=0)
        spearman = pd.read_csv(spearman_path, index_col=0)

        for name, matrix in [('Pearson', pearson), ('Spearman', spearman)]:
            if list(matrix.index) != self.INDICATORS:
                raise ValueError(f"{name} matrix has unexpected row labels: {list(matrix.index)}")
            if list(matrix.columns) != self.INDICATORS:
                raise ValueError(
                    f"{name} matrix has unexpected column labels: {list(matrix.columns)}"
                )

        self._plot_heatmap(pearson, spearman, output_path=output_path)

    def _load_indicator_scores(self):
        final_path = os.path.join(OUT_DIR, 'final_bhi_rankings.csv')
        if os.path.exists(final_path):
            df_final = pd.read_csv(final_path)
            if all(col in df_final.columns for col in ['Benchmark'] + self.INDICATORS):
                return df_final.set_index('Benchmark')[self.INDICATORS].dropna()

        df_disc = self._read_required_csv(
            os.path.join(OUT_DIR, 'capability_discrimination_final.csv'),
            ['Benchmark', 'S_Disc']
        )
        df_sat = self._read_required_csv(
            os.path.join(OUT_DIR, 'anti_saturation_results.csv'),
            ['Benchmark', 'S_Sat']
        )
        df_imp = self._read_required_csv(
            os.path.join(OUT_DIR, 'impact_results.csv'),
            ['Benchmark', 'S_Imp']
        )

        return (
            df_disc
            .merge(df_sat, on='Benchmark')
            .merge(df_imp, on='Benchmark')
            .set_index('Benchmark')[self.INDICATORS]
            .dropna()
        )

    def _plot_heatmap(self, pearson, spearman, output_path=None):
        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

        self._draw_matrix(axes[0], pearson, 'Pearson')
        im = self._draw_matrix(axes[1], spearman, 'Spearman')

        fig.suptitle('Indicator Correlation Analysis', fontsize=14)
        cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
        cbar.set_label('Correlation')

        output_path = output_path or os.path.join(
            OUT_DIR, 'indicator_correlation_heatmap.pdf'
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Indicator correlation heatmap saved to: {output_path}")

    def _draw_matrix(self, ax, matrix, title):
        values = matrix.loc[self.INDICATORS, self.INDICATORS].values
        im = ax.imshow(values, cmap='RdBu_r', vmin=-1, vmax=1)

        ax.set_title(title)
        ax.set_xticks(np.arange(len(self.INDICATORS)))
        ax.set_yticks(np.arange(len(self.INDICATORS)))
        ax.set_xticklabels(self.INDICATORS, rotation=35, ha='right')
        ax.set_yticklabels(self.INDICATORS)

        for i in range(len(self.INDICATORS)):
            for j in range(len(self.INDICATORS)):
                val = values[i, j]
                text_color = 'white' if abs(val) > 0.65 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color)

        ax.set_xticks(np.arange(-.5, len(self.INDICATORS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(self.INDICATORS), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        return im

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
