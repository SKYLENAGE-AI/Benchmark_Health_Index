import os
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config.settings import CONFIG, IN_MASTER, OUT_DIR
from src.preprocessor import MasterDataPreprocessor
from src.utils import configure_matplotlib_fonts, get_clean_name

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


class DynamicTrendPlotter:
    OUTPUT_DIR = os.path.join(OUT_DIR, 'dynamic_trend_plots')
    INDIVIDUAL_FIGSIZE = (13.5, 8.2)
    COMBINED_FIGSIZE = (19.5, 13.5)
    TITLE_FONTSIZE = 24
    AXIS_LABEL_FONTSIZE = 21
    TICK_LABEL_FONTSIZE = 19
    LEGEND_FONTSIZE = 18
    COMBINED_TITLE_FONTSIZE = 22
    COMBINED_AXIS_LABEL_FONTSIZE = 22
    COMBINED_TICK_LABEL_FONTSIZE = 18
    COMBINED_LEGEND_FONTSIZE = 17
    ANNOTATION_FONTSIZE = 20
    MISSING_MESSAGE_FONTSIZE = 19
    PLOT_START = pd.Timestamp('2025-01-01')
    PLOT_END = pd.Timestamp('2026-04-30')
    COMBINED_PLOT_END = pd.Timestamp('2026-05-01')
    SELECTED_BENCHMARKS = [
        "Humanity's Last Exam",
        'MMLU-Pro',
        'AIME 2025',
        'LiveCodeBench-V6',
    ]
    TIME_TICKS = pd.to_datetime([
        '2025-01-01',
        '2025-04-01',
        '2025-07-01',
        '2025-10-01',
        '2026-01-01',
        '2026-04-01',
    ])
    SCORE_TICKS = [20, 40, 60, 80, 100]

    def __init__(self):
        self.pre = MasterDataPreprocessor(CONFIG)
        self.annotation_filter_stats = {
            'drawn': 0,
            'outside_axes': 0,
            'title_overlap': 0,
            'annotation_overlap': 0,
        }

    def run(self):
        print("Starting dynamic anti-saturation trend plotting...")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        master = self.pre.load_and_process_master(IN_MASTER)
        sat_valid_counts = master.sat_scores.notna().sum(axis=1)
        target_benchmarks = master.sat_scores.loc[
            sat_valid_counts >= CONFIG['min_model_count']
        ].index
        model_dates = master.model_dates.reindex(master.model_columns)
        date_origin = model_dates.min()

        summary_rows = []
        used_filenames = set()

        for benchmark in target_benchmarks:
            row = self._plot_one_benchmark(
                benchmark,
                master.sat_scores.loc[benchmark],
                model_dates,
                date_origin,
                used_filenames
            )
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(self.OUTPUT_DIR, 'dynamic_trend_plot_summary.csv')
        summary.to_csv(summary_path, index=False)

        plotted = (summary['Status'] == 'Plotted').sum()
        skipped = (summary['Status'] == 'Skipped').sum()
        print(f"Dynamic trend plots complete. Plotted: {plotted}; skipped: {skipped}.")
        print(f"Summary saved to: {summary_path}")
        self._plot_selected_combined_figure(master, model_dates, date_origin)

    def plot_selected_combined(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        master = self.pre.load_and_process_master(IN_MASTER)
        model_dates = master.model_dates.reindex(master.model_columns)
        date_origin = model_dates.min()
        self._plot_selected_combined_figure(master, model_dates, date_origin)

    def _plot_one_benchmark(self, benchmark, scores, model_dates, date_origin, used_filenames):
        points = pd.DataFrame({
            'Model': scores.index,
            'Date': model_dates.reindex(scores.index),
            'Score': scores.values,
        }).dropna(subset=['Date', 'Score'])
        points = points.sort_values(['Date', 'Score']).reset_index(drop=True)

        if len(points) < 2 or pd.isna(date_origin):
            return {
                'Benchmark': benchmark,
                'N_Points': len(points),
                'Slope': np.nan,
                'Intercept': np.nan,
                'R2': np.nan,
                'Plot_File': '',
                'Status': 'Skipped',
                'Reason_If_Skipped': 'Fewer than 2 valid dated score points',
            }

        points['Days'] = (points['Date'] - date_origin).dt.days
        reg = LinearRegression().fit(points[['Days']], points['Score'] / 100.0)
        r2 = reg.score(points[['Days']], points['Score'] / 100.0)

        plot_file = self._unique_plot_file(benchmark, used_filenames)
        self._draw_plot(benchmark, points, reg, r2, date_origin, plot_file)

        return {
            'Benchmark': benchmark,
            'N_Points': len(points),
            'Slope': float(reg.coef_[0]),
            'Intercept': float(reg.intercept_),
            'R2': float(r2),
            'Plot_File': plot_file,
            'Status': 'Plotted',
            'Reason_If_Skipped': '',
        }

    def _draw_plot(self, benchmark, points, reg, r2, date_origin, plot_file):
        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, ax = plt.subplots(
            figsize=self.INDIVIDUAL_FIGSIZE,
            constrained_layout=True
        )

        ax.scatter(
            points['Date'],
            points['Score'],
            color='#0000FF',
            alpha=0.82,
            edgecolor='white',
            linewidth=0.6,
            zorder=3
        )

        line_dates = pd.date_range(self.PLOT_START, self.PLOT_END, periods=100)
        line_days = (line_dates - date_origin).days.to_numpy()
        line_scores = reg.predict(pd.DataFrame({'Days': line_days})) * 100.0
        slope_score_per_day = reg.coef_[0] * 100.0
        ax.plot(
            line_dates,
            line_scores,
            color='red',
            linestyle='--',
            linewidth=2.0,
            label=f'Linear fit (k={slope_score_per_day:.3f}, R$^2$={r2:.2f})',
            zorder=2
        )

        ax.set_title(str(benchmark), fontsize=self.TITLE_FONTSIZE, pad=10)
        ax.set_xlabel('Time', fontsize=self.AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Score', fontsize=self.AXIS_LABEL_FONTSIZE)
        ax.set_xlim(self.PLOT_START, self.PLOT_END)
        ax.set_ylim(0, 100)
        ax.set_yticks(self.SCORE_TICKS)
        ax.set_xticks(self.TIME_TICKS)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
        ax.tick_params(axis='x', labelrotation=0)
        ax.tick_params(axis='both', labelsize=self.TICK_LABEL_FONTSIZE)
        self._label_selected_points(ax, points, reg, date_origin)
        ax.legend(
            frameon=True,
            loc='lower right',
            fontsize=self.LEGEND_FONTSIZE
        )
        ax.grid(True, which='both', linestyle='--', alpha=0.45)

        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_selected_combined_figure(self, master, model_dates, date_origin):
        font_family = configure_matplotlib_fonts()
        if font_family != 'Times New Roman':
            print(f"Times New Roman not found; using {font_family} for the combined trend figure.")

        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, axes = plt.subplots(
            2,
            2,
            figsize=self.COMBINED_FIGSIZE,
            constrained_layout=True
        )

        for ax, benchmark in zip(axes.flat, self.SELECTED_BENCHMARKS):
            if benchmark not in master.sat_scores.index:
                self._draw_missing_subplot(ax, benchmark, 'Benchmark not found')
                continue

            points, reg, r2, reason = self._fit_points_for_plot(
                master.sat_scores.loc[benchmark],
                model_dates,
                date_origin
            )
            if reason:
                self._draw_missing_subplot(ax, benchmark, reason)
                continue

            self._draw_combined_subplot(ax, benchmark, points, reg, r2, date_origin)

        fig.supxlabel('Time', fontsize=self.COMBINED_AXIS_LABEL_FONTSIZE)
        fig.supylabel('Score', fontsize=self.COMBINED_AXIS_LABEL_FONTSIZE)

        output_path = os.path.join(self.OUTPUT_DIR, 'selected_dynamic_trends_2x2.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Selected dynamic trend 2x2 figure saved to: {output_path}")

    def _fit_points_for_plot(self, scores, model_dates, date_origin):
        points = pd.DataFrame({
            'Model': scores.index,
            'Date': model_dates.reindex(scores.index),
            'Score': scores.values,
        }).dropna(subset=['Date', 'Score'])
        points = points.sort_values(['Date', 'Score']).reset_index(drop=True)

        if len(points) < 2 or pd.isna(date_origin):
            return points, None, np.nan, 'Fewer than 2 valid dated score points'

        points['Days'] = (points['Date'] - date_origin).dt.days
        reg = LinearRegression().fit(points[['Days']], points['Score'] / 100.0)
        r2 = reg.score(points[['Days']], points['Score'] / 100.0)
        return points, reg, r2, ''

    def _draw_combined_subplot(self, ax, benchmark, points, reg, r2, date_origin):
        ax.scatter(
            points['Date'],
            points['Score'],
            color='#0000FF',
            alpha=0.82,
            edgecolor='white',
            linewidth=0.6,
            zorder=3
        )

        line_dates = pd.date_range(self.PLOT_START, self.COMBINED_PLOT_END, periods=100)
        line_days = (line_dates - date_origin).days.to_numpy()
        line_scores = reg.predict(pd.DataFrame({'Days': line_days})) * 100.0
        slope_score_per_day = reg.coef_[0] * 100.0
        ax.plot(
            line_dates,
            line_scores,
            color='red',
            linestyle='--',
            linewidth=2.0,
            label=f'Linear fit (k={slope_score_per_day:.3f}, R$^2$={r2:.2f})',
            zorder=2
        )

        ax.set_title(
            str(benchmark),
            fontsize=self.COMBINED_TITLE_FONTSIZE,
            pad=8
        )
        ax.set_xlim(self.PLOT_START, self.COMBINED_PLOT_END)
        ax.set_ylim(0, 100)
        ax.set_yticks(self.SCORE_TICKS)
        ax.set_xticks(self.TIME_TICKS)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
        ax.tick_params(
            axis='x',
            labelrotation=0,
            labelsize=self.COMBINED_TICK_LABEL_FONTSIZE
        )
        ax.tick_params(axis='y', labelsize=self.COMBINED_TICK_LABEL_FONTSIZE)
        self._label_selected_points(ax, points, reg, date_origin)
        ax.legend(
            frameon=True,
            loc='lower right',
            fontsize=self.COMBINED_LEGEND_FONTSIZE
        )
        ax.grid(True, which='both', linestyle='--', alpha=0.45)

    def _draw_missing_subplot(self, ax, benchmark, reason):
        ax.set_title(
            str(benchmark),
            fontsize=self.COMBINED_TITLE_FONTSIZE,
            pad=8
        )
        ax.set_xlim(self.PLOT_START, self.COMBINED_PLOT_END)
        ax.set_ylim(0, 100)
        ax.set_yticks(self.SCORE_TICKS)
        ax.set_xticks(self.TIME_TICKS)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
        ax.text(
            0.5,
            0.5,
            reason,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=self.MISSING_MESSAGE_FONTSIZE
        )
        ax.grid(True, which='both', linestyle='--', alpha=0.45)

    def _label_selected_points(self, ax, points, reg, date_origin):
        selected = self._select_label_points(points, reg, date_origin)
        bbox = {
            'boxstyle': 'round,pad=0.18',
            'fc': 'white',
            'ec': '#999999',
            'lw': 0.5,
            'alpha': 0.9,
        }

        figure = ax.figure
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axes_box = ax.get_window_extent(renderer)
        title_box = (
            ax.title.get_window_extent(renderer)
            if ax.title.get_text()
            else None
        )

        labeled_rows = []
        accepted_boxes = []
        for _, row in points.loc[selected].sort_values(['Date', 'Score']).iterrows():
            label = self._short_model_name(row['Model'])
            offset_x, offset_y = self._label_offsets(row, labeled_rows)
            annotation = ax.annotate(
                label,
                xy=(row['Date'], row['Score']),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                ha='center',
                va='top' if offset_y < 0 else 'bottom',
                fontsize=self.ANNOTATION_FONTSIZE,
                color='#222222',
                bbox=bbox,
                zorder=4
            )

            figure.canvas.draw()
            annotation_box = annotation.get_window_extent(renderer)
            outside_axes = (
                annotation_box.x0 < axes_box.x0
                or annotation_box.x1 > axes_box.x1
                or annotation_box.y0 < axes_box.y0
                or annotation_box.y1 > axes_box.y1
            )
            overlaps_title = bool(
                title_box is not None and annotation_box.overlaps(title_box)
            )
            padded_box = annotation_box.padded(5)
            overlaps_annotation = any(
                padded_box.overlaps(accepted_box.padded(5))
                for accepted_box in accepted_boxes
            )

            if outside_axes or overlaps_title or overlaps_annotation:
                annotation.remove()
                if outside_axes:
                    self.annotation_filter_stats['outside_axes'] += 1
                if overlaps_title:
                    self.annotation_filter_stats['title_overlap'] += 1
                if overlaps_annotation:
                    self.annotation_filter_stats['annotation_overlap'] += 1
                continue

            accepted_boxes.append(annotation_box)
            self.annotation_filter_stats['drawn'] += 1
            labeled_rows.append(row)

    def _select_label_points(self, points, reg, date_origin):
        visible = points[
            (points['Date'] >= self.PLOT_START)
            & (points['Date'] <= self.PLOT_END)
        ].copy()
        selected = pd.Series(False, index=points.index)

        if visible.empty:
            return selected

        visible['DisplayName'] = visible['Model'].map(self._short_model_name)
        visible['Name_Length'] = visible['DisplayName'].str.len()
        visible['Predicted'] = reg.predict(visible[['Days']]) * 100.0
        visible['Abs_Residual'] = (visible['Score'] - visible['Predicted']).abs()

        candidate_priority = {}

        self._add_label_candidate(candidate_priority, visible['Date'].idxmin(), 0)
        self._add_label_candidate(candidate_priority, visible['Date'].idxmax(), 0)

        period_bins = list(pd.date_range(self.PLOT_START, '2026-05-01', freq='2MS'))
        if period_bins[-1] < self.PLOT_END:
            period_bins.append(self.PLOT_END + pd.DateOffset(days=1))
        visible['Time_Group'] = pd.cut(visible['Date'], bins=period_bins, include_lowest=True)
        for _, group in visible.groupby('Time_Group', observed=False):
            if group.empty:
                continue
            representative = self._time_group_representative_candidate(group)
            self._add_label_candidate(candidate_priority, representative, 1)

            earliest = self._shortest_name_from_pool(group[group['Date'] == group['Date'].min()])
            latest = self._shortest_name_from_pool(group[group['Date'] == group['Date'].max()])
            self._add_label_candidate(candidate_priority, earliest, 2)
            self._add_label_candidate(candidate_priority, latest, 2)

        month_groups = visible.groupby(visible['Date'].dt.to_period('M'))
        for _, group in month_groups:
            score_spread = group['Score'].max() - group['Score'].min()
            if len(group) >= 2 and score_spread > 5:
                high_candidate = self._score_extreme_candidate(group, high=True)
                low_candidate = self._score_extreme_candidate(group, high=False)
                self._add_label_candidate(candidate_priority, high_candidate, 1)
                self._add_label_candidate(candidate_priority, low_candidate, 1)

            if len(group) >= 4 and score_spread > 12:
                outlier_candidate = group.sort_values(
                    ['Abs_Residual', 'Name_Length'],
                    ascending=[False, True]
                ).index[0]
                self._add_label_candidate(candidate_priority, outlier_candidate, 2)

        n_outliers = 3 if len(visible) < 40 else 5
        for idx in visible.sort_values('Abs_Residual', ascending=False).head(n_outliers).index:
            self._add_label_candidate(candidate_priority, idx, 2)

        ordered_candidates = sorted(
            candidate_priority,
            key=lambda idx: (
                candidate_priority[idx],
                visible.loc[idx, 'Date'],
                visible.loc[idx, 'Name_Length'],
                -visible.loc[idx, 'Abs_Residual'],
            )
        )

        chosen = []
        for idx in ordered_candidates:
            if self._is_too_close_to_existing(visible.loc[idx], visible.loc[chosen], candidate_priority[idx]):
                continue
            chosen.append(idx)

        if not chosen and not visible.empty:
            chosen.append(visible['Name_Length'].idxmin())

        selected.loc[chosen] = True

        return selected

    def _time_group_representative_candidate(self, group):
        residual_cutoff = group['Abs_Residual'].quantile(0.5)
        trend_like = group[group['Abs_Residual'] <= residual_cutoff]
        if trend_like.empty:
            trend_like = group
        return trend_like.sort_values(
            ['Name_Length', 'Abs_Residual'],
            ascending=[True, True]
        ).index[0]

    def _shortest_name_from_pool(self, group):
        return group.sort_values(
            ['Name_Length', 'Abs_Residual'],
            ascending=[True, False]
        ).index[0]

    def _score_extreme_candidate(self, group, high):
        if high:
            extreme = group['Score'].max()
            pool = group[group['Score'] >= extreme - 2]
        else:
            extreme = group['Score'].min()
            pool = group[group['Score'] <= extreme + 2]

        return pool.sort_values(
            ['Name_Length', 'Abs_Residual'],
            ascending=[True, False]
        ).index[0]

    def _add_label_candidate(self, candidate_priority, idx, priority):
        if idx not in candidate_priority or priority < candidate_priority[idx]:
            candidate_priority[idx] = priority

    def _is_too_close_to_existing(self, row, chosen_rows, priority):
        if chosen_rows.empty:
            return False

        days_apart = (chosen_rows['Date'] - row['Date']).abs().dt.days
        scores_apart = (chosen_rows['Score'] - row['Score']).abs()
        same_month = chosen_rows['Date'].dt.to_period('M') == row['Date'].to_period('M')
        same_month_near_duplicate = (same_month & (scores_apart <= 7)).any()
        close_duplicate = ((days_apart < 24) & (scores_apart <= 6)).any()
        close_cluster = ((days_apart < 12) & (scores_apart < 9)).any()
        return bool((same_month_near_duplicate and priority > 0) or close_duplicate or (close_cluster and priority > 1))

    def _label_offsets(self, row, labeled_rows):
        nearby_count = 0
        for prev in labeled_rows:
            days_apart = abs((row['Date'] - prev['Date']).days)
            score_apart = abs(row['Score'] - prev['Score'])
            if days_apart < 75 and score_apart < 12:
                nearby_count += 1

        vertical_tier = nearby_count % 6
        horizontal_tiers = (0, -18, 18, -30, 30, 0)
        horizontal_offset = horizontal_tiers[vertical_tier]
        vertical_offset = 11 + 22 * vertical_tier
        if row['Score'] >= 85:
            vertical_offset *= -1
        return horizontal_offset, vertical_offset

    def _short_model_name(self, model_name):
        clean = get_clean_name(model_name).replace('_', ' ')
        return clean if len(clean) <= 28 else f"{clean[:25]}..."

    def _unique_plot_file(self, benchmark, used_filenames):
        safe_name = get_clean_name(benchmark)
        safe_name = re.sub(r'[^A-Za-z0-9_.()-]+', '_', safe_name).strip('_')
        if not safe_name:
            safe_name = 'benchmark'

        candidate = f"{safe_name}.pdf"
        suffix = 2
        while candidate in used_filenames:
            candidate = f"{safe_name}_{suffix}.pdf"
            suffix += 1

        used_filenames.add(candidate)
        return os.path.join(self.OUTPUT_DIR, candidate)
