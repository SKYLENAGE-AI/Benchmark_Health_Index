# src/threshold_test.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from config.settings import (CONFIG, IN_MASTER,
                             THRESHOLD_LIST, BASELINE_THRESHOLD, OUT_DIR)
from src.preprocessor import MasterDataPreprocessor
from src.utils import configure_matplotlib_fonts

class ThresholdTester:
    def __init__(self):
        self.pre = MasterDataPreprocessor(CONFIG)
        self.master = self.pre.load_and_process_master(IN_MASTER)
        disc_valid_counts = self.master.disc_scores.notna().sum(axis=1)
        self.df_disc_norm = self.master.disc_scores.loc[disc_valid_counts >= CONFIG['min_model_count']]
        
        try:
            df_sat = pd.read_csv(f'{OUT_DIR}/anti_saturation_results.csv').set_index('Benchmark')
            df_imp = pd.read_csv(f'{OUT_DIR}/impact_results.csv').set_index('Benchmark')
            self.fixed_metrics = df_sat[['S_Sat']].merge(df_imp[['S_Imp']], left_index=True, right_index=True)
            self.standard_bhi = pd.read_csv(f'{OUT_DIR}/final_bhi_rankings.csv').set_index('Benchmark')['BHI']
        except FileNotFoundError:
            print("❌ Error: The basic result file was not found. Please run python main.py first to generate the standard result.")
            raise

    def calculate_bhi_with_delta(self, delta):

        disc_data = []
        for b_name, row in self.df_disc_norm.iterrows():
            v = row.dropna().values
            n = len(v)
            
            rcv = (np.percentile(v, 90) - np.percentile(v, 10)) / 100.0
            
            diffs = np.abs(v[:, None] - v)
            edr = (np.sum(diffs > delta)/2) / (n*(n-1)/2) if n > 1 else 0
            disc_data.append({'Benchmark': b_name, 'Robust_CV': rcv, 'EDR': edr})

        df_d = pd.DataFrame(disc_data).set_index('Benchmark')
        
        m_n = (df_d - df_d.min()) / (df_d.max() - df_d.min() + 1e-9)
        w_d = m_n.std() / (m_n.std().sum() + 1e-9)
        
        s_disc = (m_n['Robust_CV'] * w_d['Robust_CV'] + m_n['EDR'] * w_d['EDR'])
        
        final = pd.DataFrame({'S_Disc': s_disc}).merge(self.fixed_metrics, left_index=True, right_index=True)
        
        std = final.std()
        correlation_matrix = final.corr()
        conflict = (1 - correlation_matrix).sum()
        c_metric = std * conflict
        ws = c_metric / c_metric.sum()
        
        bhi = (final['S_Disc'] * ws['S_Disc'] + 
               final['S_Sat'] * ws['S_Sat'] + 
               final['S_Imp'] * ws['S_Imp'])
        
        return bhi, ws.to_dict()

    def execute_test(self):
        print(f"🚀 Starting Threshold Sensitivity Analysis...")
        
        baseline_bhi = self.standard_bhi
        
        results = []
        for delta in THRESHOLD_LIST:
            print(f"Processing Delta = {delta}...")
            curr_bhi, weights = self.calculate_bhi_with_delta(delta)
            
            common = baseline_bhi.index.intersection(curr_bhi.index)
            rho, _ = spearmanr(baseline_bhi.loc[common], curr_bhi.loc[common])
            top10_baseline = set(baseline_bhi.loc[common].nlargest(10).index)
            top10_current = set(curr_bhi.loc[common].nlargest(10).index)
            results.append({
                'delta': delta,
                'spearman': rho,
                'spearman_vs_standard_bhi': rho,
                'top10_overlap': len(top10_baseline & top10_current),
                'disc_weight': weights['S_Disc'],
                'weight_S_Disc': weights['S_Disc'],
                'weight_S_Sat': weights['S_Sat'],
                'weight_S_Imp': weights['S_Imp']
            })
        
        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{OUT_DIR}/threshold_ablation_results.csv", index=False)
        self.plot_results(df_res)

    def plot_results(self, df):

        plt.style.use('seaborn-v0_8-whitegrid')
        configure_matplotlib_fonts()
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_blue = '#1f77b4' 
        ax1.set_xlabel(r'Discrimination Threshold ($\delta$)', fontsize=12)
        ax1.set_ylabel("Spearman's $\\rho$ (Stability Index)", color=color_blue, fontsize=12)
        ax1.plot(df['delta'], df['spearman'], color=color_blue, marker='o', 
                 markersize=8, linewidth=2.5, label='Rank Stability')
        
        ax1.set_ylim(0, 1.05) 
        ax1.tick_params(axis='y', labelcolor=color_blue)

        ax1.axvline(x=BASELINE_THRESHOLD, color='red', linestyle='--', alpha=0.5, label=fr'Baseline ($\delta={BASELINE_THRESHOLD}$)')

        ax2 = ax1.twinx()
        color_green = '#2ca02c'
        ax2.set_ylabel('Objective Weight of $S_{Disc}$ (CRITIC)', color=color_green, fontsize=12)
        ax2.plot(df['delta'], df['disc_weight'], color=color_green, marker='s', 
                 linestyle='-.', markersize=8, linewidth=2, label='Disc Weight')
        
        ax2.set_ylim(0, 0.6) 
        ax2.tick_params(axis='y', labelcolor=color_green)


        plt.title('Threshold Sensitivity Analysis: Ranking Stability vs. Information Importance', fontsize=14, pad=15)
        

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', frameon=True, shadow=True, ncol=3)

        plt.grid(True, linestyle=':', alpha=0.6)
        
        pdf_path = f"{OUT_DIR}/threshold_sensitivity_curve.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"📊 The threshold ablation curve (in PDF vector format) has been saved to: {pdf_path}")
