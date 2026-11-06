# src/ablation_study.py

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from config.settings import ABLATION_OUTPUT_CSV, OUT_DIR

class AblationStudy:
    def __init__(self):

        df_disc = pd.read_csv(f'{OUT_DIR}/capability_discrimination_final.csv')[['Benchmark', 'S_Disc']]
        df_sat = pd.read_csv(f'{OUT_DIR}/anti_saturation_results.csv')[['Benchmark', 'S_Sat']]
        df_imp = pd.read_csv(f'{OUT_DIR}/impact_results.csv')[['Benchmark', 'S_Imp']]
        

        self.data = df_disc.merge(df_sat, on='Benchmark').merge(df_imp, on='Benchmark')
        self.metrics = ['S_Disc', 'S_Sat', 'S_Imp']

    def calculate_critic_bhi(self, df, active_metrics):

        temp_df = df[active_metrics].copy()
        

        std = temp_df.std()
        

        corr_matrix = temp_df.corr()
        conflict = (1 - corr_matrix).sum()
        

        c_metric = std * conflict
        

        weights = c_metric / (c_metric.sum() + 1e-9)
        

        bhi = (temp_df * weights).sum(axis=1)
        
        return bhi, weights.to_dict()

    def run_ablation(self):
        print("🚀 Starting Ablation Study (Leave-One-Out Strategy)...")


        print("\n[Module A] Orthogonality Check (Pearson Matrix):")
        corr_matrix = self.data[self.metrics].corr(method='pearson')
        print(corr_matrix.round(4))


        full_bhi, full_weights = self.calculate_critic_bhi(self.data, self.metrics)
        self.data['Rank_Full'] = full_bhi.rank(ascending=False)

        results_list = []
        consistency_report = []


        for metric_to_remove in self.metrics:
            active_metrics = [m for m in self.metrics if m != metric_to_remove]
            scenario_name = f"wo_{metric_to_remove.split('_')[1]}" # e.g., wo_Disc
            

            ablated_bhi, sub_weights = self.calculate_critic_bhi(self.data, active_metrics)
            ablated_rank = ablated_bhi.rank(ascending=False)
            
            self.data[f'Rank_{scenario_name}'] = ablated_rank
            

            delta = self.data['Rank_Full'] - ablated_rank
            self.data[f'Delta_{scenario_name}'] = delta.abs()
            self.data[f'Direction_{scenario_name}'] = delta.apply(lambda x: "Rose" if x > 0 else "Dropped" if x < 0 else "Stable")


            rho, _ = spearmanr(self.data['Rank_Full'], ablated_rank)
            consistency_report.append({'Scenario': scenario_name, 'Spearman_rho': rho, 'Weights': sub_weights})


        print("\n[Module C & D] Global Consistency & Sensitivity Filtering:")
        for report in consistency_report:
            s = report['Scenario']
            print(f"Scenario {s}: Spearman Rho = {report['Spearman_rho']:.4f}")
            

            top_shifted = self.data.nlargest(5, f'Delta_{s}')
            print(f"  - Top 5 Displaced Benchmarks in {s}:")
            for _, row in top_shifted.iterrows():
                print(f"    * {row['Benchmark']}: Rank {row['Rank_Full']} -> {row[f'Rank_{s}']} (Delta: {row[f'Delta_{s}']})")


        output_cols = ['Benchmark', 'Rank_Full'] + \
                      [c for c in self.data.columns if 'Rank_wo' in c or 'Delta_wo' in c or 'Direction_wo' in c]
        self.data[output_cols].to_csv(ABLATION_OUTPUT_CSV, index=False)
        print(f"\n✅ Ablation results saved to {ABLATION_OUTPUT_CSV}")
