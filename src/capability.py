import numpy as np
import pandas as pd

def calculate_lobo_capabilities(bench_data, exclude_idx, n_total):

    subset = bench_data.drop(index=exclude_idx)
    model_names = subset.columns[1:]
    

    scores = subset.iloc[:, 1:].values.astype(float)
    valid_mask = ~np.isnan(scores)
    
    total_wins = np.zeros(len(model_names))
    total_opps = np.zeros(len(model_names))
    n_part = np.zeros(len(model_names))
    
    for row_idx in range(len(scores)):
        row_scores, mask = scores[row_idx], valid_mask[row_idx]
        v_scores = row_scores[mask]
        if len(v_scores) < 2: continue
        
        diff = v_scores[:, None] - v_scores
        wins = (diff > 0).astype(float).sum(axis=1) + 0.5 * (diff == 0).astype(float).sum(axis=1) - 0.5
        
        indices = np.where(mask)[0]
        total_wins[indices] += wins
        total_opps[indices] += (len(v_scores) - 1)
        n_part[indices] += 1
        
    win_rates = np.divide(total_wins, total_opps, out=np.zeros_like(total_wins), where=total_opps > 0)

    penalty = (np.log(1 + n_part) / np.log(1 + (n_total - 1))) ** 0.25
    
    return pd.DataFrame({'Capability_Score': win_rates * penalty}, index=model_names)