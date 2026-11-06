# config/settings.py

import os
import pandas as pd


CONFIG = {
    'min_model_count': 3,
    'cut_off_column': "GitHub",
    'safety_factor': 1.25,
    'edr_threshold': 2.0,
    'tags': {
        'lib_0_1': "(Lower-Better)(0-1)",
        'lib_standard': "(Lower-Better)",
        'unbounded_hib': "(no limited)"
    }
}

CURRENT_DATE = pd.to_datetime('2025-12-31')


IN_MASTER = os.environ.get(
    'BHI_IN_MASTER',
    'data/input/benchmark_matrix.csv',
)
OUT_DIR = os.environ.get('BHI_OUT_DIR', 'data/output')
CAP_DIR = os.path.join(OUT_DIR, 'capabilities')

ABLATION_OUTPUT_CSV = os.path.join(OUT_DIR, 'ablation_study_results.csv')


BOOTSTRAP_ITERATIONS = 100   

DROP_RATIOS = [round(x * 0.05, 2) for x in range(1, 13)] 


NOISE_ITERATIONS = 100  
NOISE_SIGMAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2] 
RANDOM_SEED = 42

THRESHOLD_LIST = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
BASELINE_THRESHOLD = 2.0

WEIGHT_STATIC = 0.8
WEIGHT_DYNAMIC = 0.2
NOISE_LEVEL = 0.05
