# main.py

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from src.preprocessor import MasterDataPreprocessor
from src.capability import calculate_lobo_capabilities
from src.robustness_test import RobustnessTester
from src.robustness_test_equal import RobustnessTesterEqual
from src.noise_test import NoiseTester
from src.noise_test_equal import NoiseTesterEqual
from src.ablation_study import AblationStudy
from src.threshold_test import ThresholdTester
from src.sat_weight_sensitivity import SatWeightSensitivity
from src.dynamic_window_sensitivity import DynamicWindowSensitivity
from src.indicator_correlation import IndicatorCorrelationAnalysis
from src.coverage_analysis import CoverageBiasAnalysis
from src.dynamic_trend_plots import DynamicTrendPlotter
from src.utils import get_clean_name
from config.settings import (CONFIG, IN_MASTER, OUT_DIR,
                             WEIGHT_STATIC, WEIGHT_DYNAMIC)

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_OUTPUT_ROOT = (PROJECT_ROOT / 'data' / 'output').resolve()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _get_package_versions():
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
    }
    versions = {}
    for label, distribution_name in packages.items():
        try:
            versions[label] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _get_git_commit():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _display_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _parse_snapshot_date(value):
    try:
        parsed = datetime.strptime(value, '%Y-%m-%d').date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'snapshot date must be a valid ISO date in YYYY-MM-DD form'
        ) from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError(
            'snapshot date must be a valid ISO date in YYYY-MM-DD form'
        )
    return value


def _validate_input_path(input_path):
    resolved = Path(input_path).resolve()
    if not resolved.exists():
        raise ValueError(f'Input path does not exist: {input_path}')
    if not resolved.is_file():
        raise ValueError(f'Input path is not a file: {input_path}')
    if resolved.suffix.lower() != '.csv':
        raise ValueError(f'Input file must have a .csv extension: {input_path}')
    return resolved


def _prepare_explicit_output_directory(output_dir, input_path, overwrite):
    raw_output = os.fspath(output_dir).strip()
    if not raw_output:
        raise ValueError('Output directory must not be empty.')

    resolved = Path(raw_output).resolve()
    input_directory = Path(input_path).resolve().parent
    protected_paths = {
        PROJECT_ROOT,
        (PROJECT_ROOT / 'data').resolve(),
        (PROJECT_ROOT / 'data' / 'input').resolve(),
        PROJECT_OUTPUT_ROOT,
        Path.cwd().resolve(),
        Path(resolved.anchor).resolve(),
    }

    if resolved in protected_paths:
        raise ValueError(f'Refusing to use protected output directory: {resolved}')
    if resolved == input_directory:
        raise ValueError('Output directory must not be the same as the input directory.')
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f'Output path exists but is not a directory: {resolved}')

    has_contents = resolved.exists() and any(resolved.iterdir())
    if has_contents and not overwrite:
        raise FileExistsError(
            f'Output directory already contains files: {resolved}. '
            'Use --overwrite to safely replace this exact snapshot directory.'
        )
    if has_contents and overwrite:
        if PROJECT_OUTPUT_ROOT not in resolved.parents:
            raise ValueError(
                'Refusing to overwrite a non-empty output directory outside '
                f'{PROJECT_OUTPUT_ROOT}. Use a new or empty external directory.'
            )
        shutil.rmtree(resolved)

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_standard_pipeline(
    input_path=IN_MASTER,
    output_dir=OUT_DIR,
    snapshot_date=None,
    save_metadata=False,
    command=None,
):

    print("The standard BHI calculation process is being initiated...")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    capability_dir = output_dir / 'capabilities'
    capability_dir.mkdir(parents=True, exist_ok=True)
    
    pre = MasterDataPreprocessor(CONFIG)
    master = pre.load_and_process_master(input_path)
    min_count = CONFIG['min_model_count']

    disc_valid_counts = master.disc_scores.notna().sum(axis=1)
    sat_valid_counts = master.sat_scores.notna().sum(axis=1)
    impact_evaluated_counts = master.availability.sum(axis=1)

    # --- 1. Discrimination ---
    df_disc_norm = master.disc_scores.loc[disc_valid_counts >= min_count]
    
    disc_list = []
    for b_name, row in df_disc_norm.iterrows():
        v = row.dropna().values
        n = len(v)
        
        # RCV (Robust CV)
        rcv = (np.percentile(v, 90) - np.percentile(v, 10)) / 100.0
        # EDR 
        diffs = np.abs(v[:, None] - v)
        edr = (np.sum(diffs > CONFIG['edr_threshold'])/2) / (n*(n-1)/2) if n > 1 else 0
        
        disc_list.append({'Benchmark': b_name, 'N_Models': n, 'Robust_CV': rcv, 'EDR': edr})

    df_disc = pd.DataFrame(disc_list).set_index('Benchmark')

    m_n = (df_disc[['Robust_CV', 'EDR']] - df_disc[['Robust_CV', 'EDR']].min()) / (df_disc[['Robust_CV', 'EDR']].max() - df_disc[['Robust_CV', 'EDR']].min() + 1e-9)
    w_d = m_n.std() / m_n.std().sum()

    # S_Disc
    df_disc['S_Disc'] = (m_n['Robust_CV'] * w_d['Robust_CV'] + m_n['EDR'] * w_d['EDR'])

    df_disc = df_disc.sort_values(by='S_Disc', ascending=False)
    df_disc.to_csv(output_dir / 'capability_discrimination_final.csv')


    # --- 2. LOBO & Saturation & Impact ---
    df_sat_norm = master.sat_scores.loc[sat_valid_counts >= min_count]
    bench_data = df_sat_norm.reset_index()
    model_dates = master.model_dates.reindex(master.model_columns)
    n_total = len(bench_data)
    sat_res, imp_res = [], []

    for i in range(n_total):
        b_name = bench_data.loc[i, 'Benchmark']
        caps = calculate_lobo_capabilities(bench_data, i, n_total)
        safe_name = get_clean_name(b_name)
        caps.to_csv(capability_dir / f'capability_minus_{safe_name}.csv')
        
        scores = bench_data.iloc[i, 1:]
        valid = scores.dropna()
        common = valid.index.intersection(caps.index)
        
        # S_sta
        s_static = 1 - ( (valid[common]/100.0) * caps.loc[common, 'Capability_Score'] ).sum() / (caps.loc[common, 'Capability_Score'].sum() + 1e-9)
        
        # S_dyna
        v_reg = pd.DataFrame({'d': (model_dates - model_dates.min()).dt.days, 's': scores}).dropna()
        if len(v_reg) >= 2:
            reg = LinearRegression().fit(v_reg[['d']], v_reg['s']/100.0)
            s_dyn = max(0, 1 - (v_reg['s'].mean()/100.0 + reg.coef_[0]*30))
        else:
            s_dyn = s_static
        
        s_sat_final = WEIGHT_STATIC * s_static + WEIGHT_DYNAMIC * s_dyn
        sat_res.append({
            'Benchmark': b_name, 
            'S_Static': s_static, 
            'S_Dynamic': s_dyn, 
            'S_Sat': s_sat_final
        })
        
        # --- Impact ---
        cap = caps['Capability_Score'].reindex(master.model_columns).fillna(0.0)
        denominator = cap.sum()
        evaluated = master.availability.loc[b_name].reindex(master.model_columns).fillna(False)
        evaluated_models = evaluated[evaluated].index
        numerator = cap.loc[evaluated_models].sum()
        s_imp_raw = numerator / denominator if denominator > 0 else 0.0
        s_imp = np.sqrt(s_imp_raw)
        imp_res.append({
            'Benchmark': b_name,
            'N_Models_With_Score': int(len(evaluated_models)),
            'Total_Models': int(len(master.model_columns)),
            'Numerator_Cap': numerator,
            'Denominator_Cap': denominator,
            'S_Imp_Raw': s_imp_raw,
            'S_Imp': s_imp
        })

    df_sat = pd.DataFrame(sat_res).set_index('Benchmark')
    df_sat = df_sat.sort_values(by='S_Sat', ascending=False)
    df_sat.to_csv(output_dir / 'anti_saturation_results.csv')

    df_imp = pd.DataFrame(imp_res).set_index('Benchmark')
    df_imp = df_imp.sort_values(by='S_Imp', ascending=False)
    df_imp.to_csv(output_dir / 'impact_results.csv')


# --- 3. BHI aggregation ---
    final = df_disc[['S_Disc']].merge(df_sat[['S_Sat']], left_index=True, right_index=True) \
                               .merge(df_imp[['S_Imp']], left_index=True, right_index=True)

    inclusion_report = pd.DataFrame({
        'Benchmark': master.benchmark_names,
        'Disc_Valid_Count': disc_valid_counts.reindex(master.benchmark_names).astype(int).values,
        'Sat_Valid_Count': sat_valid_counts.reindex(master.benchmark_names).astype(int).values,
        'Impact_Evaluated_Count': impact_evaluated_counts.reindex(master.benchmark_names).astype(int).values,
    })
    inclusion_report['Disc_Eligible'] = inclusion_report['Disc_Valid_Count'] >= min_count
    inclusion_report['Sat_Eligible'] = inclusion_report['Sat_Valid_Count'] >= min_count
    inclusion_report['Impact_Eligible'] = inclusion_report['Impact_Evaluated_Count'] >= min_count
    inclusion_report['Included_In_Final'] = inclusion_report['Benchmark'].isin(final.index)
    inclusion_report['Exclusion_Reason'] = inclusion_report.apply(
        lambda row: 'Included' if row['Included_In_Final'] else '; '.join([
            reason for reason, excluded in [
                ('S_Disc valid score count < 3', not row['Disc_Eligible']),
                ('S_Sat valid score count < 3', not row['Sat_Eligible']),
                ('Impact evaluated model count < 3', not row['Impact_Eligible'])
            ] if excluded
        ]),
        axis=1
    )
    inclusion_report.to_csv(output_dir / 'benchmark_inclusion_report.csv', index=False)
    
    std = final.std()
    
    correlation_matrix = final.corr()
    conflict = (1 - correlation_matrix).sum()
    
    c_metric = std * conflict
    
    ws = c_metric / c_metric.sum()
    
    print("\n" + "="*55)
    print("CRITIC OBJECTIVE WEIGHTS (Data-Driven Strategy)")
    print("-" * 55)
    print(f"Capability Discrimination (S_Disc) : {ws['S_Disc']:.2%}")
    print(f"Anti-Saturation (S_Sat)           : {ws['S_Sat']:.2%}")
    print(f"Impact (S_Imp)                    : {ws['S_Imp']:.2%}")
    print("="*55 + "\n")
    
    final['BHI'] = (final['S_Disc'] * ws['S_Disc'] + 
                    final['S_Sat'] * ws['S_Sat'] + 
                    final['S_Imp'] * ws['S_Imp'])
    
    final_sorted = final.sort_values(by=['BHI', 'S_Imp'], ascending=False)
    final_sorted.to_csv(output_dir / 'final_bhi_rankings.csv')
    
    print("--- TOP 10 BHI RANKING (CRITIC Weighting) ---")
    print(final_sorted.head(10))

    summary = {
        's_disc_internal_weights': {
            'robust_cv': float(w_d['Robust_CV']),
            'edr': float(w_d['EDR']),
        },
        'critic_weights': {
            's_disc': float(ws['S_Disc']),
            's_sat': float(ws['S_Sat']),
            's_imp': float(ws['S_Imp']),
        },
        'total_model_count': int(len(master.model_columns)),
        'raw_benchmark_count': int(len(master.benchmark_names)),
        'disc_eligible_count': int((disc_valid_counts >= min_count).sum()),
        'sat_eligible_count': int((sat_valid_counts >= min_count).sum()),
        'impact_diagnostic_eligible_count': int(
            (impact_evaluated_counts >= min_count).sum()
        ),
        'final_included_benchmark_count': int(len(final_sorted)),
        'final_excluded_benchmark_count': int(
            len(master.benchmark_names) - len(final_sorted)
        ),
    }

    if save_metadata:
        valid_dates = master.model_dates.dropna()
        parsed_date_count = int(master.model_dates.notna().sum())
        metadata = {
            'snapshot_id': output_dir.resolve().name,
            'snapshot_date': snapshot_date,
            'input_path': _display_path(input_path),
            'absolute_input_path': str(input_path.resolve()),
            'input_sha256': _sha256_file(input_path),
            'output_directory': _display_path(output_dir),
            'absolute_output_directory': str(output_dir.resolve()),
            'command': command or subprocess.list2cmdline([sys.executable, *sys.argv]),
            'run_timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'total_model_count': summary['total_model_count'],
            'parsed_model_date_count': parsed_date_count,
            'blank_model_date_count': int(
                len(master.model_columns) - parsed_date_count
            ),
            'unparseable_nonempty_model_date_count': 0,
            'earliest_valid_model_release_date': (
                valid_dates.min().date().isoformat() if not valid_dates.empty else None
            ),
            'latest_valid_model_release_date': (
                valid_dates.max().date().isoformat() if not valid_dates.empty else None
            ),
            'raw_benchmark_count': summary['raw_benchmark_count'],
            'disc_eligible_count': summary['disc_eligible_count'],
            'sat_eligible_count': summary['sat_eligible_count'],
            'impact_diagnostic_eligible_count': (
                summary['impact_diagnostic_eligible_count']
            ),
            'final_included_benchmark_count': (
                summary['final_included_benchmark_count']
            ),
            'final_excluded_benchmark_count': (
                summary['final_excluded_benchmark_count']
            ),
            's_disc_internal_weights': summary['s_disc_internal_weights'],
            'critic_weights': summary['critic_weights'],
            'configuration': CONFIG,
            'weight_static': WEIGHT_STATIC,
            'weight_dynamic': WEIGHT_DYNAMIC,
            'random_seed': None,
            'python_version': platform.python_version(),
            'package_versions': _get_package_versions(),
            'git_commit': _get_git_commit(),
        }
        with open(output_dir / 'run_metadata.json', 'w', encoding='utf-8') as output_file:
            json.dump(metadata, output_file, ensure_ascii=False, indent=2, allow_nan=False)
            output_file.write('\n')

        summary['metadata'] = metadata

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Run the standard BHI pipeline, an isolated temporal snapshot, '
            'or one of the existing experiment commands.'
        )
    )
    parser.add_argument('--input', help='Input master CSV for the standard pipeline.')
    parser.add_argument(
        '--output-dir',
        help=(
            'New or empty output directory for an isolated run. '
            'Required when snapshot arguments are supplied.'
        ),
    )
    parser.add_argument(
        '--snapshot-date',
        type=_parse_snapshot_date,
        help='Snapshot provenance date in YYYY-MM-DD form; does not affect BHI.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Safely replace the exact explicitly supplied snapshot output directory.',
    )

    experiment_flags = {
        'robustness': '--robustness',
        'robustness_equal': '--robustness-equal',
        'noise': '--noise',
        'noise_equal': '--noise-equal',
        'ablation': '--ablation',
        'threshold': '--threshold',
        'sat_weight_sensitivity': '--sat-weight-sensitivity',
        'dynamic_window_sensitivity': '--dynamic-window-sensitivity',
        'indicator_correlation': '--indicator-correlation',
        'coverage_analysis': '--coverage-analysis',
        'plot_dynamic_trends': '--plot-dynamic-trends',
    }
    for destination, flag in experiment_flags.items():
        parser.add_argument(flag, dest=destination, action='store_true')

    args = parser.parse_args()
    selected_experiments = [
        destination for destination in experiment_flags
        if getattr(args, destination)
    ]
    if len(selected_experiments) > 1:
        parser.error('Only one experiment flag may be supplied at a time.')
    if selected_experiments and any([
        args.input,
        args.output_dir,
        args.snapshot_date,
        args.overwrite,
    ]):
        parser.error('Snapshot arguments cannot be combined with experiment flags.')

    if selected_experiments:
        flag = experiment_flags[selected_experiments[0]]
        if flag == "--robustness":
            print("Initiate the performance degradation experiment of the critic weighting method...")
            RobustnessTester().execute_decay_experiment()
        
        elif flag == "--robustness-equal":
            print("Initiate the performance degradation experiment using the [equal weight method]...")
            RobustnessTesterEqual().execute_decay_experiment()
            
        elif flag == "--noise":
            print("Initiate the critic weighted noise attenuation experiment...")
            NoiseTester().execute_noise_experiment()

        elif flag == "--noise-equal":
            print("Start the equal-weight noise attenuation experiment...")
            NoiseTesterEqual().execute_noise_experiment()
        
        elif flag == "--ablation":
            print("Start the ablation experiment (Leave-One-Out Analysis)...")
            AblationStudy().run_ablation()
        
        elif flag == "--threshold":
            print("Initiate the threshold ablation experiment (Threshold Sensitivity Analysis)...")
            ThresholdTester().execute_test()

        elif flag == "--sat-weight-sensitivity":
            print("Start the static/dynamic weight sensitivity experiment...")
            SatWeightSensitivity().run()

        elif flag == "--dynamic-window-sensitivity":
            print("Start the dynamic-window sensitivity experiment...")
            DynamicWindowSensitivity().run()

        elif flag == "--indicator-correlation":
            print("Start the indicator orthogonality analysis...")
            IndicatorCorrelationAnalysis().run()

        elif flag == "--coverage-analysis":
            print("Start the coverage bias analysis...")
            CoverageBiasAnalysis().run()

        elif flag == "--plot-dynamic-trends":
            print("Start dynamic anti-saturation trend plotting...")
            DynamicTrendPlotter().run()

        return

    explicit_snapshot_run = any([
        args.input,
        args.output_dir,
        args.snapshot_date,
        args.overwrite,
    ])
    if not explicit_snapshot_run:
        run_standard_pipeline()
        return

    if not args.output_dir:
        parser.error('--output-dir is required for an explicit snapshot run.')

    try:
        input_path = _validate_input_path(args.input or IN_MASTER)
        output_dir = _prepare_explicit_output_directory(
            args.output_dir,
            input_path,
            args.overwrite,
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))

    run_standard_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        snapshot_date=args.snapshot_date,
        save_metadata=True,
        command=subprocess.list2cmdline([sys.executable, *sys.argv]),
    )

if __name__ == "__main__":
    main()
