# Supplementary experiments

The commands below use a schema-compliant matrix at
`data/input/benchmark_matrix.csv`. Replace that path as needed. Each experiment
requires a new output directory and refuses to overwrite existing results.

The launcher configures paths for the experiment implementation and, where
needed, prepares the five baseline result tables in the experiment directory.
It retains the defined formulas, conditions, iteration counts, and seeds.

## Generate the common baseline

```bash
python main.py --input data/input/benchmark_matrix.csv --output-dir data/output/local_runs/baseline --snapshot-date YYYY-MM-DD
```

## Parameter sensitivity

```bash
python scripts/run_experiment.py edr --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/edr
python scripts/run_experiment.py rcv --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/rcv
python scripts/run_experiment.py headroom --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/headroom
python scripts/run_experiment.py anti-saturation-weight --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/sat_weights
python scripts/run_experiment.py dynamic-window --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/dynamic_window
```

The grids are unchanged: EDR 0.5/1/1.5/2/2.5/3/4/5; RCV P0-P100 through
P25-P75; headroom 1.10/1.20/1.25/1.33/1.50; dynamic window 20/25/30/35/40
days; and static/dynamic weights 0.6/0.4 through 0.9/0.1.

## Gaussian noise and model dropout

These retain seed 42 and 100 iterations per condition. Noise uses six sigma
values; dropout uses twelve fractions from 5% to 60%.

```bash
python scripts/run_experiment.py noise-critic --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/noise_critic
python scripts/run_experiment.py noise-equal --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/noise_equal
python scripts/run_experiment.py dropout-critic --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/dropout_critic
python scripts/run_experiment.py dropout-equal --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/dropout_equal
```

## Targeted score perturbations

Run zero-score first, then pass that exact output to fixed-decrement. The
original checks require at least 10 eligible benchmarks and enough positive
model scores to form target-specific capability groups.

```bash
python scripts/run_experiment.py targeted-zero --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/targeted_zero
python scripts/run_experiment.py targeted-fixed --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --prior-output-dir data/output/local_runs/targeted_zero --output-dir data/output/local_runs/targeted_fixed
```

The zero-score run retains 10 targets, 20 perturbation runs, 2,000 bootstrap
resamples, and seed 20260726. Fixed decrement retains levels 10/15/20/25/30,
100 runs, the same bootstrap size, and the same seed.

## Component and coverage diagnostics

```bash
python scripts/run_experiment.py ablation --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/ablation
python scripts/run_experiment.py correlation --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/correlation
python scripts/run_experiment.py coverage --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/coverage
```

## Cold-start / benchmark age

This analysis requires a wide score matrix plus a benchmark-level CSV
containing `Benchmark`, `Release_Date`, `Source_URL`, `Date_Precision`,
`Verification_Status`, `Manual_Check_Priority`, `Release_Scope`, and matching baseline metrics
`NUsage`, `SDisc`, `SSat`, `SImp`, and `BHI`. Accepted aliases are defined in
the script.

```bash
python scripts/analyze_cold_start_fairness.py --release-dates path/to/benchmark_dates_and_metrics.csv --master-data path/to/benchmark_matrix.csv --snapshot-date 2026-10-25 --output-dir data/output/custom_cold_start --bootstrap-samples 10000 --seed 42
```

Release dates should be independently verified. The result is observational
and does not establish that age causes BHI or coverage differences.

## Temporal comparison

Generate two snapshots with identical method settings, then compare the
completed output directories. Raw inputs are not read by the comparison.

```bash
python scripts/compare_snapshots.py --baseline-dir data/output/snapshot_a --updated-dir data/output/snapshot_b --output-dir data/output/a_vs_b --baseline-label "Snapshot A" --updated-label "Snapshot B"
```

Both snapshots need `final_bhi_rankings.csv`; `run_metadata.json` is strongly
recommended so model counts and CRITIC weights can be checked. The comparison
uses exact names and conservative aliases, 2,000 bootstrap samples, and seed
20260726. State any method-setting difference as a confound.

The paper-level values in `PAPER_RESULTS_SUMMARY.md` are dataset-specific
reference results, not expected outputs for every custom matrix. Model
population, benchmarks, dates, missingness, and score scale all affect results.
