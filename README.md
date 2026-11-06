# Benchmark Health Index (BHI)

BHI is a data-driven framework for evaluating and ranking LLM benchmarks rather than models. It combines three complementary dimensions:

- **Capability Discrimination**: Measures how effectively a benchmark distinguishes models with different capabilities using the Robust Coefficient of Variation (RCV) and Effective Differentiation Ratio (EDR).
- **Anti-Saturation**: Evaluates the remaining performance headroom and projected saturation trends of a benchmark.
- **Impact**: Measures benchmark adoption weighted by model capabilities.

The three dimensions are normalized and aggregated using the objective CRITIC weighting method to produce the final Benchmark Health Index.

This repository provides the complete BHI implementation, supplementary experimental analyses, historical research results, and documentation for applying BHI to custom evaluation datasets.

## Repository Contents

The repository contains:

- **Core BHI implementation**: Data preprocessing, LOBO capability calibration, metric calculation, CRITIC weighting, and final benchmark ranking.
- **Supplementary experiments**: Parameter sensitivity, robustness evaluation, ablation studies, correlation analysis, cold-start analysis, and temporal stability.
- **Research results**: Historical BHI rankings, component scores, experimental tables, and figures associated with the NeurIPS 2026 study.
- **Documentation**: Installation instructions, input specifications, experiment commands, and result interpretation.

### Historical Results

The `data/output/` directory organizes the research results into the following groups:

| Group | Description |
|---|---|
| `april_baseline/` | April 2026 baseline: 117 models and 158 ranked benchmarks |
| `june_historical/` | June 2026 historical snapshot: 133 models and 185 ranked benchmarks |
| `experiments/` | Sensitivity, robustness, ablation, correlation, cold-start, coverage, and temporal analyses |
| `figures/` | Research figures and visualizations |

Detailed numerical tables are available in [PAPER_RESULTS_SUMMARY.md](docs/PAPER_RESULTS_SUMMARY.md).

## Installation

The implementation was validated using Python 3.13.2 and the dependencies specified in `requirements.txt`.

Create a Python virtual environment:

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run BHI on Your Dataset

BHI accepts a wide-format CSV containing model evaluation scores and associated metadata.

Place your dataset at `data/input/benchmark_matrix.csv`, or specify another CSV path through the command-line interface.

The expected data format, metadata conventions, score normalization rules, and benchmark eligibility requirements are described in [INPUT_SCHEMA.md](docs/INPUT_SCHEMA.md).

Run the main BHI pipeline:

```bash
python main.py --input data/input/benchmark_matrix.csv --output-dir data/output/local_runs/my_snapshot --snapshot-date YYYY-MM-DD
```

Use a new or empty output directory for each execution.

The `--snapshot-date` argument records the evaluation snapshot date for provenance purposes and does not change the BHI equations.

## Output Files

The main pipeline produces the following files:

| Output | Description |
|---|---|
| `final_bhi_rankings.csv` | Final benchmark rankings and BHI component scores |
| `capability_discrimination_final.csv` | RCV, EDR, and Capability Discrimination results |
| `anti_saturation_results.csv` | Static, dynamic, and combined Anti-Saturation scores |
| `impact_results.csv` | Capability-weighted benchmark adoption and Impact scores |
| `benchmark_inclusion_report.csv` | Benchmark eligibility and exclusion information |
| `capabilities/` | LOBO-based model capability estimates |
| `run_metadata.json` | Execution settings, model counts, weights, and snapshot metadata |

BHI scores range from 0 to 1. Higher scores indicate stronger overall benchmark health under the evaluated model ecosystem.

The final ranking is determined jointly by Capability Discrimination, Anti-Saturation, and Impact. Scores and weights depend on the dataset and evaluated model population.

## Supplementary Experiments

The repository includes implementations of the supplementary analyses reported in the paper.

Available experiments include:

- EDR threshold sensitivity
- RCV percentile sensitivity
- Headroom factor sensitivity
- Dynamic-window sensitivity
- Static/dynamic Anti-Saturation weighting sensitivity
- Gaussian noise robustness
- Model dropout robustness
- Targeted score perturbation experiments
- Component ablation
- Indicator correlation
- Benchmark coverage analysis
- Cold-start fairness analysis
- Temporal snapshot robustness

First generate the baseline results:

```bash
python main.py --input data/input/benchmark_matrix.csv --output-dir data/output/local_runs/baseline --snapshot-date YYYY-MM-DD
```

Then run a supplementary experiment. For example, RCV sensitivity:

```bash
python scripts/run_experiment.py rcv --input data/input/benchmark_matrix.csv --baseline-dir data/output/local_runs/baseline --output-dir data/output/local_runs/rcv
```

For additional experiment commands, configuration options, and analysis-specific requirements, see [EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Validation

The executed validation matrix and numerical checks are summarized in
[VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md). Repository file integrity can
be checked with:

```bash
python scripts/verify_manifest.py
```

## Repository Structure

```text
config/        Configuration and experimental defaults
data/
  input/       Input datasets
  output/      Historical research results and generated outputs
docs/          Input specifications, experiment guides, and documentation
scripts/       Experiment launchers and analysis scripts
src/           Core BHI implementation and experimental methods
main.py        Main BHI command-line entry point
requirements.txt  Python dependencies
README.md      Project documentation
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
