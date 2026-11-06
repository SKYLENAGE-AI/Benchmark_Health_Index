# Validation report

Final validation was completed on 2026-10-28 using Python 3.13.2 and the pinned
dependencies in `requirements.txt`. The executed implementation retained the
BHI formulas, LOBO calculation, eligibility rules, normalization, CRITIC
weighting, experiment grids, iteration counts, and random seeds.

## Main pipeline

The main pipeline passed end-to-end checks on two generated matrices:

| Validation matrix | Models | Raw benchmarks | Ranked benchmarks | Result |
|---|---:|---:|---:|---|
| Small matrix | 6 | 12 | 11 | pass |
| Experiment matrix | 24 | 30 | 30 | pass |

All expected output files were generated. Inclusion counts matched the final
rankings, and S_Disc, S_Sat, S_Imp, and BHI remained within [0, 1]. The larger
matrix covered lower-better, 0-1 lower-better, open-upper-bound, missing-score,
and release-date cases.

## Supplementary experiments

| Analysis | Conditions | Result |
|---|---|---|
| EDR threshold | 8 thresholds | pass |
| RCV percentile range | 6 ranges; 2,000 bootstraps; seed 20260726 | pass |
| Headroom factor | 5 factors; 2,000 bootstraps; seed 20260726 | pass |
| Dynamic window | 5 windows | pass |
| Static/dynamic weighting | 4 weight pairs | pass |
| Component ablation | 3 leave-one-component-out cases | pass |
| Indicator correlation | Pearson and Spearman | pass |
| Coverage analysis | full baseline | pass |
| Gaussian noise, CRITIC | 6 sigmas x 100 iterations; seed 42 | pass; repeat hashes identical |
| Gaussian noise, equal | 6 sigmas x 100 iterations; seed 42 | pass; repeat hashes identical |
| Model dropout, CRITIC | 12 ratios x 100 iterations; seed 42 | pass; repeat hashes identical |
| Model dropout, equal | 12 ratios x 100 iterations; seed 42 | pass; repeat hashes identical |
| Targeted zero score | 10 targets; 20 perturbations; 2,000 bootstraps | pass; 220 invariants |
| Fixed decrement | 10 targets; 100 perturbations; 2,000 bootstraps | pass; 1,300 invariants |
| Cold-start / age | 10,000 bootstraps; seed 42 | pass |
| Temporal comparison | exact and conservative matching; 2,000 bootstraps | pass |

The parser-reference dependency used by the headroom and targeted-zero
experiments was validated from its final location in `config/`.

## Paper-result checks

The included tables were checked for the April 117-model/158-benchmark and June
133-model/185-benchmark configurations, default CRITIC weights, parameter
sensitivity, robustness, cold-start, coverage, component correlation,
ablation, and temporal results. All 1,520 targeted-perturbation invariants
passed.

The manuscript-reported external cohort values are 0.225 +/- 0.036 for
known-saturated benchmarks and 0.605 +/- 0.045 for active-frontier benchmarks
(mean +/- sample SD).

## Repository integrity

- Python syntax compilation: pass.
- Main command-line interface: pass.
- Supplementary launcher and standalone analysis interfaces: pass.
- Historical CSV/PDF hashes: unchanged.
- SHA-256 repository manifest: pass.
- Markdown links: pass.
- Nested README count: zero.
- `tests/` and `reference/` runtime dependency check: pass.
