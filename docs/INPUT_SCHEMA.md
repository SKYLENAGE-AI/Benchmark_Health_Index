# Input CSV schema

## Matrix layout

The BHI pipeline accepts one wide CSV:

```text
Benchmark,Model A,Model B,...,Model N,GitHub,benchmark_date,...
Vendor,...
Release Date,YYYY-MM-DD,YYYY-MM-DD,...
Benchmark 1,score,score,...
Benchmark 2,score,score,...
```

The columns have three contiguous sections:

1. The first column contains reserved metadata-row labels and benchmark names.
2. Model columns start at column 2 and end immediately before `GitHub`.
3. Benchmark metadata starts at `GitHub` and is excluded from score processing.

Column and benchmark names should be unique. Whitespace around benchmark names
is stripped, but relying on that normalization is discouraged.

## Required fields

| Field | Requirement | Meaning |
|---|---|---|
| First column | Required | Benchmark identifier; `Benchmark` is recommended |
| Model columns | At least three recommended | One column per evaluated model |
| `GitHub` | Required boundary marker | First metadata column; values may be blank |
| `Release Date` row | Strongly recommended | One parseable release date per model; ISO `YYYY-MM-DD` is recommended |
| Benchmark rows | Required | One row per benchmark and one score cell per model |

The optional `Vendor` row is ignored by the scientific calculations. If the
`Release Date` row is absent, dynamic saturation falls back to static behavior
where regression cannot be fitted.

## Score-cell syntax

| Syntax | Interpretation |
|---|---|
| `72.5` or `72.5%` | Same numeric value for discrimination and saturation; evaluated for Impact |
| `68 / 71` | First value for discrimination, second for saturation; evaluated for Impact |
| `68 [T]` | Excluded from discrimination, retained for saturation and Impact |
| Empty cell | Missing for all components and not evaluated for Impact |

Text surrounding a valid numeric score is not recommended even though the
parser can extract a number in some cases. Use explicit numeric cells whenever
possible.

## Benchmark-name normalization tags

Normalization is controlled by suffixes in the benchmark name:

| Tag | Behavior |
|---|---|
| No tag | Values in `(0, 1.05]` are multiplied by 100; other values are treated as percentage-like scores |
| `(Lower-Better)(0-1)` | `(1 - score) × 100` |
| `(Lower-Better)` | `100 - score` when the maximum is at most 100; otherwise open-upper-bound normalization is used |
| `(no limited)` | Higher-is-better open-upper-bound normalization using `max × safety_factor` |

The default open-upper-bound safety factor is 1.25.

## Eligibility

The default minimum is three valid model observations. A benchmark enters the
final ranking only when it is eligible for discrimination, saturation, and
Impact. The exact decision for every row is written to
`benchmark_inclusion_report.csv`.

## Metadata columns

Only the `GitHub` boundary is required by the core pipeline. The following
recognized metadata names may follow it and are preserved in the input but are
not used by the current BHI equations:

- `GitHub`, `HuggingFace`, `ModelScope`
- `benchmark_date`
- `gh_likes`, `gh_forks`
- `hf_likes`, `hf_downloads`, `hf_issue`
- `ms_likes`, `ms_downloads`

Additional metadata columns may also follow `GitHub`; they are ignored by the
core pipeline.

## Cold-start analysis input

`scripts/analyze_cold_start_fairness.py` uses a separate benchmark-level CSV.
It requires these canonical fields (accepted aliases are defined in the
script): `Benchmark`, `Release_Date`, `Source_URL`, `Date_Precision`,
`Verification_Status`, `Manual_Check_Priority`, `NUsage`, `SDisc`, `SSat`,
`Release_Scope`, `SImp`, and `BHI`.

Join independently verified date metadata to metrics from the matching BHI
baseline before running this analysis; do not mix metrics and dates from
different snapshots.

