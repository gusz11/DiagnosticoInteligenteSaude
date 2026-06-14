# 03 — Exploratory Data Analysis Spec

## Purpose
Produce the numerical and visual artifacts required by the relatório:
descriptive stats per feature, class balance, distributions, and
feature-vs-target relationships. EDA is read-only — it does not modify
the data and does not depend on the model.

## Inputs
- `df: pd.DataFrame` — output of Spec 01 (raw + validated).
- `output_dir: Path` — where plots and CSV summaries are written.

## Outputs
On disk, under `output_dir/eda/`:
- `summary_statistics.csv`
- `class_balance_raw.csv` (4 raw levels)
- `class_balance_remapped.csv` (3 academic classes)
- `correlation_matrix.csv`
- `dist_<feature>.png` for each numeric feature (histogram + KDE)
- `box_<feature>_by_class.png` for each numeric feature
- `pairplot_vitals.png` (heart_rate, BP, temperature, SpO2 vs class)

In memory, returned by methods:
- `EdaReport` dataclass bundling the same artifacts as DataFrames.

## Public API
```python
@dataclass(frozen=True)
class EdaReport:
    summary: pd.DataFrame
    class_balance_raw: pd.Series
    class_balance_remapped: pd.Series
    correlations: pd.DataFrame
    artifacts_dir: Path

class ExploratoryAnalysis:
    def __init__(self, output_dir: Path) -> None: ...

    def describe(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def class_balance(self, df: pd.DataFrame, *, remap: bool) -> pd.Series: ...
    def correlations(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def plot_distributions(self, df: pd.DataFrame) -> list[Path]: ...
    def plot_boxplots_by_class(self, df: pd.DataFrame) -> list[Path]: ...
    def run(self, df: pd.DataFrame) -> EdaReport: ...
```

## Invariants
- N1. No mutation of `df`.
- N2. Plotting uses matplotlib only (no seaborn dependency in v1) — keep
      the dependency footprint thin.
- N3. All written files are overwritten deterministically on rerun.
- N4. Correlation uses Pearson on numeric features only; categorical
      `arrival_mode` is reported separately as group-mean per class.

## Acceptance Criteria
- AC1. `run()` produces at least 8 histogram PNGs and 8 boxplot PNGs.
- AC2. `summary_statistics.csv` has columns
       `mean, std, min, 25%, 50%, 75%, max` for every numeric feature.
- AC3. `class_balance_remapped` sums to `len(df)` and has exactly 3 keys.

## Test Cases (tests/test_eda.py)
1. `test_describe_returns_expected_columns`
2. `test_class_balance_raw_has_four_levels`
3. `test_class_balance_remapped_has_three_levels`
4. `test_run_writes_all_expected_files` (use tmp_path)
5. `test_run_does_not_mutate_input`
