# 01 — Ingestion Spec

## Purpose
Load the triage CSV from disk into a validated pandas DataFrame with a
known schema. This module is the only place that touches the raw file —
everything downstream consumes its output.

## Inputs
- `csv_path: Path` — defaults to `config.RAW_DATA_PATH`
  (`dataset/triagem_fuzzy.csv`).

## Outputs
- `pandas.DataFrame` with the columns and dtypes declared in the schema
  table below. Row order matches the file.

## Schema (authoritative)
| Column                     | Dtype     | Range / Domain                          | Notes |
|----------------------------|-----------|------------------------------------------|-------|
| age                        | float64   | [0, 120]                                 | years |
| heart_rate                 | float64   | [20, 250]                                | bpm   |
| systolic_blood_pressure    | float64   | [50, 250]                                | mmHg  |
| oxygen_saturation          | float64   | [50, 100]                                | %     |
| body_temperature           | float64   | [30.0, 43.0]                             | °C    |
| pain_level                 | int64     | [0, 10]                                  | scale |
| chronic_disease_count      | int64     | [0, 20]                                  | count |
| previous_er_visits         | int64     | [0, 50]                                  | count |
| arrival_mode               | category  | {`walk_in`, `ambulance`, `wheelchair`}   | discovered from data 2026-06-14 |
| triage_level               | int64     | {0, 1, 2, 3}                             | raw target |

## Public API
```python
class DataIngestion:
    def __init__(self, csv_path: Path | None = None) -> None: ...
    def load(self) -> pd.DataFrame: ...
    def validate(self, df: pd.DataFrame) -> None:
        """Raise SchemaError if columns/dtypes/ranges violate the schema."""
    def load_validated(self) -> pd.DataFrame:
        """load() + validate(); preferred entry point."""

class SchemaError(ValueError): ...
```

## Invariants
- N1. `load()` never mutates the file.
- N2. `validate()` is pure (no I/O, no logging side effects beyond a
      single summary line).
- N3. Unknown columns in the CSV cause `SchemaError` — no silent drops.
- N4. Missing values: spec assumes the dataset has none; if any are found,
      raise `SchemaError` with column + row count. Imputation is **not**
      this module's responsibility.

## Acceptance Criteria
- AC1. `load_validated()` returns a DataFrame of shape `(18000, 10)` on
       the supplied dataset.
- AC2. Removing a column or corrupting a dtype in a fixture CSV raises
       `SchemaError`.
- AC3. Passing a non-existent path raises `FileNotFoundError`.

## Test Cases (tests/test_ingestion.py)
1. `test_load_returns_expected_shape_and_columns`
2. `test_validate_rejects_missing_column`
3. `test_validate_rejects_out_of_range_temperature`
4. `test_validate_rejects_unknown_arrival_mode`
5. `test_load_validated_is_idempotent` (two calls → equal DataFrames)
6. `test_missing_file_raises`
