# 02 — Preprocessing Spec

## Purpose
Transform the raw validated DataFrame into model-ready feature matrix
`X` and target vector `y`, plus a train/test split. Owns the label
remap from 4 raw classes to the 3 academic classes.

## Inputs
- `df: pd.DataFrame` — already validated by Spec 01.

## Outputs
- `X_train, X_test: pd.DataFrame` — feature matrix without the target.
- `y_train, y_test: pd.Series` — remapped triage labels.
- `feature_names: list[str]` — column order of `X_*`.
- `label_encoder: dict[int, str]` — `{0: "normal", 1: "atencao", 2: "risco"}`.

## Label Remap (D1)
```
raw triage_level  →  class id  →  class name
       0          →     0       →  normal
       1          →     1       →  atencao
       2          →     2       →  risco
       3          →     2       →  risco
```
Implemented as `config.TRIAGE_LABEL_MAP`. Class names are ASCII for
filesystem-safe artifact naming; pretty Portuguese labels live in
`config.TRIAGE_LABELS_PT`.

## Feature Engineering
- `arrival_mode` is one-hot encoded → `arrival_mode_walk_in`,
  `arrival_mode_ambulance`, `arrival_mode_wheelchair` (drop none —
  readability over rank).
- Numeric features are left in original units. Scaling is **not** applied
  (Random Forest is scale-invariant). If a future spec adds KNN, scaling
  moves into a dedicated step.
- No imputation in v1 (dataset is clean per Spec 01).
- Outliers are **not** removed; they may carry real triage signal.

## Public API
```python
@dataclass(frozen=True)
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    label_encoder: dict[int, str]

class DataPreprocessor:
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = config.RANDOM_STATE,
        stratify: bool = True,
    ) -> None: ...

    def remap_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df with new column `triage_class` (0/1/2)."""

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode arrival_mode; drop raw `triage_level`."""

    def split(self, df: pd.DataFrame) -> PreparedData: ...

    def run(self, df: pd.DataFrame) -> PreparedData:
        """remap_labels → encode_features → split."""
```

## Invariants
- I1. `run()` is deterministic given the same `random_state`.
- I2. The target column never leaks into `X_*`.
- I3. `y_*` values are in `{0, 1, 2}`; never `{0, 1, 2, 3}`.
- I4. `stratify=True` keeps class proportions equal (±1%) across splits.
- I5. `feature_names` order is stable across runs (alphabetical sort
      after one-hot, then numerics first).

## Acceptance Criteria
- AC1. On the supplied dataset, `PreparedData.X_train.shape[1] == 11`
       (8 numerics + 3 one-hot columns).
- AC2. `y_train.value_counts(normalize=True)` matches
       `y_test.value_counts(normalize=True)` within 0.01 absolute.
- AC3. Combined class counts after remap: normal=9924, atencao=4484,
       risco=3592 (= 2701 + 891).

## Test Cases (tests/test_preprocessing.py)
1. `test_remap_collapses_2_and_3_into_risco`
2. `test_encode_features_one_hots_arrival_mode`
3. `test_run_no_target_leakage`
4. `test_run_deterministic_with_seed`
5. `test_stratification_preserves_class_ratio`
6. `test_feature_names_order_stable`
