# 04 — Machine Learning Model Spec

## Purpose
Train, persist, and evaluate a multiclass Random Forest that predicts
triage class ∈ {normal, atencao, risco} from the preprocessed features.
Also expose calibrated class probabilities — these are the bridge to
the fuzzy system in Approach B (Spec 06).

## Inputs
- `PreparedData` from Spec 02.

## Outputs
- `TrainedModel` wrapper containing the fitted estimator, feature names,
  and label encoder.
- `EvaluationReport` with metrics, confusion matrix, and per-class
  feature importances.
- Model file on disk at `output_dir/models/random_forest.joblib`.

## Public API
```python
class BaseTriageClassifier(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseTriageClassifier": ...
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns array shape (n_samples, 3); columns ordered 0,1,2."""
    @abstractmethod
    def save(self, path: Path) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTriageClassifier": ...

class TriageRandomForest(BaseTriageClassifier):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        class_weight: str | dict | None = "balanced",
        random_state: int = config.RANDOM_STATE,
    ) -> None: ...

@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: pd.DataFrame      # precision, recall, f1, support per class
    confusion_matrix: pd.DataFrame  # rows=true, cols=pred, labels = names
    feature_importances: pd.Series

class ModelEvaluator:
    def __init__(self, label_encoder: dict[int, str]) -> None: ...
    def evaluate(
        self,
        model: BaseTriageClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> EvaluationReport: ...
    def plot_confusion(self, report: EvaluationReport, path: Path) -> Path: ...
```

## Hyperparameter Justification
- `class_weight="balanced"` to counter the 55/25/20 class skew without
  oversampling.
- `min_samples_leaf=2` to mildly regularise; the relatório will note
  that tuning via grid-search is a deferred improvement.
- `n_estimators=300` is a reasonable default for the dataset size; not
  swept in v1.

## Invariants
- I1. `predict_proba` columns are aligned with sorted class ids `[0,1,2]`.
- I2. `feature_importances_` length equals `len(feature_names)`.
- I3. `save()` then `load()` produces a model with identical predictions
      on a held-out sample.
- I4. Training never touches `X_test` / `y_test`.

## Acceptance Criteria
- AC1. `accuracy >= 0.75` and `macro_f1 >= 0.65` on the test split.
       (Sanity targets, not contractual — adjust after first run.)
- AC2. Confusion matrix is a 3×3 DataFrame with row/column labels
       `["normal", "atencao", "risco"]`.
- AC3. `EvaluationReport.feature_importances` sums to 1.0 ± 1e-6.

## Test Cases (tests/test_ml.py)
1. `test_fit_predict_shapes`
2. `test_predict_proba_rowsums_to_one`
3. `test_save_load_round_trip`
4. `test_evaluator_returns_three_by_three_confusion`
5. `test_feature_importances_align_with_feature_names`
6. `test_class_weight_balanced_improves_minority_recall` (vs. None)
