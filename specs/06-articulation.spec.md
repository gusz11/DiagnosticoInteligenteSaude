# 06 — Articulation Spec (Approaches A + B)

## Purpose
Articulate the ML model (Spec 04) with the Fuzzy system (Spec 05) using
both approaches mandated by the user:

- **Approach A — Comparison**: run ML and fuzzy independently on the
  same patients, compare agreement, disagreement, and per-class behavior.
- **Approach B — Integration**: feed ML's per-class probability into the
  fuzzy engine alongside vitals, producing a single integrated triage
  decision.

This module is the only place in `src/` that imports both `ml/` and
`fuzzy/` (cross-cutting invariant I-overview).

## Inputs
- `model: BaseTriageClassifier` (already fitted).
- `engine_standalone: FuzzyInferenceEngine` (vitals only).
- `engine_integrated: FuzzyInferenceEngine` (vitals + ml_risk_proba).
- `X_test: pd.DataFrame`, `y_test: pd.Series`.

## Outputs
- `ComparisonReport` and `IntegrationReport` dataclasses.
- Tabular artifacts on disk under `output_dir/articulation/`.

## Public API
```python
@dataclass(frozen=True)
class ComparisonReport:
    predictions: pd.DataFrame
    #   columns: [y_true, ml_pred, fuzzy_pred, agree]
    agreement_rate: float
    confusion_ml_vs_fuzzy: pd.DataFrame    # 3x3, ML rows vs fuzzy cols
    per_class_agreement: pd.Series         # agreement_rate per true class
    disagreement_samples: pd.DataFrame     # k=20 representative rows

class TriageComparator:
    def __init__(
        self,
        model: BaseTriageClassifier,
        fuzzy_engine: FuzzyInferenceEngine,
        label_encoder: dict[int, str],
    ) -> None: ...

    def run(self, X: pd.DataFrame, y: pd.Series) -> ComparisonReport: ...


@dataclass(frozen=True)
class IntegrationReport:
    predictions: pd.DataFrame
    #   columns: [y_true, ml_pred, ml_p_risco, integrated_score,
    #             integrated_pred]
    accuracy_ml: float
    accuracy_integrated: float
    macro_f1_ml: float
    macro_f1_integrated: float
    confusion_integrated: pd.DataFrame
    delta_per_class: pd.DataFrame
    #   recall/f1 deltas: integrated minus ml-only

class TriageIntegrator:
    def __init__(
        self,
        model: BaseTriageClassifier,
        fuzzy_engine: FuzzyInferenceEngine,
        label_encoder: dict[int, str],
        ml_proba_class: int = 2,   # default: probability of `risco`
    ) -> None: ...

    def build_fuzzy_input(
        self, row: pd.Series, proba: np.ndarray
    ) -> dict[str, float]:
        """Pack vitals + ml_risk_proba into the engine's input dict."""

    def run(self, X: pd.DataFrame, y: pd.Series) -> IntegrationReport: ...
```

## Behavior Notes
- **A**. Fuzzy is run on a fixed subset of features the engine knows
  about (`body_temperature`, `heart_rate`, `systolic_blood_pressure`).
  Other columns in `X` are ignored by the fuzzy path but still consumed
  by the ML path — the comparison is fair per patient, not per feature.
- **B**. `ml_proba_class=2` uses P(risco). Alternative: P(atencao∪risco)
  computed as `proba[:,1] + proba[:,2]`. The spec leaves this as a
  config switch (`config.ML_PROBA_AGGREGATION`).
- Both reports must include a small qualitative table (≤ 20 rows) of
  "interesting" patients: highest ML/fuzzy disagreement (A); biggest
  integrated-vs-ML class change (B). Used directly in the relatório.

## Invariants
- I1. The fuzzy engine instance for A has **no** `ml_risk_proba` input;
      the one for B **must**.
- I2. Neither comparator mutates `X` or `y`.
- I3. Predictions are aligned by index — `predictions.index == X.index`.
- I4. Class labels in all reports are the human-readable strings
      (`"normal"`, `"atencao"`, `"risco"`), not integers.

## Acceptance Criteria
- AC1. `ComparisonReport.agreement_rate ∈ [0, 1]`.
- AC2. `IntegrationReport.accuracy_integrated` reported alongside
       `accuracy_ml` for direct comparison in the relatório.
- AC3. `confusion_ml_vs_fuzzy` is a 3×3 DataFrame whose sum equals
       `len(X)`.
- AC4. `delta_per_class` has 3 rows and at least columns
       `["delta_recall", "delta_f1"]`.

## Test Cases (tests/test_articulation.py)
1. `test_comparator_alignment_with_input_index`
2. `test_comparator_agreement_rate_in_unit_interval`
3. `test_integrator_uses_proba_for_correct_class`
4. `test_integrator_reports_both_accuracies`
5. `test_disagreement_sample_table_size_capped`
6. `test_integration_requires_ml_input_variable_in_engine`
   (engine without `ml_risk_proba` raises a configuration error)
