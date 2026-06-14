# 05 — Fuzzy Inference System Spec

## Purpose
Build a Mamdani fuzzy inference system that maps three vital-sign inputs
(plus, in integration mode, the ML risk probability) into a continuous
**risk score** ∈ [0, 10], then maps that score into a discrete triage
class. The fuzzy system must be readable and explainable in the
relatório alongside the ML model.

Reference library: `scikit-fuzzy` (alias `skfuzzy`). Add to
`pyproject.toml` when implementation starts.

## Brief Compliance Map
| Requirement                              | Where satisfied |
|------------------------------------------|-----------------|
| ≥ 2 input variables                      | 3 vitals (+1 ML in B) |
| 1 output variable                        | `risk_score` ∈ [0, 10] |
| ≥ 3 linguistic terms per main variable   | each has 3       |
| ≥ 6 SE…ENTÃO rules                       | 9 rules in v1 (Spec 05 §Rule Base) |
| Triangular/trapezoidal membership fns    | trapezoidal at edges, triangular in middle |
| Fuzzification + inference + defuzzification | engine pipeline |

## Variables (Standalone Mode)
| Variable | Type   | Universe | Linguistic terms |
|----------|--------|----------|------------------|
| `body_temperature` | input  | [34, 42] °C | `baixa`, `normal`, `alta` |
| `heart_rate`       | input  | [30, 200] bpm | `baixa`, `normal`, `alta` |
| `systolic_blood_pressure` | input | [60, 220] mmHg | `baixa`, `normal`, `alta` |
| `risk_score`       | output | [0, 10]    | `baixo`, `medio`, `alto`  |

### Membership Function Parameters (initial draft, tunable)
```
body_temperature
  baixa  : trap(34, 34, 35.5, 36.3)
  normal : tri (36.0, 36.8, 37.5)
  alta   : trap(37.2, 38.0, 42, 42)

heart_rate
  baixa  : trap(30, 30, 50, 65)
  normal : tri (60, 80, 100)
  alta   : trap(95, 110, 200, 200)

systolic_blood_pressure
  baixa  : trap(60, 60, 85, 100)
  normal : tri (95, 120, 140)
  alta   : trap(135, 150, 220, 220)

risk_score
  baixo  : trap(0, 0, 2, 4)
  medio  : tri (3, 5, 7)
  alto   : trap(6, 8, 10, 10)
```

## Rule Base (v1 — 9 rules, ≥ 6 required)
```
R1.  SE temperatura alta   E batimentos alta              ENTÃO risco alto
R2.  SE temperatura alta   E pressão  baixa               ENTÃO risco alto
R3.  SE batimentos alta    E pressão  baixa               ENTÃO risco alto
R4.  SE temperatura normal E batimentos normal E pressão normal ENTÃO risco baixo
R5.  SE temperatura baixa  E pressão  baixa               ENTÃO risco alto
R6.  SE batimentos baixa   E pressão  normal              ENTÃO risco medio
R7.  SE temperatura alta   E batimentos normal E pressão normal ENTÃO risco medio
R8.  SE batimentos alta    E pressão  normal              ENTÃO risco medio
R9.  SE temperatura normal E pressão  alta                ENTÃO risco medio
```
Aggregation: `max`; defuzzification: centroid.

## Integration Mode (Approach B add-on)
Adds one input variable consumed only by `TriageIntegrator` (Spec 06):

| Variable          | Type  | Universe | Linguistic terms |
|-------------------|-------|----------|------------------|
| `ml_risk_proba`   | input | [0, 1]   | `baixo`, `medio`, `alto` |
```
ml_risk_proba
  baixo : trap(0.0, 0.0, 0.20, 0.40)
  medio : tri (0.30, 0.50, 0.70)
  alto  : trap(0.60, 0.80, 1.0, 1.0)
```
Three additional rules tying ML output to risk:
```
R10. SE ml_risk_proba alto                              ENTÃO risco alto
R11. SE ml_risk_proba medio E pressão baixa             ENTÃO risco alto
R12. SE ml_risk_proba baixo E temperatura normal        ENTÃO risco baixo
```

## Score → Class Decoder
```
risk_score ∈ [0.0, 3.5)  →  normal
risk_score ∈ [3.5, 6.5)  →  atencao
risk_score ∈ [6.5, 10.0] →  risco
```
Thresholds tunable; recorded in `config.FUZZY_DECISION_BANDS`.

## Public API
```python
@dataclass(frozen=True)
class MembershipFunction:
    name: str
    kind: Literal["tri", "trap"]
    params: tuple[float, ...]   # (a,b,c) or (a,b,c,d)

@dataclass(frozen=True)
class LinguisticVariable:
    name: str
    universe: tuple[float, float]
    resolution: int               # default 1001 points
    terms: dict[str, MembershipFunction]

@dataclass(frozen=True)
class FuzzyRule:
    antecedents: list[tuple[str, str]]   # [("body_temperature","alta"), ...]
    consequent: tuple[str, str]
    op: Literal["AND", "OR"] = "AND"

class RuleBase:
    rules: list[FuzzyRule]
    def add(self, rule: FuzzyRule) -> None: ...

class FuzzyInferenceEngine:
    def __init__(
        self,
        inputs: list[LinguisticVariable],
        output: LinguisticVariable,
        rules: RuleBase,
        defuzz_method: Literal["centroid", "mom"] = "centroid",
    ) -> None: ...

    def fuzzify(self, sample: dict[str, float]) -> dict[str, dict[str, float]]:
        """{var_name: {term: membership}}"""

    def infer(self, sample: dict[str, float]) -> float:
        """Returns crisp risk_score."""

    def classify(self, sample: dict[str, float]) -> str:
        """Score → class via FUZZY_DECISION_BANDS."""

    def explain(self, sample: dict[str, float]) -> "FuzzyTrace":
        """Per-rule firing strength + activated consequent — for relatório."""
```

## Invariants
- I1. Engine is stateless given fixed variables + rules — same input
      yields the same output across runs.
- I2. Output `risk_score` is clamped to the output universe.
- I3. Membership functions are non-negative and ≤ 1 everywhere.
- I4. Adjacent terms overlap on a non-empty interval (no gaps).
- I5. `classify()` uses thresholds from `config`, not literals.

## Acceptance Criteria
- AC1. A "clearly normal" patient (T=36.8, HR=75, BP=120) classifies as
       `normal` with risk_score < 3.5.
- AC2. A "clearly risk" patient (T=39.5, HR=130, BP=85) classifies as
       `risco` with risk_score > 6.5.
- AC3. The rule base has ≥ 6 rules in standalone mode and ≥ 9 with
       integration rules included.
- AC4. `explain()` returns a per-rule firing strength list summing to
       at least one non-zero strength on any valid input.

## Test Cases (tests/test_fuzzy.py)
1. `test_membership_in_zero_one`
2. `test_terms_overlap_on_adjacent_intervals`
3. `test_clear_normal_patient_yields_low_risk`
4. `test_clear_risco_patient_yields_high_risk`
5. `test_engine_is_deterministic`
6. `test_explain_returns_per_rule_strengths`
7. `test_integration_rules_only_active_when_ml_input_provided`
