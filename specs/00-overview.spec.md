# 00 — Project Overview Spec

## Purpose
Build an academic decision-support system that combines a Machine Learning
classifier with a Fuzzy Inference System to triage patients into three
severity classes: **normal**, **atenção**, **risco**. The project must be
modular, debuggable per module, and driven by these specs before any
implementation begins.

> Strictly academic. Not a clinical diagnostic tool.

## Locked Decisions
| # | Decision | Value | Source |
|---|----------|-------|--------|
| D1 | Target label mapping | `0→normal`, `1→atenção`, `2,3→risco` | User, 2026-06-14 |
| D2 | ML algorithm | Random Forest (multiclass) | User, 2026-06-14 |
| D3 | Articulation strategy | Both A (Comparison) and B (Integration) | User, 2026-06-14 |
| D4 | Dataset | `dataset/triagem_fuzzy.csv` (18.000 linhas) | Provided |
| D5 | Delivery format | Python modules + classes (no notebook) | User, 2026-06-14 |

## Glossary
- **Triagem**: triage; classification of urgency level on patient arrival.
- **Fuzzification**: mapping crisp numeric input to membership degrees.
- **Defuzzification**: collapsing fuzzy output set into a crisp number.
- **Linguistic variable**: e.g. `body_temperature ∈ {baixa, normal, alta}`.

## Module Map
```
src/triagem_fuzzy/
  config.py              constants, paths, thresholds, label maps
  ingestion.py           Spec 01 — DataIngestion
  preprocessing.py       Spec 02 — DataPreprocessor
  eda.py                 Spec 03 — ExploratoryAnalysis
  ml/
    base.py              Spec 04 — BaseTriageClassifier (ABC)
    random_forest.py     Spec 04 — TriageRandomForest
    evaluator.py         Spec 04 — ModelEvaluator
  fuzzy/
    variables.py         Spec 05 — LinguisticVariable, MembershipFunction
    rules.py             Spec 05 — FuzzyRule, RuleBase
    engine.py            Spec 05 — FuzzyInferenceEngine
  articulation/
    comparator.py        Spec 06 — TriageComparator (Approach A)
    integrator.py        Spec 06 — TriageIntegrator (Approach B)
  reporting.py           Spec 07 — ResultReporter
main.py                  CLI orchestrator
tests/                   pytest per module, mirrors src tree
```

## Build Order
1. `config` + `ingestion` (Spec 01)
2. `preprocessing` (Spec 02)
3. `eda` (Spec 03) — read-only, can be deferred
4. `ml/*` (Spec 04)
5. `fuzzy/*` (Spec 05)
6. `articulation/*` (Spec 06) — depends on 04 and 05
7. `reporting` (Spec 07)
8. `main.py` orchestrator

Each module must be runnable and testable in isolation. No cross-imports
between `ml/` and `fuzzy/` — only `articulation/` may import both.

## Cross-Cutting Invariants
- I1. All public functions are typed (PEP 484) and validated at boundaries.
- I2. Random seed is centralised in `config.RANDOM_STATE` (default 42).
- I3. No file in `src/` writes to disk except via `reporting`.
- I4. No `print` outside `main.py` and `reporting.py`; use `logging`.
- I5. The label set is a single source of truth: `config.TRIAGE_LABELS`.

## Acceptance Criteria (project-level)
- AC1. `pytest` passes on all modules.
- AC2. `python main.py --pipeline full` runs end-to-end on the dataset and
       produces a report artifact.
- AC3. Every spec from 01–07 has an implementation that satisfies its own
       acceptance criteria.
- AC4. The relatório (separate doc) cites which module produced which
       result; nothing is hand-authored.
