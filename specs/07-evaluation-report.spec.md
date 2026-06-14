# 07 — Evaluation & Reporting Spec

## Purpose
Collect the artifacts produced by Specs 03–06 and assemble a single
human-readable report (Markdown + plots) that maps each section to a
requirement in the academic brief. This module owns *all* disk writes
outside of `ingestion.load()`.

## Inputs
- `EdaReport` (Spec 03).
- `EvaluationReport` (Spec 04).
- `ComparisonReport` (Spec 06 — Approach A).
- `IntegrationReport` (Spec 06 — Approach B).
- `output_dir: Path`.

## Outputs
On disk, under `output_dir/report/`:
- `report.md` — the main relatório draft, structured to match the brief
  (Objetivo geral → Objetivos específicos → Parte 1/2/3 → Análise crítica).
- `confusion_ml.png`, `confusion_integrated.png`, `confusion_ml_vs_fuzzy.png`
- `feature_importances.png`
- `risk_score_distribution.png`
- `summary.json` — machine-readable digest of all metrics.

## Public API
```python
class ResultReporter:
    def __init__(self, output_dir: Path) -> None: ...

    def write_summary_json(self, payload: dict) -> Path: ...
    def write_markdown(self, sections: dict[str, str]) -> Path: ...
    def render_confusions(
        self,
        eval_report: EvaluationReport,
        cmp_report: ComparisonReport,
        int_report: IntegrationReport,
    ) -> list[Path]: ...
    def assemble(
        self,
        eda: EdaReport,
        ml: EvaluationReport,
        comparison: ComparisonReport,
        integration: IntegrationReport,
    ) -> Path:
        """Top-level entry point; returns path to report.md."""
```

## Report Outline (driven by the brief)
1. **Apresentação** — problem statement, dataset summary (from EDA).
2. **Objetivos** — copied from spec 00.
3. **Parte 1 — Machine Learning**
   - dataset stats, label remap rationale (D1)
   - algorithm choice + justification (D2)
   - metrics table, confusion matrix, feature importances
4. **Parte 2 — Sistema Fuzzy**
   - variables, membership function plots
   - rule base listing
   - example fuzzy traces from `engine.explain()`
5. **Parte 3 — Articulação**
   - 3.A Comparison: agreement, divergence cases
   - 3.B Integration: integrated metrics vs ML-only
6. **Análise Crítica** — limitations, ethics note (academic use only),
   future work.

## Invariants
- I1. Reporter is the only module performing markdown/JSON writes.
- I2. All numbers in `summary.json` round to 4 decimals; visible report
      numbers round to 3.
- I3. Plots have axis labels, a title, and a legend where applicable.
- I4. Every section in the outline corresponds to an entry in the
      `sections` dict — `assemble()` fails fast if any is missing.

## Acceptance Criteria
- AC1. `assemble()` produces a non-empty `report.md` containing all six
       outlined sections (verified by H2 header check).
- AC2. `summary.json` includes top-level keys
       `["dataset", "ml", "comparison", "integration"]`.
- AC3. Running the full pipeline twice with the same seed yields
       byte-identical `summary.json`.

## Test Cases (tests/test_reporting.py)
1. `test_write_summary_json_round_trips_payload`
2. `test_assemble_emits_all_required_sections`
3. `test_assemble_is_deterministic_under_fixed_seed`
4. `test_render_confusions_writes_three_pngs`
