"""Spec 07 acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from triagem_fuzzy.articulation.comparator import ComparisonReport
from triagem_fuzzy.articulation.integrator import IntegrationReport
from triagem_fuzzy.eda import EdaReport
from triagem_fuzzy.ml.evaluator import EvaluationReport
from triagem_fuzzy.reporting import REQUIRED_SECTIONS, ResultReporter

LABELS = ["normal", "atencao", "risco"]


def _dummy_eda(tmp_path: Path) -> EdaReport:
    return EdaReport(
        summary=pd.DataFrame(
            {col: [0.0] for col in ("mean", "std", "min", "25%", "50%", "75%", "max")},
            index=["age"],
        ),
        class_balance_raw=pd.Series([10, 5, 3, 2], index=[0, 1, 2, 3]),
        class_balance_remapped=pd.Series([10, 5, 5], index=LABELS),
        correlations=pd.DataFrame([[1.0]], index=["age"], columns=["age"]),
        artifacts_dir=tmp_path,
    )


def _dummy_eval() -> EvaluationReport:
    cm = pd.DataFrame(
        [[7, 1, 0], [2, 3, 1], [0, 1, 4]],
        index=LABELS,
        columns=LABELS,
    )
    cm.index.name = "true"
    cm.columns.name = "pred"
    return EvaluationReport(
        accuracy=0.7,
        macro_f1=0.65,
        weighted_f1=0.68,
        per_class=pd.DataFrame(
            {
                "precision": [0.78, 0.6, 0.8],
                "recall": [0.88, 0.5, 0.8],
                "f1": [0.82, 0.55, 0.8],
                "support": [8, 6, 5],
            },
            index=LABELS,
        ),
        confusion_matrix=cm,
        feature_importances=pd.Series(
            {"age": 0.3, "heart_rate": 0.4, "systolic_blood_pressure": 0.3}
        ),
    )


def _dummy_cmp() -> ComparisonReport:
    cm = pd.DataFrame(
        [[5, 1, 0], [1, 4, 1], [0, 1, 3]],
        index=LABELS,
        columns=LABELS,
    )
    cm.index.name = "ml"
    cm.columns.name = "fuzzy"
    preds = pd.DataFrame(
        {
            "y_true": ["normal", "atencao", "risco"],
            "ml_pred": ["normal", "atencao", "risco"],
            "fuzzy_pred": ["normal", "risco", "risco"],
            "agree": [True, False, True],
        }
    )
    return ComparisonReport(
        predictions=preds,
        agreement_rate=0.7,
        confusion_ml_vs_fuzzy=cm,
        per_class_agreement=pd.Series([0.9, 0.5, 0.6], index=LABELS),
        disagreement_samples=preds.head(1),
        ml_accuracy=0.7,
        fuzzy_accuracy=0.6,
    )


def _dummy_int() -> IntegrationReport:
    cm = pd.DataFrame(
        [[6, 1, 0], [1, 4, 0], [0, 1, 3]],
        index=LABELS,
        columns=LABELS,
    )
    cm.index.name = "true"
    cm.columns.name = "pred"
    return IntegrationReport(
        predictions=pd.DataFrame(
            {
                "y_true": [0, 1, 2],
                "ml_pred": [0, 1, 2],
                "ml_p_risco": [0.1, 0.4, 0.8],
                "integrated_score": [1.5, 5.2, 8.1],
                "integrated_pred": [0, 1, 2],
            }
        ),
        accuracy_ml=0.7,
        accuracy_integrated=0.72,
        macro_f1_ml=0.65,
        macro_f1_integrated=0.68,
        confusion_integrated=cm,
        delta_per_class=pd.DataFrame(
            {
                "recall_ml": [0.8, 0.5, 0.7],
                "recall_integrated": [0.82, 0.55, 0.75],
                "delta_recall": [0.02, 0.05, 0.05],
                "f1_ml": [0.78, 0.5, 0.7],
                "f1_integrated": [0.8, 0.55, 0.75],
                "delta_f1": [0.02, 0.05, 0.05],
            },
            index=LABELS,
        ),
        biggest_changes=pd.DataFrame({"a": [1]}),
    )


def test_write_summary_json_round_trips_payload(tmp_path: Path) -> None:
    reporter = ResultReporter(tmp_path)
    path = reporter.write_summary_json({"a": 1, "b": [2, 3]})
    assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}


def test_assemble_emits_all_required_sections(tmp_path: Path) -> None:
    reporter = ResultReporter(tmp_path)
    md_path = reporter.assemble(
        _dummy_eda(tmp_path), _dummy_eval(), _dummy_cmp(), _dummy_int()
    )
    text = md_path.read_text()
    for header in (
        "## Apresentação",
        "## Objetivos",
        "## Parte 1",
        "## Parte 2",
        "## Parte 3",
        "## Análise Crítica",
    ):
        assert header in text


def test_assemble_is_deterministic_under_fixed_seed(tmp_path: Path) -> None:
    r1 = ResultReporter(tmp_path / "a")
    r2 = ResultReporter(tmp_path / "b")
    r1.assemble(_dummy_eda(tmp_path), _dummy_eval(), _dummy_cmp(), _dummy_int())
    r2.assemble(_dummy_eda(tmp_path), _dummy_eval(), _dummy_cmp(), _dummy_int())
    assert (
        (r1.output_dir / "summary.json").read_bytes()
        == (r2.output_dir / "summary.json").read_bytes()
    )


def test_render_confusions_writes_three_pngs(tmp_path: Path) -> None:
    reporter = ResultReporter(tmp_path)
    paths = reporter.render_confusions(_dummy_eval(), _dummy_cmp(), _dummy_int())
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_write_markdown_rejects_missing_section(tmp_path: Path) -> None:
    reporter = ResultReporter(tmp_path)
    partial = {k: "x" for k in REQUIRED_SECTIONS[:-1]}
    with pytest.raises(ValueError, match="Missing report sections"):
        reporter.write_markdown(partial)
