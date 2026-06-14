"""Spec 06 acceptance tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from triagem_fuzzy import config
from triagem_fuzzy.articulation.comparator import TriageComparator
from triagem_fuzzy.articulation.integrator import TriageIntegrator
from triagem_fuzzy.fuzzy.factory import (
    build_integrated_engine,
    build_standalone_engine,
)
from triagem_fuzzy.ingestion import DataIngestion
from triagem_fuzzy.ml.random_forest import TriageRandomForest
from triagem_fuzzy.preprocessing import DataPreprocessor, PreparedData


@pytest.fixture(scope="module")
def small_prepared(real_dataset_path: Path) -> PreparedData:
    df = DataIngestion(real_dataset_path).load_validated()
    # Sample ~2000 rows for speed, keeping rough class balance.
    parts = [
        df[df["triage_level"] == lvl].sample(
            n=min((df["triage_level"] == lvl).sum(), 500),
            random_state=config.RANDOM_STATE,
        )
        for lvl in (0, 1, 2, 3)
    ]
    sampled = pd.concat(parts).reset_index(drop=True)
    return DataPreprocessor().run(sampled)


@pytest.fixture(scope="module")
def fitted(small_prepared: PreparedData) -> TriageRandomForest:
    model = TriageRandomForest(n_estimators=80)
    model.fit(small_prepared.X_train, small_prepared.y_train)
    return model


def test_comparator_alignment_with_input_index(
    fitted: TriageRandomForest, small_prepared: PreparedData
) -> None:
    cmp_ = TriageComparator(
        fitted, build_standalone_engine(), small_prepared.label_encoder
    )
    report = cmp_.run(small_prepared.X_test, small_prepared.y_test)
    assert report.predictions.index.equals(small_prepared.X_test.index)


def test_comparator_agreement_rate_in_unit_interval(
    fitted: TriageRandomForest, small_prepared: PreparedData
) -> None:
    cmp_ = TriageComparator(
        fitted, build_standalone_engine(), small_prepared.label_encoder
    )
    report = cmp_.run(small_prepared.X_test, small_prepared.y_test)
    assert 0.0 <= report.agreement_rate <= 1.0


def test_comparator_confusion_shape(
    fitted: TriageRandomForest, small_prepared: PreparedData
) -> None:
    cmp_ = TriageComparator(
        fitted, build_standalone_engine(), small_prepared.label_encoder
    )
    report = cmp_.run(small_prepared.X_test, small_prepared.y_test)
    assert report.confusion_ml_vs_fuzzy.shape == (3, 3)
    assert report.confusion_ml_vs_fuzzy.values.sum() == len(small_prepared.X_test)


def test_integrator_uses_proba_for_correct_class(
    fitted: TriageRandomForest, small_prepared: PreparedData
) -> None:
    integ = TriageIntegrator(
        fitted, build_integrated_engine(), small_prepared.label_encoder
    )
    row = small_prepared.X_test.iloc[0]
    proba = np.array([0.05, 0.10, 0.85])
    sample = integ.build_fuzzy_input(row, proba)
    assert sample["ml_risk_proba"] == pytest.approx(0.85)


def test_integrator_reports_both_accuracies(
    fitted: TriageRandomForest, small_prepared: PreparedData
) -> None:
    integ = TriageIntegrator(
        fitted, build_integrated_engine(), small_prepared.label_encoder
    )
    report = integ.run(small_prepared.X_test, small_prepared.y_test)
    assert 0.0 <= report.accuracy_ml <= 1.0
    assert 0.0 <= report.accuracy_integrated <= 1.0


def test_disagreement_sample_table_size_capped(
    fitted: TriageRandomForest, small_prepared: PreparedData
) -> None:
    cmp_ = TriageComparator(
        fitted, build_standalone_engine(), small_prepared.label_encoder
    )
    report = cmp_.run(small_prepared.X_test, small_prepared.y_test)
    assert len(report.disagreement_samples) <= 20


def test_integration_requires_ml_input_variable_in_engine(
    fitted: TriageRandomForest, small_prepared: PreparedData
) -> None:
    with pytest.raises(ValueError, match="ml_risk_proba"):
        TriageIntegrator(
            fitted, build_standalone_engine(), small_prepared.label_encoder
        )


def test_delta_per_class_columns(
    fitted: TriageRandomForest, small_prepared: PreparedData
) -> None:
    integ = TriageIntegrator(
        fitted, build_integrated_engine(), small_prepared.label_encoder
    )
    report = integ.run(small_prepared.X_test, small_prepared.y_test)
    for col in ("delta_recall", "delta_f1"):
        assert col in report.delta_per_class.columns
    assert len(report.delta_per_class) == 3
