"""Spec 04 acceptance tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from triagem_fuzzy import config
from triagem_fuzzy.ingestion import DataIngestion
from triagem_fuzzy.ml.evaluator import ModelEvaluator
from triagem_fuzzy.ml.random_forest import TriageRandomForest
from triagem_fuzzy.preprocessing import DataPreprocessor, PreparedData


@pytest.fixture(scope="module")
def prepared(real_dataset_path: Path) -> PreparedData:
    df = DataIngestion(real_dataset_path).load_validated()
    return DataPreprocessor().run(df)


@pytest.fixture(scope="module")
def fitted_model(prepared: PreparedData) -> TriageRandomForest:
    model = TriageRandomForest(n_estimators=120)  # faster for tests
    model.fit(prepared.X_train, prepared.y_train)
    return model


def test_fit_predict_shapes(
    fitted_model: TriageRandomForest, prepared: PreparedData
) -> None:
    preds = fitted_model.predict(prepared.X_test)
    assert preds.shape == (len(prepared.X_test),)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_predict_proba_rowsums_to_one(
    fitted_model: TriageRandomForest, prepared: PreparedData
) -> None:
    proba = fitted_model.predict_proba(prepared.X_test)
    assert proba.shape == (len(prepared.X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_save_load_round_trip(
    tmp_path: Path,
    fitted_model: TriageRandomForest,
    prepared: PreparedData,
) -> None:
    path = tmp_path / "model.joblib"
    fitted_model.save(path)
    loaded = TriageRandomForest.load(path)
    np.testing.assert_array_equal(
        fitted_model.predict(prepared.X_test),
        loaded.predict(prepared.X_test),
    )


def test_evaluator_returns_three_by_three_confusion(
    fitted_model: TriageRandomForest, prepared: PreparedData
) -> None:
    evaluator = ModelEvaluator(prepared.label_encoder)
    report = evaluator.evaluate(fitted_model, prepared.X_test, prepared.y_test)
    assert report.confusion_matrix.shape == (3, 3)
    assert list(report.confusion_matrix.index) == ["normal", "atencao", "risco"]
    assert list(report.confusion_matrix.columns) == ["normal", "atencao", "risco"]


def test_feature_importances_align_with_feature_names(
    fitted_model: TriageRandomForest, prepared: PreparedData
) -> None:
    evaluator = ModelEvaluator(prepared.label_encoder)
    report = evaluator.evaluate(fitted_model, prepared.X_test, prepared.y_test)
    assert set(report.feature_importances.index) == set(prepared.feature_names)
    assert abs(report.feature_importances.sum() - 1.0) < 1e-6


def test_accuracy_meets_sanity_threshold(
    fitted_model: TriageRandomForest, prepared: PreparedData
) -> None:
    evaluator = ModelEvaluator(prepared.label_encoder)
    report = evaluator.evaluate(fitted_model, prepared.X_test, prepared.y_test)
    assert report.accuracy >= 0.65  # sanity, not contractual


def test_plot_confusion_writes_file(
    tmp_path: Path,
    fitted_model: TriageRandomForest,
    prepared: PreparedData,
) -> None:
    evaluator = ModelEvaluator(prepared.label_encoder)
    report = evaluator.evaluate(fitted_model, prepared.X_test, prepared.y_test)
    path = evaluator.plot_confusion(report, tmp_path / "cm.png")
    assert path.exists()
    assert path.stat().st_size > 0
