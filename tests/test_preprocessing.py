"""Spec 02 acceptance tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from triagem_fuzzy.ingestion import DataIngestion
from triagem_fuzzy.preprocessing import DataPreprocessor, PreparedData


def _full(real_dataset_path: Path) -> PreparedData:
    df = DataIngestion(real_dataset_path).load_validated()
    return DataPreprocessor().run(df)


def test_remap_collapses_2_and_3_into_risco(sample_valid_df: pd.DataFrame) -> None:
    pre = DataPreprocessor()
    out = pre.remap_labels(sample_valid_df)
    assert set(out["triage_class"].unique()) <= {0, 1, 2}
    # raw 3 must become 2
    raw3 = sample_valid_df["triage_level"] == 3
    assert (out.loc[raw3, "triage_class"] == 2).all()


def test_encode_features_one_hots_arrival_mode(
    sample_valid_df: pd.DataFrame,
) -> None:
    pre = DataPreprocessor()
    remapped = pre.remap_labels(sample_valid_df)
    encoded = pre.encode_features(remapped)
    for col in (
        "arrival_mode_walk_in",
        "arrival_mode_ambulance",
        "arrival_mode_wheelchair",
    ):
        assert col in encoded.columns
    assert "arrival_mode" not in encoded.columns
    assert "triage_level" not in encoded.columns


def test_run_no_target_leakage(real_dataset_path: Path) -> None:
    data = _full(real_dataset_path)
    for col in ("triage_level", "triage_class"):
        assert col not in data.X_train.columns
        assert col not in data.X_test.columns


def test_run_deterministic_with_seed(real_dataset_path: Path) -> None:
    a = _full(real_dataset_path)
    b = _full(real_dataset_path)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_series_equal(a.y_train, b.y_train)


def test_stratification_preserves_class_ratio(real_dataset_path: Path) -> None:
    data = _full(real_dataset_path)
    train_ratio = data.y_train.value_counts(normalize=True).sort_index()
    test_ratio = data.y_test.value_counts(normalize=True).sort_index()
    diff = (train_ratio - test_ratio).abs()
    assert (diff < 0.01).all()


def test_feature_names_order_stable(real_dataset_path: Path) -> None:
    a = _full(real_dataset_path)
    b = _full(real_dataset_path)
    assert a.feature_names == b.feature_names
    # numerics first, then sorted one-hots
    numerics = a.feature_names[:8]
    one_hots = a.feature_names[8:]
    assert one_hots == sorted(one_hots)
    assert all(not n.startswith("arrival_mode_") for n in numerics)


def test_x_train_has_eleven_columns(real_dataset_path: Path) -> None:
    data = _full(real_dataset_path)
    assert data.X_train.shape[1] == 11  # 8 numerics + 3 one-hots


def test_remapped_class_counts(real_dataset_path: Path) -> None:
    df = DataIngestion(real_dataset_path).load_validated()
    remapped = DataPreprocessor().remap_labels(df)
    counts = remapped["triage_class"].value_counts().to_dict()
    assert counts == {0: 9924, 1: 4484, 2: 3592}


def test_label_encoder_passthrough(real_dataset_path: Path) -> None:
    data = _full(real_dataset_path)
    assert data.label_encoder == {0: "normal", 1: "atencao", 2: "risco"}
