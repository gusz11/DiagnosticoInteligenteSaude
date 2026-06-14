"""Spec 01 acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from triagem_fuzzy import config
from triagem_fuzzy.ingestion import DataIngestion, SchemaError


def test_load_returns_expected_shape_and_columns(real_dataset_path: Path) -> None:
    df = DataIngestion(real_dataset_path).load_validated()
    assert df.shape == (18000, 10)
    assert set(df.columns) == set(config.EXPECTED_COLUMNS)


def test_validate_rejects_missing_column(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    bad = sample_valid_df.drop(columns=["heart_rate"])
    path = tmp_path / "missing_col.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="Missing required columns"):
        DataIngestion(path).load_validated()


def test_validate_rejects_out_of_range_temperature(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    bad = sample_valid_df.copy()
    bad.loc[0, "body_temperature"] = 50.0  # outside [30, 43]
    path = tmp_path / "bad_temp.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="body_temperature"):
        DataIngestion(path).load_validated()


def test_validate_rejects_unknown_arrival_mode(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    bad = sample_valid_df.copy()
    bad.loc[0, "arrival_mode"] = "helicopter"
    path = tmp_path / "bad_mode.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="arrival_mode"):
        DataIngestion(path).load_validated()


def test_load_validated_is_idempotent(tmp_csv: Path) -> None:
    ing = DataIngestion(tmp_csv)
    a = ing.load_validated()
    b = ing.load_validated()
    pd.testing.assert_frame_equal(a, b)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DataIngestion(tmp_path / "nope.csv").load()


def test_validate_rejects_unknown_triage_level(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    bad = sample_valid_df.copy()
    bad.loc[0, "triage_level"] = 7
    path = tmp_path / "bad_level.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="triage_level"):
        DataIngestion(path).load_validated()
