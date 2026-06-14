"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from triagem_fuzzy import config


@pytest.fixture(scope="session")
def real_dataset_path() -> Path:
    if not config.RAW_DATA_PATH.exists():
        pytest.skip(f"Dataset not present at {config.RAW_DATA_PATH}")
    return config.RAW_DATA_PATH


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30.0, 75.5, 5.0],
            "heart_rate": [78.0, 110.0, 130.0],
            "systolic_blood_pressure": [120.0, 95.0, 105.0],
            "oxygen_saturation": [98.0, 91.0, 96.0],
            "body_temperature": [36.7, 39.2, 37.1],
            "pain_level": [1, 8, 3],
            "chronic_disease_count": [0, 3, 0],
            "previous_er_visits": [0, 5, 1],
            "arrival_mode": ["walk_in", "ambulance", "walk_in"],
            "triage_level": [0, 3, 1],
        }
    )


@pytest.fixture
def tmp_csv(tmp_path: Path, sample_valid_df: pd.DataFrame) -> Path:
    path = tmp_path / "valid.csv"
    sample_valid_df.to_csv(path, index=False)
    return path
