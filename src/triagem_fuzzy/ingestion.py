
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from triagem_fuzzy import config

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when the CSV violates the authoritative schema."""


class DataIngestion:
    def __init__(self, csv_path: Path | None = None) -> None:
        self.csv_path: Path = Path(csv_path) if csv_path else config.RAW_DATA_PATH

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        return pd.read_csv(self.csv_path)

    def validate(self, df: pd.DataFrame) -> None:
        self._check_columns(df)
        self._check_dtypes(df)
        self._check_no_missing(df)
        self._check_ranges(df)
        self._check_categoricals(df)
        self._check_triage_levels(df)
        logger.info(
            "Schema OK: %d rows, %d cols", len(df), len(df.columns)
        )

    def load_validated(self) -> pd.DataFrame:
        df = self.load()
        self.validate(df)
        return df

    # ── private checks ────────────────────────────────────────────────

    @staticmethod
    def _check_columns(df: pd.DataFrame) -> None:
        expected = set(config.EXPECTED_COLUMNS)
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise SchemaError(f"Missing required columns: {sorted(missing)}")
        if extra:
            raise SchemaError(f"Unknown columns present: {sorted(extra)}")

    @staticmethod
    def _check_dtypes(df: pd.DataFrame) -> None:
        for col in config.NUMERIC_FEATURES:
            if not is_numeric_dtype(df[col]):
                raise SchemaError(
                    f"Column '{col}' must be numeric, got {df[col].dtype}"
                )
        for int_col in ("pain_level", "chronic_disease_count", "previous_er_visits"):
            if not is_integer_dtype(df[int_col]):
                raise SchemaError(
                    f"Column '{int_col}' must be integer, got {df[int_col].dtype}"
                )
        if not is_integer_dtype(df["triage_level"]):
            raise SchemaError(
                f"Column 'triage_level' must be integer, got {df['triage_level'].dtype}"
            )

    @staticmethod
    def _check_no_missing(df: pd.DataFrame) -> None:
        nulls = df.isna().sum()
        offenders = nulls[nulls > 0]
        if not offenders.empty:
            raise SchemaError(
                f"Missing values found: {offenders.to_dict()}"
            )

    @staticmethod
    def _check_ranges(df: pd.DataFrame) -> None:
        for col, (lo, hi) in config.FEATURE_RANGES.items():
            series = df[col]
            if (series < lo).any() or (series > hi).any():
                bad = int(((series < lo) | (series > hi)).sum())
                raise SchemaError(
                    f"Column '{col}' has {bad} value(s) outside [{lo}, {hi}]"
                )

    @staticmethod
    def _check_categoricals(df: pd.DataFrame) -> None:
        allowed = set(config.ARRIVAL_MODES)
        seen = set(df["arrival_mode"].unique())
        unknown = seen - allowed
        if unknown:
            raise SchemaError(
                f"Unknown arrival_mode values: {sorted(unknown)}"
            )

    @staticmethod
    def _check_triage_levels(df: pd.DataFrame) -> None:
        allowed = set(config.RAW_TRIAGE_LEVELS)
        seen = set(df["triage_level"].unique().tolist())
        unknown = seen - allowed
        if unknown:
            raise SchemaError(
                f"Unknown triage_level values: {sorted(unknown)}"
            )
