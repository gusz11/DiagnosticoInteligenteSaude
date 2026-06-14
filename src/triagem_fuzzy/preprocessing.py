"""Spec 02 — Preprocessing: label remap, encoding, stratified split."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from triagem_fuzzy import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    label_encoder: dict[int, str]


class DataPreprocessor:
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = config.RANDOM_STATE,
        stratify: bool = True,
    ) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def remap_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["triage_class"] = out["triage_level"].map(config.TRIAGE_LABEL_MAP)
        if out["triage_class"].isna().any():
            unknown = sorted(
                df.loc[out["triage_class"].isna(), "triage_level"].unique()
            )
            raise ValueError(f"Unmapped triage_level values: {unknown}")
        out["triage_class"] = out["triage_class"].astype("int64")
        return out

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(columns=["triage_level"]).copy()
        # Enforce all known categories so one-hot has a stable column set.
        out["arrival_mode"] = pd.Categorical(
            out["arrival_mode"], categories=list(config.ARRIVAL_MODES)
        )
        out = pd.get_dummies(out, columns=["arrival_mode"], dtype="int64")
        return out

    def split(self, df: pd.DataFrame) -> PreparedData:
        y = df["triage_class"]
        X = df.drop(columns=["triage_class"])

        feature_names = self._stable_feature_order(X.columns.tolist())
        X = X[feature_names]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )
        logger.info(
            "Split: train=%d test=%d features=%d",
            len(X_train),
            len(X_test),
            len(feature_names),
        )
        return PreparedData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            label_encoder=dict(config.TRIAGE_LABELS),
        )

    def run(self, df: pd.DataFrame) -> PreparedData:
        remapped = self.remap_labels(df)
        encoded = self.encode_features(remapped)
        return self.split(encoded)

    @staticmethod
    def _stable_feature_order(columns: list[str]) -> list[str]:
        # Numerics first (declared order in config), then one-hots sorted.
        numerics = [c for c in config.NUMERIC_FEATURES if c in columns]
        one_hots = sorted(c for c in columns if c.startswith("arrival_mode_"))
        return numerics + one_hots
