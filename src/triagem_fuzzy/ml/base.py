"""Spec 04 — Abstract base class for triage classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseTriageClassifier(ABC):
    feature_names_: list[str]
    classes_: np.ndarray

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "BaseTriageClassifier": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTriageClassifier": ...
