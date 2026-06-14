"""Spec 04 — Random Forest classifier for triage."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from triagem_fuzzy import config
from triagem_fuzzy.ml.base import BaseTriageClassifier

logger = logging.getLogger(__name__)


class TriageRandomForest(BaseTriageClassifier):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        class_weight: str | dict | None = "balanced",
        random_state: int = config.RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        self._estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray = np.array([], dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TriageRandomForest":
        self.feature_names_ = X.columns.tolist()
        self._estimator.fit(X.values, y.values)
        self.classes_ = self._estimator.classes_
        logger.info(
            "RF trained on %d samples, %d features, classes=%s",
            len(X),
            len(self.feature_names_),
            self.classes_.tolist(),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict(X[self.feature_names_].values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict_proba(X[self.feature_names_].values)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._estimator.feature_importances_

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self._estimator,
                "feature_names": self.feature_names_,
                "classes": self.classes_,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TriageRandomForest":
        payload = joblib.load(Path(path))
        instance = cls()
        instance._estimator = payload["estimator"]
        instance.feature_names_ = payload["feature_names"]
        instance.classes_ = payload["classes"]
        return instance
