"""Spec 06 — Approach A: independent ML vs fuzzy comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from triagem_fuzzy.fuzzy.engine import FuzzyInferenceEngine
from triagem_fuzzy.ml.base import BaseTriageClassifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonReport:
    predictions: pd.DataFrame
    agreement_rate: float
    confusion_ml_vs_fuzzy: pd.DataFrame
    per_class_agreement: pd.Series
    disagreement_samples: pd.DataFrame
    ml_accuracy: float
    fuzzy_accuracy: float


class TriageComparator:
    def __init__(
        self,
        model: BaseTriageClassifier,
        fuzzy_engine: FuzzyInferenceEngine,
        label_encoder: dict[int, str],
    ) -> None:
        self.model = model
        self.fuzzy_engine = fuzzy_engine
        self.label_encoder = label_encoder
        self.id_to_name = dict(label_encoder)
        self.name_to_id = {v: k for k, v in label_encoder.items()}

    def run(self, X: pd.DataFrame, y: pd.Series) -> ComparisonReport:
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must match")

        ml_pred_ids = self.model.predict(X)
        ml_pred_names = np.array([self.id_to_name[i] for i in ml_pred_ids])

        required = self.fuzzy_engine.required_inputs()
        missing = required - set(X.columns)
        if missing:
            raise ValueError(
                f"Fuzzy engine needs columns {sorted(missing)} not present in X"
            )

        fuzzy_pred_names = np.empty(len(X), dtype=object)
        for i, (_, row) in enumerate(X.iterrows()):
            sample = {name: float(row[name]) for name in required}
            fuzzy_pred_names[i] = self.fuzzy_engine.classify(sample)

        y_names = np.array([self.id_to_name[int(v)] for v in y.values])
        agree = ml_pred_names == fuzzy_pred_names

        preds = pd.DataFrame(
            {
                "y_true": y_names,
                "ml_pred": ml_pred_names,
                "fuzzy_pred": fuzzy_pred_names,
                "agree": agree,
            },
            index=X.index,
        )

        ordered_names = list(self.label_encoder.values())
        cm = confusion_matrix(
            ml_pred_names, fuzzy_pred_names, labels=ordered_names
        )
        cm_df = pd.DataFrame(cm, index=ordered_names, columns=ordered_names)
        cm_df.index.name = "ml"
        cm_df.columns.name = "fuzzy"

        per_class = (
            preds.groupby("y_true")["agree"].mean().reindex(ordered_names)
        )

        disagreements = preds.loc[~preds["agree"]].head(20)

        ml_acc = float(accuracy_score(y_names, ml_pred_names))
        fz_acc = float(accuracy_score(y_names, fuzzy_pred_names))
        logger.info(
            "Comparator: agreement=%.3f ml_acc=%.3f fuzzy_acc=%.3f",
            agree.mean(),
            ml_acc,
            fz_acc,
        )

        return ComparisonReport(
            predictions=preds,
            agreement_rate=float(agree.mean()),
            confusion_ml_vs_fuzzy=cm_df,
            per_class_agreement=per_class,
            disagreement_samples=disagreements,
            ml_accuracy=ml_acc,
            fuzzy_accuracy=fz_acc,
        )
