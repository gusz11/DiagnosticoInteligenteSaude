"""Spec 06 — Approach B: ML probability + vitals → integrated fuzzy decision."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from triagem_fuzzy import config
from triagem_fuzzy.fuzzy.engine import FuzzyInferenceEngine
from triagem_fuzzy.ml.base import BaseTriageClassifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntegrationReport:
    predictions: pd.DataFrame
    accuracy_ml: float
    accuracy_integrated: float
    macro_f1_ml: float
    macro_f1_integrated: float
    confusion_integrated: pd.DataFrame
    delta_per_class: pd.DataFrame
    biggest_changes: pd.DataFrame


class TriageIntegrator:
    def __init__(
        self,
        model: BaseTriageClassifier,
        fuzzy_engine: FuzzyInferenceEngine,
        label_encoder: dict[int, str],
        ml_proba_class: int = 2,
    ) -> None:
        if "ml_risk_proba" not in fuzzy_engine.required_inputs():
            raise ValueError(
                "Integrator requires a fuzzy engine with 'ml_risk_proba' input"
            )
        self.model = model
        self.fuzzy_engine = fuzzy_engine
        self.label_encoder = label_encoder
        self.id_to_name = dict(label_encoder)
        self.ml_proba_class = ml_proba_class

    def build_fuzzy_input(
        self, row: pd.Series, proba: np.ndarray
    ) -> dict[str, float]:
        aggregated = self._aggregate_proba(proba)
        return {
            "body_temperature": float(row["body_temperature"]),
            "heart_rate": float(row["heart_rate"]),
            "systolic_blood_pressure": float(row["systolic_blood_pressure"]),
            "ml_risk_proba": float(aggregated),
        }

    def _aggregate_proba(self, proba: np.ndarray) -> float:
        if config.ML_PROBA_AGGREGATION == "atencao_or_risco":
            return float(proba[1] + proba[2])
        return float(proba[self.ml_proba_class])

    def run(self, X: pd.DataFrame, y: pd.Series) -> IntegrationReport:
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must match")

        proba = self.model.predict_proba(X)
        ml_pred_ids = self.model.predict(X)

        integrated_scores = np.empty(len(X), dtype=float)
        integrated_classes = np.empty(len(X), dtype=object)
        for i, (_, row) in enumerate(X.iterrows()):
            sample = self.build_fuzzy_input(row, proba[i])
            trace = self.fuzzy_engine.explain(sample)
            integrated_scores[i] = trace.crisp_output
            integrated_classes[i] = trace.classification

        ordered_names = list(self.label_encoder.values())
        name_to_id = {v: k for k, v in self.label_encoder.items()}
        integrated_ids = np.array([name_to_id[n] for n in integrated_classes])

        preds = pd.DataFrame(
            {
                "y_true": y.values,
                "ml_pred": ml_pred_ids,
                "ml_p_risco": proba[:, self.ml_proba_class],
                "integrated_score": integrated_scores,
                "integrated_pred": integrated_ids,
            },
            index=X.index,
        )

        acc_ml = float(accuracy_score(y, ml_pred_ids))
        acc_int = float(accuracy_score(y, integrated_ids))
        f1_ml = float(f1_score(y, ml_pred_ids, average="macro"))
        f1_int = float(
            f1_score(y, integrated_ids, average="macro", zero_division=0.0)
        )

        cls_ids = sorted(self.label_encoder.keys())
        cm = confusion_matrix(y, integrated_ids, labels=cls_ids)
        cm_df = pd.DataFrame(
            cm,
            index=[self.id_to_name[i] for i in cls_ids],
            columns=[self.id_to_name[i] for i in cls_ids],
        )
        cm_df.index.name = "true"
        cm_df.columns.name = "pred"

        _, recall_ml, f1_ml_pc, _ = precision_recall_fscore_support(
            y, ml_pred_ids, labels=cls_ids, zero_division=0.0
        )
        _, recall_int, f1_int_pc, _ = precision_recall_fscore_support(
            y, integrated_ids, labels=cls_ids, zero_division=0.0
        )
        delta = pd.DataFrame(
            {
                "recall_ml": recall_ml,
                "recall_integrated": recall_int,
                "delta_recall": recall_int - recall_ml,
                "f1_ml": f1_ml_pc,
                "f1_integrated": f1_int_pc,
                "delta_f1": f1_int_pc - f1_ml_pc,
            },
            index=[self.id_to_name[i] for i in cls_ids],
        )

        changes = preds.loc[preds["ml_pred"] != preds["integrated_pred"]].head(20)

        logger.info(
            "Integrator: acc_ml=%.3f acc_int=%.3f f1_ml=%.3f f1_int=%.3f",
            acc_ml,
            acc_int,
            f1_ml,
            f1_int,
        )

        return IntegrationReport(
            predictions=preds,
            accuracy_ml=acc_ml,
            accuracy_integrated=acc_int,
            macro_f1_ml=f1_ml,
            macro_f1_integrated=f1_int,
            confusion_integrated=cm_df,
            delta_per_class=delta,
            biggest_changes=changes,
        )
