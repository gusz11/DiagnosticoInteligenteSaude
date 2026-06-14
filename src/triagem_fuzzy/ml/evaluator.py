"""Spec 04 — Evaluation metrics and confusion plot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from triagem_fuzzy.ml.base import BaseTriageClassifier


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_importances: pd.Series


class ModelEvaluator:
    def __init__(self, label_encoder: dict[int, str]) -> None:
        self.label_encoder = label_encoder
        self.class_ids = sorted(label_encoder.keys())
        self.class_names = [label_encoder[i] for i in self.class_ids]

    def evaluate(
        self,
        model: BaseTriageClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> EvaluationReport:
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro = float(f1_score(y_test, y_pred, average="macro"))
        weighted = float(f1_score(y_test, y_pred, average="weighted"))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=self.class_ids,
            zero_division=0.0,
        )
        per_class = pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            },
            index=self.class_names,
        )

        cm = confusion_matrix(y_test, y_pred, labels=self.class_ids)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_df.index.name = "true"
        cm_df.columns.name = "pred"

        importances = pd.Series(
            model.feature_importances_,
            index=model.feature_names_,
            name="importance",
        ).sort_values(ascending=False)

        return EvaluationReport(
            accuracy=acc,
            macro_f1=macro,
            weighted_f1=weighted,
            per_class=per_class,
            confusion_matrix=cm_df,
            feature_importances=importances,
        )

    def plot_confusion(self, report: EvaluationReport, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cm = report.confusion_matrix.values
        labels = list(report.confusion_matrix.columns)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("predito")
        ax.set_ylabel("verdadeiro")
        ax.set_title("Matriz de confusão")
        thresh = cm.max() / 2.0 if cm.max() else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
