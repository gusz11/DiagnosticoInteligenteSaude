"""Spec 03 — Exploratory Data Analysis (read-only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from triagem_fuzzy import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdaReport:
    summary: pd.DataFrame
    class_balance_raw: pd.Series
    class_balance_remapped: pd.Series
    correlations: pd.DataFrame
    artifacts_dir: Path


class ExploratoryAnalysis:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir) / "eda"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── computations ──────────────────────────────────────────────────

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(config.NUMERIC_FEATURES)
        return df[cols].describe().T[
            ["mean", "std", "min", "25%", "50%", "75%", "max"]
        ]

    def class_balance(self, df: pd.DataFrame, *, remap: bool) -> pd.Series:
        if remap:
            mapped = df["triage_level"].map(config.TRIAGE_LABEL_MAP)
            named = mapped.map(config.TRIAGE_LABELS)
            return named.value_counts().reindex(
                list(config.TRIAGE_LABELS.values())
            )
        return df["triage_level"].value_counts().sort_index()

    def correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(config.NUMERIC_FEATURES)
        return df[cols].corr(method="pearson")

    # ── plots ─────────────────────────────────────────────────────────

    def plot_distributions(self, df: pd.DataFrame) -> list[Path]:
        paths: list[Path] = []
        for col in config.NUMERIC_FEATURES:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].plot.hist(bins=40, ax=ax, alpha=0.85, edgecolor="black")
            ax.set_title(f"Distribuição de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequência")
            path = self.output_dir / f"dist_{col}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)
        return paths

    def plot_boxplots_by_class(self, df: pd.DataFrame) -> list[Path]:
        remapped = df["triage_level"].map(config.TRIAGE_LABEL_MAP).map(
            config.TRIAGE_LABELS
        )
        paths: list[Path] = []
        for col in config.NUMERIC_FEATURES:
            fig, ax = plt.subplots(figsize=(6, 4))
            data = [
                df.loc[remapped == cls, col].values
                for cls in config.TRIAGE_LABELS.values()
            ]
            ax.boxplot(data, tick_labels=list(config.TRIAGE_LABELS.values()))
            ax.set_title(f"{col} por classe de triagem")
            ax.set_xlabel("classe")
            ax.set_ylabel(col)
            path = self.output_dir / f"box_{col}_by_class.png"
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)
        return paths

    # ── entry point ───────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> EdaReport:
        summary = self.describe(df)
        balance_raw = self.class_balance(df, remap=False)
        balance_remap = self.class_balance(df, remap=True)
        corrs = self.correlations(df)

        summary.to_csv(self.output_dir / "summary_statistics.csv")
        balance_raw.to_csv(self.output_dir / "class_balance_raw.csv")
        balance_remap.to_csv(self.output_dir / "class_balance_remapped.csv")
        corrs.to_csv(self.output_dir / "correlation_matrix.csv")

        self.plot_distributions(df)
        self.plot_boxplots_by_class(df)

        logger.info("EDA artifacts written to %s", self.output_dir)
        return EdaReport(
            summary=summary,
            class_balance_raw=balance_raw,
            class_balance_remapped=balance_remap,
            correlations=corrs,
            artifacts_dir=self.output_dir,
        )
