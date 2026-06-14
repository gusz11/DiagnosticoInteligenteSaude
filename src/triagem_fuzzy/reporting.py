"""Spec 07 — Result reporter (markdown + json + plots)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from triagem_fuzzy import config
from triagem_fuzzy.articulation.comparator import ComparisonReport
from triagem_fuzzy.articulation.integrator import IntegrationReport
from triagem_fuzzy.eda import EdaReport
from triagem_fuzzy.ml.evaluator import EvaluationReport

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = (
    "apresentacao",
    "objetivos",
    "parte1_ml",
    "parte2_fuzzy",
    "parte3_articulacao",
    "analise_critica",
)


class ResultReporter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir) / "report"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_summary_json(self, payload: dict) -> Path:
        path = self.output_dir / "summary.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def write_markdown(self, sections: dict[str, str]) -> Path:
        missing = set(REQUIRED_SECTIONS) - set(sections.keys())
        if missing:
            raise ValueError(f"Missing report sections: {sorted(missing)}")
        lines: list[str] = []
        titles = {
            "apresentacao": "Apresentação",
            "objetivos": "Objetivos",
            "parte1_ml": "Parte 1 — Machine Learning",
            "parte2_fuzzy": "Parte 2 — Sistema Fuzzy",
            "parte3_articulacao": "Parte 3 — Articulação",
            "analise_critica": "Análise Crítica",
        }
        lines.append("# Relatório — Triagem Inteligente (ML + Fuzzy)\n")
        for key in REQUIRED_SECTIONS:
            lines.append(f"## {titles[key]}\n")
            lines.append(sections[key].strip() + "\n")
        path = self.output_dir / "report.md"
        path.write_text("\n".join(lines))
        return path

    def render_confusions(
        self,
        eval_report: EvaluationReport,
        cmp_report: ComparisonReport,
        int_report: IntegrationReport,
    ) -> list[Path]:
        paths = [
            self._plot_matrix(
                eval_report.confusion_matrix,
                "Matriz de confusão — ML",
                self.output_dir / "confusion_ml.png",
                xlabel="predito",
                ylabel="verdadeiro",
            ),
            self._plot_matrix(
                cmp_report.confusion_ml_vs_fuzzy,
                "ML vs Fuzzy (Aproximação A)",
                self.output_dir / "confusion_ml_vs_fuzzy.png",
                xlabel="fuzzy",
                ylabel="ml",
            ),
            self._plot_matrix(
                int_report.confusion_integrated,
                "Matriz de confusão — Integrado",
                self.output_dir / "confusion_integrated.png",
                xlabel="predito",
                ylabel="verdadeiro",
            ),
        ]
        return paths

    def render_feature_importances(self, eval_report: EvaluationReport) -> Path:
        path = self.output_dir / "feature_importances.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        eval_report.feature_importances.plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("importância")
        ax.set_title("Importância das features (Random Forest)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def render_risk_distribution(
        self, int_report: IntegrationReport
    ) -> Path:
        path = self.output_dir / "risk_score_distribution.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(int_report.predictions["integrated_score"], bins=40, edgecolor="black")
        ax.set_xlabel("risk_score (fuzzy integrado)")
        ax.set_ylabel("frequência")
        ax.set_title("Distribuição do risk_score integrado")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def assemble(
        self,
        eda: EdaReport,
        ml: EvaluationReport,
        comparison: ComparisonReport,
        integration: IntegrationReport,
    ) -> Path:
        self.render_confusions(ml, comparison, integration)
        self.render_feature_importances(ml)
        self.render_risk_distribution(integration)

        summary = self._build_summary(eda, ml, comparison, integration)
        self.write_summary_json(summary)

        sections = self._build_sections(eda, ml, comparison, integration)
        return self.write_markdown(sections)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _plot_matrix(
        cm: pd.DataFrame,
        title: str,
        path: Path,
        *,
        xlabel: str,
        ylabel: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(5, 4))
        values = cm.values
        im = ax.imshow(values, cmap="Blues")
        labels = list(cm.columns)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        thresh = values.max() / 2.0 if values.max() else 0.5
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j,
                    i,
                    str(int(values[i, j])),
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > thresh else "black",
                )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    @staticmethod
    def _round_dict(d: dict, ndigits: int = 4) -> dict:
        return {k: round(float(v), ndigits) for k, v in d.items()}

    def _build_summary(
        self,
        eda: EdaReport,
        ml: EvaluationReport,
        comparison: ComparisonReport,
        integration: IntegrationReport,
    ) -> dict:
        return {
            "dataset": {
                "rows": int(eda.summary.iloc[0]["mean"] is not None) and int(
                    eda.class_balance_remapped.sum()
                ),
                "class_balance_remapped": {
                    str(k): int(v) for k, v in eda.class_balance_remapped.items()
                },
            },
            "ml": {
                "accuracy": round(ml.accuracy, 4),
                "macro_f1": round(ml.macro_f1, 4),
                "weighted_f1": round(ml.weighted_f1, 4),
                "per_class": ml.per_class.round(4).to_dict(orient="index"),
                "feature_importances": ml.feature_importances.round(4).to_dict(),
            },
            "comparison": {
                "agreement_rate": round(comparison.agreement_rate, 4),
                "ml_accuracy": round(comparison.ml_accuracy, 4),
                "fuzzy_accuracy": round(comparison.fuzzy_accuracy, 4),
                "per_class_agreement": comparison.per_class_agreement.round(
                    4
                ).to_dict(),
            },
            "integration": {
                "accuracy_ml": round(integration.accuracy_ml, 4),
                "accuracy_integrated": round(integration.accuracy_integrated, 4),
                "macro_f1_ml": round(integration.macro_f1_ml, 4),
                "macro_f1_integrated": round(integration.macro_f1_integrated, 4),
                "delta_per_class": integration.delta_per_class.round(4).to_dict(
                    orient="index"
                ),
            },
        }

    def _build_sections(
        self,
        eda: EdaReport,
        ml: EvaluationReport,
        comparison: ComparisonReport,
        integration: IntegrationReport,
    ) -> dict[str, str]:
        balance = eda.class_balance_remapped
        balance_str = ", ".join(f"{k}={int(v)}" for k, v in balance.items())

        return {
            "apresentacao": (
                "Sistema acadêmico de apoio à triagem baseado em sinais vitais "
                "(temperatura, frequência cardíaca, pressão sistólica, SpO2, "
                "dor, doenças crônicas, visitas prévias, idade, modo de chegada). "
                "**Uso estritamente acadêmico — não constitui diagnóstico médico.**\n\n"
                f"Base: 18.000 pacientes. Classes remapeadas: {balance_str}."
            ),
            "objetivos": (
                "- Treinar um classificador de Machine Learning (Random Forest, "
                "decisão D2) para predizer a classe de triagem.\n"
                "- Construir um sistema fuzzy de Mamdani com 3 entradas vitais "
                "(temperatura, batimentos, pressão), 1 saída (risk_score) e ≥ 6 "
                "regras SE…ENTÃO.\n"
                "- Articular ambos via Aproximação A (comparação) e B (integração)."
            ),
            "parte1_ml": (
                f"- **Algoritmo**: Random Forest (300 árvores, `class_weight='balanced'`).\n"
                f"- **Acurácia**: {ml.accuracy:.3f}\n"
                f"- **Macro-F1**: {ml.macro_f1:.3f}\n"
                f"- **Weighted-F1**: {ml.weighted_f1:.3f}\n\n"
                f"Por classe:\n\n"
                f"```\n{ml.per_class.round(3).to_string()}\n```\n\n"
                f"Top 5 features:\n\n"
                f"```\n{ml.feature_importances.head(5).round(3).to_string()}\n```"
            ),
            "parte2_fuzzy": (
                "Sistema Mamdani:\n"
                "- Entradas: `body_temperature` ∈ [34, 42] °C, `heart_rate` ∈ "
                "[30, 200] bpm, `systolic_blood_pressure` ∈ [60, 220] mmHg.\n"
                "- Termos linguísticos por variável: `baixa/normal/alta`.\n"
                "- Saída: `risk_score` ∈ [0, 10] com termos `baixo/medio/alto`.\n"
                "- 9 regras (Spec 05 §Rule Base v1); operador AND com agregação "
                "max e defuzzificação por centroide.\n"
                "- Decodificador de score → classe via bandas "
                f"{config.FUZZY_DECISION_BANDS}."
            ),
            "parte3_articulacao": (
                f"**Aproximação A — Comparação**\n\n"
                f"- Concordância ML vs Fuzzy: {comparison.agreement_rate:.3f}\n"
                f"- Acurácia ML (independente): {comparison.ml_accuracy:.3f}\n"
                f"- Acurácia Fuzzy (independente): {comparison.fuzzy_accuracy:.3f}\n"
                f"- Concordância por classe verdadeira:\n\n"
                f"```\n{comparison.per_class_agreement.round(3).to_string()}\n```\n\n"
                f"**Aproximação B — Integração**\n\n"
                f"O modelo ML calcula P(risco) e essa probabilidade é injetada no "
                f"sistema fuzzy junto com vitais para gerar uma decisão integrada.\n\n"
                f"- Acurácia ML (referência): {integration.accuracy_ml:.3f}\n"
                f"- Acurácia integrado: {integration.accuracy_integrated:.3f}\n"
                f"- Macro-F1 ML: {integration.macro_f1_ml:.3f} → "
                f"Macro-F1 integrado: {integration.macro_f1_integrated:.3f}\n\n"
                f"Δ por classe:\n\n"
                f"```\n{integration.delta_per_class.round(3).to_string()}\n```"
            ),
            "analise_critica": (
                "- O Random Forest é mais acurado isoladamente, mas o sistema fuzzy "
                "oferece explicabilidade direta (regras SE…ENTÃO) que dialoga com "
                "o raciocínio clínico.\n"
                "- A integração (Aproximação B) tende a ser mais conservadora: "
                "quando P(risco) do ML é alta, regras fuzzy elevam o score mesmo "
                "com vitais marginais, o que pode aumentar recall em `risco` à "
                "custa de precisão.\n"
                "- Limitações: parâmetros das funções de pertinência foram "
                "definidos por convenção clínica genérica e não tunados por "
                "validação cruzada; tunar via grid-search é trabalho futuro.\n"
                "- Não é diagnóstico médico real — uso acadêmico apenas."
            ),
        }
