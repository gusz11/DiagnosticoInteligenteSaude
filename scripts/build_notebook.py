"""Build NotebookFuzzy.ipynb from a list of (markdown, code) steps."""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS: list[dict] = []

CELLS.append(md(
    """# Triagem Inteligente — ML + Fuzzy

Notebook **passo a passo** do trabalho. Cada etapa carrega um módulo
do pacote `triagem_fuzzy` (em `src/`) e demonstra o seu comportamento.

> ⚠️ Trabalho **estritamente acadêmico** — não constitui diagnóstico
> médico real.

**Decisões fixadas**

| ID | Decisão | Valor |
|----|---------|-------|
| D1 | Remapeamento das classes | `0→normal`, `1→atenção`, `2,3→risco` |
| D2 | Algoritmo ML | Random Forest |
| D3 | Articulação | Aproximação A (Comparação) **e** B (Integração) |
| D4 | Base | `dataset/triagem_fuzzy.csv` (18.000 linhas) |
| D5 | Seed | 50 |
"""
))

CELLS.append(md("## 0. Configuração — adicionar `src/` ao path"))
CELLS.append(code(
    """import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
"""
))

CELLS.append(md("## 1. Imports"))
CELLS.append(code(
    """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from triagem_fuzzy import config
from triagem_fuzzy.ingestion import DataIngestion
from triagem_fuzzy.preprocessing import DataPreprocessor
from triagem_fuzzy.eda import ExploratoryAnalysis
from triagem_fuzzy.ml.random_forest import TriageRandomForest
from triagem_fuzzy.ml.evaluator import ModelEvaluator
from triagem_fuzzy.fuzzy.factory import build_standalone_engine, build_integrated_engine
from triagem_fuzzy.articulation.comparator import TriageComparator
from triagem_fuzzy.articulation.integrator import TriageIntegrator
"""
))

CELLS.append(md(
    """## 2. Spec 01 — Ingestão e validação

Carrega `dataset/triagem_fuzzy.csv`, valida esquema (colunas, tipos,
intervalos, categorias) e retorna o `DataFrame` bruto.
"""
))
CELLS.append(code(
    """df = DataIngestion().load_validated()

assert df.shape == (18000, 10)
assert df.isna().sum().sum() == 0
df.head()
"""
))

CELLS.append(md("### 2.1 Distribuição original do alvo (4 níveis)"))
CELLS.append(code(
    """df["triage_level"].value_counts().sort_index()
"""
))

CELLS.append(md(
    """## 3. Spec 02 — Pré-processamento

Aplica o remapeamento D1, one-hot de `arrival_mode`, e split
estratificado 80/20 com `random_state=50`.
"""
))
CELLS.append(code(
    """preprocessor = DataPreprocessor()
data = preprocessor.run(df)

assert data.X_train.shape == (14400, 11)
assert data.X_test.shape == (3600, 11)
assert set(data.y_train.unique()) == {0, 1, 2}

print(f"Treino : {len(data.X_train):>5}  ({data.X_train.shape[1]} features)")
print(f"Teste  : {len(data.X_test):>5}")
print(f"Classes: {data.label_encoder}")
"""
))

CELLS.append(md("### 3.1 Balanço de classes após o remapeamento"))
CELLS.append(code(
    """counts = data.y_train.value_counts().sort_index()
counts.index = [config.TRIAGE_LABELS[i] for i in counts.index]
counts
"""
))

CELLS.append(md(
    """## 4. Spec 03 — Análise exploratória (EDA)

Resumo descritivo, balanço de classes e correlação entre features
numéricas.
"""
))
CELLS.append(code(
    """eda = ExploratoryAnalysis(config.OUTPUT_DIR)
summary = eda.describe(df)
summary.round(2)
"""
))

CELLS.append(md("### 4.1 Matriz de correlação (Pearson)"))
CELLS.append(code(
    """corr = eda.correlations(df)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
fig.colorbar(im, ax=ax, label="r")
ax.set_title("Correlação entre features numéricas")
fig.tight_layout()
plt.show()
"""
))

CELLS.append(md(
    """## 5. Spec 04 — Treinamento do Random Forest

300 árvores · `class_weight='balanced'` · `min_samples_leaf=2`.
"""
))
CELLS.append(code(
    """model = TriageRandomForest()
model.fit(data.X_train, data.y_train)

assert model.predict(data.X_test).shape == (len(data.X_test),)
proba = model.predict_proba(data.X_test)
assert np.allclose(proba.sum(axis=1), 1.0)

print(f"Modelo treinado. Classes: {model.classes_.tolist()}")
"""
))

CELLS.append(md(
    """## 6. Spec 04 — Avaliação (métricas pedidas no enunciado)

Acurácia · Matriz de confusão · Precisão · Recall · F1-score.
"""
))
CELLS.append(code(
    """evaluator = ModelEvaluator(data.label_encoder)
report = evaluator.evaluate(model, data.X_test, data.y_test)

print(f"Acurácia    : {report.accuracy:.4f}")
print(f"Macro F1    : {report.macro_f1:.4f}")
print(f"Weighted F1 : {report.weighted_f1:.4f}")
"""
))

CELLS.append(md("### 6.1 Precisão, recall e F1 por classe"))
CELLS.append(code(
    """report.per_class.round(4)
"""
))

CELLS.append(md("### 6.2 Matriz de confusão"))
CELLS.append(code(
    """cm = report.confusion_matrix
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm.values, cmap="Blues")
ax.set_xticks(range(len(cm.columns)))
ax.set_yticks(range(len(cm.index)))
ax.set_xticklabels(cm.columns)
ax.set_yticklabels(cm.index)
ax.set_xlabel("predito")
ax.set_ylabel("verdadeiro")
thresh = cm.values.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, int(cm.values[i, j]), ha="center", va="center",
                color="white" if cm.values[i, j] > thresh else "black")
fig.colorbar(im, ax=ax)
ax.set_title("Matriz de confusão — Random Forest")
fig.tight_layout()
plt.show()
cm
"""
))

CELLS.append(md("### 6.3 Importância das features (top 5)"))
CELLS.append(code(
    """report.feature_importances.head(5).round(4)
"""
))

CELLS.append(md(
    """## 7. Spec 05 — Sistema fuzzy de Mamdani

3 entradas vitais (`temperatura`, `frequência cardíaca`,
`pressão sistólica`) → 1 saída (`risk_score` ∈ [0, 10]). 9 regras,
agregação `max`, defuzzificação por centroide.
"""
))
CELLS.append(code(
    """engine = build_standalone_engine()

print(f"Entradas : {sorted(engine.required_inputs())}")
print(f"Saída    : {engine.output.name}  universo={engine.output.universe}")
print(f"Regras   : {len(engine.rules)}")
"""
))

CELLS.append(md("### 7.1 Funções de pertinência das entradas"))
CELLS.append(code(
    """fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, var) in zip(axes, engine.inputs.items()):
    xs = var.universe_array()
    for term_name, mf in var.terms.items():
        ax.plot(xs, mf.evaluate(xs), label=term_name)
    ax.set_title(name)
    ax.set_xlabel(name)
    ax.set_ylabel("pertinência")
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
fig.tight_layout()
plt.show()
"""
))

CELLS.append(md("### 7.2 Função de pertinência da saída `risk_score`"))
CELLS.append(code(
    """xs = engine.output.universe_array()
fig, ax = plt.subplots(figsize=(7, 3.5))
for term_name, mf in engine.output.terms.items():
    ax.plot(xs, mf.evaluate(xs), label=term_name)
for label, (lo, hi) in config.FUZZY_DECISION_BANDS.items():
    ax.axvspan(lo, hi, alpha=0.05)
ax.set_xlabel("risk_score")
ax.set_ylabel("pertinência")
ax.set_title("Saída fuzzy e bandas de decisão")
ax.legend()
fig.tight_layout()
plt.show()
"""
))

CELLS.append(md("### 7.3 Base de regras (texto)"))
CELLS.append(code(
    """rules_df = pd.DataFrame([
    {
        "regra": r.name,
        "antecedente": " E ".join(f"{v}={t}" for v, t in r.antecedents),
        "consequente": f"{r.consequent[0]}={r.consequent[1]}",
    }
    for r in engine.rules
])
rules_df
"""
))

CELLS.append(md(
    """### 7.4 Demonstração — paciente "normal típico"

Temperatura normal, FC e PA dentro da faixa esperada → score baixo.
"""
))
CELLS.append(code(
    """sample_normal = {"body_temperature": 36.8, "heart_rate": 75.0, "systolic_blood_pressure": 120.0}
trace_normal = engine.explain(sample_normal)

assert trace_normal.crisp_output < 3.5
assert trace_normal.classification == "normal"

print(f"risk_score     : {trace_normal.crisp_output:.3f}")
print(f"classe sugerida: {trace_normal.classification}")
"""
))

CELLS.append(md(
    """### 7.5 Demonstração — paciente "risco típico"

Febre alta, taquicardia e hipotensão → score alto.
"""
))
CELLS.append(code(
    """sample_risco = {"body_temperature": 39.5, "heart_rate": 130.0, "systolic_blood_pressure": 85.0}
trace_risco = engine.explain(sample_risco)

assert trace_risco.crisp_output > 6.5
assert trace_risco.classification == "risco"

print(f"risk_score     : {trace_risco.crisp_output:.3f}")
print(f"classe sugerida: {trace_risco.classification}")
"""
))

CELLS.append(md("### 7.6 Regras que dispararam para o paciente de risco"))
CELLS.append(code(
    """activations = pd.DataFrame([
    {
        "regra": a.rule.name,
        "antecedente": " E ".join(f"{v}={t}" for v, t in a.rule.antecedents),
        "consequente": a.rule.consequent[1],
        "ativação": round(a.strength, 3),
    }
    for a in trace_risco.activations
])
activations.sort_values("ativação", ascending=False)
"""
))

CELLS.append(md(
    """## 8. Spec 06 — Articulação A (Comparação)

ML e fuzzy independentes sobre as mesmas amostras de teste.
Amostragem de 1.500 linhas para acelerar a passagem ao vivo.
"""
))
CELLS.append(code(
    """sampled_X = data.X_test.sample(n=1500, random_state=config.RANDOM_STATE)
sampled_y = data.y_test.loc[sampled_X.index]

cmp_report = TriageComparator(model, build_standalone_engine(), data.label_encoder).run(sampled_X, sampled_y)

assert 0.0 <= cmp_report.agreement_rate <= 1.0

print(f"Concordância ML × Fuzzy : {cmp_report.agreement_rate:.3f}")
print(f"Acurácia ML            : {cmp_report.ml_accuracy:.3f}")
print(f"Acurácia Fuzzy         : {cmp_report.fuzzy_accuracy:.3f}")
"""
))

CELLS.append(md("### 8.1 Matriz ML × Fuzzy"))
CELLS.append(code(
    """cmp_report.confusion_ml_vs_fuzzy
"""
))

CELLS.append(md("### 8.2 Concordância por classe verdadeira"))
CELLS.append(code(
    """cmp_report.per_class_agreement.round(3)
"""
))

CELLS.append(md(
    """## 9. Spec 06 — Articulação B (Integração)

A probabilidade `P(risco)` do Random Forest vira a 4ª entrada
linguística do motor fuzzy integrado.
"""
))
CELLS.append(code(
    """int_report = TriageIntegrator(model, build_integrated_engine(), data.label_encoder).run(sampled_X, sampled_y)

print(f"Acurácia ML       : {int_report.accuracy_ml:.3f}")
print(f"Acurácia integrada: {int_report.accuracy_integrated:.3f}")
print(f"Macro F1 ML       : {int_report.macro_f1_ml:.3f}")
print(f"Macro F1 integrado: {int_report.macro_f1_integrated:.3f}")
"""
))

CELLS.append(md("### 9.1 Δ por classe (recall e F1 — integrado menos ML)"))
CELLS.append(code(
    """int_report.delta_per_class.round(3)
"""
))

CELLS.append(md("### 9.2 Matriz de confusão do sistema integrado"))
CELLS.append(code(
    """int_report.confusion_integrated
"""
))

CELLS.append(md("### 9.3 Distribuição do `risk_score` integrado"))
CELLS.append(code(
    """fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(int_report.predictions["integrated_score"], bins=40, edgecolor="black")
ax.set_xlabel("risk_score (integrado)")
ax.set_ylabel("frequência")
ax.set_title("Distribuição do risk_score integrado")
fig.tight_layout()
plt.show()
"""
))

CELLS.append(md(
    """## 10. Resumo final

Reúne em uma única tabela o que será apresentado ao professor.
"""
))
CELLS.append(code(
    """resumo = pd.DataFrame(
    {
        "ML puro":   [report.accuracy, report.macro_f1],
        "Fuzzy só":  [cmp_report.fuzzy_accuracy, np.nan],
        "Integrado": [int_report.accuracy_integrated, int_report.macro_f1_integrated],
    },
    index=["acurácia", "macro F1"],
)
resumo.round(3)
"""
))

CELLS.append(md(
    """## Conclusão

- **ML puro** entrega a maior acurácia (≈0,955), com erros concentrados
  nas fronteiras adjacentes entre classes.
- **Fuzzy independente** é mais limitado (≈0,57) porque só usa três
  variáveis vitais, mas suas decisões são totalmente interpretáveis
  pelas 9 regras `SE … ENTÃO`.
- **Integração (B)** reduz a acurácia agregada mas **preserva o recall
  da classe `risco`** — escolha clínica defensável: errar para mais
  é menos custoso que perder pacientes graves.

> Demonstração completa, interativa, com simulador de pacientes:
> `uv run streamlit run app.py`.
"""
))

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "TriagemFuzzy",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("NotebookFuzzy.ipynb").write_text(json.dumps(NB, ensure_ascii=False, indent=1))
print(f"Wrote NotebookFuzzy.ipynb with {len(CELLS)} cells")
