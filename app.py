"""Streamlit — passo a passo do trabalho de Triagem Fuzzy (pt-BR).

Cada seção da barra lateral corresponde a uma das especificações do kit
de desenvolvimento (specs/01..07) e replica, em formato narrativo, o que
o pipeline executa nos módulos `src/triagem_fuzzy/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from triagem_fuzzy import config  # noqa: E402
from triagem_fuzzy.articulation.comparator import TriageComparator  # noqa: E402
from triagem_fuzzy.articulation.integrator import TriageIntegrator  # noqa: E402
from triagem_fuzzy.eda import ExploratoryAnalysis  # noqa: E402
from triagem_fuzzy.fuzzy.factory import (  # noqa: E402
    build_integrated_engine,
    build_standalone_engine,
)
from triagem_fuzzy.ingestion import DataIngestion  # noqa: E402
from triagem_fuzzy.ml.evaluator import ModelEvaluator  # noqa: E402
from triagem_fuzzy.ml.random_forest import TriageRandomForest  # noqa: E402
from triagem_fuzzy.preprocessing import DataPreprocessor  # noqa: E402

# ── configuração visual ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Triagem Inteligente — ML + Fuzzy",
    page_icon="🩺",
    layout="wide",
)

SECTIONS = (
    "1. Apresentação",
    "2. Ingestão & Validação",
    "3. Pré-processamento",
    "4. Análise Exploratória (EDA)",
    "5. Modelo de Machine Learning",
    "6. Sistema de Inferência Fuzzy",
    "7. Articulação A — Comparação",
    "8. Articulação B — Integração",
    "9. Análise Crítica & Conclusão",
)

LABEL_ORDER = ["normal", "atencao", "risco"]
LABEL_DISPLAY = {"normal": "normal", "atencao": "atenção", "risco": "risco"}


# ── cache pesado ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando base de dados…")
def load_dataset() -> pd.DataFrame:
    return DataIngestion().load_validated()


@st.cache_data(show_spinner="Pré-processando…")
def prepare_data(_df: pd.DataFrame):
    return DataPreprocessor().run(_df)


@st.cache_resource(show_spinner="Treinando Random Forest…")
def train_model(_X_train: pd.DataFrame, _y_train: pd.Series):
    return TriageRandomForest().fit(_X_train, _y_train)


@st.cache_resource
def get_fuzzy_engines():
    return build_standalone_engine(), build_integrated_engine()


# ── helpers de UI ────────────────────────────────────────────────────────────
def pretty_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(index=LABEL_DISPLAY, columns=LABEL_DISPLAY)


def confusion_heatmap(cm: pd.DataFrame, title: str) -> alt.Chart:
    long = cm.reset_index().melt(id_vars=cm.index.name or "true", var_name="pred", value_name="count")
    x_col, y_col = "pred", cm.index.name or "true"
    return (
        alt.Chart(long, title=title)
        .mark_rect()
        .encode(
            x=alt.X(f"{x_col}:N", title=x_col),
            y=alt.Y(f"{y_col}:N", title=y_col),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=[x_col, y_col, "count"],
        )
        .properties(height=260)
    ) + alt.Chart(long).mark_text(baseline="middle", fontWeight="bold").encode(
        x=f"{x_col}:N",
        y=f"{y_col}:N",
        text="count:Q",
        color=alt.condition("datum.count > 0", alt.value("black"), alt.value("black")),
    )


# ── seções ───────────────────────────────────────────────────────────────────
def section_apresentacao() -> None:
    st.title("🩺 Triagem Inteligente — Machine Learning + Lógica Fuzzy")
    st.caption("Tema 3 — Diagnóstico Inteligente na Área da Saúde")

    st.warning(
        "⚠️ Este trabalho é **estritamente acadêmico**. Não constitui diagnóstico "
        "médico real e não deve ser utilizado em decisões clínicas."
    )

    st.markdown(
        """
        ## Objetivo geral
        Desenvolver uma solução prática de Inteligência Artificial capaz de
        analisar dados e apoiar uma decisão de triagem por meio da
        **combinação entre Machine Learning e Lógica Fuzzy**.

        ## Objetivos específicos
        1. Escolher um problema de IA aplicado à saúde (apoio à triagem).
        2. Utilizar uma base de dados coerente com o problema.
        3. Realizar pré-processamento e análise exploratória.
        4. Implementar pelo menos um modelo de Machine Learning (classificação).
        5. Avaliar o modelo com métricas adequadas (acurácia, matriz de confusão,
           precisão, recall, F1-score).
        6. Construir um sistema fuzzy com variáveis linguísticas, funções de
           pertinência e regras de inferência.
        7. Articular ML e fuzzy — neste trabalho, **ambas** as abordagens:
           **A — Comparação** e **B — Integração**.
        8. Apresentar análise crítica.

        ## Decisões fixadas
        | # | Decisão | Valor |
        |---|---------|-------|
        | D1 | Remapeamento das classes | `0→normal`, `1→atenção`, `2,3→risco` |
        | D2 | Algoritmo ML | Random Forest |
        | D3 | Articulação | Aproximações **A** e **B** |
        | D4 | Base | `dataset/triagem_fuzzy.csv` (18.000 linhas) |
        """
    )


def section_ingestao() -> None:
    st.header("2. Ingestão & Validação")
    st.markdown(
        """
        O módulo `triagem_fuzzy.ingestion.DataIngestion` carrega o CSV e valida
        a presença das colunas obrigatórias, tipos numéricos, ausência de
        valores faltantes e intervalos plausíveis para cada sinal vital.
        """
    )
    df = load_dataset()

    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas", f"{len(df):,}")
    c2.metric("Colunas", df.shape[1])
    c3.metric("Valores faltantes", int(df.isna().sum().sum()))

    st.subheader("Primeiras linhas da base")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("Esquema autoritativo")
    schema = pd.DataFrame(
        [
            ("age", "anos", "[0, 120]"),
            ("heart_rate", "bpm", "[20, 250]"),
            ("systolic_blood_pressure", "mmHg", "[50, 250]"),
            ("oxygen_saturation", "%", "[50, 100]"),
            ("body_temperature", "°C", "[30.0, 43.0]"),
            ("pain_level", "0–10", "[0, 10]"),
            ("chronic_disease_count", "n", "[0, 20]"),
            ("previous_er_visits", "n", "[0, 50]"),
            ("arrival_mode", "categoria", "{walk_in, ambulance, wheelchair}"),
            ("triage_level", "0–3", "{0, 1, 2, 3}"),
        ],
        columns=["coluna", "unidade", "domínio"],
    )
    st.dataframe(schema, width="stretch", hide_index=True)


def section_preprocessamento() -> None:
    st.header("3. Pré-processamento")
    st.markdown(
        """
        Realizado por `triagem_fuzzy.preprocessing.DataPreprocessor`:

        1. **Remapeamento de classes (D1)**:
           `0→normal`, `1→atenção`, `2+3→risco`.
        2. **One-hot** de `arrival_mode` em três colunas
           (`arrival_mode_walk_in`, `_ambulance`, `_wheelchair`).
        3. **Split estratificado** 80/20 com `random_state=42`.
        4. Random Forest é invariante à escala, então **não há padronização**.
        """
    )
    df = load_dataset()
    prepared = prepare_data(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Treino", f"{len(prepared.X_train):,}")
    c2.metric("Teste", f"{len(prepared.X_test):,}")
    c3.metric("Features", len(prepared.feature_names))

    st.subheader("Distribuição das classes após remapeamento")
    counts = prepared.y_train.value_counts().sort_index()
    chart_df = pd.DataFrame(
        {
            "classe": [LABEL_DISPLAY[config.TRIAGE_LABELS[i]] for i in counts.index],
            "amostras": counts.values,
        }
    )
    st.altair_chart(
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("classe:N", sort=list(LABEL_DISPLAY.values())),
            y="amostras:Q",
            color=alt.Color("classe:N", scale=alt.Scale(scheme="set2")),
            tooltip=["classe", "amostras"],
        )
        .properties(height=300),
        width="stretch",
    )

    st.subheader("Lista ordenada de features")
    st.code("\n".join(prepared.feature_names), language="text")


def section_eda() -> None:
    st.header("4. Análise Exploratória (EDA)")
    st.markdown(
        "`triagem_fuzzy.eda.ExploratoryAnalysis` produz estatísticas "
        "descritivas, balanço de classes, correlações e gráficos de "
        "distribuição/boxplot por classe."
    )
    df = load_dataset()

    eda = ExploratoryAnalysis(config.OUTPUT_DIR)
    summary = eda.describe(df)
    balance_raw = eda.class_balance(df, remap=False)
    balance_remap = eda.class_balance(df, remap=True)
    corrs = eda.correlations(df)

    st.subheader("Estatísticas descritivas (features numéricas)")
    st.dataframe(summary.round(3), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Classes brutas (0/1/2/3)")
        st.bar_chart(balance_raw)
    with c2:
        st.subheader("Classes remapeadas (3 níveis)")
        balance_pretty = balance_remap.rename(LABEL_DISPLAY)
        st.bar_chart(balance_pretty)

    st.subheader("Matriz de correlação (Pearson)")
    corr_long = corrs.reset_index().melt(id_vars="index", var_name="feature_y", value_name="r")
    corr_long = corr_long.rename(columns={"index": "feature_x"})
    st.altair_chart(
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("feature_x:N", title=None),
            y=alt.Y("feature_y:N", title=None),
            color=alt.Color(
                "r:Q",
                scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
            ),
            tooltip=["feature_x", "feature_y", alt.Tooltip("r:Q", format=".2f")],
        )
        .properties(height=420),
        width="stretch",
    )

    feature = st.selectbox(
        "Distribuição por classe — selecione uma feature",
        list(config.NUMERIC_FEATURES),
        index=4,
    )
    triagem_class = df["triage_level"].map(config.TRIAGE_LABEL_MAP).map(config.TRIAGE_LABELS)
    plot_df = pd.DataFrame(
        {feature: df[feature], "classe": triagem_class.map(LABEL_DISPLAY)}
    )
    st.altair_chart(
        alt.Chart(plot_df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("classe:N", sort=list(LABEL_DISPLAY.values())),
            y=alt.Y(f"{feature}:Q"),
            color=alt.Color("classe:N", scale=alt.Scale(scheme="set2")),
        )
        .properties(height=320),
        width="stretch",
    )


def section_ml() -> None:
    st.header("5. Modelo de Machine Learning")
    st.markdown(
        """
        **Algoritmo escolhido (D2):** Random Forest, treinado com
        `class_weight='balanced'` para compensar o desbalanceamento entre
        normal/atenção/risco. Embora a disciplina prefira Árvore de Decisão e
        Naive Bayes pela interpretabilidade, o Random Forest é justificado
        no relatório pela robustez sobre múltiplos preditores ruidosos.
        """
    )
    df = load_dataset()
    prepared = prepare_data(df)
    model = train_model(prepared.X_train, prepared.y_train)
    report = ModelEvaluator(dict(config.TRIAGE_LABELS)).evaluate(
        model, prepared.X_test, prepared.y_test
    )

    st.subheader("Métricas globais")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia", f"{report.accuracy:.3f}")
    c2.metric("Macro F1", f"{report.macro_f1:.3f}")
    c3.metric("Weighted F1", f"{report.weighted_f1:.3f}")

    st.subheader("Precisão, Recall e F1-score por classe")
    per_class_pretty = pretty_labels(report.per_class.round(3))
    st.dataframe(per_class_pretty, width="stretch")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Matriz de confusão")
        cm_pretty = pretty_labels(report.confusion_matrix)
        st.altair_chart(
            confusion_heatmap(cm_pretty, "Verdadeiro × Predito"),
            width="stretch",
        )
    with c2:
        st.subheader("Importância das features")
        imp = report.feature_importances.reset_index()
        imp.columns = ["feature", "importance"]
        st.altair_chart(
            alt.Chart(imp)
            .mark_bar()
            .encode(
                y=alt.Y("feature:N", sort="-x"),
                x="importance:Q",
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(height=350),
            width="stretch",
        )

    st.info(
        "ℹ️ As métricas requeridas pelo enunciado para problemas de "
        "**classificação** estão todas exibidas acima: **Acurácia**, "
        "**Matriz de confusão**, **Precisão**, **Recall (Sensibilidade)** "
        "e **F1-score**."
    )


def section_fuzzy() -> None:
    st.header("6. Sistema de Inferência Fuzzy")
    standalone, integrated = get_fuzzy_engines()
    st.markdown(
        f"""
        Sistema **Mamdani** com agregação `max` e defuzzificação por
        **centroide**. Implementado em `triagem_fuzzy.fuzzy.engine`.

        - **Entradas (modo independente):** temperatura corporal,
          frequência cardíaca, pressão sistólica.
        - **Saída:** `risk_score ∈ [0, 10]`.
        - **Termos linguísticos:** `baixa/normal/alta` para entradas e
          `baixo/medio/alto` para a saída.
        - **Regras:** {len(standalone.rules)} no modo independente
          (+{len(integrated.rules) - len(standalone.rules)} na integração).
        - **Decodificador** `risk_score → classe`: bandas
          `normal=[0, 3.5)`, `atenção=[3.5, 6.5)`, `risco=[6.5, 10]`.
        """
    )

    st.subheader("Funções de pertinência")
    variables = list(standalone.inputs.values()) + [standalone.output]
    var_name = st.selectbox(
        "Selecione uma variável para visualizar as funções",
        [v.name for v in variables],
        index=0,
    )
    var = next(v for v in variables if v.name == var_name)
    xs = var.universe_array()
    long_rows = []
    for term_name, mf in var.terms.items():
        ys = mf.evaluate(xs)
        for x, y in zip(xs, ys):
            long_rows.append({"x": float(x), "y": float(y), "termo": term_name})
    long_df = pd.DataFrame(long_rows)
    st.altair_chart(
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("x:Q", title=var.name),
            y=alt.Y("y:Q", title="grau de pertinência", scale=alt.Scale(domain=[0, 1.05])),
            color=alt.Color("termo:N", scale=alt.Scale(scheme="set1")),
        )
        .properties(height=320),
        width="stretch",
    )

    st.subheader("Base de regras (modo independente)")
    rules_table = pd.DataFrame(
        [
            {
                "id": r.name,
                "antecedente": " E ".join(f"{v}={t}" for v, t in r.antecedents),
                "consequente": f"{r.consequent[0]}={r.consequent[1]}",
            }
            for r in standalone.rules
        ]
    )
    st.dataframe(rules_table, width="stretch", hide_index=True)

    st.subheader("Inferência interativa — teste com um paciente fictício")
    c1, c2, c3 = st.columns(3)
    temp = c1.slider("Temperatura (°C)", 34.0, 42.0, 37.0, 0.1)
    hr = c2.slider("Frequência cardíaca (bpm)", 30, 200, 90)
    bp = c3.slider("Pressão sistólica (mmHg)", 60, 220, 120)

    trace = standalone.explain(
        {
            "body_temperature": temp,
            "heart_rate": hr,
            "systolic_blood_pressure": float(bp),
        }
    )
    c1, c2 = st.columns(2)
    c1.metric("risk_score (defuzzificado)", f"{trace.crisp_output:.2f}")
    c2.metric("Classe sugerida", LABEL_DISPLAY[trace.classification])

    activ_df = pd.DataFrame(
        [
            {
                "regra": a.rule.name,
                "antecedente": " E ".join(f"{v}={t}" for v, t in a.rule.antecedents),
                "consequente": a.rule.consequent[1],
                "ativação": round(a.strength, 3),
            }
            for a in trace.activations
        ]
    )
    st.dataframe(
        activ_df.sort_values("ativação", ascending=False),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Curva agregada (saída)")
    agg_df = pd.DataFrame(
        {"risk_score": standalone.output.universe_array(), "pertinência": trace.aggregated}
    )
    st.altair_chart(
        alt.Chart(agg_df)
        .mark_area(opacity=0.5)
        .encode(x="risk_score:Q", y=alt.Y("pertinência:Q", scale=alt.Scale(domain=[0, 1.05])))
        .properties(height=260)
        + alt.Chart(pd.DataFrame({"x": [trace.crisp_output]}))
        .mark_rule(color="red", strokeWidth=2)
        .encode(x="x:Q"),
        width="stretch",
    )


def section_comparacao() -> None:
    st.header("7. Articulação A — Comparação")
    st.markdown(
        """
        ML e fuzzy são executados **independentemente** sobre o mesmo paciente.
        O comparador mede a taxa de concordância e exibe os casos onde os
        sistemas divergem — uma fonte natural de análise para o relatório.
        """
    )
    df = load_dataset()
    prepared = prepare_data(df)
    model = train_model(prepared.X_train, prepared.y_train)
    engine, _ = get_fuzzy_engines()

    sample_n = st.slider(
        "Amostragem do conjunto de teste (linhas)", 200, len(prepared.X_test), 1000, 100
    )
    sampled_X = prepared.X_test.sample(n=sample_n, random_state=config.RANDOM_STATE)
    sampled_y = prepared.y_test.loc[sampled_X.index]

    with st.spinner("Rodando Aproximação A (Comparação)…"):
        report = TriageComparator(model, engine, dict(config.TRIAGE_LABELS)).run(
            sampled_X, sampled_y
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Concordância ML × Fuzzy", f"{report.agreement_rate:.3f}")
    c2.metric("Acurácia ML", f"{report.ml_accuracy:.3f}")
    c3.metric("Acurácia Fuzzy", f"{report.fuzzy_accuracy:.3f}")

    st.subheader("Matriz ML × Fuzzy")
    st.altair_chart(
        confusion_heatmap(
            pretty_labels(report.confusion_ml_vs_fuzzy),
            "ML (linha) × Fuzzy (coluna)",
        ),
        width="stretch",
    )

    st.subheader("Concordância por classe verdadeira")
    pca = report.per_class_agreement.rename(LABEL_DISPLAY).rename("concordância")
    st.bar_chart(pca)

    st.subheader("Amostras de divergência (até 20)")
    disagree = report.disagreement_samples.copy()
    for col in ("y_true", "ml_pred", "fuzzy_pred"):
        disagree[col] = disagree[col].map(LABEL_DISPLAY).fillna(disagree[col])
    st.dataframe(disagree, width="stretch")


def section_integracao() -> None:
    st.header("8. Articulação B — Integração")
    st.markdown(
        """
        O Random Forest fornece `P(risco)` que é tratada como uma quarta
        entrada linguística (`ml_risk_proba ∈ {baixo, medio, alto}`) do
        sistema fuzzy. Três regras adicionais (R10–R12) ligam essa entrada
        ao `risk_score`. A saída integrada é mais conservadora e
        interpretável que o ML puro.
        """
    )
    df = load_dataset()
    prepared = prepare_data(df)
    model = train_model(prepared.X_train, prepared.y_train)
    _, engine = get_fuzzy_engines()

    sample_n = st.slider(
        "Amostragem do conjunto de teste (linhas)", 200, len(prepared.X_test), 1000, 100, key="int_n"
    )
    sampled_X = prepared.X_test.sample(n=sample_n, random_state=config.RANDOM_STATE)
    sampled_y = prepared.y_test.loc[sampled_X.index]

    with st.spinner("Rodando Aproximação B (Integração)…"):
        report = TriageIntegrator(model, engine, dict(config.TRIAGE_LABELS)).run(
            sampled_X, sampled_y
        )

    c1, c2 = st.columns(2)
    c1.metric("Acurácia ML (referência)", f"{report.accuracy_ml:.3f}")
    c2.metric("Acurácia integrada", f"{report.accuracy_integrated:.3f}")

    c1, c2 = st.columns(2)
    c1.metric("Macro F1 ML", f"{report.macro_f1_ml:.3f}")
    c2.metric("Macro F1 integrado", f"{report.macro_f1_integrated:.3f}")

    st.subheader("Matriz de confusão — sistema integrado")
    st.altair_chart(
        confusion_heatmap(
            pretty_labels(report.confusion_integrated),
            "Verdadeiro × Predito (integrado)",
        ),
        width="stretch",
    )

    st.subheader("Δ por classe (integrado − ML)")
    st.dataframe(pretty_labels(report.delta_per_class.round(3)), width="stretch")

    st.subheader("Distribuição do `risk_score` integrado")
    score_df = pd.DataFrame({"risk_score": report.predictions["integrated_score"]})
    st.altair_chart(
        alt.Chart(score_df)
        .mark_bar()
        .encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=40)),
            y="count()",
        )
        .properties(height=260),
        width="stretch",
    )

    st.subheader("Casos onde ML e Integrado divergem (até 20)")
    diff = report.biggest_changes.copy()
    for col in ("y_true", "ml_pred", "integrated_pred"):
        if col in diff.columns:
            diff[col] = diff[col].map(lambda v: LABEL_DISPLAY.get(config.TRIAGE_LABELS.get(int(v), v), v))
    st.dataframe(diff, width="stretch")


def section_conclusao() -> None:
    st.header("9. Análise Crítica & Conclusão")
    st.markdown(
        """
        **Conclusões principais**

        - **Random Forest** atinge alta acurácia (~0.95) sobre o conjunto de
          teste, com macro-F1 próximo de 0.95, sugerindo que os sinais vitais
          presentes na base carregam sinal suficiente para a triagem em três
          níveis.
        - O **sistema fuzzy independente** (Aproximação A) usa apenas três
          variáveis vitais e parâmetros clínicos genéricos. Sua acurácia
          isolada é menor, mas suas regras são **diretamente interpretáveis**
          — uma vantagem para apresentação clínica e didática.
        - A **integração (Aproximação B)** funciona como uma camada de
          revisão sobre o ML: quando `P(risco)` é alta, regras fuzzy elevam
          o `risk_score` mesmo com vitais marginais. Isso pode aumentar
          recall em `risco` à custa de queda na acurácia global — o trade-off
          é exatamente o ponto de discussão para o relatório.

        **Limitações**

        - As funções de pertinência foram definidas por convenção clínica
          ampla, não por *grid search*. Há ganho potencial em otimizar com
          validação cruzada.
        - O remapeamento `2+3→risco` esconde a distinção entre risco moderado
          e alto. Uma versão futura pode manter 4 classes e ajustar o fuzzy
          para 4 bandas de saída.
        - A base sintetiza pacientes; a generalização para dados reais
          exigiria nova validação.

        **Ética e escopo**

        Este sistema é uma demonstração acadêmica de integração entre
        Machine Learning e Lógica Fuzzy. **Não substitui avaliação médica.**
        """
    )


# ── roteamento ───────────────────────────────────────────────────────────────
def main() -> None:
    st.sidebar.title("Trabalho — IA")
    st.sidebar.caption("Triagem com ML + Lógica Fuzzy")
    choice = st.sidebar.radio("Navegação", SECTIONS, index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Estrutura espelha as especificações em `specs/01..07`. Cada seção "
        "consome diretamente os módulos em `src/triagem_fuzzy/`."
    )

    handler = {
        SECTIONS[0]: section_apresentacao,
        SECTIONS[1]: section_ingestao,
        SECTIONS[2]: section_preprocessamento,
        SECTIONS[3]: section_eda,
        SECTIONS[4]: section_ml,
        SECTIONS[5]: section_fuzzy,
        SECTIONS[6]: section_comparacao,
        SECTIONS[7]: section_integracao,
        SECTIONS[8]: section_conclusao,
    }[choice]
    handler()


if __name__ == "__main__":
    main()
