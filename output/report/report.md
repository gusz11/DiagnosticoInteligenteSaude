# Relatório — Triagem Inteligente (ML + Fuzzy)

## Apresentação

Sistema acadêmico de apoio à triagem baseado em sinais vitais (temperatura, frequência cardíaca, pressão sistólica, SpO2, dor, doenças crônicas, visitas prévias, idade, modo de chegada). **Uso estritamente acadêmico — não constitui diagnóstico médico.**

Base: 18.000 pacientes. Classes remapeadas: normal=9924, atencao=4484, risco=3592.

## Objetivos

- Treinar um classificador de Machine Learning (Random Forest, decisão D2) para predizer a classe de triagem.
- Construir um sistema fuzzy de Mamdani com 3 entradas vitais (temperatura, batimentos, pressão), 1 saída (risk_score) e ≥ 6 regras SE…ENTÃO.
- Articular ambos via Aproximação A (comparação) e B (integração).

## Parte 1 — Machine Learning

- **Algoritmo**: Random Forest (300 árvores, `class_weight='balanced'`).
- **Acurácia**: 0.955
- **Macro-F1**: 0.949
- **Weighted-F1**: 0.955

Por classe:

```
         precision  recall     f1  support
normal       0.974   0.969  0.971     1985
atencao      0.905   0.916  0.911      897
risco        0.967   0.965  0.966      718
```

Top 5 features:

```
pain_level           0.416
body_temperature     0.151
heart_rate           0.113
oxygen_saturation    0.096
age                  0.075
```

## Parte 2 — Sistema Fuzzy

Sistema Mamdani:
- Entradas: `body_temperature` ∈ [34, 42] °C, `heart_rate` ∈ [30, 200] bpm, `systolic_blood_pressure` ∈ [60, 220] mmHg.
- Termos linguísticos por variável: `baixa/normal/alta`.
- Saída: `risk_score` ∈ [0, 10] com termos `baixo/medio/alto`.
- 9 regras (Spec 05 §Rule Base v1); operador AND com agregação max e defuzzificação por centroide.
- Decodificador de score → classe via bandas {'normal': (0.0, 3.5), 'atencao': (3.5, 6.5), 'risco': (6.5, 10.0)}.

## Parte 3 — Articulação

**Aproximação A — Comparação**

- Concordância ML vs Fuzzy: 0.578
- Acurácia ML (independente): 0.955
- Acurácia Fuzzy (independente): 0.571
- Concordância por classe verdadeira:

```
y_true
normal     0.509
atencao    0.652
risco      0.689
```

**Aproximação B — Integração**

O modelo ML calcula P(risco) e essa probabilidade é injetada no sistema fuzzy junto com vitais para gerar uma decisão integrada.

- Acurácia ML (referência): 0.955
- Acurácia integrado: 0.691
- Macro-F1 ML: 0.950 → Macro-F1 integrado: 0.683

Δ por classe:

```
         recall_ml  recall_integrated  delta_recall  f1_ml  f1_integrated  delta_f1
normal       0.965              0.715        -0.250  0.969          0.744    -0.226
atencao      0.926              0.422        -0.504  0.910          0.409    -0.501
risco        0.961              0.965         0.004  0.970          0.897    -0.073
```

## Análise Crítica

- O Random Forest é mais acurado isoladamente, mas o sistema fuzzy oferece explicabilidade direta (regras SE…ENTÃO) que dialoga com o raciocínio clínico.
- A integração (Aproximação B) tende a ser mais conservadora: quando P(risco) do ML é alta, regras fuzzy elevam o score mesmo com vitais marginais, o que pode aumentar recall em `risco` à custa de precisão.
- Limitações: parâmetros das funções de pertinência foram definidos por convenção clínica genérica e não tunados por validação cruzada; tunar via grid-search é trabalho futuro.
- Não é diagnóstico médico real — uso acadêmico apenas.
