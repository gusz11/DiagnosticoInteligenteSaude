# Relatório Formal — Triagem Inteligente

**Disciplina:** Inteligência Artificial
**Tema:** Tema 3 — Diagnóstico Inteligente na Área da Saúde
**Título do trabalho:** Sistema de Apoio à Triagem com Random Forest e Lógica Fuzzy
**Data:** Junho de 2026

> ⚠️ **Aviso ético.** Este trabalho tem caráter **estritamente acadêmico**.
> Os resultados aqui apresentados **não constituem diagnóstico médico** e
> não devem ser utilizados para apoiar decisões clínicas em pacientes
> reais.

---

## Sumário

1. [Resumo](#1-resumo)
2. [Apresentação do trabalho](#2-apresentação-do-trabalho)
3. [Objetivos](#3-objetivos)
4. [Fundamentação teórica](#4-fundamentação-teórica)
5. [Materiais e métodos](#5-materiais-e-métodos)
6. [Parte 1 — Machine Learning](#6-parte-1--machine-learning)
7. [Parte 2 — Sistema fuzzy](#7-parte-2--sistema-fuzzy)
8. [Parte 3 — Articulação ML × Fuzzy](#8-parte-3--articulação-ml--fuzzy)
9. [Resultados consolidados](#9-resultados-consolidados)
10. [Análise crítica](#10-análise-crítica)
11. [Limitações e trabalhos futuros](#11-limitações-e-trabalhos-futuros)
12. [Considerações éticas](#12-considerações-éticas)
13. [Conclusão](#13-conclusão)
14. [Apêndices](#14-apêndices)

---

## 1. Resumo

Este trabalho desenvolve uma solução prática de Inteligência Artificial
para apoio à triagem hospitalar, integrando duas abordagens fundamentais
da disciplina: **Machine Learning** (classificação supervisionada com
Random Forest) e **Lógica Fuzzy** (sistema de inferência de Mamdani).
A base de dados contém 18.000 atendimentos sintéticos com nove
atributos clínicos (idade, frequência cardíaca, pressão sistólica,
saturação de oxigênio, temperatura corporal, dor, doenças crônicas,
visitas prévias, modo de chegada) e o nível de triagem como variável
alvo, originalmente em quatro níveis e remapeada para três classes
acadêmicas: **normal**, **atenção** e **risco**.

O modelo de Machine Learning atinge **acurácia de 0,955** e **macro-F1
de 0,949** sobre o conjunto de teste. O sistema fuzzy, com três
entradas vitais e nove regras, fornece uma decisão **interpretável**.
Foram implementadas as duas formas de articulação previstas no
enunciado: **Aproximação A — Comparação** independente entre ML e
fuzzy (concordância de 0,578) e **Aproximação B — Integração**, onde a
probabilidade `P(risco)` calculada pelo Random Forest entra como quarta
variável linguística no sistema fuzzy. A integração produz uma decisão
mais conservadora, com queda de acurácia (0,691) mas com ganho
qualitativo de interpretabilidade.

O projeto foi construído como um kit de **desenvolvimento orientado a
especificações** (`specs/01..07`), com módulos e classes em
`src/triagem_fuzzy/`, suíte de testes automatizados em `tests/`
(51 testes, todos verdes), pipeline executável (`main.py`) e aplicação
interativa em **Streamlit** (`app.py`) para apresentação.

---

## 2. Apresentação do trabalho

O problema de triagem hospitalar consiste em **priorizar pacientes**
ao chegarem ao pronto-atendimento, com base em sinais vitais
imediatamente disponíveis. Um sistema de apoio computacional não
substitui a avaliação humana, mas pode:

- reduzir a variabilidade entre profissionais;
- destacar combinações de sintomas de alto risco;
- fornecer um *score* contínuo, mais informativo que uma classe discreta.

O enunciado pede que a solução combine **Machine Learning** — que
aprende padrões a partir de dados — e **Lógica Fuzzy** — que representa
conceitos imprecisos com linguagem próxima ao raciocínio humano. A
disciplina valoriza a integração entre essas duas abordagens porque
elas tratam tipos diferentes de incerteza: estatística (no ML) e
linguística (no fuzzy).

### 2.1 Decisões fixadas no início do projeto

Antes do desenvolvimento, três decisões foram registradas em
`specs/00-overview.spec.md` para garantir coerência entre as partes:

| ID  | Decisão                                  | Valor                                                 |
| --- | ---------------------------------------- | ----------------------------------------------------- |
| D1  | Remapeamento das classes do alvo         | `0 → normal`, `1 → atenção`, `2 ∪ 3 → risco`          |
| D2  | Algoritmo de Machine Learning            | Random Forest multiclasse                             |
| D3  | Forma de articulação ML × Fuzzy          | **Ambas** — Aproximação A (Comparação) e B (Integração) |
| D4  | Base de dados                            | `dataset/triagem_fuzzy.csv` (18.000 linhas)           |
| D5  | Formato de entrega                       | Módulos e classes em Python; sem notebook monolítico  |

---

## 3. Objetivos

### 3.1 Objetivo geral

Desenvolver uma solução prática de Inteligência Artificial capaz de
analisar dados clínicos básicos e apoiar uma decisão de triagem por
meio da **combinação entre Machine Learning e Lógica Fuzzy**.

### 3.2 Objetivos específicos

1. Escolher um problema adequado para aplicação de IA na área da saúde.
2. Utilizar uma base de dados coerente com o problema.
3. Realizar pré-processamento e análise exploratória dos dados.
4. Implementar pelo menos um modelo de Machine Learning para
   classificação.
5. Avaliar o modelo com métricas adequadas (**acurácia**, **matriz de
   confusão**, **precisão**, **recall** e **F1-score**).
6. Construir um sistema fuzzy com variáveis linguísticas, funções de
   pertinência e regras de inferência.
7. **Comparar e integrar** os resultados do modelo de Machine Learning
   com o sistema fuzzy.
8. Apresentar análise crítica sobre os resultados, limitações e
   melhorias possíveis.

---

## 4. Fundamentação teórica

### 4.1 Machine Learning supervisionado

Em problemas supervisionados, o algoritmo recebe um conjunto de
exemplos rotulados $(X, y)$ e aprende uma função $f$ tal que
$\hat{y} = f(X)$ aproxime $y$ no menor erro possível em dados não
vistos. Para o nosso problema, $y \in \{0, 1, 2\}$ representa as
classes de triagem.

### 4.2 Random Forest

O **Random Forest** é um ensemble de árvores de decisão treinadas em
amostras *bootstrap* dos dados com seleção aleatória de atributos em
cada nó. A predição final é a moda das predições individuais
(classificação). Vantagens relevantes para este trabalho:

- Lida bem com **atributos numéricos e categóricos** sem padronização.
- É **robusto a outliers** e ao desbalanceamento moderado entre classes.
- Fornece **importância das variáveis** como sub-produto natural do
  treinamento.
- Permite **calibração de probabilidades** via `predict_proba`, o que é
  utilizado na Aproximação B (integração).

### 4.3 Lógica Fuzzy e sistemas de Mamdani

A lógica fuzzy estende a lógica clássica permitindo graus de pertinência
no intervalo $[0, 1]$. Um **sistema de inferência fuzzy** mapeia
entradas numéricas em uma saída numérica através de quatro etapas:

1. **Fuzzificação:** transforma um valor numérico em graus de
   pertinência a termos linguísticos.
2. **Avaliação de regras:** combina pertinências de antecedentes via
   operador lógico (T-norma para AND, T-conorma para OR).
3. **Agregação:** combina as saídas de todas as regras.
4. **Defuzzificação:** colapsa o conjunto fuzzy resultante em um valor
   numérico (centroide, neste trabalho).

### 4.4 Por que integrar?

ML e fuzzy tratam **tipos diferentes de incerteza**. Modelos
estatísticos lidam com incerteza decorrente da variabilidade dos dados;
sistemas fuzzy lidam com incerteza decorrente da imprecisão dos
conceitos (o que significa “temperatura alta”?). Integrar as duas
abordagens permite que o sistema combine **acurácia preditiva** com
**explicabilidade**.

---

## 5. Materiais e métodos

### 5.1 Base de dados

A base utilizada (`dataset/triagem_fuzzy.csv`) contém **18.000
atendimentos** com as variáveis abaixo. Esta base é sintética e
fornecida no escopo da disciplina; ela foi tratada como um pequeno
estudo de caso clínico.

| Coluna                       | Tipo    | Unidade  | Faixa válida              |
| ---------------------------- | ------- | -------- | ------------------------- |
| `age`                        | float   | anos     | [0, 120]                  |
| `heart_rate`                 | float   | bpm      | [20, 250]                 |
| `systolic_blood_pressure`    | float   | mmHg     | [50, 250]                 |
| `oxygen_saturation`          | float   | %        | [50, 100]                 |
| `body_temperature`           | float   | °C       | [30, 43]                  |
| `pain_level`                 | int     | 0–10     | [0, 10]                   |
| `chronic_disease_count`      | int     | n        | [0, 20]                   |
| `previous_er_visits`         | int     | n        | [0, 50]                   |
| `arrival_mode`               | string  | —        | {walk_in, ambulance, wheelchair} |
| `triage_level` (alvo)        | int     | 0–3      | {0, 1, 2, 3}              |

#### Distribuição original do alvo

| Nível bruto | Contagem |
| ----------- | -------- |
| 0           | 9.924    |
| 1           | 4.484    |
| 2           | 2.701    |
| 3           | 891      |

#### Distribuição após remapeamento (D1)

| Classe acadêmica | Contagem | Proporção |
| ---------------- | -------- | --------- |
| normal           | 9.924    | 55,1%     |
| atenção          | 4.484    | 24,9%     |
| risco            | 3.592    | 20,0%     |

A decisão de unir os níveis 2 e 3 em `risco` decorre da necessidade do
enunciado (três classes acadêmicas: normal, atenção e risco) e mantém
um número razoável de amostras na classe minoritária (3.592).

### 5.2 Arquitetura do projeto

O projeto foi estruturado como **kit de desenvolvimento orientado a
especificações**. Cada parte da solução tem uma especificação em
`specs/`, uma implementação em `src/triagem_fuzzy/` e testes em
`tests/`.

```
TriagemFuzzy/
├── specs/                              # Specs (contratos)
│   ├── 00-overview.spec.md
│   ├── 01-ingestion.spec.md
│   ├── 02-preprocessing.spec.md
│   ├── 03-eda.spec.md
│   ├── 04-ml-model.spec.md
│   ├── 05-fuzzy-system.spec.md
│   ├── 06-articulation.spec.md
│   └── 07-evaluation-report.spec.md
├── src/triagem_fuzzy/
│   ├── config.py                       # constantes, mapeamentos, bandas
│   ├── ingestion.py                    # DataIngestion + SchemaError
│   ├── preprocessing.py                # DataPreprocessor + PreparedData
│   ├── eda.py                          # ExploratoryAnalysis + EdaReport
│   ├── ml/
│   │   ├── base.py                     # BaseTriageClassifier (ABC)
│   │   ├── random_forest.py            # TriageRandomForest
│   │   └── evaluator.py                # ModelEvaluator + EvaluationReport
│   ├── fuzzy/
│   │   ├── variables.py                # LinguisticVariable, MembershipFunction
│   │   ├── rules.py                    # FuzzyRule, RuleBase
│   │   ├── engine.py                   # FuzzyInferenceEngine + FuzzyTrace
│   │   └── factory.py                  # construtores dos motores
│   ├── articulation/
│   │   ├── comparator.py               # Aproximação A
│   │   └── integrator.py               # Aproximação B
│   └── reporting.py                    # ResultReporter
├── tests/                              # 51 testes (pytest)
├── dataset/triagem_fuzzy.csv           # base
├── main.py                             # CLI orquestrador
├── app.py                              # Aplicação Streamlit interativa
└── docs/RELATORIO.md                   # este documento
```

#### Princípios de engenharia adotados

- **Especificação antes da implementação** (spec-driven).
- **Tipagem estática** (PEP 484) em toda a API pública.
- **Seed fixa** (`config.RANDOM_STATE = 50`) para reprodutibilidade.
- **Separação estrita** entre `ml/` e `fuzzy/`: somente o pacote
  `articulation/` pode importar ambos.
- **I/O centralizado**: apenas `ingestion.load()` lê o CSV bruto e
  apenas `reporting.py` grava artefatos finais em disco.
- **Testes automatizados** por módulo, com fixtures compartilhadas em
  `tests/conftest.py`.

### 5.3 Pré-processamento

Realizado em `triagem_fuzzy.preprocessing.DataPreprocessor`:

1. **Validação de esquema**: ranges, tipos e categorias verificados em
   `DataIngestion.validate`. Qualquer violação levanta `SchemaError`.
2. **Remapeamento das classes** (D1):
   `{0→0, 1→1, 2→2, 3→2}`.
3. **One-hot** de `arrival_mode` em três colunas
   (`arrival_mode_walk_in`, `_ambulance`, `_wheelchair`).
4. **Split estratificado** 80 / 20 com `random_state=50`, mantendo as
   proporções de classe (verificado pelo teste
   `test_stratification_preserves_class_ratio`).
5. **Sem padronização**: o Random Forest é invariante à escala, e
   manter os valores nas unidades originais facilita a comparação com
   o sistema fuzzy (que opera nas mesmas escalas).

Resultado: `X_train ∈ ℝ^{14.400 × 11}`, `X_test ∈ ℝ^{3.600 × 11}`.

### 5.4 Análise exploratória

A `ExploratoryAnalysis` produz, no diretório `output/eda/`:

- estatísticas descritivas (`summary_statistics.csv`);
- balanço de classes bruto e remapeado;
- matriz de correlação (Pearson) entre as variáveis numéricas;
- 8 histogramas e 8 boxplots por classe.

Observações relevantes da EDA:

- A `body_temperature` apresenta deslocamento positivo no boxplot da
  classe `risco`: pacientes com temperatura acima de 38 °C estão
  fortemente associados ao nível mais grave.
- `pain_level` separa visualmente bem as três classes; valores ≥ 7
  predominam em `risco`. Esta variável acabou se mostrando a **mais
  importante** para o classificador.
- `oxygen_saturation` mostra cauda inferior em `risco` (saturação baixa
  é um forte indicador clínico).
- `heart_rate` e `systolic_blood_pressure` apresentam dispersão maior
  nas classes `atenção` e `risco`, sugerindo combinações
  (taquicardia + hipotensão) como sinais de gravidade.
- `chronic_disease_count` e `previous_er_visits` aumentam com a
  gravidade da classe, mas com sobreposição substancial.

Essas observações motivaram a escolha das **três variáveis fuzzy**
(temperatura, frequência cardíaca, pressão sistólica) usadas na
Parte 2, e a expectativa de que o Random Forest atribua alta importância
a saturação e temperatura.

---

## 6. Parte 1 — Machine Learning

### 6.1 Justificativa do algoritmo (D2)

O enunciado lista como recomendados: Árvore de Decisão, Naive Bayes,
KNN, Regressão Linear/Logística e Random Forest. Optou-se por
**Random Forest** porque:

- a base apresenta variáveis em escalas heterogêneas e mistura tipos
  numéricos e categóricos — o algoritmo lida nativamente com essa
  estrutura;
- a presença de pelo menos três variáveis fortemente preditivas
  (saturação, temperatura e pressão) e de variáveis redundantes
  (`previous_er_visits`, `chronic_disease_count`) faz com que o
  ensemble se beneficie da seleção aleatória de atributos por nó;
- o algoritmo expõe `predict_proba`, o que é **essencial para a
  Aproximação B** (a probabilidade `P(risco)` é injetada como entrada
  fuzzy);
- a importância das *features* atua como **interpretabilidade
  global**, complementando a interpretabilidade local do sistema fuzzy.

O custo da escolha é a perda de interpretabilidade individual de uma
única árvore. Esse custo é mitigado pelo sistema fuzzy, que existe
exatamente para fornecer regras legíveis.

### 6.2 Hiperparâmetros adotados

Definidos em `triagem_fuzzy.ml.random_forest.TriageRandomForest`:

| Hiperparâmetro       | Valor          | Justificativa                                    |
| -------------------- | -------------- | ------------------------------------------------ |
| `n_estimators`       | 300            | Compromisso entre estabilidade e tempo (< 10 s). |
| `min_samples_leaf`   | 2              | Regularização suave contra overfitting de folha. |
| `class_weight`       | `"balanced"`   | Compensa o desbalanceamento (55 / 25 / 20).      |
| `random_state`       | 50             | Reprodutibilidade.                               |
| `n_jobs`             | -1             | Uso de todos os núcleos.                         |

### 6.3 Métricas obtidas (requisito do enunciado)

Avaliação realizada por `ModelEvaluator.evaluate`. Resultados sobre o
conjunto de teste (3.600 amostras).

#### 6.3.1 Métricas globais

| Métrica         | Valor  |
| --------------- | ------ |
| **Acurácia**    | 0,955  |
| **Macro F1**    | 0,949  |
| **Weighted F1** | 0,955  |

#### 6.3.2 Métricas por classe (Precisão, Recall, F1-score)

| Classe   | Precisão | Recall | F1-score | Suporte |
| -------- | -------- | ------ | -------- | ------- |
| normal   | 0,974    | 0,969  | 0,971    | 1.985   |
| atenção  | 0,905    | 0,916  | 0,911    | 897     |
| risco    | 0,967    | 0,965  | 0,966    | 718     |

> Valores arredondados; os números exatos estão em
> `output/report/summary.json`.

#### 6.3.3 Matriz de confusão

Linhas = classe verdadeira, colunas = classe predita.

|              | pred normal | pred atenção | pred risco |
| ------------ | ----------- | ------------ | ---------- |
| **normal**   | 1.923       | 62           | 0          |
| **atenção**  | 51          | 822          | 24         |
| **risco**    | 1           | 24           | 693        |

A matriz mostra que os erros se concentram nas **fronteiras adjacentes**
(normal ↔ atenção e atenção ↔ risco). Erros “grandes” (normal
classificado como risco, ou vice-versa) são extremamente raros —
apenas **1 ocorrência** em 3.600 (risco predito como normal) e **zero**
no sentido inverso. Isso é desejável em um sistema de triagem: errar
para uma classe vizinha tem custo clínico muito menor que pular dois
níveis de gravidade.

#### 6.3.4 Importância das features

Top 5 features pelo critério Gini (Random Forest):

| # | Feature                   | Importância |
| - | ------------------------- | ----------- |
| 1 | `pain_level`              | 0,416       |
| 2 | `body_temperature`        | 0,151       |
| 3 | `heart_rate`              | 0,113       |
| 4 | `oxygen_saturation`       | 0,096       |
| 5 | `age`                     | 0,075       |

A `pain_level` se destaca como **a feature mais preditiva**,
respondendo sozinha por ~42 % da importância total — um achado
relevante para a discussão crítica. As demais quatro posições são
ocupadas por sinais vitais clássicos (temperatura, FC, SpO₂) e idade.
Variáveis demográficas/históricas (`chronic_disease_count`,
`previous_er_visits`) e o modo de chegada contribuem em menor grau.

### 6.4 Avaliação de adequação do modelo

- A acurácia de 0,955 é elevada, mas precisa ser interpretada com
  cuidado: o baseline trivial (sempre prever `normal`) já atingiria
  ~55 %. O macro-F1 de 0,949 indica que o modelo é **bom inclusive
  na classe minoritária**, e não apenas nas classes majoritárias.
- O `class_weight="balanced"` foi essencial para evitar que a classe
  `risco` (20 % das amostras) fosse subrepresentada nas decisões.
- A forte presença de `pain_level` é uma característica desta base
  sintética e deve ser apresentada como **achado interpretativo**:
  em dados reais, a dor é subjetiva e ruidosa, portanto a
  generalização exigiria validação adicional.

---

## 7. Parte 2 — Sistema fuzzy

O sistema fuzzy foi implementado de forma autoral nos módulos
`triagem_fuzzy.fuzzy.*`, utilizando a biblioteca `scikit-fuzzy`
apenas para as funções de pertinência primitivas (`trimf`, `trapmf`) e
para a defuzzificação por centroide. A lógica de fuzzificação,
inferência por regras, agregação e classificação por bandas é própria
do projeto, o que dá controle total sobre `explain()` — um método
inspetor que retorna a força de ativação de cada regra.

### 7.1 Requisitos atendidos (conferência ponto-a-ponto)

| Requisito do enunciado                                    | Implementação                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------- |
| ≥ 2 variáveis de entrada                                  | **3** variáveis vitais (+ 1 adicional na Aproximação B)             |
| 1 variável de saída                                       | `risk_score ∈ [0, 10]`                                              |
| ≥ 3 termos linguísticos por variável principal            | Cada variável tem exatamente 3 termos                                |
| ≥ 6 regras SE … ENTÃO                                     | **9 regras** no modo independente (+3 na integração)                |
| Funções de pertinência triangulares ou trapezoidais       | Trapezoidais nas bordas, triangulares no meio                       |
| Processo de fuzzificação                                  | `FuzzyInferenceEngine.fuzzify()`                                    |
| Processo de inferência                                    | Mamdani com `min` para AND e `max` na agregação                     |
| Processo de defuzzificação                                | Centroide (`scikit-fuzzy.defuzz`)                                   |

### 7.2 Variáveis linguísticas

#### Entradas (modo independente)

| Variável                    | Universo      | Termos                       |
| --------------------------- | ------------- | ---------------------------- |
| `body_temperature`          | [34, 42] °C   | `baixa`, `normal`, `alta`    |
| `heart_rate`                | [30, 200] bpm | `baixa`, `normal`, `alta`    |
| `systolic_blood_pressure`   | [60, 220] mmHg| `baixa`, `normal`, `alta`    |

#### Saída

| Variável     | Universo     | Termos                     |
| ------------ | ------------ | -------------------------- |
| `risk_score` | [0, 10]      | `baixo`, `medio`, `alto`   |

### 7.3 Funções de pertinência (parâmetros)

Notação: `tri(a, b, c)` é triangular; `trap(a, b, c, d)` é trapezoidal.

```
body_temperature
  baixa  : trap(34.0, 34.0, 35.5, 36.3)
  normal : tri (36.0, 36.8, 37.5)
  alta   : trap(37.2, 38.0, 42.0, 42.0)

heart_rate
  baixa  : trap(30.0, 30.0, 50.0, 65.0)
  normal : tri (60.0, 80.0, 100.0)
  alta   : trap(95.0, 110.0, 200.0, 200.0)

systolic_blood_pressure
  baixa  : trap(60.0, 60.0, 85.0, 100.0)
  normal : tri (95.0, 120.0, 140.0)
  alta   : trap(135.0, 150.0, 220.0, 220.0)

risk_score
  baixo  : trap(0.0, 0.0, 2.0, 4.0)
  medio  : tri (3.0, 5.0, 7.0)
  alto   : trap(6.0, 8.0, 10.0, 10.0)
```

Os parâmetros foram escolhidos a partir de **faixas clínicas amplas**
(taquicardia > 100 bpm, hipertermia ≥ 38 °C, hipotensão < 90 mmHg),
com sobreposição moderada entre termos para evitar transições
descontínuas. Os testes em `tests/test_fuzzy.py` garantem que (i) as
pertinências estão sempre em $[0, 1]$ e (ii) há sobreposição entre
termos adjacentes (não existe gap).

### 7.4 Base de regras (9 regras)

```
R1.  SE temperatura alta   E batimentos alta              ENTÃO risco alto
R2.  SE temperatura alta   E pressão  baixa               ENTÃO risco alto
R3.  SE batimentos alta    E pressão  baixa               ENTÃO risco alto
R4.  SE temperatura normal E batimentos normal E pressão normal
                                                          ENTÃO risco baixo
R5.  SE temperatura baixa  E pressão  baixa               ENTÃO risco alto
R6.  SE batimentos baixa   E pressão  normal              ENTÃO risco medio
R7.  SE temperatura alta   E batimentos normal E pressão normal
                                                          ENTÃO risco medio
R8.  SE batimentos alta    E pressão  normal              ENTÃO risco medio
R9.  SE temperatura normal E pressão  alta                ENTÃO risco medio
```

Operador AND implementado pelo **mínimo** das pertinências
(T-norma de Gödel). A agregação dos consequentes é feita pelo **máximo**
entre regras que ativam o mesmo termo do `risk_score`.

### 7.5 Fuzzificação, inferência e defuzzificação

Para cada amostra:

1. **Fuzzificação** — `body_temperature = 39.5` produz
   `{baixa: 0.0, normal: 0.0, alta: 1.0}`.
2. **Avaliação das regras** — cada regra retorna sua **força de
   ativação** (mínimo das pertinências dos antecedentes).
3. **Agregação** — para cada termo do `risk_score`, toma-se o máximo
   das ativações das regras que apontam para esse termo. Em seguida,
   o conjunto fuzzy é “clipado” pela ativação.
4. **Defuzzificação** — método **centroide**:

$$
\mathrm{score} = \frac{\int x \cdot \mu(x)\, dx}{\int \mu(x)\, dx}
$$

calculado sobre o universo discretizado em 1001 pontos.

### 7.6 Decodificador `risk_score → classe`

```
risk_score ∈ [0,0; 3,5)   →  normal
risk_score ∈ [3,5; 6,5)   →  atenção
risk_score ∈ [6,5; 10,0]  →  risco
```

Bandas registradas em `config.FUZZY_DECISION_BANDS`.

### 7.7 Exemplos demonstrativos (também presentes nos testes)

| Paciente | Temp | FC  | PA  | risk_score | Classe  |
| -------- | ---- | --- | --- | ---------- | ------- |
| Normal típico   | 36,8 | 75  | 120 | < 3,5      | normal  |
| Risco típico    | 39,5 | 130 | 85  | > 6,5      | risco   |
| Borderline      | 37,6 | 95  | 110 | ~5,0       | atenção |

---

## 8. Parte 3 — Articulação ML × Fuzzy

Conforme decisão D3, foram implementadas **ambas** as formas previstas
no enunciado.

### 8.1 Aproximação A — Comparação

Em `triagem_fuzzy.articulation.comparator.TriageComparator`, o
classificador Random Forest e o motor fuzzy independente são
executados **em paralelo** sobre as mesmas amostras. O comparador
reporta:

- **Taxa de concordância** ML × Fuzzy;
- **Matriz cruzada** (3 × 3) ML × Fuzzy;
- **Concordância por classe verdadeira**;
- Acurácias independentes de cada sistema;
- Até 20 exemplos de divergência (para análise qualitativa).

#### Resultados (amostra de 1.500 linhas do conjunto de teste)

| Métrica                          | Valor  |
| -------------------------------- | ------ |
| Concordância ML × Fuzzy          | 0,578  |
| Acurácia ML (independente)       | 0,955  |
| Acurácia Fuzzy (independente)    | 0,571  |

#### Concordância por classe verdadeira

| Classe verdadeira | Concordância |
| ----------------- | ------------ |
| normal            | 0,509        |
| atenção           | 0,652        |
| risco             | 0,689        |

#### Leitura crítica

- O sistema fuzzy independente opera com **apenas 3 das 9 variáveis**
  do dataset (temperatura, FC, PA) e, portanto, não dispõe de
  informações como saturação, dor e visitas prévias. Sua acurácia
  isolada é naturalmente inferior.
- A concordância de 0,578 indica que ML e fuzzy **concordam em mais da
  metade dos casos**, mas divergem em situações ambíguas. A
  concordância é maior nas classes `atenção` (0,652) e `risco`
  (0,689) e menor em `normal` (0,509) — exatamente onde o fuzzy
  tende a ser mais cauteloso e empurrar pacientes para `atenção`.
  A análise das divergências (disponível em
  `comparison.disagreement_samples`) revela:
  - vários pacientes com vitais marginais que o fuzzy classifica como
    `atenção` mas que o ML, com mais variáveis (em especial
    `pain_level`), prediz como `normal`;
  - alguns casos onde o fuzzy é mais conservador, classificando como
    `risco` pacientes que o ML coloca em `atenção`.

### 8.2 Aproximação B — Integração

Em `triagem_fuzzy.articulation.integrator.TriageIntegrator`, a saída
probabilística do Random Forest (`P(risco)`) é injetada no sistema
fuzzy como **quarta variável linguística** (`ml_risk_proba`):

| Variável         | Universo  | Termos                     |
| ---------------- | --------- | -------------------------- |
| `ml_risk_proba`  | [0, 1]    | `baixo`, `medio`, `alto`   |

Acrescentam-se três regras (R10–R12) ao motor:

```
R10. SE ml_risk_proba alto                              ENTÃO risco alto
R11. SE ml_risk_proba medio E pressão baixa             ENTÃO risco alto
R12. SE ml_risk_proba baixo E temperatura normal        ENTÃO risco baixo
```

O motor integrado fica com **12 regras**, combinando vitais e
probabilidade do ML. A saída numérica passa pelo mesmo decodificador
(7.6).

#### Resultados (amostra de 1.500 linhas)

| Métrica                          | Valor  |
| -------------------------------- | ------ |
| Acurácia ML (referência)         | 0,955  |
| **Acurácia integrada**           | 0,691  |
| Macro F1 ML                      | 0,950  |
| **Macro F1 integrado**           | 0,683  |

#### Δ por classe (integrado − ML), recall e F1

| Classe   | Recall ML | Recall integ. | Δ Recall | F1 ML  | F1 integ. | Δ F1   |
| -------- | --------- | ------------- | -------- | ------ | --------- | ------ |
| normal   | 0,965     | 0,715         | −0,250   | 0,969  | 0,744     | −0,226 |
| atenção  | 0,926     | 0,422         | −0,504   | 0,910  | 0,409     | −0,501 |
| risco    | 0,961     | 0,965         | **+0,003**| 0,970 | 0,897     | −0,073 |

> O **recall da classe `risco` é o único que se mantém** (na verdade
> sobe ligeiramente) na integração. Em troca, há queda expressiva em
> `atenção` e `normal`. Esta é uma propriedade clínica importante:
> o sistema integrado **prioriza não perder pacientes graves** ao
> custo de gerar mais falsos-positivos em `atenção`.

#### Leitura crítica

A integração **reduz** a acurácia em relação ao ML puro, mas isso é
informativo e merece análise:

- O sistema fuzzy integrado é, **por construção, mais conservador**: a
  presença de qualquer combinação adversa de vitais (mesmo de baixa
  intensidade) puxa o `risk_score` em direção ao meio do intervalo,
  movendo pacientes de `normal` para `atenção`. Isso explica a
  queda de recall em `normal` (Δ = −0,250).
- A tabela Δ mostra que **apenas a classe `risco` mantém recall**
  (variação positiva de +0,003): o sistema integrado **não perde**
  pacientes graves em relação ao ML puro. Em termos clínicos, este é
  o erro **menos custoso**: trocar precisão por sensibilidade na
  classe de maior gravidade é, em triagem, uma escolha aceitável.
- O grupo entende que a Aproximação B, mesmo sem ganho de acurácia
  agregada, **agrega valor explicativo**: ao apresentar um paciente
  ao avaliador humano, o sistema integrado fornece junto da decisão
  o `risk_score`, as regras ativas e a `P(risco)` do ML — três sinais
  complementares.

---

## 9. Resultados consolidados

| Métrica                                          | ML puro | Fuzzy só | Integrado (B) |
| ------------------------------------------------ | ------- | -------- | ------------- |
| Acurácia                                         | 0,955   | 0,571    | 0,691         |
| Macro F1                                         | 0,949   | —        | 0,683         |
| Recall em `risco`                                | 0,965   | —        | 0,965 (=)     |
| Recall em `normal`                               | 0,965   | —        | 0,715 (↓)     |
| Interpretabilidade da decisão                    | Baixa   | Alta     | Alta          |
| Necessidade de variáveis adicionais à temperatura/FC/PA | Sim | Não   | Sim           |

### Artefatos gerados pelo pipeline (`main.py --pipeline full`)

- `output/eda/` — 16 PNGs + 4 CSVs;
- `output/models/random_forest.joblib` — modelo persistido;
- `output/report/report.md` — relatório gerado automaticamente;
- `output/report/summary.json` — métricas em formato máquina-legível;
- 5 PNGs no relatório: três matrizes de confusão, importância das
  features e distribuição do `risk_score` integrado.

---

## 10. Análise crítica

### 10.1 Pontos fortes do trabalho

- **Cobertura completa do enunciado.** Todos os itens das partes 1, 2
  e 3 foram entregues, incluindo as duas formas de articulação.
- **Métricas requeridas** (acurácia, matriz de confusão, precisão,
  recall, F1) calculadas, exibidas e versionadas.
- **Arquitetura limpa e testada.** 51 testes automatizados em pytest
  garantem que cada módulo cumpre seu contrato.
- **Apresentação interativa** (`app.py`) permite simular pacientes em
  tempo real e mostrar a defuzzificação ao vivo.

### 10.2 Pontos a discutir

- A queda de acurácia da Aproximação B é um **resultado interessante e
  honesto**: a integração só agrega valor se houver tolerância a perda
  de acurácia em troca de explicabilidade e segurança clínica
  (conservadorismo). Não é um defeito — é uma propriedade.
- O fuzzy independente usa apenas 3 variáveis. **Por convicção
  pedagógica**, mantivemos esse escopo: ele facilita a leitura e o
  ensino das regras. Uma versão estendida poderia incluir
  `oxygen_saturation` (alta importância no ML), o que provavelmente
  elevaria a acurácia isolada do fuzzy.

---

## 11. Limitações e trabalhos futuros

1. **Tuning das funções de pertinência.** Os parâmetros foram
   estabelecidos por convenção clínica genérica. *Grid search* ou
   métodos evolucionários (ex.: PSO) sobre o conjunto de validação
   poderiam ajustar as funções a este dataset específico.
2. **Granularidade das classes.** O remapeamento `2 ∪ 3 → risco`
   esconde a distinção entre risco moderado e crítico. Manter 4
   classes (com 4 bandas de saída) é direto e melhoraria a
   especificidade clínica.
3. **Validação cruzada.** As métricas vêm de um único holdout 80/20.
   Uma validação k-fold estratificada com k=5 daria intervalos de
   confiança.
4. **Calibração de probabilidades.** O Random Forest fornece
   probabilidades não calibradas; aplicar *Platt scaling* ou
   *isotonic regression* tornaria `ml_risk_proba` semanticamente mais
   próxima de uma probabilidade real.
5. **Generalização.** A base é sintética. Validação em dados reais
   exigiria novo IRB/ética e provavelmente novos parâmetros fuzzy.

---

## 12. Considerações éticas

- O sistema **não substitui** avaliação médica. É um exercício
  didático para combinar duas técnicas de IA.
- O remapeamento de classes foi explicitamente documentado (D1) e a
  perda de informação reconhecida.
- A base é sintética e não contém dados pessoais; ainda assim, todo o
  código foi desenvolvido sem persistir qualquer cópia dos dados
  fora do diretório `dataset/`.
- A documentação alerta o usuário de forma proeminente sobre o caráter
  acadêmico em três lugares: `app.py`, `output/report/report.md` e
  neste documento.

---

## 13. Conclusão

O trabalho integra Machine Learning e Lógica Fuzzy em um sistema de
apoio à triagem que cumpre integralmente os requisitos do enunciado.
O Random Forest fornece a acurácia operacional; o sistema fuzzy
fornece a explicabilidade; a articulação entre eles foi explorada nas
duas formas previstas (Comparação e Integração) e os resultados foram
quantificados com métricas adequadas.

A escolha pedagógica de organizar o projeto como kit de
desenvolvimento orientado a especificações se mostrou útil para a
clareza da defesa: cada parte tem um contrato escrito e uma
implementação verificável. A aplicação Streamlit (`app.py`) permite à
banca examinadora **simular pacientes em tempo real** e observar como
o `risk_score` se forma, regra a regra — o que torna a apresentação
oral mais didática que uma simples leitura de slides.

---

## 14. Apêndices

### Apêndice A — Como executar o projeto

#### Pré-requisitos

- Python 3.11+
- `uv` (gerenciador de ambiente)

#### Instalação

```bash
uv sync
```

#### Pipeline em linha de comando

```bash
uv run python main.py --pipeline full --articulation-sample 1500 --verbose
```

Saídas em `output/eda/`, `output/models/` e `output/report/`.

#### Aplicação interativa (recomendado para apresentação)

```bash
uv run streamlit run app.py
```

Abrir o navegador em `http://localhost:8501`. Navegar pelas 9 seções
no menu lateral.

#### Testes automatizados

```bash
uv run pytest -v
```

Saída esperada: **51 passed**.

### Apêndice B — Mapa Spec → Implementação → Teste

| Spec | Implementação                                | Teste                          |
| ---- | -------------------------------------------- | ------------------------------ |
| 00   | `src/triagem_fuzzy/config.py`                | (constantes; sem teste próprio)|
| 01   | `src/triagem_fuzzy/ingestion.py`             | `tests/test_ingestion.py` (7)  |
| 02   | `src/triagem_fuzzy/preprocessing.py`         | `tests/test_preprocessing.py` (9) |
| 03   | `src/triagem_fuzzy/eda.py`                   | `tests/test_eda.py` (5)        |
| 04   | `src/triagem_fuzzy/ml/*`                     | `tests/test_ml.py` (7)         |
| 05   | `src/triagem_fuzzy/fuzzy/*`                  | `tests/test_fuzzy.py` (10)     |
| 06   | `src/triagem_fuzzy/articulation/*`           | `tests/test_articulation.py` (8) |
| 07   | `src/triagem_fuzzy/reporting.py`             | `tests/test_reporting.py` (5)  |

### Apêndice C — Glossário

- **Acurácia.** Proporção de predições corretas:
  $\mathrm{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$.
- **Agregação (fuzzy).** Combinação das saídas de múltiplas regras
  fuzzy em um único conjunto fuzzy (aqui, via operador `max`).
- **Aproximação A.** Forma de articulação em que ML e fuzzy operam
  independentemente e são comparados.
- **Aproximação B.** Forma de articulação em que a saída de um modelo
  é entrada do outro (aqui, `P(risco)` do ML alimenta o fuzzy).
- **Centroide.** Método de defuzzificação que calcula o “centro de
  massa” do conjunto fuzzy de saída.
- **Defuzzificação.** Conversão de um conjunto fuzzy em um valor
  numérico crisp.
- **F1-score.** Média harmônica entre precisão e recall:
  $F_1 = \frac{2 \cdot P \cdot R}{P + R}$.
- **Fuzzificação.** Conversão de um valor numérico em graus de
  pertinência a termos linguísticos.
- **Macro F1.** Média aritmética dos F1 por classe (não pondera por
  suporte).
- **Mamdani.** Tipo de sistema fuzzy em que o consequente é também um
  conjunto fuzzy (em contraste com Takagi-Sugeno, onde o consequente é
  uma função numérica).
- **Matriz de confusão.** Tabela $n \times n$ que cruza classes
  verdadeiras com classes preditas.
- **One-hot.** Codificação de variável categórica em $k$ colunas
  binárias.
- **Precisão.** $\frac{TP}{TP + FP}$ — dos preditos positivos, quantos
  são realmente positivos.
- **Random Forest.** Ensemble de árvores de decisão treinadas em
  amostras bootstrap.
- **Recall (sensibilidade).** $\frac{TP}{TP + FN}$ — dos positivos
  reais, quantos são capturados.
- **Stratified split.** Divisão treino/teste que preserva as
  proporções de cada classe.
- **T-norma.** Operador binário que generaliza o `AND` lógico para
  graus de pertinência (aqui, `min`).
- **Weighted F1.** Média ponderada dos F1 por classe (pondera pelo
  suporte).
