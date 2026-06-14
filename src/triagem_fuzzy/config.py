"""Central configuration: paths, constants, label maps, decision bands.

Single source of truth referenced by every module (Spec 00 / I5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset"
RAW_DATA_PATH: Final[Path] = DATASET_DIR / "triagem_fuzzy.csv"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "output"

RANDOM_STATE: Final[int] = 50

RAW_TRIAGE_LEVELS: Final[tuple[int, ...]] = (0, 1, 2, 3)

# D1 — Label remap: 0→normal, 1→atencao, 2 & 3→risco.
TRIAGE_LABEL_MAP: Final[dict[int, int]] = {0: 0, 1: 1, 2: 2, 3: 2}
TRIAGE_LABELS: Final[dict[int, str]] = {0: "normal", 1: "atencao", 2: "risco"}
TRIAGE_LABELS_PT: Final[dict[int, str]] = {
    0: "normal",
    1: "atenção",
    2: "risco",
}

ARRIVAL_MODES: Final[tuple[str, ...]] = ("walk_in", "ambulance", "wheelchair")

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "age",
    "heart_rate",
    "systolic_blood_pressure",
    "oxygen_saturation",
    "body_temperature",
    "pain_level",
    "chronic_disease_count",
    "previous_er_visits",
)

# Spec 01 — authoritative schema. (min, max) inclusive ranges.
FEATURE_RANGES: Final[dict[str, tuple[float, float]]] = {
    "age": (0.0, 120.0),
    "heart_rate": (20.0, 250.0),
    "systolic_blood_pressure": (50.0, 250.0),
    "oxygen_saturation": (50.0, 100.0),
    "body_temperature": (30.0, 43.0),
    "pain_level": (0.0, 10.0),
    "chronic_disease_count": (0.0, 20.0),
    "previous_er_visits": (0.0, 50.0),
}

EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    *NUMERIC_FEATURES,
    "arrival_mode",
    "triage_level",
)

# Spec 05 — score → class decoder.
FUZZY_DECISION_BANDS: Final[dict[str, tuple[float, float]]] = {
    "normal": (0.0, 3.5),
    "atencao": (3.5, 6.5),
    "risco": (6.5, 10.0),
}

# Spec 06 — which ML probability flows into fuzzy in Approach B.
# "risco_only" → P(class=2); "atencao_or_risco" → P(1) + P(2).
ML_PROBA_AGGREGATION: Final[str] = "risco_only"
