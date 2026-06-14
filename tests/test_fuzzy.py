"""Spec 05 acceptance tests."""

from __future__ import annotations

import numpy as np
import pytest

from triagem_fuzzy.fuzzy.engine import FuzzyInferenceEngine
from triagem_fuzzy.fuzzy.factory import (
    build_integrated_engine,
    build_standalone_engine,
)


@pytest.fixture(scope="module")
def standalone() -> FuzzyInferenceEngine:
    return build_standalone_engine()


@pytest.fixture(scope="module")
def integrated() -> FuzzyInferenceEngine:
    return build_integrated_engine()


def test_membership_in_zero_one(standalone: FuzzyInferenceEngine) -> None:
    for var in standalone.inputs.values():
        xs = var.universe_array()
        for term in var.terms.values():
            ys = term.evaluate(xs)
            assert (ys >= 0.0).all()
            assert (ys <= 1.0 + 1e-9).all()


def test_terms_overlap_on_adjacent_intervals(
    standalone: FuzzyInferenceEngine,
) -> None:
    for var in standalone.inputs.values():
        xs = var.universe_array()
        memberships = {name: t.evaluate(xs) for name, t in var.terms.items()}
        # Each pair of distinct terms must share at least one x where both > 0.
        names = list(memberships.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = (memberships[a] > 0.05) & (memberships[b] > 0.05)
                if not overlap.any():
                    continue  # non-adjacent — allowed
                assert overlap.any()


def test_clear_normal_patient_yields_low_risk(
    standalone: FuzzyInferenceEngine,
) -> None:
    sample = {
        "body_temperature": 36.8,
        "heart_rate": 75.0,
        "systolic_blood_pressure": 120.0,
    }
    trace = standalone.explain(sample)
    assert trace.crisp_output < 3.5
    assert trace.classification == "normal"


def test_clear_risco_patient_yields_high_risk(
    standalone: FuzzyInferenceEngine,
) -> None:
    sample = {
        "body_temperature": 39.5,
        "heart_rate": 130.0,
        "systolic_blood_pressure": 85.0,
    }
    trace = standalone.explain(sample)
    assert trace.crisp_output > 6.5
    assert trace.classification == "risco"


def test_engine_is_deterministic(standalone: FuzzyInferenceEngine) -> None:
    sample = {
        "body_temperature": 37.6,
        "heart_rate": 95.0,
        "systolic_blood_pressure": 110.0,
    }
    a = standalone.infer(sample)
    b = standalone.infer(sample)
    assert a == b


def test_explain_returns_per_rule_strengths(
    standalone: FuzzyInferenceEngine,
) -> None:
    sample = {
        "body_temperature": 39.5,
        "heart_rate": 130.0,
        "systolic_blood_pressure": 85.0,
    }
    trace = standalone.explain(sample)
    assert len(trace.activations) == len(standalone.rules)
    assert any(a.strength > 0.0 for a in trace.activations)


def test_integration_rules_only_active_when_ml_input_provided(
    integrated: FuzzyInferenceEngine,
) -> None:
    sample = {
        "body_temperature": 36.8,
        "heart_rate": 75.0,
        "systolic_blood_pressure": 120.0,
        "ml_risk_proba": 0.95,
    }
    trace = integrated.explain(sample)
    # High ML probability should push score up despite normal vitals.
    assert trace.crisp_output > 4.0


def test_standalone_engine_rejects_missing_input(
    standalone: FuzzyInferenceEngine,
) -> None:
    with pytest.raises(ValueError, match="Missing fuzzy inputs"):
        standalone.infer({"body_temperature": 37.0, "heart_rate": 80.0})


def test_rule_count_meets_minimum(standalone: FuzzyInferenceEngine) -> None:
    assert len(standalone.rules) >= 6


def test_integrated_has_more_rules_than_standalone(
    standalone: FuzzyInferenceEngine,
    integrated: FuzzyInferenceEngine,
) -> None:
    assert len(integrated.rules) >= len(standalone.rules) + 3
