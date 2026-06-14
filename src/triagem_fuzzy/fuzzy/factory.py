"""Spec 05 — Default variable / rule definitions and engine builders."""

from __future__ import annotations

from triagem_fuzzy.fuzzy.engine import FuzzyInferenceEngine
from triagem_fuzzy.fuzzy.rules import FuzzyRule, RuleBase
from triagem_fuzzy.fuzzy.variables import LinguisticVariable, MembershipFunction


def _body_temperature() -> LinguisticVariable:
    return LinguisticVariable(
        name="body_temperature",
        universe=(34.0, 42.0),
        terms={
            "baixa": MembershipFunction("baixa", "trap", (34.0, 34.0, 35.5, 36.3)),
            "normal": MembershipFunction("normal", "tri", (36.0, 36.8, 37.5)),
            "alta": MembershipFunction("alta", "trap", (37.2, 38.0, 42.0, 42.0)),
        },
    )


def _heart_rate() -> LinguisticVariable:
    return LinguisticVariable(
        name="heart_rate",
        universe=(30.0, 200.0),
        terms={
            "baixa": MembershipFunction("baixa", "trap", (30.0, 30.0, 50.0, 65.0)),
            "normal": MembershipFunction("normal", "tri", (60.0, 80.0, 100.0)),
            "alta": MembershipFunction("alta", "trap", (95.0, 110.0, 200.0, 200.0)),
        },
    )


def _systolic_bp() -> LinguisticVariable:
    return LinguisticVariable(
        name="systolic_blood_pressure",
        universe=(60.0, 220.0),
        terms={
            "baixa": MembershipFunction("baixa", "trap", (60.0, 60.0, 85.0, 100.0)),
            "normal": MembershipFunction("normal", "tri", (95.0, 120.0, 140.0)),
            "alta": MembershipFunction("alta", "trap", (135.0, 150.0, 220.0, 220.0)),
        },
    )


def _risk_score() -> LinguisticVariable:
    return LinguisticVariable(
        name="risk_score",
        universe=(0.0, 10.0),
        terms={
            "baixo": MembershipFunction("baixo", "trap", (0.0, 0.0, 2.0, 4.0)),
            "medio": MembershipFunction("medio", "tri", (3.0, 5.0, 7.0)),
            "alto": MembershipFunction("alto", "trap", (6.0, 8.0, 10.0, 10.0)),
        },
    )


def _ml_risk_proba() -> LinguisticVariable:
    return LinguisticVariable(
        name="ml_risk_proba",
        universe=(0.0, 1.0),
        terms={
            "baixo": MembershipFunction("baixo", "trap", (0.0, 0.0, 0.20, 0.40)),
            "medio": MembershipFunction("medio", "tri", (0.30, 0.50, 0.70)),
            "alto": MembershipFunction("alto", "trap", (0.60, 0.80, 1.0, 1.0)),
        },
    )


def _standalone_rules() -> RuleBase:
    R = lambda i, ants, cons: FuzzyRule(  # noqa: E731
        antecedents=tuple(ants),
        consequent=cons,
        op="AND",
        name=f"R{i}",
    )
    rb = RuleBase()
    rb.add(R(1, [("body_temperature", "alta"), ("heart_rate", "alta")],
            ("risk_score", "alto")))
    rb.add(R(2, [("body_temperature", "alta"),
                 ("systolic_blood_pressure", "baixa")],
            ("risk_score", "alto")))
    rb.add(R(3, [("heart_rate", "alta"),
                 ("systolic_blood_pressure", "baixa")],
            ("risk_score", "alto")))
    rb.add(R(4, [("body_temperature", "normal"),
                 ("heart_rate", "normal"),
                 ("systolic_blood_pressure", "normal")],
            ("risk_score", "baixo")))
    rb.add(R(5, [("body_temperature", "baixa"),
                 ("systolic_blood_pressure", "baixa")],
            ("risk_score", "alto")))
    rb.add(R(6, [("heart_rate", "baixa"),
                 ("systolic_blood_pressure", "normal")],
            ("risk_score", "medio")))
    rb.add(R(7, [("body_temperature", "alta"),
                 ("heart_rate", "normal"),
                 ("systolic_blood_pressure", "normal")],
            ("risk_score", "medio")))
    rb.add(R(8, [("heart_rate", "alta"),
                 ("systolic_blood_pressure", "normal")],
            ("risk_score", "medio")))
    rb.add(R(9, [("body_temperature", "normal"),
                 ("systolic_blood_pressure", "alta")],
            ("risk_score", "medio")))
    return rb


def _integration_rules(rb: RuleBase) -> RuleBase:
    R = lambda i, ants, cons: FuzzyRule(  # noqa: E731
        antecedents=tuple(ants),
        consequent=cons,
        op="AND",
        name=f"R{i}",
    )
    rb.add(R(10, [("ml_risk_proba", "alto")], ("risk_score", "alto")))
    rb.add(R(11, [("ml_risk_proba", "medio"),
                  ("systolic_blood_pressure", "baixa")],
            ("risk_score", "alto")))
    rb.add(R(12, [("ml_risk_proba", "baixo"),
                  ("body_temperature", "normal")],
            ("risk_score", "baixo")))
    return rb


def build_standalone_engine() -> FuzzyInferenceEngine:
    return FuzzyInferenceEngine(
        inputs=[_body_temperature(), _heart_rate(), _systolic_bp()],
        output=_risk_score(),
        rules=_standalone_rules(),
    )


def build_integrated_engine() -> FuzzyInferenceEngine:
    rules = _integration_rules(_standalone_rules())
    return FuzzyInferenceEngine(
        inputs=[
            _body_temperature(),
            _heart_rate(),
            _systolic_bp(),
            _ml_risk_proba(),
        ],
        output=_risk_score(),
        rules=rules,
    )
