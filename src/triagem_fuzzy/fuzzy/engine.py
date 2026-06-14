"""Spec 05 — Mamdani fuzzy inference engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import skfuzzy as fuzz

from triagem_fuzzy import config
from triagem_fuzzy.fuzzy.rules import FuzzyRule, RuleBase
from triagem_fuzzy.fuzzy.variables import LinguisticVariable

logger = logging.getLogger(__name__)

DefuzzMethod = Literal["centroid", "mom", "bisector"]


@dataclass(frozen=True)
class RuleActivation:
    rule: FuzzyRule
    strength: float


@dataclass(frozen=True)
class FuzzyTrace:
    fuzzified: dict[str, dict[str, float]]
    activations: list[RuleActivation]
    aggregated: np.ndarray
    crisp_output: float
    classification: str


class FuzzyInferenceEngine:
    def __init__(
        self,
        inputs: list[LinguisticVariable],
        output: LinguisticVariable,
        rules: RuleBase,
        defuzz_method: DefuzzMethod = "centroid",
    ) -> None:
        self.inputs = {var.name: var for var in inputs}
        self.output = output
        self.rules = rules
        self.defuzz_method = defuzz_method
        self._universe = output.universe_array()

    # ── API ───────────────────────────────────────────────────────────

    def required_inputs(self) -> set[str]:
        return set(self.inputs.keys())

    def fuzzify(self, sample: dict[str, float]) -> dict[str, dict[str, float]]:
        self._check_sample(sample)
        return {
            name: var.fuzzify(sample[name]) for name, var in self.inputs.items()
        }

    def infer(self, sample: dict[str, float]) -> float:
        return self.explain(sample).crisp_output

    def classify(self, sample: dict[str, float]) -> str:
        return self.explain(sample).classification

    def explain(self, sample: dict[str, float]) -> FuzzyTrace:
        fuzzified = self.fuzzify(sample)
        activations: list[RuleActivation] = []
        consequent_strengths: dict[str, float] = {
            term: 0.0 for term in self.output.terms
        }

        for rule in self.rules:
            strength = self._rule_strength(rule, fuzzified)
            activations.append(RuleActivation(rule=rule, strength=strength))
            cons_var, cons_term = rule.consequent
            if cons_var != self.output.name:
                raise ValueError(
                    f"Rule consequent {cons_var} != output {self.output.name}"
                )
            if strength > consequent_strengths[cons_term]:
                consequent_strengths[cons_term] = strength

        aggregated = self._aggregate(consequent_strengths)
        crisp = self._defuzzify(aggregated)
        classification = self._classify_score(crisp)
        return FuzzyTrace(
            fuzzified=fuzzified,
            activations=activations,
            aggregated=aggregated,
            crisp_output=crisp,
            classification=classification,
        )

    # ── internals ─────────────────────────────────────────────────────

    def _check_sample(self, sample: dict[str, float]) -> None:
        missing = self.required_inputs() - set(sample.keys())
        if missing:
            raise ValueError(
                f"Missing fuzzy inputs: {sorted(missing)}; got {sorted(sample.keys())}"
            )

    def _rule_strength(
        self,
        rule: FuzzyRule,
        fuzzified: dict[str, dict[str, float]],
    ) -> float:
        degrees = [fuzzified[var][term] for var, term in rule.antecedents]
        if not degrees:
            return 0.0
        if rule.op == "AND":
            return float(min(degrees))
        return float(max(degrees))

    def _aggregate(self, consequent_strengths: dict[str, float]) -> np.ndarray:
        result = np.zeros_like(self._universe)
        for term_name, strength in consequent_strengths.items():
            if strength <= 0.0:
                continue
            mf_curve = self.output.terms[term_name].evaluate(self._universe)
            clipped = np.minimum(mf_curve, strength)
            result = np.maximum(result, clipped)
        return result

    def _defuzzify(self, aggregated: np.ndarray) -> float:
        if not aggregated.any():
            # No rule fired with non-zero strength — return midpoint of universe.
            return float((self._universe[0] + self._universe[-1]) / 2.0)
        return float(
            fuzz.defuzz(self._universe, aggregated, self.defuzz_method)
        )

    @staticmethod
    def _classify_score(score: float) -> str:
        for label, (lo, hi) in config.FUZZY_DECISION_BANDS.items():
            # `normal` band is [lo, hi); `risco` band is closed on top.
            if label == "risco":
                if lo <= score <= hi:
                    return label
            elif lo <= score < hi:
                return label
        # Fallback (numerical edge cases).
        return "risco"
