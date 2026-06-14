"""Spec 05 — Linguistic variables and membership functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import skfuzzy as fuzz

Kind = Literal["tri", "trap"]


@dataclass(frozen=True)
class MembershipFunction:
    name: str
    kind: Kind
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.kind == "tri" and len(self.params) != 3:
            raise ValueError(
                f"tri MF '{self.name}' requires 3 params, got {self.params}"
            )
        if self.kind == "trap" and len(self.params) != 4:
            raise ValueError(
                f"trap MF '{self.name}' requires 4 params, got {self.params}"
            )

    def evaluate(self, x: np.ndarray | float) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        if self.kind == "tri":
            return fuzz.trimf(arr, list(self.params))
        return fuzz.trapmf(arr, list(self.params))


@dataclass(frozen=True)
class LinguisticVariable:
    name: str
    universe: tuple[float, float]
    terms: dict[str, MembershipFunction]
    resolution: int = 1001

    def universe_array(self) -> np.ndarray:
        return np.linspace(self.universe[0], self.universe[1], self.resolution)

    def membership(self, term: str, value: float) -> float:
        clipped = float(np.clip(value, self.universe[0], self.universe[1]))
        mf = self.terms[term]
        return float(mf.evaluate(clipped)[0])

    def fuzzify(self, value: float) -> dict[str, float]:
        return {term: self.membership(term, value) for term in self.terms}
