"""Spec 05 — Fuzzy rule representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class FuzzyRule:
    antecedents: tuple[tuple[str, str], ...]
    consequent: tuple[str, str]
    op: Literal["AND", "OR"] = "AND"
    name: str = ""


@dataclass
class RuleBase:
    rules: list[FuzzyRule] = field(default_factory=list)

    def add(self, rule: FuzzyRule) -> None:
        self.rules.append(rule)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)
