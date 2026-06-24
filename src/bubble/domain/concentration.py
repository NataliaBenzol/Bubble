from __future__ import annotations

from enum import Enum


class Concentration(Enum):
    SOLUTION_0PCT = 0
    SOLUTION_5PCT = 1
    SOLUTION_12_5PCT = 2
    SOLUTION_25PCT = 3
    SOLUTION_50PCT = 4
    SOLUTION_75PCT = 5
    SOLUTION_100PCT = 6

    @property
    def class_id(self) -> int:
        return self.value

    @property
    def label(self) -> str:
        return _LABEL_BY_CONCENTRATION[self]

    @classmethod
    def from_class_id(cls, class_id: int) -> Concentration:
        for member in cls:
            if member.value == class_id:
                return member
        raise ValueError(f"Unknown class_id: {class_id}")

    @classmethod
    def from_label(cls, label: str) -> Concentration:
        for member, lbl in _LABEL_BY_CONCENTRATION.items():
            if lbl == label:
                return member
        raise ValueError(f"Unknown concentration label: {label!r}")

    @classmethod
    def all_labels(cls) -> list[str]:
        return [_LABEL_BY_CONCENTRATION[m] for m in cls]

    @classmethod
    def class_mapping(cls) -> dict[str, int]:
        return {m.label: m.class_id for m in cls}


_LABEL_BY_CONCENTRATION: dict[Concentration, str] = {
    Concentration.SOLUTION_0PCT: "solution_0pct",
    Concentration.SOLUTION_5PCT: "solution_5pct",
    Concentration.SOLUTION_12_5PCT: "solution_12.5pct",
    Concentration.SOLUTION_25PCT: "solution_25pct",
    Concentration.SOLUTION_50PCT: "solution_50pct",
    Concentration.SOLUTION_75PCT: "solution_75pct",
    Concentration.SOLUTION_100PCT: "solution_100pct",
}