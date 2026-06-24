from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .concentration import Concentration


@dataclass(frozen=True, slots=True)
class FrameSample:
    path: Path
    concentration: Concentration

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))

    @property
    def class_id(self) -> int:
        return self.concentration.class_id

    @property
    def label(self) -> str:
        return self.concentration.label

    def as_tuple(self) -> tuple[str, int]:
        return (str(self.path), self.class_id)