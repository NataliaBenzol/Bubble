from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    def total_frames(self) -> int: ...

    def read_frames(self) -> Iterator[tuple[int, np.ndarray]]: ...

    def close(self) -> None: ...


@runtime_checkable
class ImageRepository(Protocol):
    @property
    def root(self) -> Path: ...

    def keys(self) -> list[Path]: ...

    def load(self, path: Path) -> np.ndarray | None: ...

    def save(self, path: Path, image: np.ndarray) -> None: ...