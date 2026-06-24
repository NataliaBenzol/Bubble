from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FileSystemImageRepository:
    EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def keys(self) -> list[Path]:
        paths: list[Path] = []
        for ext in self.EXTENSIONS:
            paths.extend(self._root.rglob(f"*{ext}"))
            paths.extend(self._root.rglob(f"*{ext.upper()}"))
        return sorted(set(paths))

    def load(self, path: Path) -> np.ndarray | None:
        return cv2.imread(str(path))

    def save(self, path: Path, image: np.ndarray) -> None:
        absolute = self._root / path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(absolute), image)