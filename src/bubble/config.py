from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bubble.domain import ImageSize


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    brightness_threshold: int = 10
    min_nonblack_ratio: float = 0.05
    max_frames_per_video: int = 4000


@dataclass(frozen=True, slots=True)
class TrainConfig:
    data_dir: str
    output_dir: str
    architectures: tuple[str, ...] = ("resnet50",)
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    image_size: ImageSize = ImageSize(224, 224)
    num_classes: int = 7
    random_state: int = 42
    patience: int = 10
    output_path: Path = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image_size, ImageSize):
            raise TypeError(f"image_size must be ImageSize, got {type(self.image_size).__name__}")
        if not self.architectures:
            raise ValueError("architectures must contain at least one backbone name")
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "output_path", path)


def load_extraction_config(path: str | Path = "configs/default.toml") -> ExtractionConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    extraction_data = data.get("extraction", {})
    return ExtractionConfig(**extraction_data)


def load_train_config(path: str | Path = "configs/default.toml", **overrides: Any) -> TrainConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    train_data = data.get("train", {})
    train_data.update(overrides)
    if "image_size" in train_data and isinstance(train_data["image_size"], list):
        train_data["image_size"] = ImageSize(*train_data["image_size"])
    if "architectures" in train_data and isinstance(train_data["architectures"], list):
        train_data["architectures"] = tuple(train_data["architectures"])
    return TrainConfig(**train_data)