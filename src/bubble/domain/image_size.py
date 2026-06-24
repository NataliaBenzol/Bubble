from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError(
                f"ImageSize dimensions must be integers, got width={self.width!r}, height={self.height!r}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"ImageSize dimensions must be positive, got width={self.width}, height={self.height}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> ImageSize:
        if not isinstance(data, dict):
            raise ValueError(f"ImageSize.from_dict expects a dict, got {type(data).__name__}")
        missing = {"width", "height"} - data.keys()
        if missing:
            raise ValueError(f"ImageSize dict is missing required keys: {sorted(missing)}")
        return cls(width=int(data["width"]), height=int(data["height"]))

    def as_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)

    def to_dict(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}