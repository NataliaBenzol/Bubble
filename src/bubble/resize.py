from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bubble.domain import ImageSize
from bubble.repositories import FileSystemImageRepository, ImageRepository


def resize_with_padding(
    img: np.ndarray,
    target_size: ImageSize,
    padding_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    h, w = img.shape[:2]
    target_w = target_size.width
    target_h = target_size.height
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = img_resized
    return canvas


@dataclass(frozen=True, slots=True)
class ResizerConfig:
    output_size: ImageSize
    keep_aspect_ratio: bool = True
    padding_color: tuple[int, int, int] = (0, 0, 0)


class ImageResizer:
    def __init__(
        self,
        source: ImageRepository,
        destination: ImageRepository,
        config: ResizerConfig,
    ) -> None:
        self._source = source
        self._destination = destination
        self._config = config

    def resize(self) -> int:
        resized_count = 0
        target_tuple = self._config.output_size.as_tuple()

        for src_path in self._source.keys():
            img = self._source.load(src_path)
            if img is None:
                continue

            if self._config.keep_aspect_ratio:
                img_resized = resize_with_padding(
                    img,
                    target_size=self._config.output_size,
                    padding_color=self._config.padding_color,
                )
            else:
                img_resized = cv2.resize(img, target_tuple, interpolation=cv2.INTER_AREA)

            relative_path = src_path.relative_to(self._source.root)
            self._destination.save(relative_path, img_resized)
            resized_count += 1

        return resized_count


def resize_images(
    input_folder: str,
    output_folder: str,
    output_size: ImageSize,
) -> int:
    source = FileSystemImageRepository(input_folder)
    destination = FileSystemImageRepository(output_folder)
    resizer = ImageResizer(
        source=source,
        destination=destination,
        config=ResizerConfig(output_size=output_size),
    )
    return resizer.resize()