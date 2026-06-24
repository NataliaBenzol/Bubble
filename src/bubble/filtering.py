from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bubble.repositories import FileSystemImageRepository, ImageRepository


def is_black_image(
    img: np.ndarray,
    threshold: int = 10,
    min_nonblack_ratio: float = 0.02,
) -> bool:
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (np.sum(gray > threshold) / gray.size) < min_nonblack_ratio


def has_bubbles(
    img: np.ndarray,
    bubble_threshold: int = 80,
    min_bubble_area: int = 50,
    min_bubble_count: int = 3,
) -> bool:
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin = 5
    if h <= 2 * margin or w <= 2 * margin:
        return False
    gray_cropped = gray[margin : h - margin, margin : w - margin]
    _, binary = cv2.threshold(gray_cropped, bubble_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len([cnt for cnt in contours if cv2.contourArea(cnt) > min_bubble_area]) >= min_bubble_count


@dataclass(frozen=True, slots=True)
class FilterConfig:
    black_threshold: int = 10
    min_nonblack_ratio: float = 0.02
    bubble_threshold: int = 80
    min_bubble_area: int = 50
    min_bubble_count: int = 3


class BubbleFilter:
    def __init__(
        self,
        source: ImageRepository,
        destination: ImageRepository,
        config: FilterConfig,
    ) -> None:
        self._source = source
        self._destination = destination
        self._config = config

    def _should_keep(self, img: np.ndarray) -> bool:
        if is_black_image(img, self._config.black_threshold, self._config.min_nonblack_ratio):
            return False
        return has_bubbles(
            img,
            self._config.bubble_threshold,
            self._config.min_bubble_area,
            self._config.min_bubble_count,
        )

    def filter(self) -> tuple[int, int]:
        kept = 0
        removed = 0

        for src_path in self._source.keys():
            img = self._source.load(src_path)
            if not self._should_keep(img):
                removed += 1
                continue

            relative_path = src_path.relative_to(self._source.root)
            self._destination.save(relative_path, img)
            kept += 1

        return kept, removed


def filter_images(
    input_folder: str,
    output_folder: str,
) -> tuple[int, int]:
    source = FileSystemImageRepository(input_folder)
    destination = FileSystemImageRepository(output_folder)
    bubble_filter = BubbleFilter(
        source=source,
        destination=destination,
        config=FilterConfig(),
    )
    kept, removed = bubble_filter.filter()
    print(f"Kept: {kept}, Removed: {removed}")
    return kept, removed