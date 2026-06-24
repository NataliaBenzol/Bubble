from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from bubble.config import ExtractionConfig
from bubble.repositories import FileSystemImageRepository, FrameSource, ImageRepository


def is_informative_frame(
    frame: np.ndarray,
    threshold: int,
    min_nonblack_ratio: float,
) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    non_black_pixels = int(np.sum(gray > threshold))
    total_pixels = gray.size
    return (non_black_pixels / total_pixels) > min_nonblack_ratio


class VideoFrameSource:
    def __init__(self, video_path: str) -> None:
        self._video_path = video_path
        self._cap: cv2.VideoCapture | None = cv2.VideoCapture(video_path)
        if self._cap is None or not self._cap.isOpened():
            if self._cap is not None:
                self._cap.release()
            raise RuntimeError(f"Cannot open video: {video_path}")

    def total_frames(self) -> int:
        if self._cap is None:
            raise RuntimeError("VideoCapture is closed")
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frames(self):
        if self._cap is None:
            raise RuntimeError("VideoCapture is closed")
        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                yield frame_idx, frame
                frame_idx += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoFrameSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class FrameExtractor:
    def __init__(
        self,
        source: FrameSource,
        repository: ImageRepository,
        config: ExtractionConfig,
    ) -> None:
        self._source = source
        self._repository = repository
        self._config = config

    def _compute_step(self) -> int:
        total = self._source.total_frames()
        if total <= 0 or self._config.max_frames_per_video <= 0:
            return 1
        return max(1, total // self._config.max_frames_per_video)

    def extract(self, output_prefix: str = "") -> int:
        step = self._compute_step()
        saved_count = 0

        for frame_idx, frame in self._source.read_frames():
            if saved_count >= self._config.max_frames_per_video:
                break

            if frame_idx % step != 0:
                continue

            if not is_informative_frame(
                frame,
                self._config.brightness_threshold,
                self._config.min_nonblack_ratio,
            ):
                continue

            relative_path = Path(f"{output_prefix}frame_{saved_count:05d}.jpg")
            self._repository.save(relative_path, frame)
            saved_count += 1

        return saved_count


def process_videos(
    video_list: list[tuple[str, str]],
    output_root: str,
    config: ExtractionConfig,
) -> int:
    os.makedirs(output_root, exist_ok=True)
    frames_per_video = max(1, config.max_frames_per_video // len(video_list))
    total_saved = 0

    for video_path, solution_name in video_list:
        video_output_dir = os.path.join(output_root, solution_name)
        os.makedirs(video_output_dir, exist_ok=True)

        with VideoFrameSource(video_path) as source:
            repository = FileSystemImageRepository(video_output_dir)
            extractor = FrameExtractor(
                source=source,
                repository=repository,
                config=ExtractionConfig(
                    max_frames_per_video=frames_per_video,
                    brightness_threshold=config.brightness_threshold,
                    min_nonblack_ratio=config.min_nonblack_ratio,
                ),
            )
            saved = extractor.extract()
            total_saved += saved
            print(f"{Path(video_path).name}: saved {saved} frames")

    print(f"\nTotal saved: {total_saved} (target: ~{config.max_frames_per_video})")
    return total_saved