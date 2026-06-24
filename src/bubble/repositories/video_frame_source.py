from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


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

    def read_frames(self) -> Iterator[tuple[int, np.ndarray]]:
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

    def __enter__(self) -> VideoFrameSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()