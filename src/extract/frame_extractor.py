from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from src.utils import get_config, get_data_path

logger = logging.getLogger(__name__)


class FrameExtractor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.target_fps = float(self.config["video"].get("frame_sampling_fps", 1.0))
        self.max_frames = int(self.config["video"].get("max_frames", 180))
        self.max_frame_width = int(self.config["video"].get("max_frame_width", 960))
        self.max_frame_height = int(self.config["video"].get("max_frame_height", 540))

        if self.target_fps <= 0:
            logger.warning("Invalid frame_sampling_fps=%s. Defaulting to 1.0", self.target_fps)
            self.target_fps = 1.0

        self.output_root = Path(
            get_data_path(self.config["paths"].get("interim_frames_dir", "data/interim/frames"))
        )
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _resize_if_needed(self, frame):
        h, w = frame.shape[:2]
        scale = min(
            self.max_frame_width / max(w, 1),
            self.max_frame_height / max(h, 1),
            1.0,
        )
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def _cleanup_old_frames(self, output_dir: Path) -> int:
        removed = 0
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for old_file in output_dir.glob(pattern):
                if old_file.is_file():
                    try:
                        old_file.unlink()
                        removed += 1
                    except Exception as e:
                        logger.warning("Could not remove old frame %s: %s", old_file, e)
        return removed

    def extract_frames(self, video_path: str) -> str:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_dir = self.output_root / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        removed_count = self._cleanup_old_frames(output_dir)
        if removed_count > 0:
            logger.info("Removed %d old frame files from %s", removed_count, output_dir)

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps is None or video_fps <= 0:
            logger.warning("Invalid FPS detected for '%s'. Defaulting to 1.0", video_file.name)
            video_fps = 1.0

        frame_interval = max(1, int(video_fps / self.target_fps))
        frame_count = 0
        saved_count = 0
        failed_writes = 0

        logger.info(
            "Extracting frames from %s | target_fps=%.2f | max_frames=%d",
            video_file.name,
            self.target_fps,
            self.max_frames,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                if saved_count >= self.max_frames:
                    logger.info("Reached max_frames=%d for %s", self.max_frames, video_file.name)
                    break

                timestamp = frame_count / video_fps if video_fps > 0 else 0.0
                frame_name = f"frame_{saved_count:04d}_{timestamp:.2f}s.jpg"
                frame_path = output_dir / frame_name

                frame = self._resize_if_needed(frame)
                ok = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                if ok:
                    saved_count += 1
                else:
                    failed_writes += 1
                    logger.warning("Failed to save frame: %s", frame_path)

            frame_count += 1

        cap.release()

        logger.info(
            "Saved %d frames to %s (failed writes: %d)",
            saved_count,
            output_dir,
            failed_writes,
        )
        return str(output_dir)