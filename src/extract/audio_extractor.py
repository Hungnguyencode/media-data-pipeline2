from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import get_config, get_data_path

logger = logging.getLogger(__name__)


class AudioExtractor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.output_dir = Path(
            get_data_path(self.config["paths"].get("interim_audio_dir", "data/interim/audio"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(self, video_path: str) -> str:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg and add it to PATH.")

        output_path = self.output_dir / f"{video_file.stem}.wav"

        if output_path.exists():
            logger.info("Audio output already exists and will be overwritten: %s", output_path)

        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_file),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]

        logger.info("Extracting audio from %s to %s", video_file.name, output_path)

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            logger.debug("ffmpeg stdout: %s", result.stdout)
            logger.debug("ffmpeg stderr: %s", result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error("Audio extraction failed for %s: %s", video_file.name, e.stderr)
            raise RuntimeError(f"Failed to extract audio from {video_path}: {e.stderr}") from e

        if not output_path.exists():
            raise RuntimeError(f"Audio extraction completed but output file not found: {output_path}")

        logger.info("Audio extracted successfully: %s", output_path)
        return str(output_path)