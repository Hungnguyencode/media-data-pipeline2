from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import get_config, get_data_path

logger = logging.getLogger(__name__)


class NoAudioStreamError(RuntimeError):
    pass


class AudioExtractor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.output_dir = Path(
            get_data_path(self.config["paths"].get("interim_audio_dir", "data/interim/audio"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_ffmpeg_path(self) -> str:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg and add it to PATH.")
        return ffmpeg_path

    def _get_ffprobe_path(self) -> str:
        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            raise RuntimeError("ffprobe not found in PATH. Please install ffmpeg/ffprobe and add them to PATH.")
        return ffprobe_path

    def _probe_audio_streams(self, video_file: Path) -> list[dict]:
        ffprobe_path = self._get_ffprobe_path()

        cmd = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_streams",
            "-of",
            "json",
            str(video_file),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout or "{}")
            streams = data.get("streams", [])
            if isinstance(streams, list):
                return streams
            return []
        except subprocess.CalledProcessError as e:
            logger.error("ffprobe failed for %s: %s", video_file.name, e.stderr)
            raise RuntimeError(f"Failed to inspect audio streams for {video_file}: {e.stderr}") from e
        except json.JSONDecodeError as e:
            logger.error("ffprobe returned invalid JSON for %s", video_file.name)
            raise RuntimeError(f"Failed to parse ffprobe output for {video_file}: {e}") from e

    def has_audio_stream(self, video_path: str) -> bool:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        streams = self._probe_audio_streams(video_file)
        return len(streams) > 0

    def extract_audio(self, video_path: str) -> str:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ffmpeg_path = self._get_ffmpeg_path()
        output_path = self.output_dir / f"{video_file.stem}.wav"

        if output_path.exists():
            logger.info("Audio output already exists and will be overwritten: %s", output_path)

        audio_streams = self._probe_audio_streams(video_file)
        if not audio_streams:
            logger.warning("No audio stream detected in video: %s", video_file.name)
            raise NoAudioStreamError(f"No audio stream found in video: {video_path}")

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