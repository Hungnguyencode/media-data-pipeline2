from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.ingest.base_ingestor import BaseIngestor
from src.utils import get_config, infer_has_audio_from_video_path


class LocalFileIngestor(BaseIngestor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()

    def ingest(self, source: str) -> Dict[str, Any]:
        video_path = Path(source).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Local video file not found: {video_path}")

        title = video_path.stem.replace("_", " ").replace("-", " ").strip() or video_path.name
        has_audio = infer_has_audio_from_video_path(str(video_path))

        return {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "source_platform": "local",
            "source_url": "",
            "video_title": title,
            "video_description": "",
            "thumbnail_url": "",
            "video_tags": [],
            "ingest_method": "local_file",
            "has_audio": has_audio,
        }