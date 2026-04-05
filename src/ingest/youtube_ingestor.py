from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL

from src.ingest.base_ingestor import BaseIngestor
from src.utils import get_config, get_data_path, sanitize_filename_component

logger = logging.getLogger(__name__)


class _YTDLPLogger:
    def debug(self, msg):
        if msg and str(msg).strip():
            logger.debug("yt-dlp: %s", msg)

    def info(self, msg):
        if msg and str(msg).strip():
            logger.info("yt-dlp: %s", msg)

    def warning(self, msg):
        if msg and str(msg).strip():
            logger.warning("yt-dlp: %s", msg)

    def error(self, msg):
        if msg and str(msg).strip():
            logger.error("yt-dlp: %s", msg)


class YouTubeIngestor(BaseIngestor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.raw_dir = Path(get_data_path(self.config["paths"].get("raw_dir", "data/raw")))
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _get_ffprobe_path(self) -> str:
        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            raise RuntimeError("ffprobe not found in PATH. Please install ffmpeg/ffprobe and add them to PATH.")
        return ffprobe_path

    def _has_audio_stream(self, file_path: Path) -> bool:
        ffprobe_path = self._get_ffprobe_path()
        cmd = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "a",
            "-show_streams",
            "-of", "json",
            str(file_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        import json
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        return bool(streams)

    def _extract_video_id(self, video_url: str) -> str:
        parsed = urlparse((video_url or "").strip())
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""
        query = parse_qs(parsed.query or "")

        if "youtu.be" in host:
            return path.strip("/").split("/")[0].strip()

        if "youtube.com" in host or "www.youtube.com" in host or "m.youtube.com" in host:
            if "/shorts/" in path:
                raise ValueError("YouTube Shorts URLs are not supported in this version")
            if path == "/watch":
                return (query.get("v") or [""])[0].strip()

        return ""

    def canonicalize_url(self, video_url: str) -> str:
        video_id = self._extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL or missing video ID")
        return f"https://www.youtube.com/watch?v={video_id}"

    def validate_url(self, video_url: str) -> str:
        raw = (video_url or "").strip()
        if not raw:
            raise ValueError("video_url must not be empty")

        parsed = urlparse(raw)
        host = (parsed.netloc or "").lower()

        if not host:
            raise ValueError("Invalid URL")

        if not any(domain in host for domain in ("youtube.com", "youtu.be")):
            raise ValueError("Only YouTube URLs are supported in this version")

        if "list=" in (parsed.query or ""):
            logger.info("Playlist parameters detected. Canonicalizing to single watch URL.")

        return self.canonicalize_url(raw)

    def _safe_output_stem(self, title: str, video_id: str) -> str:
        safe_title = sanitize_filename_component(title or "youtube_video", fallback="youtube_video")
        safe_title = safe_title[:70].strip("_") or "youtube_video"
        return f"{safe_title}_{video_id}"

    def _extract_metadata(self, canonical_url: str) -> Dict[str, Any]:
        logger.info("YouTube ingest: extracting metadata for %s", canonical_url)

        ydl_opts = {
            "quiet": True,
            "no_warnings": False,
            "noplaylist": True,
            "logger": _YTDLPLogger(),
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(canonical_url, download=False)

        if not info:
            raise RuntimeError("Could not extract YouTube metadata")
        if info.get("_type") == "playlist":
            raise ValueError("Playlist URLs are not supported in this version")

        return info

    def _resolve_downloaded_file(self, prepared: Path, output_stem: str) -> Path:
        if prepared.suffix.lower() != ".mp4":
            mp4_candidate = prepared.with_suffix(".mp4")
            if mp4_candidate.exists():
                prepared = mp4_candidate

        if prepared.exists():
            return prepared.resolve()

        candidates = sorted(self.raw_dir.glob(f"{output_stem}.*"))
        if not candidates:
            raise RuntimeError("Download finished but output file was not found")

        return candidates[0].resolve()

    def _download_once(self, canonical_url: str, output_stem: str, format_selector: str) -> Path:
        outtmpl = str(self.raw_dir / f"{output_stem}.%(ext)s")
        ydl_opts = {
            "format": format_selector,
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": False,
            "logger": _YTDLPLogger(),
            "retries": 3,
            "fragment_retries": 3,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(canonical_url, download=True)
            prepared = Path(ydl.prepare_filename(info))

        return self._resolve_downloaded_file(prepared, output_stem)

    def _download_video(self, canonical_url: str, output_stem: str) -> Path:
        logger.info("YouTube ingest: starting download for %s", canonical_url)

        format_candidates = [
            "best[ext=mp4][acodec!=none]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
            "bestvideo[ext=mp4]+bestaudio/best",
            "best",
        ]

        last_error: Exception | None = None

        for idx, fmt in enumerate(format_candidates, start=1):
            try:
                logger.info("YouTube ingest: trying format strategy %d", idx)
                downloaded_path = self._download_once(canonical_url, output_stem, fmt)

                if self._has_audio_stream(downloaded_path):
                    logger.info("YouTube ingest: download finished with audio -> %s", downloaded_path)
                    return downloaded_path

                logger.warning(
                    "YouTube ingest: downloaded file has no audio stream with strategy %d: %s",
                    idx,
                    downloaded_path,
                )
            except Exception as e:
                last_error = e
                logger.warning("YouTube ingest: strategy %d failed: %s", idx, e)

        if last_error:
            raise RuntimeError(f"Failed to download YouTube video with audio: {last_error}")
        raise RuntimeError("Failed to download YouTube video with audio")

    def ingest(self, video_url: str) -> Dict[str, Any]:
        canonical_url = self.validate_url(video_url)
        info = self._extract_metadata(canonical_url)

        video_id = str(info.get("id") or "").strip()
        if not video_id:
            raise RuntimeError("Could not determine YouTube video ID")

        title = str(info.get("title") or f"youtube_{video_id}").strip()
        description = str(info.get("description") or "").strip()
        thumbnail_url = str(info.get("thumbnail") or "").strip()
        tags = info.get("tags") or []
        if not isinstance(tags, list):
            tags = []

        uploader = str(info.get("uploader") or "").strip()
        channel = str(info.get("channel") or "").strip()
        duration_sec = info.get("duration")

        normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        seen = {t.lower() for t in normalized_tags}
        for maybe_extra in (uploader, channel):
            if maybe_extra and maybe_extra.lower() not in seen:
                normalized_tags.append(maybe_extra)
                seen.add(maybe_extra.lower())

        output_stem = self._safe_output_stem(title, video_id)
        downloaded_path = self._download_video(canonical_url, output_stem)

        return {
            "video_path": str(downloaded_path),
            "video_name": downloaded_path.name,
            "source_platform": "youtube",
            "source_url": canonical_url,
            "video_title": title,
            "video_description": description,
            "thumbnail_url": thumbnail_url,
            "video_tags": normalized_tags[:15],
            "youtube_video_id": video_id,
            "ingest_method": "youtube_url",
            "duration_sec": duration_sec,
            "has_audio": True,
        }