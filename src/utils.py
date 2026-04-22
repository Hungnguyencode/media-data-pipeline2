from __future__ import annotations

import gc
import hashlib
import json
import logging
import logging.config
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
import yaml
import shutil

_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_VIDEO_CATALOG_CACHE: Optional[List[Dict[str, Any]]] = None


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_config_path(relative_path: str = "configs/config.yaml") -> Path:
    return get_project_root() / relative_path


def get_data_path(relative_path: str) -> Path:
    return get_project_root() / relative_path


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(force_reload: bool = False) -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or force_reload:
        _CONFIG_CACHE = load_yaml(get_config_path("configs/config.yaml"))
    return _CONFIG_CACHE


def reload_config() -> Dict[str, Any]:
    return get_config(force_reload=True)


def get_video_catalog_path(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or get_config()
    relative_path = cfg.get("paths", {}).get("video_catalog_path", "data/video_catalog.json")
    return get_data_path(relative_path)


def load_video_catalog(
    force_reload: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    global _VIDEO_CATALOG_CACHE

    if _VIDEO_CATALOG_CACHE is not None and not force_reload:
        return _VIDEO_CATALOG_CACHE

    catalog_path = get_video_catalog_path(config)
    if not catalog_path.exists():
        _VIDEO_CATALOG_CACHE = []
        return _VIDEO_CATALOG_CACHE

    with open(catalog_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Video catalog must be a list, got: {type(raw)}")

    cleaned: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            cleaned.append(item)

    _VIDEO_CATALOG_CACHE = cleaned
    return _VIDEO_CATALOG_CACHE


def save_video_catalog(
    catalog: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    global _VIDEO_CATALOG_CACHE

    catalog_path = get_video_catalog_path(config)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned: List[Dict[str, Any]] = []
    for item in catalog:
        if isinstance(item, dict):
            cleaned.append(item)

    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    _VIDEO_CATALOG_CACHE = cleaned
    return catalog_path


def get_video_catalog_entry(
    video_name: str,
    force_reload: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    target = (video_name or "").strip()
    if not target:
        return None

    catalog = load_video_catalog(force_reload=force_reload, config=config)
    for item in catalog:
        name = str(item.get("video_name", "")).strip()
        if name == target:
            return dict(item)

    return None


def _now_iso_string() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _normalize_catalog_tags(tags: Any) -> List[str]:
    if tags is None:
        return []

    if isinstance(tags, list):
        return [str(tag).strip() for tag in tags if str(tag).strip()]

    raw = str(tags).strip()
    if not raw:
        return []

    if "|" in raw:
        return [part.strip() for part in raw.split("|") if part.strip()]

    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]

    return [raw]


def _to_project_relative_path(path_value: str | Path) -> str:
    path_obj = Path(path_value).resolve()
    project_root = get_project_root().resolve()

    try:
        return str(path_obj.relative_to(project_root)).replace("\\", "/")
    except Exception:
        return str(path_obj).replace("\\", "/")


def sanitize_filename_component(value: str, fallback: str = "video") -> str:
    cleaned = (value or "").strip().lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"[-\s]+", "_", cleaned).strip("_")
    return cleaned or fallback


def build_auto_video_catalog_entry(
    video_path: str,
    source_url: str = "",
    source_platform: str = "local",
    existing_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    video_file = Path(video_path).resolve()
    video_name = video_file.name
    stem_title = video_file.stem.replace("_", " ").replace("-", " ").strip()
    now_iso = _now_iso_string()

    existing = existing_entry or {}
    existing_tags = _normalize_catalog_tags(existing.get("tags"))
    created_at = str(existing.get("created_at", "")).strip() or now_iso

    entry = {
        "video_name": video_name,
        "local_video_path": _to_project_relative_path(video_file),
        "source_platform": str(existing.get("source_platform", "")).strip()
        or str(source_platform).strip()
        or "local",
        "source_url": str(existing.get("source_url", "")).strip() or str(source_url).strip(),
        "title": str(existing.get("title", "")).strip() or stem_title or video_name,
        "description": str(existing.get("description", "")).strip(),
        "thumbnail_url": str(existing.get("thumbnail_url", "")).strip(),
        "tags": existing_tags,
        "created_at": created_at,
        "ingested_at": now_iso,
    }
    return entry


def build_video_catalog_entry_from_metadata(
    *,
    video_path: str,
    source_platform: str,
    source_url: str,
    title: str,
    description: str = "",
    thumbnail_url: str = "",
    tags: Optional[List[str]] = None,
    existing_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    video_file = Path(video_path).resolve()
    video_name = video_file.name
    now_iso = _now_iso_string()

    existing = existing_entry or {}
    created_at = str(existing.get("created_at", "")).strip() or now_iso

    merged_tags = _normalize_catalog_tags(existing.get("tags"))
    incoming_tags = _normalize_catalog_tags(tags or [])
    seen_lower = {tag.lower() for tag in merged_tags}
    for tag in incoming_tags:
        if tag.lower() not in seen_lower:
            merged_tags.append(tag)
            seen_lower.add(tag.lower())

    entry = {
        "video_name": video_name,
        "local_video_path": _to_project_relative_path(video_file),
        "source_platform": str(source_platform).strip() or str(existing.get("source_platform", "")).strip() or "local",
        "source_url": str(source_url).strip() or str(existing.get("source_url", "")).strip(),
        "title": str(title).strip() or str(existing.get("title", "")).strip() or video_name,
        "description": str(description).strip() or str(existing.get("description", "")).strip(),
        "thumbnail_url": str(thumbnail_url).strip() or str(existing.get("thumbnail_url", "")).strip(),
        "tags": merged_tags,
        "created_at": created_at,
        "ingested_at": now_iso,
    }
    return entry


def upsert_video_catalog_entry(
    entry: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise TypeError("entry must be a dict")

    video_name = str(entry.get("video_name", "")).strip()
    if not video_name:
        raise ValueError("Catalog entry must include a non-empty video_name")

    catalog = load_video_catalog(force_reload=True, config=config)
    updated_catalog: List[Dict[str, Any]] = []
    replaced = False

    for item in catalog:
        existing_name = str(item.get("video_name", "")).strip()
        if existing_name == video_name:
            merged = dict(item)
            merged.update(entry)

            if not merged.get("created_at"):
                merged["created_at"] = item.get("created_at") or _now_iso_string()
            if "tags" in merged:
                merged["tags"] = _normalize_catalog_tags(merged.get("tags"))

            updated_catalog.append(merged)
            replaced = True
        else:
            updated_catalog.append(item)

    if not replaced:
        new_entry = dict(entry)
        new_entry["tags"] = _normalize_catalog_tags(new_entry.get("tags"))
        updated_catalog.append(new_entry)

    save_video_catalog(updated_catalog, config=config)

    saved_entry = get_video_catalog_entry(
        video_name,
        force_reload=True,
        config=config,
    )
    return saved_entry or dict(entry)


def ensure_video_catalog_entry(
    video_path: str,
    config: Optional[Dict[str, Any]] = None,
    source_url: str = "",
    source_platform: str = "local",
) -> Dict[str, Any]:
    video_file = Path(video_path).resolve()
    existing = get_video_catalog_entry(
        video_file.name,
        force_reload=True,
        config=config,
    )

    auto_entry = build_auto_video_catalog_entry(
        video_path=str(video_file),
        source_url=source_url,
        source_platform=source_platform,
        existing_entry=existing,
    )

    return upsert_video_catalog_entry(auto_entry, config=config)


def ensure_video_catalog_entry_from_metadata(
    *,
    video_path: str,
    source_platform: str,
    source_url: str,
    title: str,
    description: str = "",
    thumbnail_url: str = "",
    tags: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    video_file = Path(video_path).resolve()
    existing = get_video_catalog_entry(
        video_file.name,
        force_reload=True,
        config=config,
    )

    entry = build_video_catalog_entry_from_metadata(
        video_path=str(video_file),
        source_platform=source_platform,
        source_url=source_url,
        title=title,
        description=description,
        thumbnail_url=thumbnail_url,
        tags=tags,
        existing_entry=existing,
    )

    return upsert_video_catalog_entry(entry, config=config)


def normalize_device(device: Any = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    raise TypeError(f"Unsupported device type: {type(device)}")


def setup_logging() -> None:
    project_root = get_project_root()
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging_config_path = project_root / "configs" / "logging.yaml"
    if logging_config_path.exists():
        config = load_yaml(logging_config_path)

        handlers = config.get("handlers", {})
        for _, handler in handlers.items():
            filename = handler.get("filename")
            if filename:
                handler["filename"] = str(project_root / filename)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    # Chặn log telemetry rác của ChromaDB
    logging.getLogger("chromadb.telemetry").disabled = True
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    logging.getLogger("chromadb.telemetry.product.events").disabled = True


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


@contextmanager
def stage_timer(stage_name: str, stage_metrics: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.getLogger(__name__).info("Stage '%s' finished in %.2f sec", stage_name, elapsed)
        if stage_metrics is not None:
            stage_metrics[stage_name] = round(elapsed, 4)


def md5_of_file(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def format_timestamp(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None

    try:
        total_seconds = max(0, int(float(seconds)))
    except Exception:
        return None

    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def delete_video_catalog_entry(
    video_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    target = (video_name or "").strip()
    if not target:
        return False

    catalog = load_video_catalog(force_reload=True, config=config)
    kept = [item for item in catalog if str(item.get("video_name", "")).strip() != target]
    changed = len(kept) != len(catalog)

    if changed:
        save_video_catalog(kept, config=config)

    return changed


def _safe_unlink(path: Path) -> bool:
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except Exception:
        pass
    return False


def _safe_rmtree(path: Path) -> bool:
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            return True
    except Exception:
        pass
    return False


def get_video_artifact_paths(
    video_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    cfg = config or get_config()
    safe_video_name = (video_name or "").strip()
    if not safe_video_name:
        raise ValueError("video_name must not be empty")

    stem = Path(safe_video_name).stem
    entry = get_video_catalog_entry(safe_video_name, force_reload=True, config=cfg) or {}

    raw_candidates: List[Path] = []
    local_video_path = str(entry.get("local_video_path", "")).strip()
    if local_video_path:
        raw_candidates.append(get_data_path(local_video_path))
    raw_candidates.append(get_data_path(cfg["paths"].get("raw_dir", "data/raw")) / safe_video_name)

    unique_raw: List[str] = []
    seen = set()
    for candidate in raw_candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            unique_raw.append(str(candidate))

    interim_audio_dir = get_data_path(cfg["paths"].get("interim_audio_dir", "data/interim/audio"))
    interim_frames_dir = get_data_path(cfg["paths"].get("interim_frames_dir", "data/interim/frames"))
    interim_transcripts_dir = get_data_path(cfg["paths"].get("interim_transcripts_dir", "data/interim/transcripts"))
    interim_captions_dir = get_data_path(cfg["paths"].get("interim_captions_dir", "data/interim/captions"))
    processed_dir = get_data_path(cfg["paths"].get("processed_dir", "data/processed"))

    return {
        "raw_video": unique_raw,
        "audio_files": [str(interim_audio_dir / f"{stem}.wav")],
        "frame_dirs": [str(interim_frames_dir / stem)],
        "transcript_files": [
            str(interim_transcripts_dir / f"{stem}_transcript.json"),
            str(processed_dir / f"{stem}_transcript_processed.json"),
        ],
        "caption_files": [
            str(interim_captions_dir / f"{stem}_captions.json"),
            str(processed_dir / f"{stem}_captions_processed.json"),
        ],
        "processed_files": [
            str(processed_dir / f"{stem}_merged_output.json"),
            str(processed_dir / f"{stem}_run_metadata.json"),
        ],
    }


def cleanup_video_artifacts(
    video_name: str,
    *,
    delete_raw: bool = False,
    delete_audio: bool = False,
    delete_frames: bool = False,
    delete_interim_json: bool = False,
    delete_processed: bool = False,
    keep_catalog: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or get_config()
    artifact_paths = get_video_artifact_paths(video_name, config=cfg)

    deleted: Dict[str, List[str]] = {
        "raw_video": [],
        "audio_files": [],
        "frame_dirs": [],
        "transcript_files": [],
        "caption_files": [],
        "processed_files": [],
    }
    missing: Dict[str, List[str]] = {
        "raw_video": [],
        "audio_files": [],
        "frame_dirs": [],
        "transcript_files": [],
        "caption_files": [],
        "processed_files": [],
    }

    if delete_raw:
        for p in artifact_paths["raw_video"]:
            path = Path(p)
            if _safe_unlink(path):
                deleted["raw_video"].append(str(path))
            else:
                missing["raw_video"].append(str(path))

    if delete_audio:
        for p in artifact_paths["audio_files"]:
            path = Path(p)
            if _safe_unlink(path):
                deleted["audio_files"].append(str(path))
            else:
                missing["audio_files"].append(str(path))

    if delete_frames:
        for p in artifact_paths["frame_dirs"]:
            path = Path(p)
            if _safe_rmtree(path):
                deleted["frame_dirs"].append(str(path))
            else:
                missing["frame_dirs"].append(str(path))

    if delete_interim_json:
        for p in artifact_paths["transcript_files"]:
            path = Path(p)
            if path.name.endswith("_transcript_processed.json"):
                continue
            if _safe_unlink(path):
                deleted["transcript_files"].append(str(path))
            else:
                missing["transcript_files"].append(str(path))

        for p in artifact_paths["caption_files"]:
            path = Path(p)
            if path.name.endswith("_captions_processed.json"):
                continue
            if _safe_unlink(path):
                deleted["caption_files"].append(str(path))
            else:
                missing["caption_files"].append(str(path))

    if delete_processed:
        for p in artifact_paths["processed_files"]:
            path = Path(p)
            if _safe_unlink(path):
                deleted["processed_files"].append(str(path))
            else:
                missing["processed_files"].append(str(path))

        for p in artifact_paths["transcript_files"]:
            path = Path(p)
            if path.name.endswith("_transcript_processed.json"):
                if _safe_unlink(path):
                    deleted["transcript_files"].append(str(path))
                else:
                    missing["transcript_files"].append(str(path))

        for p in artifact_paths["caption_files"]:
            path = Path(p)
            if path.name.endswith("_captions_processed.json"):
                if _safe_unlink(path):
                    deleted["caption_files"].append(str(path))
                else:
                    missing["caption_files"].append(str(path))

    removed_from_catalog = False
    if not keep_catalog:
        removed_from_catalog = delete_video_catalog_entry(video_name, config=cfg)

    deleted_count = sum(len(v) for v in deleted.values())
    missing_count = sum(len(v) for v in missing.values())

    return {
        "video_name": video_name,
        "deleted": deleted,
        "missing_or_not_deleted": missing,
        "deleted_count": deleted_count,
        "missing_count": missing_count,
        "removed_from_catalog": removed_from_catalog,
    }

def infer_has_audio_from_video_path(video_path: str) -> bool:
    try:
        from src.extract.audio_extractor import AudioExtractor
        extractor = AudioExtractor(get_config())
        return extractor.has_audio_stream(video_path)
    except Exception:
        return False

def infer_video_duration_sec(video_path: str) -> Optional[float]:
    try:
        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            return None

        import subprocess

        cmd = [
            ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        raw = (result.stdout or "").strip()
        if not raw:
            return None

        duration = float(raw)
        if duration < 0:
            return None

        return round(duration, 3)
    except Exception:
        return None

def infer_video_type_from_source_info(source_info: Optional[Dict[str, Any]] = None) -> str:
    info = source_info or {}
    text = " ".join(
        [
            str(info.get("video_title", "") or ""),
            str(info.get("video_description", "") or ""),
            " ".join(_normalize_catalog_tags(info.get("video_tags"))),
        ]
    ).lower()

    if any(token in text for token in ["ted", "talk", "lecture", "presentation", "speaker"]):
        return "talk"
    if any(token in text for token in ["cook", "recipe", "egg", "kitchen", "tutorial", "how to"]):
        return "tutorial"
    if any(token in text for token in ["wildlife", "nature", "animal", "forest", "ocean", "documentary"]):
        return "visual_story"
    if any(token in text for token in ["music", "official video", "cinematic", "vlog", "film", "short film"]):
        return "cinematic"
    return get_config().get("metadata", {}).get("default_video_type", "generic")


def infer_content_style_from_source_info(source_info: Optional[Dict[str, Any]] = None) -> str:
    info = source_info or {}
    text = " ".join(
        [
            str(info.get("video_title", "") or ""),
            str(info.get("video_description", "") or ""),
            " ".join(_normalize_catalog_tags(info.get("video_tags"))),
        ]
    ).lower()

    if any(token in text for token in ["ted", "talk", "lecture", "speech", "presentation", "motivation"]):
        return "talk"
    if any(token in text for token in ["cook", "recipe", "egg", "kitchen", "tutorial", "how to"]):
        return "action"
    if any(token in text for token in ["wildlife", "nature", "animal", "forest", "ocean", "documentary"]):
        return "visual"
    if any(token in text for token in ["music", "official video", "cinematic", "vlog", "film", "short film"]):
        return "cinematic_music"
    return get_config().get("metadata", {}).get("default_estimated_content_style", "generic")


def infer_recommended_search_mode(source_info: Optional[Dict[str, Any]] = None) -> str:
    style = infer_content_style_from_source_info(source_info)
    if style == "talk":
        return "Talk mode"
    if style == "action":
        return "Action mode"
    if style == "visual":
        return "Visual mode"
    if style == "cinematic_music":
        return "Visual mode"
    return "Manual"


def normalize_source_metadata_for_pipeline(
    source_metadata: Optional[Dict[str, Any]],
    *,
    video_path: str,
    fallback_platform: str = "local",
) -> Dict[str, Any]:
    raw = dict(source_metadata or {})
    video_file = Path(video_path).resolve()

    normalized: Dict[str, Any] = {
        "video_path": str(video_file),
        "video_name": video_file.name,
        "source_platform": str(raw.get("source_platform", fallback_platform)).strip() or fallback_platform,
        "source_url": str(raw.get("source_url", "")).strip(),
        "video_title": str(raw.get("video_title", video_file.stem)).strip() or video_file.stem,
        "video_description": str(raw.get("video_description", "")).strip(),
        "thumbnail_url": str(raw.get("thumbnail_url", "")).strip(),
        "video_tags": _normalize_catalog_tags(raw.get("video_tags")),
        "local_video_path": _to_project_relative_path(video_file),
        "created_at": str(raw.get("created_at", "")).strip(),
        "ingested_at": str(raw.get("ingested_at", "")).strip(),
        "ingest_method": str(raw.get("ingest_method", "local_file")).strip() or "local_file",
        "duration_sec": raw.get("duration_sec")
        if raw.get("duration_sec") is not None
        else infer_video_duration_sec(str(video_file)),
    }

    if "has_audio" in raw:
        normalized["has_audio"] = bool(raw.get("has_audio"))
    else:
        normalized["has_audio"] = infer_has_audio_from_video_path(str(video_file))

    normalized["video_type"] = str(
        raw.get("video_type") or infer_video_type_from_source_info(normalized)
    ).strip()

    normalized["estimated_content_style"] = str(
        raw.get("estimated_content_style") or infer_content_style_from_source_info(normalized)
    ).strip()

    normalized["recommended_search_mode"] = str(
        raw.get("recommended_search_mode") or infer_recommended_search_mode(normalized)
    ).strip()

    return normalized


def looks_like_pytest_temp_path(path_value: str) -> bool:
    normalized = str(path_value or "").replace("\\", "/").lower()
    return "pytest-of-" in normalized or "/temp/pytest-" in normalized


def sanitize_video_catalog(
    *,
    remove_stale_test_entries: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or get_config()
    catalog = load_video_catalog(force_reload=True, config=cfg)
    raw_dir = get_data_path(cfg["paths"].get("raw_dir", "data/raw"))

    cleaned: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for item in catalog:
        if not isinstance(item, dict):
            continue

        video_name = str(item.get("video_name", "")).strip()
        local_video_path = str(item.get("local_video_path", "")).strip()

        should_remove = False

        if remove_stale_test_entries and local_video_path and looks_like_pytest_temp_path(local_video_path):
            fallback = raw_dir / video_name
            if not fallback.exists():
                should_remove = True

        if should_remove:
            removed.append(
                {
                    "video_name": video_name,
                    "local_video_path": local_video_path,
                    "reason": "stale_pytest_temp_path",
                }
            )
            continue

        cleaned.append(item)

    save_video_catalog(cleaned, config=cfg)

    return {
        "removed_count": len(removed),
        "removed": removed,
        "remaining_count": len(cleaned),
    }


def resolve_video_path_from_catalog_entry(
    video_name: str,
    entry: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    cfg = config or get_config()
    local_video_path = str(entry.get("local_video_path", "")).strip()

    if local_video_path:
        rel_candidate = get_data_path(local_video_path)
        if rel_candidate.exists():
            return rel_candidate

        abs_candidate = Path(local_video_path)
        if abs_candidate.is_absolute() and abs_candidate.exists():
            return abs_candidate

    fallback = get_data_path(cfg["paths"].get("raw_dir", "data/raw")) / video_name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not resolve video path for '{video_name}'. "
        f"Catalog path='{local_video_path}', fallback='{fallback}'"
    )
