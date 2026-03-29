from __future__ import annotations

import gc
import hashlib
import json
import logging
import logging.config
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
import yaml


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