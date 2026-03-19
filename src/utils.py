from __future__ import annotations

import gc
import hashlib
import json
import logging
import logging.config
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


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


def normalize_device(device=None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    raise TypeError(f"Unsupported device type: {type(device)}")


def setup_logging():
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
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


@contextmanager
def stage_timer(stage_name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.getLogger(__name__).info("Stage '%s' finished in %.2f sec", stage_name, elapsed)


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


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)