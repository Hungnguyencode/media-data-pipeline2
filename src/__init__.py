"""
Core package for the media data pipeline project.
Keep this file minimal to avoid import-time breakage.
"""

from .utils import get_config, get_config_path, get_data_path, normalize_device, setup_logging

__all__ = [
    "get_config",
    "get_config_path",
    "get_data_path",
    "normalize_device",
    "setup_logging",
]