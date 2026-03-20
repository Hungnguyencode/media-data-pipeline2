from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.utils import get_config, get_data_path, normalize_device, release_memory

logger = logging.getLogger(__name__)


class VisionProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None, device=None):
        self.config = config or get_config()
        self.device = normalize_device(device)

        model_name = self.config["models"]["vision"]["name"]
        self.model_name = model_name
        self.max_length = int(self.config["models"]["vision"].get("max_length", 40))
        self.image_size = int(self.config["models"]["vision"].get("image_size", 384))
        self.output_language = self.config["models"]["vision"].get("output_language", "en")
        self.fallback_to_cpu_on_oom = bool(
            self.config["models"]["vision"].get("fallback_to_cpu_on_oom", True)
        )
        self.pipeline_version = str(self.config.get("pipeline", {}).get("version", "1.0.0"))

        self.processor = None
        self.model = None

        self.output_dir = Path(
            get_data_path(self.config["paths"].get("interim_captions_dir", "data/interim/captions"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processed_dir = Path(
            get_data_path(self.config["paths"].get("processed_dir", "data/processed"))
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, device=None):
        target_device = normalize_device(device or self.device)
        logger.info("Loading vision model: %s on %s", self.model_name, target_device)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(target_device)
        self.model.eval()
        self.device = target_device

    def unload_model(self):
        self.processor = None
        self.model = None
        release_memory()

    def _extract_timestamp_from_filename(self, filename: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)s", filename)
        return float(match.group(1)) if match else 0.0

    def _format_timestamp(self, seconds: float) -> str:
        total_seconds = int(seconds)
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _clean_caption(self, caption: str) -> str:
        caption = " ".join(caption.strip().split())
        if caption:
            caption = caption[0].upper() + caption[1:]
        return caption

    def _save_json(self, data: Any, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _prepare_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((self.image_size, self.image_size))
        return image

    def _generate_once(self, image_path: str) -> str:
        if self.model is None or self.processor is None:
            self._load_model()

        image = self._prepare_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=self.max_length)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()

    def generate_caption(self, image_path: str) -> str:
        try:
            return self._generate_once(image_path)
        except RuntimeError as e:
            err = str(e).lower()
            if "out of memory" in err and self.device.type == "cuda" and self.fallback_to_cpu_on_oom:
                logger.warning("CUDA OOM in BLIP. Falling back to CPU for %s", Path(image_path).name)
                self.unload_model()
                self._load_model(device="cpu")
                return self._generate_once(image_path)
            raise

    def process_frames(self, frames_dir: str, video_name: str) -> List[Dict[str, Any]]:
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        image_files = sorted(
            [
                p for p in frames_path.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        results: List[Dict[str, Any]] = []
        for image_file in image_files:
            caption = self._clean_caption(self.generate_caption(str(image_file)))
            timestamp = self._extract_timestamp_from_filename(image_file.name)

            results.append(
                {
                    "video_name": video_name,
                    "frame_name": image_file.name,
                    "image_path": str(image_file),
                    "timestamp": timestamp,
                    "timestamp_str": self._format_timestamp(timestamp),
                    "caption": caption,
                    "model_name": getattr(self, "model_name", "unknown"),
                    "device_used": str(getattr(self, "device", "cpu")),
                    "language": getattr(self, "output_language", "en"),
                    "pipeline_version": getattr(self, "pipeline_version", "1.0.0"),
                    "source_modality": "image",
                }
            )

        interim_output = self.output_dir / f"{Path(video_name).stem}_captions.json"
        processed_output = self.processed_dir / f"{Path(video_name).stem}_captions_processed.json"

        self._save_json(results, interim_output)
        self._save_json(results, processed_output)

        logger.info("Generated %d captions for '%s'", len(results), video_name)
        logger.info("Saved captions to %s and %s", interim_output, processed_output)

        return results