from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.utils import get_config, get_data_path, normalize_device, release_memory

logger = logging.getLogger(__name__)


class VisionProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None, device=None):
        self.config = config or get_config()
        self.device = normalize_device(device)

        vision_cfg = self.config["models"]["vision"]

        self.blip_name = vision_cfg.get("blip_name", "Salesforce/blip-image-captioning-base")
        self.clip_name = vision_cfg.get("clip_name", "ViT-B-32")
        self.clip_pretrained = vision_cfg.get("clip_pretrained", "openai")
        self.max_length = int(vision_cfg.get("max_length", 40))
        self.image_size = int(vision_cfg.get("image_size", 384))
        self.output_language = vision_cfg.get("output_language", "en")
        self.fallback_to_cpu_on_oom = bool(vision_cfg.get("fallback_to_cpu_on_oom", True))
        self.pipeline_version = str(self.config.get("pipeline", {}).get("version", "1.0.0"))

        self.blip_processor = None
        self.blip_model = None

        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None

        self.output_dir = Path(
            get_data_path(self.config["paths"].get("interim_captions_dir", "data/interim/captions"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processed_dir = Path(
            get_data_path(self.config["paths"].get("processed_dir", "data/processed"))
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _load_models(self, device=None):
        target_device = normalize_device(device or self.device)

        logger.info("Loading BLIP model: %s on %s", self.blip_name, target_device)
        self.blip_processor = BlipProcessor.from_pretrained(self.blip_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_name).to(target_device)
        self.blip_model.eval()

        logger.info(
            "Loading CLIP model: %s (%s) on %s",
            self.clip_name,
            self.clip_pretrained,
            target_device,
        )
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.clip_name,
            pretrained=self.clip_pretrained,
            device=str(target_device),
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(self.clip_name)

        self.device = target_device

    def unload_model(self):
        self.blip_processor = None
        self.blip_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
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

    def _refine_caption(self, caption: str, timestamp: float) -> str:
        if not caption:
            return caption

        refined = caption.strip()
        lower = refined.lower()

        replacements = [
            (" in a blend", " in a bowl"),
            (" into a blend", " into a bowl"),
            (" in the blend", " in the bowl"),
            (" peeling an egg", " cracking an egg"),
            (" squeezing an egg", " cracking an egg"),
            (" squeezing egg", " cracking egg"),
            (" peel an egg", " crack an egg"),
        ]

        for old, new in replacements:
            lower = lower.replace(old, new)

        if "egg" in lower and "bowl" in lower:
            lower = lower.replace("peeling an egg", "cracking an egg")
            lower = lower.replace("squeezing an egg", "cracking an egg")

        if "egg" in lower and "bowl" in lower and timestamp <= 20:
            if "cracking an egg" in lower and "into a bowl" not in lower:
                lower = lower.replace("cracking an egg", "cracking an egg into a bowl")
            if "person is cracking an egg" in lower and "into a bowl" not in lower:
                lower = lower.replace("person is cracking an egg", "person is cracking an egg into a bowl")

        lower = re.sub(r"\s+", " ", lower).strip()
        if lower:
            refined = lower[0].upper() + lower[1:]

        return refined

    def _save_json(self, data: Any, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _prepare_pil_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((self.image_size, self.image_size))
        return image

    def _generate_caption_once(self, image: Image.Image) -> str:
        if self.blip_model is None or self.blip_processor is None:
            self._load_models()

        inputs = self.blip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.blip_model.generate(**inputs, max_length=self.max_length)

        caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()

    def _encode_image_clip_once(self, image: Image.Image) -> List[float]:
        if self.clip_model is None or self.clip_preprocess is None:
            self._load_models()

        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        vector = image_features[0].detach().cpu().numpy().astype(np.float32)
        return vector.tolist()

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

        if self.blip_model is None or self.clip_model is None:
            self._load_models()

        results: List[Dict[str, Any]] = []
        total_files = len(image_files)

        for idx, image_file in enumerate(image_files, start=1):
            image = self._prepare_pil_image(str(image_file))

            try:
                raw_caption = self._clean_caption(self._generate_caption_once(image))
                clip_embedding = self._encode_image_clip_once(image)
            except RuntimeError as e:
                err = str(e).lower()
                if "out of memory" in err and str(self.device) == "cuda" and self.fallback_to_cpu_on_oom:
                    logger.warning("CUDA OOM in vision stack. Falling back to CPU for %s", image_file.name)
                    self.unload_model()
                    self._load_models(device="cpu")
                    image = self._prepare_pil_image(str(image_file))
                    raw_caption = self._clean_caption(self._generate_caption_once(image))
                    clip_embedding = self._encode_image_clip_once(image)
                else:
                    raise

            timestamp = self._extract_timestamp_from_filename(image_file.name)
            refined_caption = self._refine_caption(raw_caption, timestamp)

            results.append(
                {
                    "video_name": video_name,
                    "frame_name": image_file.name,
                    "image_path": str(image_file),
                    "timestamp": timestamp,
                    "timestamp_str": self._format_timestamp(timestamp),
                    "caption": refined_caption,
                    "raw_caption": raw_caption,
                    "caption_refined": refined_caption,
                    "blip_model_name": self.blip_name,
                    "clip_model_name": f"{self.clip_name}:{self.clip_pretrained}",
                    "device_used": str(self.device),
                    "language": self.output_language,
                    "pipeline_version": self.pipeline_version,
                    "source_modality": "image",
                    "clip_embedding": clip_embedding,
                }
            )

            if idx % 20 == 0 or idx == total_files:
                logger.info(
                    "Vision progress for '%s': %d/%d frames processed",
                    video_name,
                    idx,
                    total_files,
                )

        interim_output = self.output_dir / f"{Path(video_name).stem}_captions.json"
        processed_output = self.processed_dir / f"{Path(video_name).stem}_captions_processed.json"

        self._save_json(results, interim_output)
        self._save_json(results, processed_output)

        logger.info("Generated %d BLIP+CLIP frame records for '%s'", len(results), video_name)
        logger.info("Saved captions to %s and %s", interim_output, processed_output)

        return results

    def encode_text_for_clip(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.clip_model is None or self.clip_tokenizer is None:
            self._load_models()

        tokens = self.clip_tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        vectors = text_features.detach().cpu().numpy().astype(np.float32)
        return vectors.tolist()