from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
        self.pipeline_version = str(self.config.get("pipeline", {}).get("version", "2.1.0"))

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

        self.action_families: Dict[str, List[str]] = {
            "break_open": [
                "break",
                "breaking",
                "break open",
                "breaking open",
                "crack",
                "cracking",
                "split",
                "splitting",
                "split open",
                "open",
                "opening",
                "separate",
                "separating",
                "shell",
                "shelling",
            ],
            "peel_remove_outer": [
                "peel",
                "peeling",
                "remove peel",
                "removing peel",
                "remove shell",
                "removing shell",
                "strip",
                "stripping",
            ],
            "cut_divide": [
                "cut",
                "cutting",
                "slice",
                "slicing",
                "chop",
                "chopping",
                "dice",
                "dicing",
                "halve",
                "halving",
            ],
            "mix_agitate": [
                "mix",
                "mixing",
                "stir",
                "stirring",
                "whisk",
                "whisking",
                "beat",
                "beating",
                "blend",
                "blending",
            ],
            "pour_transfer": [
                "pour",
                "pouring",
                "add",
                "adding",
                "transfer",
                "transferring",
                "empty",
                "emptying",
            ],
            "hold_pick_place": [
                "hold",
                "holding",
                "pick up",
                "picking up",
                "place",
                "placing",
                "put",
                "putting",
                "grab",
                "grabbing",
            ],
            "squeeze_press": [
                "squeeze",
                "squeezing",
                "press",
                "pressing",
                "pinch",
                "pinching",
            ],
        }

        self.openable_object_cues: Set[str] = {
            "egg",
            "eggs",
            "shell",
            "package",
            "packet",
            "bag",
            "box",
            "carton",
            "jar",
            "bottle",
            "can",
            "capsule",
            "pod",
            "fruit",
            "orange",
            "coconut",
            "nut",
            "garlic",
            "onion",
            "avocado",
            "oyster",
            "clam",
        }

        self.container_cues: Set[str] = {
            "bowl",
            "cup",
            "glass",
            "plate",
            "pan",
            "pot",
            "tray",
            "container",
            "jar",
            "spoon",
        }

        self.pourable_object_cues: Set[str] = {
            "milk",
            "water",
            "oil",
            "juice",
            "sauce",
            "syrup",
            "cream",
            "liquid",
        }

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

    def _remove_repeated_phrases(self, text: str) -> str:
        tokens = text.split()
        if not tokens:
            return text

        changed = True
        while changed:
            changed = False
            for n in (3, 2, 1):
                i = 0
                new_tokens: List[str] = []
                while i < len(tokens):
                    if i + 2 * n <= len(tokens) and tokens[i : i + n] == tokens[i + n : i + 2 * n]:
                        new_tokens.extend(tokens[i : i + n])
                        i += 2 * n
                        changed = True
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

        return " ".join(tokens)

    def _normalize_for_matching(self, text: str) -> str:
        normalized = (text or "").strip().lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _has_any_phrase(self, text: str, phrases: Set[str]) -> bool:
        normalized = f" {self._normalize_for_matching(text)} "
        for phrase in phrases:
            if f" {phrase} " in normalized:
                return True
        return False

    def _detect_objects_and_context(self, caption: str) -> Dict[str, bool]:
        lower = self._normalize_for_matching(caption)
        tokens = set(lower.split())

        openable_object_cues = getattr(
            self,
            "openable_object_cues",
            {
                "egg",
                "eggs",
                "shell",
                "package",
                "packet",
                "bag",
                "box",
                "carton",
                "jar",
                "bottle",
                "can",
                "capsule",
                "pod",
                "fruit",
                "orange",
                "coconut",
                "nut",
                "garlic",
                "onion",
                "avocado",
                "oyster",
                "clam",
            },
        )

        container_cues = getattr(
            self,
            "container_cues",
            {
                "bowl",
                "cup",
                "glass",
                "plate",
                "pan",
                "pot",
                "tray",
                "container",
                "jar",
                "spoon",
            },
        )

        pourable_object_cues = getattr(
            self,
            "pourable_object_cues",
            {
                "milk",
                "water",
                "oil",
                "juice",
                "sauce",
                "syrup",
                "cream",
                "liquid",
            },
        )

        return {
            "has_openable_object": any(word in tokens for word in openable_object_cues),
            "has_container": any(word in tokens for word in container_cues),
            "has_person": bool(tokens.intersection({"person", "hand", "hands", "someone"})),
            "has_pourable_object": any(word in tokens for word in pourable_object_cues),
        }

    def _replace_action_phrase(self, text: str, source_phrase: str, target_phrase: str) -> str:
        pattern = re.compile(rf"\b{re.escape(source_phrase)}\b", flags=re.IGNORECASE)
        if not pattern.search(text):
            return ""

        replaced = pattern.sub(target_phrase, text, count=1)
        replaced = re.sub(r"\s+", " ", replaced).strip()
        if replaced:
            replaced = replaced[0].upper() + replaced[1:]
        return replaced

    def _sanitize_action_hallucination(self, caption: str, timestamp: float) -> str:
        _ = timestamp

        if not caption:
            return caption

        normalized = self._normalize_for_matching(caption)
        if not normalized:
            return caption

        action_families = getattr(self, "action_families", {})
        pour_terms = set(action_families.get("pour_transfer", []))
        break_terms = set(action_families.get("break_open", []))
        peel_terms = set(action_families.get("peel_remove_outer", []))

        has_pour_action = self._has_any_phrase(normalized, pour_terms)
        has_break_or_peel = self._has_any_phrase(normalized, break_terms.union(peel_terms))

        context = self._detect_objects_and_context(caption)
        tokens = set(normalized.split())

        has_person = context["has_person"]
        has_container = context["has_container"]
        has_pourable_object = context["has_pourable_object"]
        has_openable_object = context["has_openable_object"]

        if has_pour_action:
            # Chỉ tin action đổ khi có đủ tín hiệu mạnh
            strong_pour_signal = has_person and has_container and has_pourable_object
            medium_pour_signal = has_person and has_container

            if strong_pour_signal:
                return caption

            # Nếu thiếu người/tay hoặc thiếu chất lỏng rõ, hạ caption về trung tính
            if not medium_pour_signal:
                if "egg" in tokens or "eggs" in tokens:
                    return "A bowl with eggs in it"
                if "spoon" in tokens:
                    return "A bowl with a spoon"
                if "bowl" in tokens:
                    return "A bowl on a table"
                if "cup" in tokens or "glass" in tokens:
                    return "A container on a table"
                return "A container on a table"

            # Có person nhưng không có vật thể lỏng rõ -> vẫn trung tính hơn
            if has_person and has_container and not has_pourable_object:
                if "bowl" in tokens:
                    return "A person near a bowl"
                if "spoon" in tokens:
                    return "A person holding a spoon"
                return "A person near a container"

        if has_break_or_peel and has_openable_object:
            return caption

        return caption

    def _refine_caption(self, caption: str, timestamp: float) -> str:
        _ = timestamp

        if not caption:
            return caption

        refined = caption.strip().lower()
        refined = re.sub(r"\s+", " ", refined).strip()

        general_replacements = [
            (" in a blend", " in a bowl"),
            (" into a blend", " into a bowl"),
            (" in the blend", " in the bowl"),
            (" on a blend", " on a bowl"),
        ]
        for old, new in general_replacements:
            refined = refined.replace(old, new)

        refined = self._remove_repeated_phrases(refined)
        refined = re.sub(r"\b(\w+)( \1\b)+", r"\1", refined)
        refined = re.sub(r"\s+", " ", refined).strip()

        if refined:
            refined = refined[0].upper() + refined[1:]

        refined = self._sanitize_action_hallucination(refined, timestamp)
        return refined

    def _extract_action_aliases(self, caption: str) -> List[str]:
        if not caption:
            return []

        cleaned = self._normalize_for_matching(caption)
        context = self._detect_objects_and_context(caption)
        aliases: List[str] = []

        action_families = getattr(
            self,
            "action_families",
            {
                "break_open": [
                    "break",
                    "breaking",
                    "break open",
                    "breaking open",
                    "crack",
                    "cracking",
                    "split",
                    "splitting",
                    "split open",
                    "open",
                    "opening",
                    "separate",
                    "separating",
                    "shell",
                    "shelling",
                ],
                "peel_remove_outer": [
                    "peel",
                    "peeling",
                    "remove peel",
                    "removing peel",
                    "remove shell",
                    "removing shell",
                    "strip",
                    "stripping",
                ],
                "cut_divide": [
                    "cut",
                    "cutting",
                    "slice",
                    "slicing",
                    "chop",
                    "chopping",
                    "dice",
                    "dicing",
                    "halve",
                    "halving",
                ],
                "mix_agitate": [
                    "mix",
                    "mixing",
                    "stir",
                    "stirring",
                    "whisk",
                    "whisking",
                    "beat",
                    "beating",
                    "blend",
                    "blending",
                ],
                "pour_transfer": [
                    "pour",
                    "pouring",
                    "add",
                    "adding",
                    "transfer",
                    "transferring",
                    "empty",
                    "emptying",
                ],
                "hold_pick_place": [
                    "hold",
                    "holding",
                    "pick up",
                    "picking up",
                    "place",
                    "placing",
                    "put",
                    "putting",
                    "grab",
                    "grabbing",
                ],
                "squeeze_press": [
                    "squeeze",
                    "squeezing",
                    "press",
                    "pressing",
                    "pinch",
                    "pinching",
                ],
            },
        )

        family_by_phrase: Dict[str, str] = {}
        for family, phrases in action_families.items():
            for phrase in phrases:
                family_by_phrase[phrase] = family

        matched_phrases = [phrase for phrase in family_by_phrase if f" {phrase} " in f" {cleaned} "]

        for phrase in matched_phrases:
            family = family_by_phrase[phrase]
            sibling_phrases = action_families.get(family, [])

            for sibling in sibling_phrases:
                if sibling == phrase:
                    continue
                alias = self._replace_action_phrase(caption, phrase, sibling)
                if alias:
                    aliases.append(alias)

            if family in {"break_open", "peel_remove_outer"} and context["has_openable_object"]:
                bridge_targets = ["break_open", "peel_remove_outer"]
                for target_family in bridge_targets:
                    for sibling in action_families.get(target_family, []):
                        if sibling == phrase:
                            continue
                        alias = self._replace_action_phrase(caption, phrase, sibling)
                        if alias:
                            aliases.append(alias)

            if (
                family == "pour_transfer"
                and context["has_pourable_object"]
                and context["has_container"]
                and context["has_person"]
            ):
                for sibling in action_families.get("pour_transfer", []):
                    if sibling == phrase:
                        continue
                    alias = self._replace_action_phrase(caption, phrase, sibling)
                    if alias:
                        aliases.append(alias)

        deduped: List[str] = []
        seen = set()
        original_norm = self._normalize_for_matching(caption)
        for alias in aliases:
            norm = self._normalize_for_matching(alias)
            if not norm or norm == original_norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(alias)

        return deduped[:5]

    def _build_search_text(self, caption: str, action_aliases: List[str]) -> str:
        base = caption.strip()
        if not action_aliases:
            return base

        alias_block = " | ".join(action_aliases[:2]).strip()
        if not alias_block:
            return base

        return f"{base} [Action aliases] {alias_block}"

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
            action_aliases = self._extract_action_aliases(refined_caption)
            search_text = self._build_search_text(refined_caption, action_aliases)

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
                    "action_aliases": action_aliases,
                    "search_text": search_text,
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