from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import whisper

from src.utils import get_config, get_data_path, normalize_device, release_memory

logger = logging.getLogger(__name__)


class WhisperProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None, device=None):
        self.config = config or get_config()
        self.device = normalize_device(device)

        whisper_cfg = self.config.get("models", {}).get("whisper", {})
        self.model_name = whisper_cfg.get("name", "base")
        self.language = whisper_cfg.get("language", "auto")
        self.use_fp16 = bool(whisper_cfg.get("use_fp16", True))
        self.fallback_to_cpu_on_oom = bool(whisper_cfg.get("fallback_to_cpu_on_oom", True))
        self.pipeline_version = str(self.config.get("pipeline", {}).get("version", "1.0.0"))

        self.output_dir = Path(
            get_data_path(self.config["paths"].get("interim_transcripts_dir", "data/interim/transcripts"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processed_dir = Path(
            get_data_path(self.config["paths"].get("processed_dir", "data/processed"))
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.model = None

    def _save_json(self, data: Dict[str, Any], output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _clean_text(self, text: str) -> str:
        return " ".join((text or "").strip().split())

    def _load_model(self, device=None):
        target_device = normalize_device(device or self.device)
        logger.info("Loading Whisper model '%s' on %s", self.model_name, target_device)
        self.model = whisper.load_model(self.model_name, device=str(target_device))
        self.device = target_device

    def unload_model(self):
        self.model = None
        release_memory()

    def _transcribe_once(self, audio_path: str) -> Dict[str, Any]:
        if self.model is None:
            self._load_model()

        transcribe_kwargs = {
            "audio": str(audio_path),
            "task": "transcribe",
            "fp16": (self.use_fp16 and getattr(self.device, "type", str(self.device)) == "cuda"),
            "verbose": False,
        }

        language = (self.language or "").strip().lower()
        if language and language != "auto":
            transcribe_kwargs["language"] = language

        return self.model.transcribe(**transcribe_kwargs)

    def _should_fallback_to_cpu(self, error: Exception) -> bool:
        if not self.fallback_to_cpu_on_oom:
            return False

        device_type = getattr(self.device, "type", str(self.device))
        if device_type != "cuda":
            return False

        err = str(error).lower()
        fallback_markers = [
            "out of memory",
            "cuda",
            "cublas",
            "cudnn",
            "device-side assert",
            "device unavailable",
            "no kernel image is available",
            "failed to initialize cuda",
        ]
        return any(marker in err for marker in fallback_markers)

    def _transcribe_with_fallback(self, audio_path: str) -> Dict[str, Any]:
        try:
            return self._transcribe_once(audio_path)
        except Exception as e:
            if not self._should_fallback_to_cpu(e):
                raise

            logger.warning(
                "Whisper failed on CUDA for %s with error: %s. Falling back to CPU.",
                Path(audio_path).name,
                e,
            )

            self.unload_model()
            self._load_model(device="cpu")

            try:
                return self._transcribe_once(audio_path)
            except Exception as cpu_error:
                logger.error(
                    "Whisper transcription failed on both CUDA and CPU for %s: %s",
                    Path(audio_path).name,
                    cpu_error,
                )
                raise RuntimeError(
                    f"Failed to transcribe audio {audio_path} after CPU fallback: {cpu_error}"
                ) from cpu_error

    def transcribe(self, audio_path: str, video_name: Optional[str] = None) -> Dict[str, Any]:
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        video_name = video_name or audio_file.stem
        logger.info("Transcribing audio: %s", audio_file.name)

        try:
            raw_result = self._transcribe_with_fallback(str(audio_file))
        except RuntimeError:
            raise
        except Exception as e:
            logger.error("Whisper transcription failed for %s: %s", audio_file.name, e)
            raise RuntimeError(f"Failed to transcribe audio {audio_path}: {e}") from e

        detected_language = raw_result.get("language")
        fallback_language = self.language if (self.language and self.language != "auto") else "auto"

        segments: List[Dict[str, Any]] = []
        for seg in raw_result.get("segments", []):
            text = self._clean_text(seg.get("text", ""))
            if not text:
                continue
            segments.append(
                {
                    "id": seg.get("id"),
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": text,
                }
            )

        result = {
            "video_name": video_name,
            "audio_path": str(audio_file),
            "language": detected_language or fallback_language,
            "full_text": self._clean_text(raw_result.get("text", "")),
            "segments": segments,
            "model_name": getattr(self, "model_name", "unknown"),
            "device_used": str(getattr(self, "device", "cpu")),
            "pipeline_version": getattr(self, "pipeline_version", "1.0.0"),
            "source_modality": "audio",
        }

        interim_output = self.output_dir / f"{Path(video_name).stem}_transcript.json"
        processed_output = self.processed_dir / f"{Path(video_name).stem}_transcript_processed.json"

        self._save_json(result, interim_output)
        self._save_json(result, processed_output)

        logger.info("Transcription completed for '%s': %d segments", video_name, len(segments))
        logger.info("Saved transcription to %s and %s", interim_output, processed_output)

        return result