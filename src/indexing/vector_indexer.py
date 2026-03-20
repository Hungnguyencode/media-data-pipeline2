from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.utils import format_timestamp, get_config, get_data_path, normalize_device, safe_float

logger = logging.getLogger(__name__)


class VectorIndexer:
    def __init__(self, config: Optional[Dict[str, Any]] = None, device=None):
        self.config = config or get_config()
        self.device = normalize_device(device)

        vector_db_dir = self.config["paths"].get("vector_db_dir", "data/vector_db")
        self.persist_dir = Path(get_data_path(vector_db_dir))
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = self.config["vector_db"].get("collection_name", "video_semantic_search")
        self.distance_metric = self.config["vector_db"].get("distance_metric", "cosine")

        self.embedding_model_name = self.config["models"]["embedding"]["name"]
        self.batch_size = int(self.config["models"]["embedding"].get("batch_size", 32))
        self.normalize_embeddings = bool(
            self.config["models"]["embedding"].get("normalize_embeddings", True)
        )

        self.segment_window = int(self.config.get("pipeline", {}).get("segment_window", 3))
        self.segment_overlap = int(self.config.get("pipeline", {}).get("segment_overlap", 1))
        self.caption_merge_window_sec = float(
            self.config.get("pipeline", {}).get("caption_merge_window_sec", 3.0)
        )
        self.enable_multimodal_documents = bool(
            self.config.get("pipeline", {}).get("enable_multimodal_documents", True)
        )
        self.pipeline_version = str(self.config.get("pipeline", {}).get("version", "1.0.0"))

        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=str(self.device))

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            logger.info("Collection '%s' not found. Creating new one.", self.collection_name)
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )

    def _stable_id(self, prefix: str, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"{prefix}_{digest}"

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        batch_size = getattr(self, "batch_size", 32)
        normalize_embeddings = getattr(self, "normalize_embeddings", True)

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
        )

        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()

        return [list(vec) for vec in embeddings]

    def _base_metadata(
        self,
        *,
        video_name: str,
        content_type: str,
        source_modality: str,
        model_name: Optional[str] = None,
        timestamp: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        frame_name: Optional[str] = None,
        image_path: Optional[str] = None,
        document_language: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pipeline_version = getattr(self, "pipeline_version", "1.0.0")

        metadata: Dict[str, Any] = {
            "video_name": video_name,
            "content_type": content_type,
            "source_modality": source_modality,
            "model_name": model_name,
            "pipeline_version": pipeline_version,
            "timestamp": timestamp,
            "timestamp_str": format_timestamp(timestamp),
            "start_time": start_time,
            "start_time_str": format_timestamp(start_time),
            "end_time": end_time,
            "end_time_str": format_timestamp(end_time),
            "frame_name": frame_name,
            "image_path": image_path,
            "document_language": document_language,
        }

        if extra:
            metadata.update(extra)

        return metadata

    def delete_video_data(self, video_name: str) -> int:
        logger.info("Deleting existing indexed data for video: %s", video_name)
        try:
            results = self.collection.get(where={"video_name": video_name})
            ids = results.get("ids", []) if results else []
            if ids:
                self.collection.delete(ids=ids)
                logger.info("Deleted %d records for video '%s'", len(ids), video_name)
            return len(ids)
        except Exception as e:
            logger.warning("Could not delete old data for '%s': %s", video_name, e)
            return 0

    def _build_segment_chunks(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            cleaned.append(
                {
                    "start": safe_float(seg.get("start", 0.0)),
                    "end": safe_float(seg.get("end", 0.0)),
                    "text": text,
                }
            )

        if not cleaned:
            return []

        chunks: List[Dict[str, Any]] = []
        segment_window = getattr(self, "segment_window", 3)
        segment_overlap = getattr(self, "segment_overlap", 1)
        step = max(1, segment_window - segment_overlap)

        for i in range(0, len(cleaned), step):
            group = cleaned[i : i + segment_window]
            if not group:
                continue

            merged_text = " ".join(item["text"] for item in group).strip()
            chunks.append(
                {
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "text": merged_text,
                }
            )

            if i + segment_window >= len(cleaned):
                break

        return chunks

    def _find_nearby_caption_texts(
        self,
        captions_data: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> List[str]:
        matched: List[str] = []
        window_sec = getattr(self, "caption_merge_window_sec", 3.0)

        for item in captions_data:
            ts = safe_float(item.get("timestamp"), -1.0)
            if ts < 0:
                continue
            if (start_time - window_sec) <= ts <= (end_time + window_sec):
                caption = (item.get("caption") or "").strip()
                if caption:
                    matched.append(caption)
        return matched

    def index_transcriptions(self, transcription_data: Dict[str, Any]) -> int:
        video_name = transcription_data["video_name"]
        full_text = transcription_data.get("full_text", "").strip()
        segments = transcription_data.get("segments", [])

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        if full_text:
            payload = {
                "video_name": video_name,
                "type": "transcription",
                "text": full_text,
            }
            ids.append(self._stable_id("transcription", payload))
            texts.append(full_text)
            metadatas.append(
                self._base_metadata(
                    video_name=video_name,
                    content_type="transcription",
                    source_modality="audio",
                    model_name=transcription_data.get("model_name"),
                    document_language=transcription_data.get("language"),
                )
            )

        chunks = self._build_segment_chunks(segments)
        for chunk in chunks:
            payload = {
                "video_name": video_name,
                "type": "segment_chunk",
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
            }
            ids.append(self._stable_id("segment_chunk", payload))
            texts.append(chunk["text"])
            metadatas.append(
                self._base_metadata(
                    video_name=video_name,
                    content_type="segment_chunk",
                    source_modality="audio",
                    model_name=transcription_data.get("model_name"),
                    timestamp=chunk["start"],
                    start_time=chunk["start"],
                    end_time=chunk["end"],
                    document_language=transcription_data.get("language"),
                )
            )

        if not texts:
            logger.warning("No transcription text found for video '%s'", video_name)
            return 0

        embeddings = self._embed_texts(texts)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Indexed %d transcription records for '%s'", len(ids), video_name)
        return len(ids)

    def index_captions(self, captions_data: List[Dict[str, Any]]) -> int:
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for item in captions_data:
            caption = (item.get("caption") or "").strip()
            if not caption:
                continue

            video_name = item["video_name"]
            frame_name = item.get("frame_name")
            image_path = item.get("image_path")
            timestamp = safe_float(item.get("timestamp"), 0.0)

            payload = {
                "video_name": video_name,
                "type": "caption",
                "frame_name": frame_name,
                "timestamp": timestamp,
                "caption": caption,
            }

            ids.append(self._stable_id("caption", payload))
            texts.append(caption)
            metadatas.append(
                self._base_metadata(
                    video_name=video_name,
                    content_type="caption",
                    source_modality="image",
                    model_name=item.get("model_name"),
                    timestamp=timestamp,
                    frame_name=frame_name,
                    image_path=image_path,
                    document_language=item.get("language", "en"),
                )
            )

        if not texts:
            logger.warning("No captions to index.")
            return 0

        embeddings = self._embed_texts(texts)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Indexed %d caption records", len(ids))
        return len(ids)

    def index_multimodal_documents(
        self,
        transcription_data: Dict[str, Any],
        captions_data: List[Dict[str, Any]],
    ) -> int:
        if not getattr(self, "enable_multimodal_documents", True):
            return 0

        video_name = transcription_data["video_name"]
        chunks = self._build_segment_chunks(transcription_data.get("segments", []))

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for chunk in chunks:
            nearby_captions = self._find_nearby_caption_texts(
                captions_data,
                start_time=chunk["start"],
                end_time=chunk["end"],
            )
            if not nearby_captions:
                continue

            merged_doc = (
                f"[Speech] {chunk['text']} "
                f"[Visual] {' | '.join(nearby_captions[:3])}"
            ).strip()

            payload = {
                "video_name": video_name,
                "type": "multimodal",
                "start": chunk["start"],
                "end": chunk["end"],
                "text": merged_doc,
            }

            ids.append(self._stable_id("multimodal", payload))
            texts.append(merged_doc)
            metadatas.append(
                self._base_metadata(
                    video_name=video_name,
                    content_type="multimodal",
                    source_modality="audio+image",
                    timestamp=chunk["start"],
                    start_time=chunk["start"],
                    end_time=chunk["end"],
                    document_language="vi+en",
                    extra={
                        "num_attached_captions": len(nearby_captions[:3]),
                    },
                )
            )

        if not texts:
            logger.info("No multimodal documents created for '%s'", video_name)
            return 0

        embeddings = self._embed_texts(texts)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Indexed %d multimodal records for '%s'", len(ids), video_name)
        return len(ids)

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_dir": str(self.persist_dir),
                "embedding_model": self.embedding_model_name,
                "distance_metric": self.distance_metric,
                "pipeline_version": self.pipeline_version,
            }
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {"error": str(e)}