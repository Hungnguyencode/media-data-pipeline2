from __future__ import annotations

import hashlib
import json
import logging
import re
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
            "pipeline_version": pipeline_version,
        }

        optional_fields = {
            "model_name": model_name,
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

        for key, value in optional_fields.items():
            if value is not None:
                metadata[key] = value

        if extra:
            for key, value in extra.items():
                if value is not None:
                    metadata[key] = value

        return metadata

    def _normalize_caption_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = text.strip().lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _deduplicate_caption_records(
        self,
        captions_data: List[Dict[str, Any]],
        min_time_gap_sec: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate captions within the same video by normalized text and nearby timestamp.
        Keep the earliest caption when the same normalized caption appears repeatedly
        in a short time window.
        """
        deduped: List[Dict[str, Any]] = []
        last_seen_by_video_and_text: Dict[tuple[str, str], float] = {}

        sorted_captions = sorted(
            captions_data,
            key=lambda x: (
                str(x.get("video_name", "")),
                float(x.get("timestamp", 0.0) or 0.0),
            ),
        )

        for item in sorted_captions:
            caption_text = (item.get("caption") or "").strip()
            if not caption_text:
                continue

            video_name = str(item.get("video_name", "")).strip()
            normalized = self._normalize_caption_text(caption_text)
            if not normalized:
                continue

            timestamp = float(item.get("timestamp", 0.0) or 0.0)
            key = (video_name, normalized)
            last_timestamp = last_seen_by_video_and_text.get(key)

            if last_timestamp is not None and abs(timestamp - last_timestamp) < min_time_gap_sec:
                continue

            deduped.append(item)
            last_seen_by_video_and_text[key] = timestamp

        return deduped

    def _deduplicate_texts_preserve_order(self, texts: List[str]) -> List[str]:
        seen: set[str] = set()
        deduped: List[str] = []

        for text in texts:
            cleaned = (text or "").strip()
            if not cleaned:
                continue

            normalized = self._normalize_caption_text(cleaned)
            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            deduped.append(cleaned)

        return deduped

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

    def _safe_collection_get(self, **kwargs) -> Dict[str, Any]:
        data = self.collection.get(**kwargs)
        return data or {}

    def list_videos(self) -> List[str]:
        try:
            data = self._safe_collection_get(include=["metadatas"])
            metadatas = data.get("metadatas", []) or []
            names = set()

            for meta in metadatas:
                if isinstance(meta, dict):
                    video_name = (meta.get("video_name") or "").strip()
                    if video_name:
                        names.add(video_name)

            return sorted(names)
        except Exception as e:
            logger.error("Failed to list videos: %s", e)
            return []

    def get_video_inventory(self, video_name: str) -> Dict[str, Any]:
        video_name = (video_name or "").strip()
        if not video_name:
            raise ValueError("video_name must not be empty")

        try:
            data = self._safe_collection_get(
                where={"video_name": video_name},
                include=["metadatas"],
            )
            metadatas = data.get("metadatas", []) or []
            ids = data.get("ids", []) or []

            content_type_counts = {
                "transcription": 0,
                "segment_chunk": 0,
                "caption": 0,
                "multimodal": 0,
            }
            source_modality_counts: Dict[str, int] = {}
            languages = set()
            pipeline_versions = set()

            min_timestamp = None
            max_timestamp = None
            min_start_time = None
            max_end_time = None

            for meta in metadatas:
                if not isinstance(meta, dict):
                    continue

                content_type = meta.get("content_type")
                if content_type in content_type_counts:
                    content_type_counts[content_type] += 1

                source_modality = (meta.get("source_modality") or "").strip()
                if source_modality:
                    source_modality_counts[source_modality] = (
                        source_modality_counts.get(source_modality, 0) + 1
                    )

                language = (meta.get("document_language") or "").strip()
                if language:
                    languages.add(language)

                pipeline_version = (meta.get("pipeline_version") or "").strip()
                if pipeline_version:
                    pipeline_versions.add(pipeline_version)

                timestamp = meta.get("timestamp")
                if timestamp is not None:
                    try:
                        timestamp = float(timestamp)
                        min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
                        max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)
                    except Exception:
                        pass

                start_time = meta.get("start_time")
                if start_time is not None:
                    try:
                        start_time = float(start_time)
                        min_start_time = (
                            start_time if min_start_time is None else min(min_start_time, start_time)
                        )
                    except Exception:
                        pass

                end_time = meta.get("end_time")
                if end_time is not None:
                    try:
                        end_time = float(end_time)
                        max_end_time = (
                            end_time if max_end_time is None else max(max_end_time, end_time)
                        )
                    except Exception:
                        pass

            return {
                "video_name": video_name,
                "exists": len(ids) > 0,
                "total_records": len(ids),
                "content_type_counts": content_type_counts,
                "source_modality_counts": source_modality_counts,
                "languages": sorted(languages),
                "pipeline_versions": sorted(pipeline_versions),
                "time_range": {
                    "min_timestamp": min_timestamp,
                    "max_timestamp": max_timestamp,
                    "min_start_time": min_start_time,
                    "max_end_time": max_end_time,
                },
            }
        except Exception as e:
            logger.error("Failed to get inventory for video '%s': %s", video_name, e)
            raise

    def get_all_videos_inventory(self) -> Dict[str, Any]:
        try:
            videos = self.list_videos()
            items = [self.get_video_inventory(video_name) for video_name in videos]

            return {
                "total_videos": len(videos),
                "videos": items,
            }
        except Exception as e:
            logger.error("Failed to get all video inventory: %s", e)
            raise

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
        captions_data = self._deduplicate_caption_records(captions_data, min_time_gap_sec=10.0)

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

        logger.info("Indexed %d deduplicated caption records", len(ids))
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
            nearby_captions = self._deduplicate_texts_preserve_order(nearby_captions)

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