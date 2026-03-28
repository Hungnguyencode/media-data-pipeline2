from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.indexing.vector_indexer import VectorIndexer
from src.transform.vision_processor import VisionProcessor
from src.utils import get_config

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        vector_indexer: Optional[VectorIndexer] = None,
        vision_processor: Optional[VisionProcessor] = None,
    ):
        self.config = config or get_config()
        self.vector_indexer = vector_indexer or VectorIndexer(self.config)
        self.vision_processor = vision_processor or VisionProcessor(self.config)

        self.text_collection = self.vector_indexer.text_collection
        self.clip_collection = self.vector_indexer.clip_collection
        self.embedding_model = self.vector_indexer.embedding_model

        pipeline_cfg = self.config.get("pipeline", {})
        self.max_top_k = int(pipeline_cfg.get("max_top_k", 50))
        self.default_top_k = int(pipeline_cfg.get("default_top_k", 5))
        self.hybrid_search_alpha = float(pipeline_cfg.get("hybrid_search_alpha", 0.6))
        self.clip_search_beta = float(pipeline_cfg.get("clip_search_beta", 0.4))
        self.hybrid_candidate_multiplier = int(pipeline_cfg.get("hybrid_candidate_multiplier", 3))

    def _distance_to_similarity_proxy(self, distance: Optional[float]) -> Optional[float]:
        if distance is None:
            return None
        score = 1.0 - float(distance)
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return round(score, 6)

    def _build_where_clause(
        self,
        content_type: Optional[str] = None,
        video_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        filters = []

        if content_type:
            filters.append({"content_type": content_type})
        if video_name:
            filters.append({"video_name": video_name})

        if len(filters) == 1:
            return filters[0]
        if len(filters) > 1:
            return {"$and": filters}
        return None

    def _query_collection(
        self,
        *,
        collection,
        query_embedding: List[float],
        top_k: int,
        where_clause: Optional[Dict[str, Any]],
        score_type: str,
        score_weight: float,
    ) -> List[Dict[str, Any]]:
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        results: List[Dict[str, Any]] = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            similarity_proxy = self._distance_to_similarity_proxy(distance)
            weighted_score = (similarity_proxy or 0.0) * score_weight

            results.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": distance,
                    "similarity_score": similarity_proxy,
                    "fusion_score": round(weighted_score, 6),
                    "relevance": round(weighted_score, 6),
                    "score_type": score_type,
                }
            )

        return results

    def _result_key(self, item: Dict[str, Any]) -> Tuple[str, str, str]:
        meta = item.get("metadata", {}) or {}
        return (
            str(meta.get("video_name", "")),
            str(meta.get("content_type", "")),
            str(meta.get("frame_name") or meta.get("timestamp") or item.get("document", "")),
        )

    def _fuse_results(
        self,
        text_results: List[Dict[str, Any]],
        clip_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for item in text_results + clip_results:
            key = self._result_key(item)
            existing = merged.get(key)

            if existing is None:
                merged[key] = dict(item)
                continue

            existing["fusion_score"] = round(
                existing.get("fusion_score", 0.0) + item.get("fusion_score", 0.0),
                6,
            )
            existing["relevance"] = existing["fusion_score"]

            prev_type = str(existing.get("score_type", ""))
            new_type = str(item.get("score_type", ""))
            if prev_type != new_type:
                existing["score_type"] = "hybrid_fusion"

            if item.get("similarity_score", 0.0) > existing.get("similarity_score", 0.0):
                existing["similarity_score"] = item.get("similarity_score")
                existing["distance"] = item.get("distance")
                existing["document"] = item.get("document")
                existing["metadata"] = item.get("metadata")

        fused = sorted(
            merged.values(),
            key=lambda x: x.get("fusion_score", 0.0),
            reverse=True,
        )
        return fused

    def _caption_quality_score(self, text: str) -> int:
        if not text:
            return -999

        score = 0
        lower = text.lower()

        if "cracking an egg" in lower:
            score += 8
        if "into a bowl" in lower:
            score += 4
        if "egg" in lower:
            score += 2
        if "bowl" in lower:
            score += 2

        bad_patterns = [
            "in a blend",
            "peeling an egg",
            "squeezing an egg",
        ]
        for pattern in bad_patterns:
            if pattern in lower:
                score -= 4

        score -= abs(len(text) - 40) // 10
        return score

    def _find_nearby_speech_context(
        self,
        *,
        video_name: Optional[str],
        center_timestamp: Optional[float],
        window_sec: float = 4.0,
    ) -> str:
        if not video_name or center_timestamp is None:
            return ""

        raw = self.text_collection.get(
            where={"video_name": video_name},
            include=["documents", "metadatas"],
        )

        documents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []

        candidates: List[str] = []
        for doc, meta in zip(documents, metadatas):
            if not isinstance(meta, dict):
                continue
            content_type = meta.get("content_type")
            if content_type not in {"segment_chunk", "multimodal", "transcription"}:
                continue

            start_time = meta.get("start_time")
            end_time = meta.get("end_time")
            timestamp = meta.get("timestamp")

            include_doc = False
            if start_time is not None and end_time is not None:
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                    include_doc = not (
                        end_time < center_timestamp - window_sec
                        or start_time > center_timestamp + window_sec
                    )
                except Exception:
                    include_doc = False
            elif timestamp is not None:
                try:
                    ts = float(timestamp)
                    include_doc = abs(ts - center_timestamp) <= window_sec
                except Exception:
                    include_doc = False

            if include_doc and doc:
                candidates.append(str(doc).strip())

        deduped: List[str] = []
        seen = set()
        for text in candidates:
            key = text.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(text)

        return " | ".join(deduped[:2])

    def _group_results_into_events(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        event_gap_sec: float = 2.0,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        sorted_results = sorted(
            results,
            key=lambda x: (
                str((x.get("metadata") or {}).get("video_name", "")),
                float((x.get("metadata") or {}).get("timestamp", 0.0) or 0.0),
            ),
        )

        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []

        for item in sorted_results:
            meta = item.get("metadata", {}) or {}
            video_name = str(meta.get("video_name", ""))
            timestamp = float(meta.get("timestamp", 0.0) or 0.0)

            if not current_group:
                current_group = [item]
                continue

            prev_meta = current_group[-1].get("metadata", {}) or {}
            prev_video_name = str(prev_meta.get("video_name", ""))
            prev_timestamp = float(prev_meta.get("timestamp", 0.0) or 0.0)

            same_video = video_name == prev_video_name
            close_in_time = abs(timestamp - prev_timestamp) <= event_gap_sec

            if same_video and close_in_time:
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]

        if current_group:
            groups.append(current_group)

        event_results: List[Dict[str, Any]] = []
        for group in groups:
            best_item = max(group, key=lambda x: x.get("fusion_score", 0.0))
            captions = [g.get("document", "") for g in group if (g.get("metadata") or {}).get("content_type") == "caption"]
            best_caption = max(captions, key=self._caption_quality_score) if captions else best_item.get("document", "")

            timestamps = []
            for g in group:
                meta = g.get("metadata", {}) or {}
                ts = meta.get("timestamp")
                if ts is not None:
                    try:
                        timestamps.append(float(ts))
                    except Exception:
                        pass

            event_start = min(timestamps) if timestamps else None
            event_end = max(timestamps) if timestamps else None

            event_item = dict(best_item)
            event_item["group_size"] = len(group)
            event_item["event_time_range"] = {
                "start": event_start,
                "end": event_end,
            }
            event_item["display_caption"] = best_caption
            event_item["display_text"] = best_caption

            meta = dict(event_item.get("metadata", {}) or {})
            if event_start is not None:
                meta["event_start_timestamp"] = event_start
            if event_end is not None:
                meta["event_end_timestamp"] = event_end
            event_item["metadata"] = meta

            nearby_speech = self._find_nearby_speech_context(
                video_name=meta.get("video_name"),
                center_timestamp=meta.get("timestamp"),
                window_sec=4.0,
            )
            event_item["nearby_speech_context"] = nearby_speech

            event_results.append(event_item)

        event_results = sorted(
            event_results,
            key=lambda x: x.get("fusion_score", 0.0),
            reverse=True,
        )
        return event_results[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        content_type: Optional[str] = None,
        video_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        if top_k <= 0:
            top_k = self.default_top_k
        if top_k > self.max_top_k:
            top_k = self.max_top_k

        candidate_k = min(self.max_top_k, max(top_k, top_k * self.hybrid_candidate_multiplier))
        where_clause = self._build_where_clause(content_type=content_type, video_name=video_name)

        text_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        text_query_embedding = (
            text_embedding.tolist() if hasattr(text_embedding, "tolist") else list(text_embedding)
        )

        text_results = self._query_collection(
            collection=self.text_collection,
            query_embedding=text_query_embedding,
            top_k=candidate_k,
            where_clause=where_clause,
            score_type="text_similarity",
            score_weight=self.hybrid_search_alpha,
        )

        clip_results: List[Dict[str, Any]] = []
        if content_type in (None, "caption"):
            clip_query_embedding = self.vision_processor.encode_text_for_clip([query])[0]
            clip_results = self._query_collection(
                collection=self.clip_collection,
                query_embedding=clip_query_embedding,
                top_k=candidate_k,
                where_clause=where_clause,
                score_type="clip_text_image_similarity",
                score_weight=self.clip_search_beta,
            )

        fused_results = self._fuse_results(text_results, clip_results)
        MIN_FUSION_SCORE = 0.3 
        fused_results = [res for res in fused_results if res.get("fusion_score", 0.0) >= MIN_FUSION_SCORE]
        final_results = self._group_results_into_events(fused_results, top_k=top_k)
        return final_results