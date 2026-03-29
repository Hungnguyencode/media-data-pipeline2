from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
        self.hybrid_search_alpha = float(pipeline_cfg.get("hybrid_search_alpha", 0.35))
        self.clip_search_beta = float(pipeline_cfg.get("clip_search_beta", 0.65))
        self.hybrid_candidate_multiplier = int(pipeline_cfg.get("hybrid_candidate_multiplier", 3))

        self.query_action_groups: Dict[str, Set[str]] = {
            "break_open": {"break", "breaking", "crack", "cracking", "open", "opening", "split", "splitting", "separate", "separating", "shell", "shelling"},
            "cut_divide": {"cut", "cutting", "slice", "slicing", "chop", "chopping", "dice", "dicing", "halve", "halving"},
            "mix_agitate": {"mix", "mixing", "stir", "stirring", "whisk", "whisking", "beat", "beating", "blend", "blending"},
            "pour_transfer": {"pour", "pouring", "add", "adding", "transfer", "transferring", "empty", "emptying"},
            "peel_remove_outer": {"peel", "peeling", "remove", "removing", "strip", "stripping"},
            "hold_pick_place": {"hold", "holding", "pick", "picking", "place", "placing", "put", "putting", "grab", "grabbing"},
            "squeeze_press": {"squeeze", "squeezing", "press", "pressing", "pinch", "pinching"},
        }

        self.object_like_tokens: Set[str] = {
            "egg",
            "eggs",
            "milk",
            "water",
            "oil",
            "juice",
            "sauce",
            "cream",
            "bowl",
            "spoon",
            "cup",
            "glass",
            "knife",
            "pan",
            "jar",
            "bottle",
            "onion",
            "garlic",
            "tomato",
            "fruit",
            "whisk",
            "yolk",
            "white",
        }

        self.audio_only_content_types = {"transcription", "segment_chunk"}

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

    def _normalize_text(self, text: str) -> str:
        cleaned = (text or "").strip().lower()
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _tokenize(self, text: str) -> Set[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        return set(normalized.split())

    def _extract_query_action_groups(self, query: str) -> Set[str]:
        tokens = self._tokenize(query)
        matched_groups: Set[str] = set()
        for group, vocab in self.query_action_groups.items():
            if tokens.intersection(vocab):
                matched_groups.add(group)
        return matched_groups

    def _extract_action_groups_from_result(self, item: Dict[str, Any]) -> Set[str]:
        meta = item.get("metadata", {}) or {}

        search_space = " ".join(
            [
                str(meta.get("caption_text_original", "") or ""),
                str(meta.get("action_aliases", "") or ""),
            ]
        )

        tokens = self._tokenize(search_space)
        matched_groups: Set[str] = set()
        for group, vocab in self.query_action_groups.items():
            if tokens.intersection(vocab):
                matched_groups.add(group)
        return matched_groups

    def _display_text_for_result(self, item: Dict[str, Any]) -> str:
        meta = item.get("metadata", {}) or {}
        original = str(meta.get("caption_text_original", "") or "").strip()
        if original:
            return original
        return str(item.get("document", "") or "").strip()

    def _is_static_object_caption(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False

        static_patterns = [
            "a bowl with",
            "a plate with",
            "a cup with",
            "an egg in it",
            "eggs in a bowl",
            "a bowl of",
            "food in a bowl",
            "a glass bowl with",
            "two eggs are in",
            "a whisked egg in it",
            "empty bowl",
            "an empty bowl",
            "a bowl on a table",
        ]
        return any(pattern in normalized for pattern in static_patterns)

    def _has_person_or_hand_signal(self, item: Dict[str, Any]) -> bool:
        meta = item.get("metadata", {}) or {}
        text = " ".join(
            [
                str(meta.get("caption_text_original", "") or ""),
                str(meta.get("search_text", "") or ""),
            ]
        )
        tokens = self._tokenize(text)
        return bool(tokens.intersection({"person", "hand", "hands", "someone"}))

    def _modality_bonus(self, item: Dict[str, Any], has_action_query: bool, has_action_match: bool) -> float:
        meta = item.get("metadata", {}) or {}
        content_type = str(meta.get("content_type", "") or "")

        if not has_action_query:
            return 0.0

        if content_type == "caption":
            return 0.03 if has_action_match else 0.0
        if content_type == "multimodal":
            return 0.02 if has_action_match else 0.0
        if content_type in self.audio_only_content_types:
            return -0.04
        return 0.0

    def _soft_semantic_bonus(self, item: Dict[str, Any], query: str) -> float:
        meta = item.get("metadata", {}) or {}

        original_caption = str(meta.get("caption_text_original", "") or "")
        search_text = str(meta.get("search_text", "") or "")
        action_aliases = str(meta.get("action_aliases", "") or "")
        doc_text = str(item.get("document", "") or "")

        query_tokens = self._tokenize(query)
        original_tokens = self._tokenize(original_caption)
        alias_tokens = self._tokenize(action_aliases)
        combined_tokens = self._tokenize(" ".join([doc_text, original_caption, search_text, action_aliases]))

        if not query_tokens or not combined_tokens:
            return 0.0

        query_groups = self._extract_query_action_groups(query)
        result_groups = self._extract_action_groups_from_result(item)

        original_overlap = query_tokens.intersection(original_tokens)
        alias_overlap = query_tokens.intersection(alias_tokens)
        object_overlap = query_tokens.intersection(self.object_like_tokens)

        has_action_query = bool(query_groups)
        has_action_match = bool(query_groups.intersection(result_groups))
        has_object_overlap = bool(object_overlap)
        has_person_signal = self._has_person_or_hand_signal(item)

        token_bonus = min(0.03, len(original_overlap) * 0.012)
        alias_bonus = min(0.015, len(alias_overlap) * 0.005)
        action_bonus = 0.0
        object_bonus = 0.0
        static_penalty = 0.0
        modality_bonus = self._modality_bonus(item, has_action_query, has_action_match)

        if has_action_query:
            if has_action_match:
                action_bonus += 0.10
                if has_object_overlap:
                    object_bonus += 0.02
                if len(original_overlap) >= 2:
                    token_bonus += 0.015

                if not has_person_signal and self._is_static_object_caption(original_caption):
                    static_penalty -= 0.08
            else:
                if self._is_static_object_caption(original_caption):
                    static_penalty -= 0.12
                if has_object_overlap:
                    object_bonus += 0.002
                    static_penalty -= 0.04
                token_bonus = min(token_bonus, 0.01)
                alias_bonus = min(alias_bonus, 0.005)
        else:
            if has_object_overlap:
                object_bonus += 0.015

        final_bonus = token_bonus + alias_bonus + action_bonus + object_bonus + static_penalty + modality_bonus
        return round(final_bonus, 6)

    def _apply_soft_rerank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        reranked: List[Dict[str, Any]] = []

        for item in results:
            bonus = self._soft_semantic_bonus(item, query)
            updated = dict(item)
            updated["fusion_score"] = round(updated.get("fusion_score", 0.0) + bonus, 6)
            updated["relevance"] = updated["fusion_score"]
            if bonus != 0:
                updated["score_type"] = "hybrid_fusion+rerank"
            reranked.append(updated)

        reranked.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)
        return reranked

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

            best_document = self._display_text_for_result(best_item)
            event_item["display_caption"] = best_document
            event_item["display_text"] = best_document

            meta = dict(event_item.get("metadata") or {})
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
        reranked_results = self._apply_soft_rerank(fused_results, query)
        final_results = self._group_results_into_events(reranked_results, top_k=top_k)
        return final_results