from __future__ import annotations
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
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

        retrieval_cfg = self.config.get("retrieval", {})
        self.topic_bonus_for_talk = float(retrieval_cfg.get("topic_bonus_for_talk", 0.08))
        self.action_bonus_for_action_video = float(retrieval_cfg.get("action_bonus_for_action_video", 0.06))
        self.visual_bonus_for_visual_video = float(retrieval_cfg.get("visual_bonus_for_visual_video", 0.06))
        self.audio_penalty_for_cinematic_music = float(retrieval_cfg.get("audio_penalty_for_cinematic_music", 0.08))
        self.multimodal_bonus_for_talk = float(retrieval_cfg.get("multimodal_bonus_for_talk", 0.04))

        self.query_action_groups: Dict[str, Set[str]] = {
            "break_open": {
                "break", "breaking", "crack", "cracking", "open", "opening",
                "split", "splitting", "separate", "separating", "shell", "shelling",
            },
            "cut_divide": {
                "cut", "cutting", "slice", "slicing", "chop", "chopping",
                "dice", "dicing", "halve", "halving",
            },
            "mix_agitate": {
                "mix", "mixing", "stir", "stirring", "whisk", "whisking",
                "beat", "beating", "blend", "blending",
            },
            "pour_transfer": {
                "pour", "pouring", "add", "adding", "transfer", "transferring",
                "empty", "emptying",
            },
            "peel_remove_outer": {
                "peel", "peeling", "remove", "removing", "strip", "stripping",
            },
            "hold_pick_place": {
                "hold", "holding", "pick", "picking", "place", "placing",
                "put", "putting", "grab", "grabbing",
            },
            "squeeze_press": {
                "squeeze", "squeezing", "press", "pressing", "pinch", "pinching",
            },
        }

        self.object_like_tokens: Set[str] = {
            "egg", "eggs", "milk", "water", "oil", "juice", "sauce", "cream",
            "bowl", "spoon", "cup", "glass", "knife", "pan", "jar", "bottle",
            "onion", "garlic", "tomato", "fruit", "whisk", "yolk", "white",
            "bird", "sky", "forest", "road", "car", "beach", "ocean", "animal",
            "bed", "camera", "face", "portrait",
        }

        self.person_like_tokens: Set[str] = {
            "person", "people",
            "man", "men",
            "woman", "women",
            "girl", "girls",
            "boy", "boys",
            "someone", "face",
            "hand", "hands",
        }

        self.human_scene_tokens: Set[str] = {
            "camera", "smile", "smiling", "wave", "waving",
            "bed", "lying", "laying",
            "portrait", "selfie", "face",
            "close", "up",
        }

        self.audio_only_content_types = {"transcription", "segment_chunk"}
        self.visual_content_types = {"caption"}
        self.multimodal_content_types = {"multimodal"}
        self.cross_encoder_reranker = CrossEncoderReranker(self.config)

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
            meta = meta or {}
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
            str(meta.get("frame_name") or meta.get("timestamp") or meta.get("start_time") or item.get("document", "")),
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

    def _contains_number_two(self, text: str) -> bool:
        normalized = f" {self._normalize_text(text)} "
        return (" two " in normalized) or (" 2 " in normalized)

    def _classify_query_type(self, query: str) -> str:
        tokens = self._tokenize(query)
        if not tokens:
            return "generic"

        topic_tokens = {
            "happiness", "relationship", "relationships", "connection", "motivation",
            "strategy", "speech", "idea", "meaning", "topic", "human", "love",
            "hanh", "phuc", "ket", "noi", "moi", "quan", "he",
        }
        visual_tokens = {
            "sky", "forest", "bird", "animal", "ocean", "beach", "road", "scene",
            "cảnh", "bầu", "trời", "chim", "rừng",
        }
        human_visual_tokens = {
            "camera", "face", "portrait", "smile", "smiling", "waving", "wave", "bed",
            "guy", "guys", "girl", "girls", "boy", "boys", "man", "men", "woman", "women",
            "people", "person",
        }

        action_groups = self._extract_query_action_groups(query)

        if action_groups:
            return "action"
        if tokens.intersection(topic_tokens):
            return "topic"
        if tokens.intersection(visual_tokens):
            return "visual"
        if tokens.intersection(human_visual_tokens):
            return "visual"
        if any(token in tokens for token in {"speech", "audio", "voice", "nói", "giọng"}):
            return "audio"
        return "generic"

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
                str(item.get("document", "") or ""),
            ]
        )
        tokens = self._tokenize(text)
        return bool(tokens.intersection(self.person_like_tokens))

    def _has_human_scene_signal(self, item: Dict[str, Any]) -> bool:
        meta = item.get("metadata", {}) or {}
        text = " ".join(
            [
                str(meta.get("caption_text_original", "") or ""),
                str(meta.get("search_text", "") or ""),
                str(item.get("document", "") or ""),
            ]
        )
        tokens = self._tokenize(text)
        if tokens.intersection(self.human_scene_tokens):
            return True

        normalized = self._normalize_text(text)
        return "close up" in normalized

    def _has_multiple_people_signal(self, item: Dict[str, Any], query: str) -> bool:
        meta = item.get("metadata", {}) or {}
        text = " ".join(
            [
                query,
                str(meta.get("caption_text_original", "") or ""),
                str(meta.get("search_text", "") or ""),
                str(item.get("document", "") or ""),
            ]
        )
        normalized = self._normalize_text(text)
        return (
            "two people" in normalized
            or "two men" in normalized
            or "two women" in normalized
            or "two girls" in normalized
            or "two boys" in normalized
            or "two guys" in normalized
            or "2 people" in normalized
            or "2 men" in normalized
            or "2 women" in normalized
            or "2 guys" in normalized
            or self._contains_number_two(text)
        )

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

    def _metadata_bonus(self, item: Dict[str, Any], query_type: str) -> float:
        meta = item.get("metadata", {}) or {}
        style = str(meta.get("estimated_content_style", "") or "").strip()
        content_type = str(meta.get("content_type", "") or "").strip()

        bonus = 0.0

        if query_type == "topic":
            if style == "talk":
                bonus += self.topic_bonus_for_talk
            if content_type == "multimodal":
                bonus += self.multimodal_bonus_for_talk

        elif query_type == "action":
            if style == "action":
                bonus += self.action_bonus_for_action_video
            if content_type in {"caption", "multimodal"}:
                bonus += 0.02

        elif query_type == "visual":
            if style in {"visual", "cinematic_music"}:
                bonus += self.visual_bonus_for_visual_video
            if content_type == "caption":
                bonus += 0.02
            if content_type == "multimodal":
                bonus += 0.01

        elif query_type == "audio":
            if style == "talk" and content_type in {"transcription", "segment_chunk", "multimodal"}:
                bonus += 0.03

        return round(bonus, 6)

    def _style_aware_modality_bonus(self, item: Dict[str, Any], query_type: str) -> float:
        meta = item.get("metadata", {}) or {}
        style = str(meta.get("estimated_content_style", "") or "").strip()
        content_type = str(meta.get("content_type", "") or "").strip()

        if style == "cinematic_music" and content_type in self.audio_only_content_types:
            return round(-self.audio_penalty_for_cinematic_music, 6)

        if style == "talk" and query_type == "topic" and content_type in {"segment_chunk", "multimodal"}:
            return 0.02

        if style == "action" and query_type == "action" and content_type in {"caption", "multimodal"}:
            return 0.02

        return 0.0

    def _human_scene_bonus(self, item: Dict[str, Any], query: str) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        human_query_tokens = {
            "camera", "face", "portrait", "smile", "smiling", "wave", "waving",
            "bed", "lying", "laying",
            "guy", "guys", "girl", "girls", "boy", "boys",
            "man", "men", "woman", "women", "people", "person",
        }

        if not query_tokens.intersection(human_query_tokens) and not self._contains_number_two(query):
            return 0.0

        bonus = 0.0

        has_person_signal = self._has_person_or_hand_signal(item)
        has_human_scene_signal = self._has_human_scene_signal(item)
        has_multiple_people_signal = self._has_multiple_people_signal(item, query)

        if has_person_signal:
            bonus += 0.02
        if has_human_scene_signal:
            bonus += 0.03
        if self._contains_number_two(query) and has_multiple_people_signal:
            bonus += 0.02

        return round(bonus, 6)

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

        human_scene_bonus = self._human_scene_bonus(item, query)

        final_bonus = (
            token_bonus
            + alias_bonus
            + action_bonus
            + object_bonus
            + static_penalty
            + modality_bonus
            + human_scene_bonus
        )
        return round(final_bonus, 6)

    def _apply_soft_rerank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        reranked: List[Dict[str, Any]] = []
        query_type = self._classify_query_type(query)

        for item in results:
            semantic_bonus = self._soft_semantic_bonus(item, query)
            metadata_bonus = self._metadata_bonus(item, query_type)
            style_bonus = self._style_aware_modality_bonus(item, query_type)
            total_bonus = semantic_bonus + metadata_bonus + style_bonus

            updated = dict(item)
            updated["fusion_score"] = round(updated.get("fusion_score", 0.0) + total_bonus, 6)
            updated["relevance"] = updated["fusion_score"]
            updated["query_type"] = query_type

            if total_bonus != 0:
                updated["score_type"] = "hybrid_fusion+rerank"

            reranked.append(updated)

        reranked.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)
        return reranked

    def _get_anchor_timestamp(self, meta: Dict[str, Any]) -> Optional[float]:
        timestamp = meta.get("timestamp")
        if timestamp is not None:
            try:
                return float(timestamp)
            except Exception:
                pass

        start_time = meta.get("start_time")
        end_time = meta.get("end_time")

        try:
            if start_time is not None and end_time is not None:
                return (float(start_time) + float(end_time)) / 2.0
        except Exception:
            pass

        try:
            if start_time is not None:
                return float(start_time)
        except Exception:
            pass

        try:
            if end_time is not None:
                return float(end_time)
        except Exception:
            pass

        return None

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

        sortable: List[Tuple[str, float, Dict[str, Any]]] = []
        remainder: List[Dict[str, Any]] = []

        for item in results:
            meta = item.get("metadata", {}) or {}
            video_name = str(meta.get("video_name", ""))
            anchor_ts = self._get_anchor_timestamp(meta)
            if anchor_ts is None:
                remainder.append(item)
            else:
                sortable.append((video_name, anchor_ts, item))

        sortable.sort(key=lambda x: (x[0], x[1]))

        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []
        current_video = ""
        current_anchor = None

        for video_name, anchor_ts, item in sortable:
            if not current_group:
                current_group = [item]
                current_video = video_name
                current_anchor = anchor_ts
                continue

            same_video = video_name == current_video
            close_in_time = current_anchor is not None and abs(anchor_ts - current_anchor) <= event_gap_sec

            if same_video and close_in_time:
                current_group.append(item)
                current_anchor = anchor_ts
            else:
                groups.append(current_group)
                current_group = [item]
                current_video = video_name
                current_anchor = anchor_ts

        if current_group:
            groups.append(current_group)

        for item in remainder:
            groups.append([item])

        event_results: List[Dict[str, Any]] = []
        for group in groups:
            best_item = max(group, key=lambda x: x.get("fusion_score", 0.0))

            starts = []
            ends = []
            anchors = []

            for g in group:
                meta = g.get("metadata", {}) or {}
                start_time = meta.get("start_time")
                end_time = meta.get("end_time")
                anchor_ts = self._get_anchor_timestamp(meta)

                if anchor_ts is not None:
                    anchors.append(anchor_ts)

                try:
                    if start_time is not None:
                        starts.append(float(start_time))
                except Exception:
                    pass

                try:
                    if end_time is not None:
                        ends.append(float(end_time))
                except Exception:
                    pass

            if not starts and anchors:
                starts = anchors[:]
            if not ends and anchors:
                ends = anchors[:]

            event_start = min(starts) if starts else None
            event_end = max(ends) if ends else None

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
                center_timestamp=self._get_anchor_timestamp(meta),
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
        text_query_embedding = text_embedding.tolist() if hasattr(text_embedding, "tolist") else list(text_embedding)

        text_results = self._query_collection(
            collection=self.text_collection,
            query_embedding=text_query_embedding,
            top_k=candidate_k,
            where_clause=where_clause,
            score_type="text_similarity",
            score_weight=self.hybrid_search_alpha,
        )

        clip_results: List[Dict[str, Any]] = []
        if content_type in (None, "caption", "multimodal"):
            clip_where = self._build_where_clause(content_type="caption", video_name=video_name)
            clip_query_embedding = self.vision_processor.encode_text_for_clip([query])[0]
            clip_results = self._query_collection(
                collection=self.clip_collection,
                query_embedding=clip_query_embedding,
                top_k=candidate_k,
                where_clause=clip_where,
                score_type="clip_text_image_similarity",
                score_weight=self.clip_search_beta,
            )

            fused_results = self._fuse_results(text_results, clip_results)
            reranked_results = self._apply_soft_rerank(fused_results, query)

            # Vòng 2: Cross-Encoder reranking
            reranked_results = self.cross_encoder_reranker.rerank(query, reranked_results)

            final_results = self._group_results_into_events(reranked_results, top_k=top_k)
            return final_results