from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from src.utils import get_config

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()

        reranker_cfg = self.config.get("reranker", {})
        self.enabled = bool(reranker_cfg.get("enabled", True))
        self.model_name = reranker_cfg.get(
            "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.top_k_rerank = int(reranker_cfg.get("top_k_rerank", 50))
        self.score_weight = float(reranker_cfg.get("score_weight", 0.5))

        self.model: Optional[CrossEncoder] = None

    def _load_model(self):
        if self.model is None:
            logger.info("Loading CrossEncoder model: %s", self.model_name)
            self.model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder model loaded.")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return results

        if not results:
            return results

        self._load_model()

        # Lấy text đại diện cho mỗi kết quả
        pairs = []
        for item in results:
            doc_text = (
                item.get("display_text")
                or item.get("document")
                or ""
            ).strip()
            pairs.append([query, doc_text])

        # Chạy Cross-Encoder chấm điểm từng cặp (query, document)
        ce_scores = self.model.predict(pairs)

        # Gắn điểm cross-encoder vào từng kết quả
        for item, ce_score in zip(results, ce_scores):
            item["cross_encoder_score"] = round(float(ce_score), 6)

        # Kết hợp: fusion_score cũ + cross_encoder_score (có trọng số)
        for item in results:
            old_score = item.get("fusion_score", 0.0)
            ce_score = item.get("cross_encoder_score", 0.0)

            # Normalize CE score về khoảng [0, 1] bằng sigmoid
            import math
            ce_normalized = 1 / (1 + math.exp(-ce_score))

            combined = (
                (1 - self.score_weight) * old_score
                + self.score_weight * ce_normalized
            )
            item["fusion_score"] = round(combined, 6)
            item["relevance"] = item["fusion_score"]
            item["score_type"] = "hybrid_fusion+rerank+cross_encoder"

        # Sắp xếp lại theo điểm mới
        reranked = sorted(
            results,
            key=lambda x: x.get("fusion_score", 0.0),
            reverse=True,
        )

        logger.info(
            "CrossEncoder reranked %d results for query: '%s'",
            len(reranked),
            query,
        )
        return reranked