from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.indexing.vector_indexer import VectorIndexer
from src.utils import get_config

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        vector_indexer: Optional[VectorIndexer] = None,
    ):
        self.config = config or get_config()
        self.vector_indexer = vector_indexer or VectorIndexer(self.config)
        self.collection = self.vector_indexer.collection
        self.embedding_model = self.vector_indexer.embedding_model
        self.max_top_k = int(self.config.get("pipeline", {}).get("max_top_k", 50))
        self.default_top_k = int(self.config.get("pipeline", {}).get("default_top_k", 5))

    def _distance_to_similarity_proxy(self, distance: Optional[float]) -> Optional[float]:
        if distance is None:
            return None
        score = 1.0 - float(distance)
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return round(score, 6)

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

        embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        query_embedding = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        where_clause: Optional[Dict[str, Any]] = None
        filters = []

        if content_type:
            filters.append({"content_type": content_type})
        if video_name:
            filters.append({"video_name": video_name})

        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}

        raw = self.collection.query(
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
            results.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": distance,
                    "similarity_score": similarity_proxy,
                    "relevance": similarity_proxy,
                    "score_type": "similarity_proxy_from_distance",
                }
            )

        return results