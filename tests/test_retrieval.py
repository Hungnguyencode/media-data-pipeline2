import numpy as np
import pytest

from src.retrieval.search_engine import SearchEngine


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        batch_size=None,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            length = float(len(text))
            vectors.append([length, 1.0, 0.5])

        return np.array(vectors, dtype=float)


class FakeCollection:
    def __init__(self):
        self.last_where = None
        self.last_n_results = None

    def query(self, query_embeddings, n_results=5, where=None, include=None):
        self.last_where = where
        self.last_n_results = n_results

        return {
            "documents": [[
                "giới thiệu về trí tuệ nhân tạo",
                "một người đang đứng trước bảng trình chiếu",
            ]],
            "metadatas": [[
                {
                    "video_name": "demo.mp4",
                    "content_type": "transcription",
                    "timestamp": 1.23,
                    "timestamp_str": "00:00:01",
                    "source_modality": "audio",
                },
                {
                    "video_name": "demo.mp4",
                    "content_type": "caption",
                    "timestamp": 5.0,
                    "timestamp_str": "00:00:05",
                    "frame_name": "frame_0001_5.00s.jpg",
                    "source_modality": "image",
                },
            ]],
            "distances": [[0.1, 0.3]],
        }


class FakeVectorIndexer:
    def __init__(self):
        self.collection = FakeCollection()
        self.embedding_model = FakeEmbeddingModel()


def test_search_returns_results_with_score_type():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(config={}, vector_indexer=indexer)

    results = engine.search(query="trí tuệ nhân tạo", top_k=2)

    assert len(results) == 2
    assert results[0]["document"] == "giới thiệu về trí tuệ nhân tạo"
    assert results[0]["metadata"]["video_name"] == "demo.mp4"
    assert results[0]["score_type"] == "similarity_proxy_from_distance"
    assert results[0]["similarity_score"] == pytest.approx(0.9)
    assert indexer.collection.last_n_results == 2


def test_search_empty_query_returns_empty_list():
    engine = SearchEngine(config={}, vector_indexer=FakeVectorIndexer())
    assert engine.search("   ") == []


def test_search_with_both_filters_builds_and_where_clause():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(config={}, vector_indexer=indexer)

    results = engine.search(
        query="bảng trình chiếu",
        top_k=2,
        content_type="caption",
        video_name="demo.mp4",
    )

    assert len(results) == 2
    assert indexer.collection.last_where == {
        "$and": [
            {"content_type": "caption"},
            {"video_name": "demo.mp4"},
        ]
    }


def test_search_with_single_filter_builds_simple_where_clause():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(config={}, vector_indexer=indexer)

    engine.search(
        query="caption",
        top_k=2,
        content_type="caption",
    )

    assert indexer.collection.last_where == {"content_type": "caption"}


def test_search_top_k_is_capped():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(config={"pipeline": {"max_top_k": 10}}, vector_indexer=indexer)

    engine.search(query="demo", top_k=999)

    assert indexer.collection.last_n_results == 10