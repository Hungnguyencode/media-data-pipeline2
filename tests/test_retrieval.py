import numpy as np
import pytest

from src.retrieval.search_engine import SearchEngine


class FakeCollection:
    def __init__(self, documents, metadatas, distances):
        self.documents = documents
        self.metadatas = metadatas
        self.distances = distances
        self.last_where = None
        self.last_n_results = None

    def query(self, query_embeddings, n_results=5, where=None, include=None):
        self.last_where = where
        self.last_n_results = n_results
        return {
            "documents": [self.documents],
            "metadatas": [self.metadatas],
            "distances": [self.distances],
        }

    def get(self, where=None, include=None):
        filtered_documents = []
        filtered_metadatas = []

        for doc, meta in zip(self.documents, self.metadatas):
            matched = True

            if where:
                for key, value in where.items():
                    if meta.get(key) != value:
                        matched = False
                        break

            if matched:
                filtered_documents.append(doc)
                filtered_metadatas.append(meta)

        result = {}
        if include:
            if "documents" in include:
                result["documents"] = filtered_documents
            if "metadatas" in include:
                result["metadatas"] = filtered_metadatas

        return result


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


class FakeVisionProcessor:
    def encode_text_for_clip(self, texts):
        return [[0.9, 0.8, 0.7] for _ in texts]


class FakeVectorIndexer:
    def __init__(self):
        self.embedding_model = FakeEmbeddingModel()

        self.text_collection = FakeCollection(
            documents=[
                "giới thiệu về trí tuệ nhân tạo",
                "một người đang đứng trước bảng trình chiếu",
                "xin chào đây là video demo",
            ],
            metadatas=[
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
                {
                    "video_name": "demo.mp4",
                    "content_type": "segment_chunk",
                    "timestamp": 4.5,
                    "start_time": 4.0,
                    "end_time": 6.0,
                    "start_time_str": "00:00:04",
                    "end_time_str": "00:00:06",
                    "source_modality": "audio",
                },
            ],
            distances=[0.1, 0.3, 0.2],
        )

        self.clip_collection = FakeCollection(
            documents=[
                "một người đang đứng trước bảng trình chiếu",
                "một người đang thuyết trình",
            ],
            metadatas=[
                {
                    "video_name": "demo.mp4",
                    "content_type": "caption",
                    "timestamp": 5.0,
                    "timestamp_str": "00:00:05",
                    "frame_name": "frame_0001_5.00s.jpg",
                    "source_modality": "image",
                },
                {
                    "video_name": "demo.mp4",
                    "content_type": "caption",
                    "timestamp": 7.0,
                    "timestamp_str": "00:00:07",
                    "frame_name": "frame_0002_7.00s.jpg",
                    "source_modality": "image",
                },
            ],
            distances=[0.05, 0.2],
        )


def test_search_returns_results_with_score_type():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(
        config={},
        vector_indexer=indexer,
        vision_processor=FakeVisionProcessor(),
    )

    results = engine.search(query="trí tuệ nhân tạo", top_k=2)

    assert len(results) >= 1
    assert results[0]["metadata"]["video_name"] == "demo.mp4"
    assert results[0]["score_type"] in {
        "hybrid_fusion",
        "text_similarity",
        "clip_text_image_similarity",
        "hybrid_fusion+rerank+cross_encoder",
    }
    assert results[0]["similarity_score"] is not None
    assert "display_text" in results[0]
    assert "nearby_speech_context" in results[0]
    assert indexer.text_collection.last_n_results == 6
    assert indexer.clip_collection.last_n_results == 6


def test_search_empty_query_returns_empty_list():
    engine = SearchEngine(
        config={},
        vector_indexer=FakeVectorIndexer(),
        vision_processor=FakeVisionProcessor(),
    )
    assert engine.search("   ") == []


def test_search_with_both_filters_builds_and_where_clause():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(
        config={},
        vector_indexer=indexer,
        vision_processor=FakeVisionProcessor(),
    )

    results = engine.search(
        query="bảng trình chiếu",
        top_k=2,
        content_type="caption",
        video_name="demo.mp4",
    )

    assert len(results) >= 1
    expected = {
        "$and": [
            {"content_type": "caption"},
            {"video_name": "demo.mp4"},
        ]
    }
    assert indexer.text_collection.last_where == expected
    assert indexer.clip_collection.last_where == expected


def test_search_with_single_filter_builds_simple_where_clause():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(
        config={},
        vector_indexer=indexer,
        vision_processor=FakeVisionProcessor(),
    )

    engine.search(
        query="caption",
        top_k=2,
        content_type="caption",
    )

    assert indexer.text_collection.last_where == {"content_type": "caption"}
    assert indexer.clip_collection.last_where == {"content_type": "caption"}


def test_search_top_k_is_capped():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(
        config={"pipeline": {"max_top_k": 10}},
        vector_indexer=indexer,
        vision_processor=FakeVisionProcessor(),
    )

    engine.search(query="demo", top_k=999)

    assert indexer.text_collection.last_n_results == 10
    assert indexer.clip_collection.last_n_results == 10


def test_search_skips_clip_for_non_caption_content_type():
    indexer = FakeVectorIndexer()
    engine = SearchEngine(
        config={},
        vector_indexer=indexer,
        vision_processor=FakeVisionProcessor(),
    )

    engine.search(query="audio topic", top_k=2, content_type="transcription")

    assert indexer.text_collection.last_n_results == 6
    assert indexer.clip_collection.last_n_results is None


def test_distance_to_similarity_proxy():
    engine = SearchEngine(
        config={},
        vector_indexer=FakeVectorIndexer(),
        vision_processor=FakeVisionProcessor(),
    )

    assert engine._distance_to_similarity_proxy(None) is None
    assert engine._distance_to_similarity_proxy(0.1) == pytest.approx(0.9)
    assert engine._distance_to_similarity_proxy(2.0) == 0.0

def test_metadata_aware_rerank_prefers_talk_for_topic_queries():
    indexer = FakeVectorIndexer()

    # override metadata cho rõ style
    indexer.text_collection.metadatas[0]["estimated_content_style"] = "talk"
    indexer.text_collection.metadatas[0]["content_type"] = "segment_chunk"

    engine = SearchEngine(
        config={
            "retrieval": {
                "topic_bonus_for_talk": 0.08,
                "action_bonus_for_action_video": 0.06,
                "visual_bonus_for_visual_video": 0.06,
                "audio_penalty_for_cinematic_music": 0.08,
                "multimodal_bonus_for_talk": 0.04,
            }
        },
        vector_indexer=indexer,
        vision_processor=FakeVisionProcessor(),
    )

    results = engine.search(query="human connection", top_k=3)
    assert len(results) >= 1
    assert results[0]["query_type"] in {"topic", "generic"}