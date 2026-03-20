import numpy as np

from src.indexing.vector_indexer import VectorIndexer


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        batch_size=None,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ):
        return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=float)


class FakeCollection:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, ids, documents, embeddings, metadatas):
        self.upsert_calls.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

    def get(self, where=None):
        return {"ids": []}

    def count(self):
        return 0


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name):
        return self.collection

    def create_collection(self, name, metadata=None):
        return self.collection


class TestableVectorIndexer(VectorIndexer):
    __test__ = False

    def __init__(self):
        self.config = {
            "paths": {"vector_db_dir": "data/vector_db"},
            "vector_db": {
                "collection_name": "video_semantic_search",
                "distance_metric": "cosine",
            },
            "models": {
                "embedding": {
                    "name": "fake-model",
                    "batch_size": 32,
                }
            },
            "pipeline": {
                "segment_window": 2,
                "segment_overlap": 1,
                "caption_merge_window_sec": 3.0,
                "enable_multimodal_documents": True,
                "version": "1.2.0",
            },
        }
        self.device = "cpu"
        self.persist_dir = "data/vector_db"
        self.collection_name = "video_semantic_search"
        self.distance_metric = "cosine"
        self.embedding_model_name = "fake-model"
        self.batch_size = 32
        self.segment_window = 2
        self.segment_overlap = 1
        self.caption_merge_window_sec = 3.0
        self.enable_multimodal_documents = True
        self.pipeline_version = "1.2.0"
        self.embedding_model = FakeEmbeddingModel()
        self.collection = FakeCollection()
        self.client = FakeClient(self.collection)


def test_index_transcriptions_upserts_records():
    indexer = TestableVectorIndexer()

    transcription_data = {
        "video_name": "demo.mp4",
        "full_text": "Xin chao day la video demo",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Xin chao"},
            {"start": 1.0, "end": 2.0, "text": "day la"},
            {"start": 2.0, "end": 3.0, "text": "video demo"},
        ],
        "model_name": "whisper-base",
    }

    count = indexer.index_transcriptions(transcription_data)

    assert count >= 2
    assert len(indexer.collection.upsert_calls) == 1
    docs = indexer.collection.upsert_calls[0]["documents"]
    metas = indexer.collection.upsert_calls[0]["metadatas"]
    assert any("Xin chao day la video demo" in doc for doc in docs)
    assert any(meta["content_type"] == "transcription" for meta in metas)
    assert any(meta["content_type"] == "segment_chunk" for meta in metas)


def test_index_captions_upserts_caption_records():
    indexer = TestableVectorIndexer()

    captions_data = [
        {
            "video_name": "demo.mp4",
            "frame_name": "frame_0001_1.00s.jpg",
            "image_path": "data/interim/frames/demo/frame_0001_1.00s.jpg",
            "caption": "A person standing in front of a screen",
            "timestamp": 1.0,
            "timestamp_str": "00:00:01",
            "model_name": "blip-base",
        }
    ]

    count = indexer.index_captions(captions_data)

    assert count == 1
    assert len(indexer.collection.upsert_calls) == 1
    meta = indexer.collection.upsert_calls[0]["metadatas"][0]
    assert meta["content_type"] == "caption"
    assert meta["source_modality"] == "image"


def test_index_multimodal_documents_upserts_when_caption_nearby():
    indexer = TestableVectorIndexer()

    transcription_data = {
        "video_name": "demo.mp4",
        "full_text": "demo",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "gioi thieu bai thuyet trinh"},
            {"start": 2.0, "end": 4.0, "text": "ve tri tue nhan tao"},
        ],
        "model_name": "whisper-base",
    }

    captions_data = [
        {
            "video_name": "demo.mp4",
            "frame_name": "frame_0001_1.00s.jpg",
            "image_path": "frame.jpg",
            "caption": "A presentation slide on the screen",
            "timestamp": 1.0,
            "timestamp_str": "00:00:01",
        }
    ]

    count = indexer.index_multimodal_documents(transcription_data, captions_data)

    assert count >= 1
    assert len(indexer.collection.upsert_calls) == 1
    doc = indexer.collection.upsert_calls[0]["documents"][0]
    meta = indexer.collection.upsert_calls[0]["metadatas"][0]
    assert "[Speech]" in doc
    assert "[Visual]" in doc
    assert meta["content_type"] == "multimodal"
    assert meta["source_modality"] == "audio+image"