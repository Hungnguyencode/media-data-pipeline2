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
        self.deleted_ids = []
        self.items = []

    def upsert(self, ids, documents, embeddings, metadatas):
        for metadata in metadatas:
            for key, value in metadata.items():
                assert value is not None, f"Metadata field '{key}' must not be None"

        self.upsert_calls.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

        for i in range(len(ids)):
            self.items.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                }
            )

    def get(self, where=None, include=None, limit=None, offset=None):
        filtered = self.items

        if where and "video_name" in where:
            filtered = [item for item in filtered if item["metadata"].get("video_name") == where["video_name"]]

        ids = [item["id"] for item in filtered]
        documents = [item["document"] for item in filtered]
        metadatas = [item["metadata"] for item in filtered]

        result = {"ids": ids}
        if include:
            if "documents" in include:
                result["documents"] = documents
            if "metadatas" in include:
                result["metadatas"] = metadatas
        return result

    def delete(self, ids):
        self.deleted_ids.extend(ids)
        self.items = [item for item in self.items if item["id"] not in ids]

    def count(self):
        return len(self.items)


class TestableVectorIndexer(VectorIndexer):
    def __init__(self):
        self.config = {
            "paths": {
                "vector_db_dir": "data/vector_db",
            },
            "models": {
                "embedding": {
                    "name": "fake-embedding-model",
                    "batch_size": 32,
                    "normalize_embeddings": True,
                }
            },
            "pipeline": {
                "version": "1.2.0",
                "segment_window": 3,
                "segment_overlap": 1,
                "caption_merge_window_sec": 3.0,
                "enable_multimodal_documents": True,
            },
            "vector_db": {
                "collection_name": "video_semantic_search",
                "distance_metric": "cosine",
            },
        }

        self.device = "cpu"
        self.persist_dir = None
        self.collection_name = "video_semantic_search"
        self.distance_metric = "cosine"
        self.embedding_model_name = "fake-embedding-model"
        self.batch_size = 32
        self.normalize_embeddings = True
        self.segment_window = 3
        self.segment_overlap = 1
        self.caption_merge_window_sec = 3.0
        self.enable_multimodal_documents = True
        self.pipeline_version = "1.2.0"
        self.embedding_model = FakeEmbeddingModel()
        self.collection = FakeCollection()

    def _get_or_create_collection(self):
        return self.collection


def test_base_metadata_does_not_include_none_values():
    indexer = TestableVectorIndexer()

    metadata = indexer._base_metadata(
        video_name="demo.mp4",
        content_type="transcription",
        source_modality="audio",
        model_name=None,
        timestamp=None,
        start_time=None,
        end_time=None,
        frame_name=None,
        image_path=None,
        document_language="vi",
    )

    assert metadata["video_name"] == "demo.mp4"
    assert metadata["content_type"] == "transcription"
    assert metadata["source_modality"] == "audio"
    assert metadata["document_language"] == "vi"

    assert "model_name" not in metadata
    assert "timestamp" not in metadata
    assert "timestamp_str" not in metadata
    assert "start_time" not in metadata
    assert "start_time_str" not in metadata
    assert "end_time" not in metadata
    assert "end_time_str" not in metadata
    assert "frame_name" not in metadata
    assert "image_path" not in metadata


def test_index_transcriptions():
    indexer = TestableVectorIndexer()

    transcription_data = {
        "video_name": "demo.mp4",
        "language": "vi",
        "full_text": "Xin chao day la video demo",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Xin chao"},
            {"start": 1.0, "end": 2.0, "text": "day la"},
            {"start": 2.0, "end": 3.0, "text": "video demo"},
        ],
        "model_name": "whisper-base",
    }

    count = indexer.index_transcriptions(transcription_data)

    assert count == 2
    assert len(indexer.collection.upsert_calls) == 1

    call = indexer.collection.upsert_calls[0]
    assert len(call["ids"]) == 2
    assert call["metadatas"][0]["content_type"] == "transcription"
    assert call["metadatas"][1]["content_type"] == "segment_chunk"


def test_index_captions():
    indexer = TestableVectorIndexer()

    captions_data = [
        {
            "video_name": "demo.mp4",
            "frame_name": "frame_0001_1.00s.jpg",
            "image_path": "data/interim/frames/demo/frame_0001_1.00s.jpg",
            "caption": "A person giving a presentation",
            "timestamp": 1.0,
            "model_name": "blip-base",
            "language": "en",
        }
    ]

    count = indexer.index_captions(captions_data)

    assert count == 1
    assert len(indexer.collection.upsert_calls) == 1

    call = indexer.collection.upsert_calls[0]
    assert len(call["ids"]) == 1
    assert call["metadatas"][0]["content_type"] == "caption"
    assert call["metadatas"][0]["source_modality"] == "image"


def test_index_multimodal_documents():
    indexer = TestableVectorIndexer()

    transcription_data = {
        "video_name": "demo.mp4",
        "language": "vi",
        "full_text": "Xin chao day la video demo",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Xin chao"},
            {"start": 1.0, "end": 2.0, "text": "day la"},
            {"start": 2.0, "end": 3.0, "text": "video demo"},
        ],
        "model_name": "whisper-base",
    }

    captions_data = [
        {
            "video_name": "demo.mp4",
            "frame_name": "frame_0001_1.00s.jpg",
            "image_path": "data/interim/frames/demo/frame_0001_1.00s.jpg",
            "caption": "A person giving a presentation",
            "timestamp": 1.0,
            "model_name": "blip-base",
            "language": "en",
        }
    ]

    count = indexer.index_multimodal_documents(transcription_data, captions_data)

    assert count == 1
    assert len(indexer.collection.upsert_calls) == 1

    call = indexer.collection.upsert_calls[0]
    assert len(call["ids"]) == 1
    assert call["metadatas"][0]["content_type"] == "multimodal"
    assert call["metadatas"][0]["source_modality"] == "audio+image"


def test_delete_video_data():
    indexer = TestableVectorIndexer()

    indexer.collection.items = [
        {
            "id": "id_1",
            "document": "doc 1",
            "metadata": {"video_name": "demo.mp4"},
        },
        {
            "id": "id_2",
            "document": "doc 2",
            "metadata": {"video_name": "other.mp4"},
        },
    ]

    deleted = indexer.delete_video_data("demo.mp4")

    assert deleted == 1
    assert "id_1" in indexer.collection.deleted_ids
    assert "id_2" not in indexer.collection.deleted_ids


def test_list_videos_and_inventory():
    indexer = TestableVectorIndexer()

    indexer.collection.items = [
        {
            "id": "id_1",
            "document": "doc 1",
            "metadata": {
                "video_name": "demo.mp4",
                "content_type": "transcription",
                "source_modality": "audio",
                "document_language": "vi",
                "pipeline_version": "1.2.0",
            },
        },
        {
            "id": "id_2",
            "document": "doc 2",
            "metadata": {
                "video_name": "demo.mp4",
                "content_type": "caption",
                "source_modality": "image",
                "document_language": "en",
                "pipeline_version": "1.2.0",
                "timestamp": 1.0,
            },
        },
    ]

    videos = indexer.list_videos()
    assert videos == ["demo.mp4"]

    inventory = indexer.get_video_inventory("demo.mp4")
    assert inventory["exists"] is True
    assert inventory["total_records"] == 2
    assert inventory["content_type_counts"]["transcription"] == 1
    assert inventory["content_type_counts"]["caption"] == 1


def test_get_all_videos_inventory():
    indexer = TestableVectorIndexer()

    indexer.collection.items = [
        {
            "id": "id_1",
            "document": "doc 1",
            "metadata": {
                "video_name": "demo.mp4",
                "content_type": "transcription",
                "source_modality": "audio",
                "document_language": "vi",
                "pipeline_version": "1.2.0",
            },
        },
        {
            "id": "id_2",
            "document": "doc 2",
            "metadata": {
                "video_name": "lesson.mp4",
                "content_type": "caption",
                "source_modality": "image",
                "document_language": "en",
                "pipeline_version": "1.2.0",
                "timestamp": 2.0,
            },
        },
    ]

    inventory = indexer.get_all_videos_inventory()
    assert inventory["total_videos"] == 2
    assert len(inventory["videos"]) == 2