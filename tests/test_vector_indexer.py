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
            filtered = [
                item for item in filtered
                if item["metadata"].get("video_name") == where["video_name"]
            ]

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


class DummyVectorIndexer(VectorIndexer):
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
                "version": "2.0.0",
                "segment_window": 3,
                "segment_overlap": 1,
                "caption_merge_window_sec": 3.0,
                "enable_multimodal_documents": True,
            },
            "vector_db": {
                "text_collection_name": "video_semantic_search_text",
                "clip_collection_name": "video_semantic_search_clip",
                "distance_metric": "cosine",
            },
        }

        self.device = "cpu"
        self.persist_dir = None
        self.text_collection_name = "video_semantic_search_text"
        self.clip_collection_name = "video_semantic_search_clip"
        self.distance_metric = "cosine"
        self.embedding_model_name = "fake-embedding-model"
        self.batch_size = 32
        self.normalize_embeddings = True
        self.segment_window = 3
        self.segment_overlap = 1
        self.caption_merge_window_sec = 3.0
        self.enable_multimodal_documents = True
        self.pipeline_version = "2.0.0"
        self.embedding_model = FakeEmbeddingModel()
        self.text_collection = FakeCollection()
        self.clip_collection = FakeCollection()


VIDEO_SOURCE_INFO = {
    "source_platform": "youtube",
    "source_url": "https://youtube.com/example",
    "video_title": "Demo Video",
    "video_description": "A demo video for testing.",
    "thumbnail_url": "",
    "video_tags": "demo|test",
    "local_video_path": "data/raw/demo.mp4",
    "created_at": "2026-03-27T00:00:00",
    "ingested_at": "2026-03-27T00:00:00",
    "ingest_method": "youtube_url",
    "has_audio": True,
    "video_type": "talk",
    "estimated_content_style": "talk",
    "recommended_search_mode": "Talk mode",
    "duration_sec": 180,
}


def test_base_metadata_does_not_include_none_values():
    indexer = DummyVectorIndexer()

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
        extra=indexer._prepare_source_extra(VIDEO_SOURCE_INFO),
    )

    assert metadata["video_name"] == "demo.mp4"
    assert metadata["content_type"] == "transcription"
    assert metadata["source_modality"] == "audio"
    assert metadata["document_language"] == "vi"
    assert metadata["source_platform"] == "youtube"
    assert metadata["source_url"] == "https://youtube.com/example"
    assert metadata["video_title"] == "Demo Video"
    assert metadata["ingest_method"] == "youtube_url"
    assert metadata["has_audio"] is True
    assert metadata["video_type"] == "talk"
    assert metadata["estimated_content_style"] == "talk"
    assert metadata["recommended_search_mode"] == "Talk mode"
    assert metadata["duration_sec"] == 180

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
    indexer = DummyVectorIndexer()

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

    count = indexer.index_transcriptions(transcription_data, video_source_info=VIDEO_SOURCE_INFO)

    assert count == 2
    assert len(indexer.text_collection.upsert_calls) == 1
    assert len(indexer.clip_collection.upsert_calls) == 0

    call = indexer.text_collection.upsert_calls[0]
    assert len(call["ids"]) == 2
    assert call["metadatas"][0]["content_type"] == "transcription"
    assert call["metadatas"][1]["content_type"] == "segment_chunk"
    assert call["metadatas"][0]["source_platform"] == "youtube"
    assert call["metadatas"][0]["video_title"] == "Demo Video"


def test_index_captions():
    indexer = DummyVectorIndexer()

    captions_data = [
        {
            "video_name": "demo.mp4",
            "frame_name": "frame_0001_1.00s.jpg",
            "image_path": "data/interim/frames/demo/frame_0001_1.00s.jpg",
            "caption": "A person giving a presentation",
            "timestamp": 1.0,
            "blip_model_name": "blip-base",
            "clip_model_name": "ViT-B-32:openai",
            "language": "en",
            "clip_embedding": [0.1, 0.2, 0.3],
        }
    ]

    count = indexer.index_captions(captions_data, video_source_info=VIDEO_SOURCE_INFO)

    assert count == 2
    assert len(indexer.text_collection.upsert_calls) == 1
    assert len(indexer.clip_collection.upsert_calls) == 1

    text_call = indexer.text_collection.upsert_calls[0]
    clip_call = indexer.clip_collection.upsert_calls[0]

    assert len(text_call["ids"]) == 1
    assert len(clip_call["ids"]) == 1
    assert text_call["metadatas"][0]["content_type"] == "caption"
    assert clip_call["metadatas"][0]["content_type"] == "caption"
    assert text_call["metadatas"][0]["source_modality"] == "image"
    assert text_call["metadatas"][0]["source_url"] == "https://youtube.com/example"


def test_index_multimodal_documents():
    indexer = DummyVectorIndexer()

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
            "blip_model_name": "blip-base",
            "language": "en",
        }
    ]

    count = indexer.index_multimodal_documents(
        transcription_data,
        captions_data,
        video_source_info=VIDEO_SOURCE_INFO,
    )

    assert count == 1
    assert len(indexer.text_collection.upsert_calls) == 1
    assert len(indexer.clip_collection.upsert_calls) == 0

    call = indexer.text_collection.upsert_calls[0]
    assert len(call["ids"]) == 1
    assert call["metadatas"][0]["content_type"] == "multimodal"
    assert call["metadatas"][0]["source_modality"] == "audio+image"
    assert call["metadatas"][0]["source_platform"] == "youtube"


def test_delete_video_data():
    indexer = DummyVectorIndexer()

    indexer.text_collection.items = [
        {
            "id": "text_1",
            "document": "doc 1",
            "metadata": {"video_name": "demo.mp4"},
        },
        {
            "id": "text_2",
            "document": "doc 2",
            "metadata": {"video_name": "other.mp4"},
        },
    ]
    indexer.clip_collection.items = [
        {
            "id": "clip_1",
            "document": "doc 3",
            "metadata": {"video_name": "demo.mp4"},
        }
    ]

    deleted = indexer.delete_video_data("demo.mp4")

    assert deleted == 2
    assert "text_1" in indexer.text_collection.deleted_ids
    assert "clip_1" in indexer.clip_collection.deleted_ids
    assert "text_2" not in indexer.text_collection.deleted_ids


def test_list_videos_and_inventory():
    indexer = DummyVectorIndexer()

    indexer.text_collection.items = [
        {
            "id": "id_1",
            "document": "doc 1",
            "metadata": {
                "video_name": "demo.mp4",
                "content_type": "transcription",
                "source_modality": "audio",
                "document_language": "vi",
                "pipeline_version": "2.0.0",
                "source_platform": "youtube",
                "source_url": "https://youtube.com/example",
                "video_title": "Demo Video",
                "video_description": "A demo video for testing.",
                "video_tags": "demo|test",
            },
        }
    ]
    indexer.clip_collection.items = [
        {
            "id": "id_2",
            "document": "doc 2",
            "metadata": {
                "video_name": "demo.mp4",
                "content_type": "caption",
                "source_modality": "image",
                "document_language": "en",
                "pipeline_version": "2.0.0",
                "timestamp": 1.0,
                "source_platform": "youtube",
                "source_url": "https://youtube.com/example",
                "video_title": "Demo Video",
                "video_description": "A demo video for testing.",
                "video_tags": "demo|test",
            },
        }
    ]

    videos = indexer.list_videos()
    assert videos == ["demo.mp4"]

    inventory = indexer.get_video_inventory("demo.mp4")
    assert inventory["exists"] is True
    assert inventory["total_records"] == 2
    assert inventory["content_type_counts"]["transcription"] == 1
    assert inventory["content_type_counts"]["caption"] == 1
    assert inventory["source_info"]["source_platform"] == "youtube"
    assert inventory["source_info"]["source_url"] == "https://youtube.com/example"
    assert inventory["source_info"]["video_title"] == "Demo Video"


def test_get_all_videos_inventory():
    indexer = DummyVectorIndexer()

    indexer.text_collection.items = [
        {
            "id": "id_1",
            "document": "doc 1",
            "metadata": {
                "video_name": "demo.mp4",
                "content_type": "transcription",
                "source_modality": "audio",
                "document_language": "vi",
                "pipeline_version": "2.0.0",
                "source_platform": "youtube",
                "source_url": "https://youtube.com/example",
                "video_title": "Demo Video",
            },
        }
    ]
    indexer.clip_collection.items = [
        {
            "id": "id_2",
            "document": "doc 2",
            "metadata": {
                "video_name": "lesson.mp4",
                "content_type": "caption",
                "source_modality": "image",
                "document_language": "en",
                "pipeline_version": "2.0.0",
                "timestamp": 2.0,
                "source_platform": "facebook",
                "source_url": "https://facebook.com/example",
                "video_title": "Lesson Video",
            },
        }
    ]

    inventory = indexer.get_all_videos_inventory()
    assert inventory["total_videos"] == 2
    assert len(inventory["videos"]) == 2


def test_get_stats():
    indexer = DummyVectorIndexer()

    indexer.text_collection.items = [{"id": "a", "document": "x", "metadata": {}}]
    indexer.clip_collection.items = [{"id": "b", "document": "y", "metadata": {}}]

    stats = indexer.get_stats()

    assert stats["text_collection_name"] == "video_semantic_search_text"
    assert stats["clip_collection_name"] == "video_semantic_search_clip"
    assert stats["text_total_documents"] == 1
    assert stats["clip_total_documents"] == 1