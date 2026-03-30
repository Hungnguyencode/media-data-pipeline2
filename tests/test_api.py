from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main


class FakeSearchEngine:
    def search(self, query, top_k=5, content_type=None, video_name=None):
        return [
            {
                "document": "artificial intelligence introduction",
                "display_text": "artificial intelligence introduction",
                "display_caption": "artificial intelligence introduction",
                "nearby_speech_context": "",
                "group_size": 1,
                "event_time_range": {"start": 1.0, "end": 2.0},
                "metadata": {
                    "video_name": video_name or "demo.mp4",
                    "content_type": content_type or "transcription",
                    "timestamp": 1.23,
                    "timestamp_str": "00:00:01",
                    "source_platform": "youtube",
                    "source_url": "https://youtube.com/example",
                    "video_title": "Demo Video",
                    "video_description": "A demo video for testing.",
                    "video_tags": "demo|test",
                },
                "distance": 0.1,
                "relevance": 0.9,
                "score_type": "hybrid_fusion",
            }
        ]


class FakeVectorIndexer:
    def get_stats(self):
        return {
            "text_collection_name": "video_semantic_search_text",
            "clip_collection_name": "video_semantic_search_clip",
            "text_total_documents": 10,
            "clip_total_documents": 4,
            "persist_dir": "data/vector_db",
            "embedding_model": "fake-model",
            "distance_metric": "cosine",
            "pipeline_version": "2.0.0",
        }

    def list_videos(self):
        return ["demo.mp4", "lesson_01.mp4"]

    def get_video_inventory(self, video_name):
        if video_name == "missing.mp4":
            return {
                "video_name": video_name,
                "exists": False,
                "total_records": 0,
                "content_type_counts": {
                    "transcription": 0,
                    "segment_chunk": 0,
                    "caption": 0,
                    "multimodal": 0,
                },
                "source_modality_counts": {},
                "languages": [],
                "pipeline_versions": [],
                "source_info": {
                    "source_platform": "",
                    "source_url": "",
                    "video_title": "",
                    "video_description": "",
                    "thumbnail_url": "",
                    "video_tags": "",
                    "local_video_path": "",
                    "created_at": "",
                    "ingested_at": "",
                },
                "time_range": {
                    "min_timestamp": None,
                    "max_timestamp": None,
                    "min_start_time": None,
                    "max_end_time": None,
                },
            }

        return {
            "video_name": video_name,
            "exists": True,
            "total_records": 6,
            "content_type_counts": {
                "transcription": 1,
                "segment_chunk": 2,
                "caption": 2,
                "multimodal": 1,
            },
            "source_modality_counts": {
                "audio": 3,
                "image": 2,
                "audio+image": 1,
            },
            "languages": ["en", "vi", "vi+en"],
            "pipeline_versions": ["2.0.0"],
            "source_info": {
                "source_platform": "youtube",
                "source_url": "https://youtube.com/example",
                "video_title": "Demo Video",
                "video_description": "A demo video for testing.",
                "thumbnail_url": "",
                "video_tags": "demo|test",
                "local_video_path": "data/raw/demo.mp4",
                "created_at": "2026-03-27T00:00:00",
                "ingested_at": "2026-03-27T00:00:00",
            },
            "time_range": {
                "min_timestamp": 0.0,
                "max_timestamp": 12.0,
                "min_start_time": 0.0,
                "max_end_time": 15.0,
            },
        }

    def get_all_videos_inventory(self):
        return {
            "total_videos": 2,
            "videos": [
                self.get_video_inventory("demo.mp4"),
                self.get_video_inventory("lesson_01.mp4"),
            ],
        }

    def delete_video_data(self, video_name):
        if video_name == "missing.mp4":
            return 0
        return 6


class FakePipeline:
    def __init__(self):
        self.search_engine = FakeSearchEngine()
        self.vector_indexer = FakeVectorIndexer()

    def search(self, query, top_k=5, content_type=None, video_name=None):
        return self.search_engine.search(
            query=query,
            top_k=top_k,
            content_type=content_type,
            video_name=video_name,
        )

    def process_video(self, video_path, reset_index=False, source_metadata=None):
        return {
            "video_name": Path(video_path).name,
            "video_path": video_path,
            "audio_path": "data/interim/audio/demo.wav",
            "frames_dir": "data/interim/frames/demo",
            "transcription_records": 3,
            "caption_records": 5,
            "multimodal_records": 2,
            "merged_output_path": "data/processed/demo_merged_output.json",
            "run_metadata_path": "data/processed/demo_run_metadata.json",
            "video_source_info": {
                "source_platform": (source_metadata or {}).get("source_platform", "youtube"),
                "source_url": (source_metadata or {}).get("source_url", "https://youtube.com/example"),
                "video_title": (source_metadata or {}).get("video_title", "Demo Video"),
                "video_description": (source_metadata or {}).get("video_description", "A demo video for testing."),
                "thumbnail_url": "",
                "video_tags": "demo|test",
                "local_video_path": "data/raw/demo.mp4",
                "created_at": "2026-03-27T00:00:00",
                "ingested_at": "2026-03-27T00:00:00",
            },
            "stage_status": {
                "extract_audio": "done",
                "extract_frames": "done",
                "transcribe": "done",
                "caption": "done",
                "index": "done",
            },
        }


class FakeYouTubeIngestor:
    def ingest(self, video_url):
        return {
            "video_path": "data/raw/demo_youtube.mp4",
            "video_name": "demo_youtube.mp4",
            "source_platform": "youtube",
            "source_url": "https://www.youtube.com/watch?v=abc123",
            "video_title": "Demo YouTube Video",
            "video_description": "A youtube demo video.",
            "thumbnail_url": "https://img.youtube.com/vi/abc123/default.jpg",
            "video_tags": ["demo", "youtube"],
            "youtube_video_id": "abc123",
        }


api_main._pipeline = FakePipeline()
api_main._youtube_ingestor = FakeYouTubeIngestor()
client = TestClient(api_main.app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "/search" in data["endpoints"]
    assert "/ingest-youtube" in data["endpoints"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["text_collection_name"] == "video_semantic_search_text"
    assert data["clip_collection_name"] == "video_semantic_search_clip"
    assert data["text_total_documents"] == 10
    assert data["clip_total_documents"] == 4


def test_list_videos():
    response = client.get("/videos")
    assert response.status_code == 200
    data = response.json()
    assert data["total_videos"] == 2
    assert "demo.mp4" in data["videos"]


def test_all_videos_inventory():
    response = client.get("/videos/inventory")
    assert response.status_code == 200
    data = response.json()
    assert data["total_videos"] == 2
    assert len(data["videos"]) == 2
    assert data["videos"][0]["source_info"]["source_platform"] == "youtube"


def test_get_video_inventory_success():
    response = client.get("/videos/demo.mp4")
    assert response.status_code == 200
    data = response.json()
    assert data["video_name"] == "demo.mp4"
    assert data["exists"] is True
    assert data["total_records"] == 6
    assert data["source_info"]["video_title"] == "Demo Video"


def test_get_video_inventory_not_found():
    response = client.get("/videos/missing.mp4")
    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found in index: missing.mp4"


def test_delete_video_success():
    response = client.delete("/videos/demo.mp4")
    assert response.status_code == 200
    data = response.json()
    assert data["video_name"] == "demo.mp4"
    assert data["deleted_records"] == 6


def test_delete_video_missing():
    response = client.delete("/videos/missing.mp4")
    assert response.status_code == 200
    data = response.json()
    assert data["video_name"] == "missing.mp4"
    assert data["deleted_records"] == 0


def test_search_success():
    response = client.post(
        "/search",
        json={
            "query": "AI introduction",
            "top_k": 5,
            "content_type": "transcription",
            "video_name": "demo.mp4",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["document"] == "artificial intelligence introduction"
    assert data["results"][0]["score_type"] == "hybrid_fusion"
    assert data["results"][0]["similarity_score"] == 0.9
    assert data["results"][0]["metadata"]["source_platform"] == "youtube"
    assert data["results"][0]["metadata"]["source_url"] == "https://youtube.com/example"


def test_search_rejects_empty_query():
    response = client.post(
        "/search",
        json={
            "query": "   ",
            "top_k": 5,
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Query must not be empty"


def test_search_rejects_invalid_content_type():
    response = client.post(
        "/search",
        json={
            "query": "AI",
            "top_k": 5,
            "content_type": "segment",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid content_type"


def test_process_video_success():
    response = client.post(
        "/process-video",
        json={
            "video_path": "data/raw/demo.mp4",
            "reset_index": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["video_name"] == "demo.mp4"
    assert data["multimodal_records"] == 2
    assert data["stage_status"]["index"] == "done"
    assert data["video_source_info"]["source_platform"] == "youtube"


def test_ingest_youtube_success():
    response = client.post(
        "/ingest-youtube",
        json={
            "video_url": "https://www.youtube.com/watch?v=abc123&list=whatever",
            "reset_index": True,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert "ingest_result" in data
    assert "result" in data
    assert data["ingest_result"]["source_platform"] == "youtube"
    assert data["ingest_result"]["source_url"] == "https://www.youtube.com/watch?v=abc123"
    assert data["result"]["video_name"] == "demo_youtube.mp4"
    assert data["result"]["video_source_info"]["video_title"] == "Demo YouTube Video"