from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main


class FakeSearchEngine:
    def search(self, query, top_k=5, content_type=None, video_name=None):
        return [
            {
                "document": "artificial intelligence introduction",
                "metadata": {
                    "video_name": "demo.mp4",
                    "content_type": "transcription",
                    "timestamp": 1.23,
                    "timestamp_str": "00:00:01",
                },
                "distance": 0.1,
                "relevance": 0.9,
                "score_type": "similarity_proxy_from_distance",
            }
        ]


class FakeVectorIndexer:
    def get_stats(self):
        return {
            "collection_name": "video_semantic_search",
            "total_documents": 10,
            "persist_dir": "data/vector_db",
            "embedding_model": "fake-model",
            "distance_metric": "cosine",
            "pipeline_version": "1.1.0",
        }


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

    def process_video(self, video_path, reset_index=False):
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
            "stage_status": {
                "extract_audio": "done",
                "extract_frames": "done",
                "transcribe": "done",
                "caption": "done",
                "index": "done",
            },
        }


api_main._pipeline = FakePipeline()
client = TestClient(api_main.app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "/search" in data["endpoints"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["collection_name"] == "video_semantic_search"
    assert data["total_documents"] == 10


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
    assert data["results"][0]["score_type"] == "similarity_proxy_from_distance"


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