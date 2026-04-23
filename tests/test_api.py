from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main


class FakeSearchEngine:
    def search(self, query, top_k=5, content_type=None, video_name=None, search_mode="auto"):
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
                    "content_type": content_type or "segment_chunk",
                    "timestamp_str": "00:00:01",
                    "start_time_str": "00:00:01",
                    "end_time_str": "00:00:02",
                    "source_modality": "audio",
                    "model_name": "whisper-base",
                    "document_language": "en",
                    "video_title": "Demo Video",
                    "video_description": "A demo video for testing.",
                    "thumbnail_url": "https://img.youtube.com/demo.jpg",
                    "video_tags": "demo|test",
                    "source_platform": "youtube",
                    "source_url": "https://youtube.com/example",
                    "estimated_content_style": "talk",
                    "recommended_search_mode": "Talk mode",
                },
                "distance": 0.2,
                "similarity_score": 0.8,
                "score_type": "hybrid_fusion+rerank",
            }
        ]


class FakeVectorIndexer:
    def __init__(self):
        self.deleted_video_name = None

    def get_stats(self):
        return {
            "persist_directory": "data/vector_db",
            "total_collections": 2,
            "collections": ["video_semantic_search_text", "video_semantic_search_clip"],
        }

    def list_videos(self):
        return ["demo.mp4", "demo_youtube.mp4"]

    def get_video_inventory(self, video_name):
        if video_name == "missing.mp4":
            return {"exists": False, "video_name": video_name}

        return {
            "exists": True,
            "video_name": video_name,
            "total_records": 12,
            "content_type_counts": {
                "transcription": 2,
                "segment_chunk": 3,
                "caption": 4,
                "multimodal": 3,
            },
            "source_modality_counts": {
                "audio": 5,
                "image": 4,
                "audio+image": 3,
            },
            "languages": ["en"],
            "pipeline_versions": ["2.2.0"],
            "time_range": {
                "min_timestamp": 1.0,
                "max_timestamp": 10.0,
                "min_start_time": 1.0,
                "max_end_time": 12.0,
            },
            "source_info": {
                "source_platform": "youtube",
                "source_url": "https://youtube.com/example",
                "video_title": "Demo Video",
                "video_description": "A demo video for testing.",
                "thumbnail_url": "https://img.youtube.com/demo.jpg",
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
            },
        }

    def get_all_videos_inventory(self):
        return {
            "total_videos": 2,
            "videos": [
                self.get_video_inventory("demo.mp4"),
                self.get_video_inventory("demo_youtube.mp4"),
            ],
        }

    def delete_video_data(self, video_name):
        self.deleted_video_name = video_name
        return 7


class FakePipeline:
    def __init__(self):
        self.search_engine = FakeSearchEngine()
        self.vector_indexer = FakeVectorIndexer()

    def search(self, query, top_k=5, content_type=None, video_name=None, search_mode="auto"):
        return self.search_engine.search(
            query=query,
            top_k=top_k,
            content_type=content_type,
            video_name=video_name,
            search_mode=search_mode,
        )

    def process_video(self, video_path, reset_index=False, source_metadata=None):
        source_metadata = source_metadata or {}
        video_name = Path(video_path).name

        return {
            "video_name": video_name,
            "video_path": str(video_path),
            "audio_path": "data/interim/audio/demo.wav",
            "frames_dir": "data/interim/frames/demo",
            "transcription_records": 2,
            "caption_records": 1,
            "multimodal_records": 1,
            "merged_output_path": "data/processed/demo_merged_output.json",
            "run_metadata_path": "data/processed/demo_run_metadata.json",
            "stage_metrics_sec": {
                "extract_audio": 0.1,
                "extract_frames": 0.2,
                "transcribe": 0.3,
                "caption": 0.4,
                "index": 0.1,
            },
            "data_summary": {
                "indexed_total_records": 4,
                "transcription_records": 2,
                "caption_records": 1,
                "multimodal_records": 1,
                "has_audio": source_metadata.get("has_audio", True),
                "video_type": source_metadata.get("video_type", "talk"),
                "estimated_content_style": source_metadata.get("estimated_content_style", "talk"),
                "recommended_search_mode": source_metadata.get("recommended_search_mode", "Talk mode"),
            },
            "stage_status": {
                "extract_audio": "done",
                "extract_frames": "done",
                "transcribe": "done",
                "caption": "done",
                "index": "done",
            },
            "video_source_info": {
                "source_platform": source_metadata.get("source_platform", "youtube"),
                "source_url": source_metadata.get("source_url", "https://youtube.com/example"),
                "video_title": source_metadata.get("video_title", "Demo Video"),
                "video_description": source_metadata.get("video_description", "A demo video for testing."),
                "thumbnail_url": source_metadata.get("thumbnail_url", "https://img.youtube.com/demo.jpg"),
                "video_tags": source_metadata.get("video_tags", ["demo", "test"]),
                "local_video_path": str(video_path),
                "created_at": "2026-03-27T00:00:00",
                "ingested_at": "2026-03-27T00:00:00",
                "ingest_method": source_metadata.get("ingest_method", "local_file"),
                "has_audio": source_metadata.get("has_audio", True),
                "video_type": source_metadata.get("video_type", "talk"),
                "estimated_content_style": source_metadata.get("estimated_content_style", "talk"),
                "recommended_search_mode": source_metadata.get("recommended_search_mode", "Talk mode"),
                "duration_sec": source_metadata.get("duration_sec", 180),
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
            "video_description": "A demo YouTube video.",
            "thumbnail_url": "https://img.youtube.com/demo.jpg",
            "video_tags": ["demo", "youtube"],
            "youtube_video_id": "abc123",
            "ingest_method": "youtube_url",
            "has_audio": True,
            "video_type": "talk",
            "estimated_content_style": "talk",
            "recommended_search_mode": "Talk mode",
            "duration_sec": 180,
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


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_collections"] == 2


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


def test_get_video_inventory_success():
    response = client.get("/videos/demo.mp4")
    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is True
    assert data["video_name"] == "demo.mp4"
    assert data["source_info"]["ingest_method"] == "youtube_url"
    assert data["source_info"]["has_audio"] is True
    assert data["source_info"]["video_type"] == "talk"
    assert data["source_info"]["estimated_content_style"] == "talk"
    assert data["source_info"]["recommended_search_mode"] == "Talk mode"
    assert data["source_info"]["duration_sec"] == 180


def test_get_video_inventory_not_found():
    response = client.get("/videos/missing.mp4")
    assert response.status_code == 404


def test_delete_video():
    response = client.delete("/videos/demo.mp4")
    assert response.status_code == 200
    data = response.json()
    assert data["video_name"] == "demo.mp4"
    assert data["deleted_records"] == 7


def test_search_success():
    response = client.post(
        "/search",
        json={
            "query": "artificial intelligence",
            "top_k": 5,
            "content_type": "segment_chunk",
            "video_name": "demo.mp4",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["score_type"] == "hybrid_fusion+rerank"


def test_search_empty_query():
    response = client.post(
        "/search",
        json={"query": "   ", "top_k": 5},
    )
    assert response.status_code == 400


def test_search_invalid_content_type():
    response = client.post(
        "/search",
        json={
            "query": "test",
            "top_k": 5,
            "content_type": "invalid",
        },
    )
    assert response.status_code == 400


def test_process_video_success(tmp_path):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake video content")

    response = client.post(
        "/process-video",
        json={"video_path": str(video_path), "reset_index": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["video_name"] == "demo.mp4"
    assert data["data_summary"]["indexed_total_records"] == 4
    assert data["video_source_info"]["source_platform"] == "youtube"


def test_cleanup_video_success(monkeypatch):
    def fake_cleanup_video_artifacts(
        video_name,
        *,
        delete_raw=False,
        delete_audio=False,
        delete_frames=False,
        delete_interim_json=False,
        delete_processed=False,
        keep_catalog=True,
        config=None,
    ):
        assert video_name == "demo.mp4"
        assert delete_raw is True
        assert delete_audio is False
        assert delete_frames is True
        assert delete_interim_json is False
        assert delete_processed is True
        assert keep_catalog is True

        return {
            "video_name": video_name,
            "deleted": {
                "raw_video": ["data/raw/demo.mp4"],
                "audio_files": [],
                "frame_dirs": ["data/interim/frames/demo"],
                "transcript_files": [],
                "caption_files": [],
                "processed_files": [
                    "data/processed/demo_merged_output.json",
                    "data/processed/demo_run_metadata.json",
                ],
            },
            "missing_or_not_deleted": {
                "raw_video": [],
                "audio_files": [],
                "frame_dirs": [],
                "transcript_files": [],
                "caption_files": [],
                "processed_files": [],
            },
            "deleted_count": 4,
            "missing_count": 0,
            "removed_from_catalog": False,
        }

    monkeypatch.setattr(api_main, "cleanup_video_artifacts", fake_cleanup_video_artifacts)

    response = client.post(
        "/videos/demo.mp4/cleanup",
        json={
            "delete_raw": True,
            "delete_audio": False,
            "delete_frames": True,
            "delete_interim_json": False,
            "delete_processed": True,
            "keep_catalog": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Cleanup completed for 'demo.mp4'"
    assert data["cleanup_result"]["video_name"] == "demo.mp4"
    assert data["cleanup_result"]["deleted_count"] == 4
    assert data["cleanup_result"]["removed_from_catalog"] is False


def test_cleanup_video_remove_catalog(monkeypatch):
    def fake_cleanup_video_artifacts(
        video_name,
        *,
        delete_raw=False,
        delete_audio=False,
        delete_frames=False,
        delete_interim_json=False,
        delete_processed=False,
        keep_catalog=True,
        config=None,
    ):
        return {
            "video_name": video_name,
            "deleted": {
                "raw_video": [],
                "audio_files": [],
                "frame_dirs": [],
                "transcript_files": [],
                "caption_files": [],
                "processed_files": [],
            },
            "missing_or_not_deleted": {
                "raw_video": [],
                "audio_files": [],
                "frame_dirs": [],
                "transcript_files": [],
                "caption_files": [],
                "processed_files": [],
            },
            "deleted_count": 0,
            "missing_count": 0,
            "removed_from_catalog": True,
        }

    monkeypatch.setattr(api_main, "cleanup_video_artifacts", fake_cleanup_video_artifacts)

    response = client.post(
        "/videos/demo.mp4/cleanup",
        json={
            "delete_raw": False,
            "delete_audio": False,
            "delete_frames": False,
            "delete_interim_json": False,
            "delete_processed": False,
            "keep_catalog": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["cleanup_result"]["removed_from_catalog"] is True


def test_reindex_video_success(monkeypatch, tmp_path):
    demo_video = tmp_path / "demo.mp4"
    demo_video.write_bytes(b"fake video content")

    monkeypatch.setattr(
        api_main,
        "get_video_catalog_entry",
        lambda video_name, force_reload=True, config=None: {
            "video_name": "demo.mp4",
            "local_video_path": str(demo_video),
            "source_platform": "youtube",
            "source_url": "https://youtube.com/example",
            "title": "Demo Video",
            "description": "A demo video for testing.",
            "thumbnail_url": "https://img.youtube.com/demo.jpg",
            "tags": ["demo", "test"],
            "ingest_method": "youtube_url",
            "has_audio": True,
            "video_type": "talk",
            "estimated_content_style": "talk",
            "recommended_search_mode": "Talk mode",
            "duration_sec": 180,
        },
    )

    monkeypatch.setattr(api_main, "get_data_path", lambda p: Path(p))

    response = client.post(
        "/videos/demo.mp4/reindex",
        json={"reset_index": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Re-indexed 'demo.mp4' successfully"
    assert data["result"]["video_name"] == "demo.mp4"
    assert data["result"]["video_source_info"]["source_platform"] == "youtube"
    assert data["result"]["video_source_info"]["video_title"] == "Demo Video"
    assert data["result"]["video_source_info"]["ingest_method"] == "youtube_url"
    assert data["result"]["video_source_info"]["estimated_content_style"] == "talk"
    assert data["result"]["video_source_info"]["recommended_search_mode"] == "Talk mode"


def test_reindex_video_catalog_not_found(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "get_video_catalog_entry",
        lambda video_name, force_reload=True, config=None: None,
    )

    response = client.post(
        "/videos/missing.mp4/reindex",
        json={"reset_index": True},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found in catalog: missing.mp4"


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