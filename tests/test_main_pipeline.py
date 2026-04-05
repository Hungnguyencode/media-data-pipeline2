from pathlib import Path

from main_pipeline import MediaDataPipeline
from src.extract.audio_extractor import NoAudioStreamError


class FakeAudioExtractor:
    def extract_audio(self, video_path):
        return "data/interim/audio/demo.wav"


class FakeNoAudioExtractor:
    def extract_audio(self, video_path):
        raise NoAudioStreamError(f"No audio stream found in video: {video_path}")


class FakeFrameExtractor:
    def extract_frames(self, video_path):
        return "data/interim/frames/demo"


class FakeWhisperProcessor:
    def transcribe(self, audio_path, video_name=None):
        return {
            "video_name": video_name or "demo.mp4",
            "audio_path": audio_path,
            "language": "vi",
            "full_text": "Xin chao day la video demo",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Xin chao day la"},
                {"start": 2.0, "end": 4.0, "text": "video demo"},
            ],
            "model_name": "whisper-base",
        }

    def unload_model(self):
        return None


class FakeVisionProcessor:
    def process_frames(self, frames_dir, video_name=None):
        return [
            {
                "video_name": video_name or "demo.mp4",
                "frame_name": "frame_0001_1.00s.jpg",
                "image_path": f"{frames_dir}/frame_0001_1.00s.jpg",
                "caption": "A person giving a presentation",
                "timestamp": 1.0,
                "timestamp_str": "00:00:01",
                "blip_model_name": "blip-base",
                "clip_model_name": "ViT-B-32:openai",
                "clip_embedding": [0.1, 0.2, 0.3],
                "language": "en",
                "search_text": "person presentation talk speech",
                "action_aliases": "",
            }
        ]

    def encode_text_for_clip(self, texts):
        return [[0.9, 0.8, 0.7] for _ in texts]

    def unload_model(self):
        return None


class FakeVectorIndexer:
    def delete_video_data(self, video_name):
        return 0

    def index_transcriptions(self, transcription_data, video_source_info=None):
        return 2

    def index_captions(self, captions_data, video_source_info=None):
        return 1

    def index_multimodal_documents(self, transcription_data, captions_data, video_source_info=None):
        return 1


class FakeNoAudioVectorIndexer:
    def delete_video_data(self, video_name):
        return 0

    def index_transcriptions(self, transcription_data, video_source_info=None):
        return 0

    def index_captions(self, captions_data, video_source_info=None):
        return 1

    def index_multimodal_documents(self, transcription_data, captions_data, video_source_info=None):
        return 0


class FakeSearchEngine:
    def search(self, query, top_k=5, content_type=None, video_name=None):
        return [
            {
                "document": "demo semantic result",
                "display_text": "demo semantic result",
                "display_caption": "demo semantic result",
                "nearby_speech_context": "",
                "group_size": 1,
                "event_time_range": {"start": 1.0, "end": 2.0},
                "metadata": {
                    "video_name": video_name or "demo.mp4",
                    "content_type": content_type or "segment_chunk",
                },
                "distance": 0.2,
                "similarity_score": 0.8,
                "score_type": "hybrid_fusion+rerank",
            }
        ]


class TestablePipeline(MediaDataPipeline):
    def __init__(self, tmp_path: Path):
        self.config = {
            "paths": {"processed_dir": str(tmp_path / "processed")},
            "pipeline": {"version": "2.2.0", "save_run_metadata": True},
        }
        self.audio_extractor = FakeAudioExtractor()
        self.frame_extractor = FakeFrameExtractor()
        self.whisper_processor = FakeWhisperProcessor()
        self.vision_processor = FakeVisionProcessor()
        self.vector_indexer = FakeVectorIndexer()
        self.search_engine = FakeSearchEngine()
        self.processed_dir = tmp_path / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_version = "2.2.0"
        self.save_run_metadata = True

    def _build_video_source_info(self, video_name: str, video_path: str):
        return {
            "source_platform": "youtube",
            "source_url": "https://youtube.com/example",
            "video_title": "Demo Video",
            "video_description": "A demo video for testing.",
            "thumbnail_url": "",
            "video_tags": "demo|test",
            "local_video_path": "data/raw/demo.mp4",
            "created_at": "2026-03-27T00:00:00",
            "ingested_at": "2026-03-27T00:00:00",
            "ingest_method": "local_file",
            "has_audio": True,
            "video_type": "talk",
            "estimated_content_style": "talk",
            "recommended_search_mode": "Talk mode",
            "duration_sec": 180,
        }


class TestableNoAudioPipeline(MediaDataPipeline):
    def __init__(self, tmp_path: Path):
        self.config = {
            "paths": {"processed_dir": str(tmp_path / "processed")},
            "pipeline": {"version": "2.2.0", "save_run_metadata": True},
        }
        self.audio_extractor = FakeNoAudioExtractor()
        self.frame_extractor = FakeFrameExtractor()
        self.whisper_processor = FakeWhisperProcessor()
        self.vision_processor = FakeVisionProcessor()
        self.vector_indexer = FakeNoAudioVectorIndexer()
        self.search_engine = FakeSearchEngine()
        self.processed_dir = tmp_path / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_version = "2.2.0"
        self.save_run_metadata = True

    def _build_video_source_info(self, video_name: str, video_path: str):
        return {
            "source_platform": "local",
            "source_url": "",
            "video_title": "No Audio Demo",
            "video_description": "",
            "thumbnail_url": "",
            "video_tags": "demo|visual",
            "local_video_path": "data/raw/demo.mp4",
            "created_at": "2026-03-27T00:00:00",
            "ingested_at": "2026-03-27T00:00:00",
            "ingest_method": "local_file",
            "has_audio": False,
            "video_type": "visual_story",
            "estimated_content_style": "visual",
            "recommended_search_mode": "Visual mode",
            "duration_sec": 60,
        }


def test_process_video_returns_expected_summary(tmp_path, monkeypatch):
    import main_pipeline as mp

    monkeypatch.setattr(mp, "ensure_video_catalog_entry", lambda *args, **kwargs: {})
    monkeypatch.setattr(mp, "get_video_catalog_entry", lambda *args, **kwargs: {})
    monkeypatch.setattr(mp, "release_memory", lambda: None)
    monkeypatch.setattr(mp, "md5_of_file", lambda _: "fake-md5")

    def fake_save_json(data, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mp, "save_json", fake_save_json)

    pipeline = TestablePipeline(tmp_path)

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake video content")

    result = pipeline.process_video(str(video_path), reset_index=True)

    assert result["video_name"] == "demo.mp4"
    assert result["transcription_records"] == 2
    assert result["caption_records"] == 1
    assert result["multimodal_records"] == 1
    assert result["data_summary"]["indexed_total_records"] == 4
    assert result["video_source_info"]["has_audio"] is True
    assert result["video_source_info"]["video_type"] == "talk"
    assert result["video_source_info"]["estimated_content_style"] == "talk"
    assert result["video_source_info"]["recommended_search_mode"] == "Talk mode"
    assert Path(result["merged_output_path"]).exists()
    assert Path(result["run_metadata_path"]).exists()


def test_process_video_visual_only_path(tmp_path, monkeypatch):
    import main_pipeline as mp

    monkeypatch.setattr(mp, "ensure_video_catalog_entry", lambda *args, **kwargs: {})
    monkeypatch.setattr(mp, "get_video_catalog_entry", lambda *args, **kwargs: {})
    monkeypatch.setattr(mp, "release_memory", lambda: None)
    monkeypatch.setattr(mp, "md5_of_file", lambda _: "fake-md5")

    def fake_save_json(data, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mp, "save_json", fake_save_json)

    pipeline = TestableNoAudioPipeline(tmp_path)

    video_path = tmp_path / "demo_no_audio.mp4"
    video_path.write_bytes(b"fake video content")

    result = pipeline.process_video(str(video_path), reset_index=True)

    assert result["video_name"] == "demo_no_audio.mp4"
    assert result["transcription_records"] == 0
    assert result["caption_records"] == 1
    assert result["multimodal_records"] == 0
    assert result["data_summary"]["has_audio"] is False
    assert result["video_source_info"]["estimated_content_style"] == "visual"
    assert result["video_source_info"]["recommended_search_mode"] == "Visual mode"
    assert Path(result["merged_output_path"]).exists()
    assert Path(result["run_metadata_path"]).exists()


def test_search_delegates_to_search_engine(tmp_path):
    pipeline = TestablePipeline(tmp_path)
    results = pipeline.search(query="demo query", top_k=3, content_type="segment_chunk", video_name="demo.mp4")

    assert len(results) == 1
    assert results[0]["score_type"] == "hybrid_fusion+rerank"
    assert results[0]["metadata"]["video_name"] == "demo.mp4"