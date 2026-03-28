from pathlib import Path

from main_pipeline import MediaDataPipeline


class FakeAudioExtractor:
    def extract_audio(self, video_path):
        return "data/interim/audio/demo.wav"


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
        assert video_source_info is not None
        return 3

    def index_captions(self, captions_data, video_source_info=None):
        assert video_source_info is not None
        return 2

    def index_multimodal_documents(self, transcription_data, captions_data, video_source_info=None):
        assert video_source_info is not None
        return 1


class FakeSearchEngine:
    def __init__(self, config=None, vector_indexer=None, vision_processor=None):
        self.vector_indexer = vector_indexer
        self.vision_processor = vision_processor

    def search(self, query, top_k=5, content_type=None, video_name=None):
        return []


class DummyMediaDataPipeline(MediaDataPipeline):
    __test__ = False

    def __init__(self, tmp_path):
        self.config = {
            "paths": {
                "processed_dir": str(tmp_path / "processed"),
                "video_catalog_path": str(tmp_path / "video_catalog.json"),
            },
            "pipeline": {
                "version": "2.0.0",
                "save_run_metadata": True,
            },
            "models": {
                "whisper": {"language": "vi", "name": "base"},
            },
        }

        catalog_path = Path(self.config["paths"]["video_catalog_path"])
        catalog_path.write_text(
            """
[
  {
    "video_name": "demo.mp4",
    "local_video_path": "data/raw/demo.mp4",
    "source_platform": "youtube",
    "source_url": "https://youtube.com/example",
    "title": "Demo Video",
    "description": "A demo video for testing.",
    "thumbnail_url": "",
    "tags": ["demo", "test"],
    "created_at": "2026-03-27T00:00:00",
    "ingested_at": "2026-03-27T00:00:00"
  }
]
""".strip(),
            encoding="utf-8",
        )

        self.audio_extractor = FakeAudioExtractor()
        self.frame_extractor = FakeFrameExtractor()
        self.whisper_processor = FakeWhisperProcessor()
        self.vision_processor = FakeVisionProcessor()
        self.vector_indexer = FakeVectorIndexer()
        self.search_engine = FakeSearchEngine(
            config=self.config,
            vector_indexer=self.vector_indexer,
            vision_processor=self.vision_processor,
        )
        self.processed_dir = Path(self.config["paths"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_version = "2.0.0"
        self.save_run_metadata = True


def test_process_video_returns_expected_summary(tmp_path):
    pipeline = DummyMediaDataPipeline(tmp_path)

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake video content")

    result = pipeline.process_video(str(video_path), reset_index=True)

    assert result["video_name"] == "demo.mp4"
    assert result["audio_path"] == "data/interim/audio/demo.wav"
    assert result["frames_dir"] == "data/interim/frames/demo"
    assert result["transcription_records"] == 3
    assert result["caption_records"] == 2
    assert result["multimodal_records"] == 1
    assert result["stage_status"]["extract_audio"] == "done"
    assert result["stage_status"]["extract_frames"] == "done"
    assert result["stage_status"]["transcribe"] == "done"
    assert result["stage_status"]["caption"] == "done"
    assert result["stage_status"]["index"] == "done"
    assert result["merged_output_path"] is not None
    assert result["run_metadata_path"] is not None
    assert result["data_summary"]["indexed_total_records"] == 6

    assert result["video_source_info"]["source_platform"] == "youtube"
    assert result["video_source_info"]["source_url"] == "https://youtube.com/example"
    assert result["video_source_info"]["video_title"] == "Demo Video"

    merged_output_path = Path(result["merged_output_path"])
    run_metadata_path = Path(result["run_metadata_path"])

    assert merged_output_path.exists()
    assert run_metadata_path.exists()

    merged_text = merged_output_path.read_text(encoding="utf-8")
    run_text = run_metadata_path.read_text(encoding="utf-8")

    assert "youtube" in merged_text
    assert "https://youtube.com/example" in merged_text
    assert "Demo Video" in merged_text

    assert "youtube" in run_text
    assert "https://youtube.com/example" in run_text
    assert "Demo Video" in run_text