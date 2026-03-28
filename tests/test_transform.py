import json
from pathlib import Path

from PIL import Image

from src.transform.vision_processor import VisionProcessor
from src.transform.whisper_processor import WhisperProcessor


class DummyVisionProcessor(VisionProcessor):
    def __init__(self, tmp_path):
        self.config = {
            "models": {
                "vision": {
                    "blip_name": "fake-blip",
                    "clip_name": "ViT-B-32",
                    "clip_pretrained": "openai",
                    "max_length": 40,
                    "image_size": 384,
                    "fallback_to_cpu_on_oom": True,
                    "output_language": "en",
                }
            },
            "paths": {
                "interim_captions_dir": str(tmp_path / "interim_captions"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "pipeline": {
                "version": "2.0.0",
            },
        }
        self.device = "cpu"
        self.blip_name = "fake-blip"
        self.clip_name = "ViT-B-32"
        self.clip_pretrained = "openai"
        self.max_length = 40
        self.image_size = 384
        self.output_language = "en"
        self.fallback_to_cpu_on_oom = True
        self.pipeline_version = "2.0.0"

        self.blip_processor = object()
        self.blip_model = object()
        self.clip_model = object()
        self.clip_preprocess = None
        self.clip_tokenizer = None

        self.output_dir = tmp_path / "interim_captions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = tmp_path / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _load_models(self, device=None):
        return None

    def _generate_caption_once(self, image):
        return "a person giving a presentation"

    def _encode_image_clip_once(self, image):
        return [0.1, 0.2, 0.3]


class DummyWhisperProcessor(WhisperProcessor):
    def __init__(self, tmp_path):
        self.config = {
            "models": {
                "whisper": {
                    "name": "base",
                    "language": "vi",
                    "use_fp16": False,
                    "fallback_to_cpu_on_oom": True,
                }
            },
            "paths": {
                "interim_transcripts_dir": str(tmp_path / "interim_transcripts"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "pipeline": {
                "version": "2.0.0",
            },
        }
        self.device = "cpu"
        self.model_name = "base"
        self.language = "vi"
        self.use_fp16 = False
        self.fallback_to_cpu_on_oom = True
        self.pipeline_version = "2.0.0"
        self.output_dir = tmp_path / "interim_transcripts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = tmp_path / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def _transcribe_once(self, audio_path: str):
        return {
            "language": "vi",
            "text": "  Xin chào   đây là video demo   ",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.5, "text": "  Xin chào "},
                {"id": 1, "start": 1.5, "end": 3.0, "text": " đây là video demo  "},
            ],
        }

    def unload_model(self):
        return None


def test_vision_processor_process_frames(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_path = frames_dir / "frame_0001_1.00s.jpg"
    Image.new("RGB", (64, 64), color="white").save(frame_path)

    processor = DummyVisionProcessor(tmp_path)
    results = processor.process_frames(str(frames_dir), "demo.mp4")

    assert len(results) == 1
    assert results[0]["video_name"] == "demo.mp4"
    assert results[0]["frame_name"] == "frame_0001_1.00s.jpg"
    assert results[0]["caption"] == "A person giving a presentation"
    assert results[0]["timestamp"] == 1.0
    assert results[0]["clip_embedding"] == [0.1, 0.2, 0.3]
    assert results[0]["clip_model_name"] == "ViT-B-32:openai"

    interim_file = tmp_path / "interim_captions" / "demo_captions.json"
    processed_file = tmp_path / "processed" / "demo_captions_processed.json"
    assert interim_file.exists()
    assert processed_file.exists()


def test_whisper_processor_transcribe_and_save_json(tmp_path):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio-data")

    processor = DummyWhisperProcessor(tmp_path)
    result = processor.transcribe(str(audio_path), video_name="demo.mp4")

    assert result["video_name"] == "demo.mp4"
    assert result["language"] == "vi"
    assert result["full_text"] == "Xin chào đây là video demo"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["text"] == "Xin chào"
    assert result["segments"][1]["text"] == "đây là video demo"

    interim_file = tmp_path / "interim_transcripts" / "demo_transcript.json"
    processed_file = tmp_path / "processed" / "demo_transcript_processed.json"
    assert interim_file.exists()
    assert processed_file.exists()

    with open(interim_file, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["full_text"] == "Xin chào đây là video demo"
    assert saved["segments"][0]["text"] == "Xin chào"