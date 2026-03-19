from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.extract.audio_extractor import AudioExtractor
from src.extract.frame_extractor import FrameExtractor


def create_dummy_video(video_path: Path, num_frames: int = 10, fps: int = 5):
    height, width = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


def test_frame_extraction(tmp_path):
    video_path = tmp_path / "dummy.mp4"
    create_dummy_video(video_path)

    config = {
        "video": {"frame_sampling_fps": 1.0},
        "paths": {"interim_frames_dir": str(tmp_path / "frames")}
    }

    extractor = FrameExtractor(config)
    frames_dir = extractor.extract_frames(str(video_path))

    frame_files = list(Path(frames_dir).glob("*.jpg"))
    assert len(frame_files) > 0


@patch("src.extract.audio_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
@patch("src.extract.audio_extractor.subprocess.run")
def test_audio_extraction(mock_run, mock_which, tmp_path):
    video_path = tmp_path / "dummy.mp4"
    create_dummy_video(video_path)

    output_dir = tmp_path / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_audio = output_dir / "dummy.wav"

    def fake_run(*args, **kwargs):
        expected_audio.write_bytes(b"fake wav content")
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    mock_run.side_effect = fake_run

    config = {
        "paths": {"interim_audio_dir": str(output_dir)}
    }

    extractor = AudioExtractor(config)
    audio_path = extractor.extract_audio(str(video_path))

    assert Path(audio_path).exists()
    assert Path(audio_path).suffix == ".wav"