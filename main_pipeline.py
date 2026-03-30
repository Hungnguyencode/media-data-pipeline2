from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.extract.audio_extractor import AudioExtractor, NoAudioStreamError
from src.extract.frame_extractor import FrameExtractor
from src.indexing.vector_indexer import VectorIndexer
from src.retrieval.search_engine import SearchEngine
from src.transform.vision_processor import VisionProcessor
from src.transform.whisper_processor import WhisperProcessor
from src.utils import (
    ensure_video_catalog_entry,
    ensure_video_catalog_entry_from_metadata,
    get_config,
    get_data_path,
    get_video_catalog_entry,
    md5_of_file,
    release_memory,
    save_json,
    setup_logging,
    stage_timer,
)

logger = logging.getLogger(__name__)


class MediaDataPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()

        self.audio_extractor = AudioExtractor(self.config)
        self.frame_extractor = FrameExtractor(self.config)
        self.whisper_processor = WhisperProcessor(self.config)
        self.vision_processor = VisionProcessor(self.config)

        self.vector_indexer = VectorIndexer(self.config)
        self.search_engine = SearchEngine(
            config=self.config,
            vector_indexer=self.vector_indexer,
            vision_processor=self.vision_processor,
        )

        self.processed_dir = Path(
            get_data_path(self.config["paths"].get("processed_dir", "data/processed"))
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_version = str(self.config.get("pipeline", {}).get("version", "1.0.0"))
        self.save_run_metadata = bool(self.config.get("pipeline", {}).get("save_run_metadata", True))

    def _build_base_result(self, video_path: str) -> Dict[str, Any]:
        video_name = Path(video_path).name
        return {
            "video_name": video_name,
            "video_path": video_path,
            "audio_path": None,
            "frames_dir": None,
            "transcription_records": 0,
            "caption_records": 0,
            "multimodal_records": 0,
            "merged_output_path": None,
            "run_metadata_path": None,
            "stage_metrics_sec": {},
            "data_summary": {},
            "video_source_info": {},
            "stage_status": {
                "extract_audio": "pending",
                "extract_frames": "pending",
                "transcribe": "pending",
                "caption": "pending",
                "index": "pending",
            },
        }

    def _mark_stage(self, result: Dict[str, Any], stage_name: str, status: str) -> None:
        result["stage_status"][stage_name] = status

    def _build_video_source_info(self, video_name: str, video_path: str) -> Dict[str, Any]:
        entry = get_video_catalog_entry(
            video_name,
            force_reload=True,
            config=self.config,
        ) or {}

        local_video_path = entry.get("local_video_path") or video_path
        tags = entry.get("tags") or []

        if isinstance(tags, list):
            normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
            video_tags = "|".join(normalized_tags)
        else:
            video_tags = str(tags).strip()

        return {
            "video_name": video_name,
            "local_video_path": str(local_video_path),
            "source_platform": str(entry.get("source_platform", "local")).strip() or "local",
            "source_url": str(entry.get("source_url", "")).strip(),
            "video_title": str(entry.get("title", video_name)).strip() or video_name,
            "video_description": str(entry.get("description", "")).strip(),
            "thumbnail_url": str(entry.get("thumbnail_url", "")).strip(),
            "video_tags": video_tags,
            "created_at": str(entry.get("created_at", "")).strip(),
            "ingested_at": str(entry.get("ingested_at", "")).strip(),
        }

    def process_video(
        self,
        video_path: str,
        reset_index: bool = False,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        video_path = str(Path(video_path).resolve())
        video_name = Path(video_path).name

        logger.info("Starting pipeline for video: %s", video_name)
        result = self._build_base_result(video_path)

        audio_path = None
        frames_dir = None
        transcription_data = None
        captions_data = None
        trans_count = 0
        cap_count = 0
        multi_count = 0

        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if source_metadata:
            logger.info("Applying external source metadata for %s", video_name)
            ensure_video_catalog_entry_from_metadata(
                video_path=video_path,
                source_platform=str(source_metadata.get("source_platform", "local")).strip() or "local",
                source_url=str(source_metadata.get("source_url", "")).strip(),
                title=str(source_metadata.get("video_title", video_name)).strip() or video_name,
                description=str(source_metadata.get("video_description", "")).strip(),
                thumbnail_url=str(source_metadata.get("thumbnail_url", "")).strip(),
                tags=source_metadata.get("video_tags") or [],
                config=self.config,
            )
        else:
            ensure_video_catalog_entry(
                video_path=video_path,
                config=self.config,
                source_platform="local",
            )

        video_source_info = self._build_video_source_info(video_name, video_path)
        result["video_source_info"] = video_source_info

        runtime_metadata = {
            "video_name": video_name,
            "video_path": video_path,
            "pipeline_version": self.pipeline_version,
            "video_size_bytes": video_file.stat().st_size,
            "video_checksum_md5": md5_of_file(video_path),
            "video_source_info": video_source_info,
        }

        stage_metrics = result["stage_metrics_sec"]

        try:
            with stage_timer("extract_audio", stage_metrics):
                try:
                    audio_path = self.audio_extractor.extract_audio(video_path)
                    result["audio_path"] = audio_path
                    self._mark_stage(result, "extract_audio", "done")
                except NoAudioStreamError:
                    logger.warning("No audio stream for %s. Continuing with visual-only pipeline.", video_name)
                    audio_path = None
                    result["audio_path"] = None
                    self._mark_stage(result, "extract_audio", "skipped_no_audio")
        except Exception:
            self._mark_stage(result, "extract_audio", "failed")
            raise

        try:
            with stage_timer("extract_frames", stage_metrics):
                frames_dir = self.frame_extractor.extract_frames(video_path)
                result["frames_dir"] = frames_dir
                self._mark_stage(result, "extract_frames", "done")
        except Exception:
            self._mark_stage(result, "extract_frames", "failed")
            raise

        if audio_path:
            try:
                with stage_timer("transcribe", stage_metrics):
                    transcription_data = self.whisper_processor.transcribe(audio_path, video_name=video_name)
                    transcription_data["video_source_info"] = video_source_info
                    self._mark_stage(result, "transcribe", "done")
            except Exception:
                self._mark_stage(result, "transcribe", "failed")
                raise
        else:
            self._mark_stage(result, "transcribe", "skipped_no_audio")

        try:
            with stage_timer("caption", stage_metrics):
                captions_data = self.vision_processor.process_frames(frames_dir, video_name=video_name)
                self._mark_stage(result, "caption", "done")
        except Exception:
            self._mark_stage(result, "caption", "failed")
            raise

        try:
            with stage_timer("index", stage_metrics):
                if reset_index:
                    self.vector_indexer.delete_video_data(video_name)

                if transcription_data:
                    trans_count = self.vector_indexer.index_transcriptions(
                        transcription_data,
                        video_source_info=video_source_info,
                    )
                else:
                    trans_count = 0

                cap_count = self.vector_indexer.index_captions(
                    captions_data or [],
                    video_source_info=video_source_info,
                )

                if transcription_data:
                    multi_count = self.vector_indexer.index_multimodal_documents(
                        transcription_data,
                        captions_data or [],
                        video_source_info=video_source_info,
                    )
                else:
                    multi_count = 0

                self._mark_stage(result, "index", "done")
        except Exception:
            self._mark_stage(result, "index", "failed")
            raise

        data_summary = {
            "indexed_total_records": trans_count + cap_count + multi_count,
            "transcription_records": trans_count,
            "caption_records": cap_count,
            "multimodal_records": multi_count,
            "num_transcript_segments": len(transcription_data.get("segments", [])) if transcription_data else 0,
            "num_caption_frames": len(captions_data) if captions_data else 0,
            "audio_available": audio_path is not None,
        }

        merged_output = {
            "video_name": video_name,
            "video_path": video_path,
            "pipeline_version": self.pipeline_version,
            "video_source_info": video_source_info,
            "runtime_metadata": runtime_metadata,
            "stage_metrics_sec": stage_metrics,
            "stage_status": result["stage_status"],
            "indexing_summary": data_summary,
            "outputs": {
                "audio_path": audio_path,
                "frames_dir": frames_dir,
                "transcription": transcription_data,
                "captions": captions_data,
            },
        }

        merged_output_path = self.processed_dir / f"{Path(video_name).stem}_merged_output.json"
        save_json(merged_output, merged_output_path)

        result["transcription_records"] = trans_count
        result["caption_records"] = cap_count
        result["multimodal_records"] = multi_count
        result["merged_output_path"] = str(merged_output_path)
        result["data_summary"] = data_summary

        if self.save_run_metadata:
            run_metadata = {
                "video_name": video_name,
                "pipeline_version": self.pipeline_version,
                "stage_status": result["stage_status"],
                "stage_metrics_sec": stage_metrics,
                "runtime_metadata": runtime_metadata,
                "video_source_info": video_source_info,
                "indexing_summary": data_summary,
            }
            run_metadata_path = self.processed_dir / f"{Path(video_name).stem}_run_metadata.json"
            save_json(run_metadata, run_metadata_path)
            result["run_metadata_path"] = str(run_metadata_path)

        release_memory()

        logger.info(
            "Pipeline completed for '%s' | transcription=%d | caption=%d | multimodal=%d",
            video_name,
            trans_count,
            cap_count,
            multi_count,
        )

        return result

    def search(
        self,
        query: str,
        top_k: int = 5,
        content_type: Optional[str] = None,
        video_name: Optional[str] = None,
    ):
        return self.search_engine.search(
            query=query,
            top_k=top_k,
            content_type=content_type,
            video_name=video_name,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Media data pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Delete previous indexed records of this video before re-indexing",
    )
    return parser


def main() -> None:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    pipeline = MediaDataPipeline(get_config())
    result = pipeline.process_video(args.video, reset_index=args.reset_index)

    print("\n=== PIPELINE RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()