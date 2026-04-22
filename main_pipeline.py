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
    normalize_source_metadata_for_pipeline,
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

        self.pipeline_version = str(self.config.get("pipeline", {}).get("version", "2.2.0"))
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
        entry = get_video_catalog_entry(video_name, force_reload=True, config=self.config) or {}
        return {
            "source_platform": str(entry.get("source_platform", "")).strip(),
            "source_url": str(entry.get("source_url", "")).strip(),
            "video_title": str(entry.get("title", video_name)).strip(),
            "video_description": str(entry.get("description", "")).strip(),
            "thumbnail_url": str(entry.get("thumbnail_url", "")).strip(),
            "video_tags": entry.get("tags") or [],
            "local_video_path": str(entry.get("local_video_path", "")).strip() or str(video_path),
            "created_at": str(entry.get("created_at", "")).strip(),
            "ingested_at": str(entry.get("ingested_at", "")).strip(),
            "ingest_method": str(entry.get("ingest_method", "local_file")).strip() or "local_file",
            "has_audio": entry.get("has_audio"),
            "video_type": str(entry.get("video_type", "")).strip(),
            "estimated_content_style": str(entry.get("estimated_content_style", "")).strip(),
            "recommended_search_mode": str(entry.get("recommended_search_mode", "")).strip(),
            "duration_sec": entry.get("duration_sec"),
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

        normalized_source = normalize_source_metadata_for_pipeline(
            source_metadata,
            video_path=video_path,
            fallback_platform="local",
        )

        if source_metadata:
            logger.info("Applying external source metadata for %s", video_name)
            ensure_video_catalog_entry_from_metadata(
                video_path=video_path,
                source_platform=normalized_source["source_platform"],
                source_url=normalized_source["source_url"],
                title=normalized_source["video_title"],
                description=normalized_source["video_description"],
                thumbnail_url=normalized_source["thumbnail_url"],
                tags=normalized_source["video_tags"],
                config=self.config,
            )
        else:
            ensure_video_catalog_entry(
                video_path=video_path,
                config=self.config,
                source_platform=normalized_source["source_platform"],
                source_url=normalized_source["source_url"],
            )

        # upsert lại metadata mở rộng
        extended_entry = get_video_catalog_entry(video_name, force_reload=True, config=self.config) or {}
        extended_entry.update(
            {
                "video_name": video_name,
                "local_video_path": normalized_source["local_video_path"],
                "source_platform": normalized_source["source_platform"],
                "source_url": normalized_source["source_url"],
                "title": normalized_source["video_title"],
                "description": normalized_source["video_description"],
                "thumbnail_url": normalized_source["thumbnail_url"],
                "tags": normalized_source["video_tags"],
                "ingest_method": normalized_source["ingest_method"],
                "has_audio": normalized_source["has_audio"],
                "video_type": normalized_source["video_type"],
                "estimated_content_style": normalized_source["estimated_content_style"],
                "recommended_search_mode": normalized_source["recommended_search_mode"],
                "duration_sec": normalized_source["duration_sec"],
            }
        )
        from src.utils import upsert_video_catalog_entry
        upsert_video_catalog_entry(extended_entry, config=self.config)

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
                    video_source_info["has_audio"] = True
                except NoAudioStreamError:
                    logger.warning("No audio stream for %s. Continuing with visual-only pipeline.", video_name)
                    audio_path = None
                    self._mark_stage(result, "extract_audio", "skipped_no_audio")
                    video_source_info["has_audio"] = False

            with stage_timer("extract_frames", stage_metrics):
                frames_dir = self.frame_extractor.extract_frames(video_path)
                result["frames_dir"] = frames_dir
                self._mark_stage(result, "extract_frames", "done")

            if reset_index:
                self.vector_indexer.delete_video_data(video_name)

            if audio_path:
                with stage_timer("transcribe", stage_metrics):
                    transcription_data = self.whisper_processor.transcribe(audio_path, video_name=video_name)
                    trans_count = self.vector_indexer.index_transcriptions(
                        transcription_data,
                        video_source_info=video_source_info,
                    )
                    self._mark_stage(result, "transcribe", "done")
            else:
                self._mark_stage(result, "transcribe", "skipped_no_audio")

            with stage_timer("caption", stage_metrics):
                captions_data = self.vision_processor.process_frames(frames_dir, video_name=video_name)
                cap_count = self.vector_indexer.index_captions(
                    captions_data,
                    video_source_info=video_source_info,
                )
                self._mark_stage(result, "caption", "done")

            with stage_timer("index", stage_metrics):
                if transcription_data and captions_data:
                    multi_count = self.vector_indexer.index_multimodal_documents(
                        transcription_data,
                        captions_data,
                        video_source_info=video_source_info,
                    )
                else:
                    multi_count = 0
                self._mark_stage(result, "index", "done")

        except Exception:
            release_memory()
            raise

        data_summary = {
            "indexed_total_records": trans_count + cap_count + multi_count,
            "transcription_records": trans_count,
            "caption_records": cap_count,
            "multimodal_records": multi_count,
            "has_audio": video_source_info.get("has_audio"),
            "video_type": video_source_info.get("video_type"),
            "estimated_content_style": video_source_info.get("estimated_content_style"),
            "recommended_search_mode": video_source_info.get("recommended_search_mode"),
        }

        merged_output = {
            "video_name": video_name,
            "pipeline_version": self.pipeline_version,
            "video_source_info": video_source_info,
            "data_summary": data_summary,
            "available_fields": {
                "frame_name": "Tên frame ảnh",
                "timestamp": "Mốc thời gian theo giây",
                "content_type": "Loại tài liệu index",
                "source_modality": "Nguồn dữ liệu: audio/image/audio+image",
                "model_name": "Tên model dùng để sinh dữ liệu",
                "pipeline_version": "Phiên bản pipeline",
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
                "indexing_summary": data_summary,
            }
            run_metadata_path = self.processed_dir / f"{Path(video_name).stem}_run_metadata.json"
            save_json(run_metadata, run_metadata_path)
            result["run_metadata_path"] = str(run_metadata_path)

        release_memory()

        logger.info(
            "Pipeline finished for '%s'. Indexed %d transcription, %d caption, %d multimodal records.",
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
        search_mode: Optional[str] = "auto",
    ):
        return self.search_engine.search(
            query=query,
            top_k=top_k,
            content_type=content_type,
            video_name=video_name,
            search_mode=search_mode,
        )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Media Data Pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--query", type=str, help="Semantic query")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results")
    parser.add_argument("--content-type", type=str, default=None, help="Filter by content type")
    parser.add_argument("--video-name", type=str, default=None, help="Filter by video name")
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Delete previous indexed data for this video before re-index",
    )
    args = parser.parse_args()

    pipeline = MediaDataPipeline()

    if args.video:
        result = pipeline.process_video(args.video, reset_index=args.reset_index)
        print(result)
    elif args.query:
        results = pipeline.search(
            query=args.query,
            top_k=args.top_k,
            content_type=args.content_type,
            video_name=args.video_name,
        )
        for item in results:
            print(item)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()