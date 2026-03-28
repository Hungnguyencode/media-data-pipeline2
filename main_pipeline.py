from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.extract.audio_extractor import AudioExtractor
from src.extract.frame_extractor import FrameExtractor
from src.indexing.vector_indexer import VectorIndexer
from src.retrieval.search_engine import SearchEngine
from src.transform.vision_processor import VisionProcessor
from src.transform.whisper_processor import WhisperProcessor
from src.utils import (
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

    def process_video(self, video_path: str, reset_index: bool = False) -> Dict[str, Any]:
        video_path = str(Path(video_path).resolve())
        video_name = Path(video_path).name

        logger.info("Starting pipeline for video: %s", video_name)
        result = self._build_base_result(video_path)

        audio_path = None
        frames_dir = None
        transcription_data = None
        captions_data = None

        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

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
                audio_path = self.audio_extractor.extract_audio(video_path)
                result["audio_path"] = audio_path
                self._mark_stage(result, "extract_audio", "done")
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

        try:
            with stage_timer("transcribe", stage_metrics):
                transcription_data = self.whisper_processor.transcribe(audio_path, video_name=video_name)
                transcription_data["video_source_info"] = video_source_info
                self._mark_stage(result, "transcribe", "done")
        except Exception:
            self._mark_stage(result, "transcribe", "failed")
            raise
        finally:
            self.whisper_processor.unload_model()

        if not transcription_data:
            transcription_data = {
                "video_name": video_name,
                "audio_path": audio_path,
                "language": self.config["models"]["whisper"].get("language", "vi"),
                "full_text": "",
                "segments": [],
                "model_name": self.config["models"]["whisper"].get("name", "base"),
                "video_source_info": video_source_info,
            }

        try:
            with stage_timer("caption", stage_metrics):
                captions_data = self.vision_processor.process_frames(frames_dir, video_name=video_name)
                for item in captions_data:
                    item["video_source_info"] = video_source_info
                self._mark_stage(result, "caption", "done")
        except Exception:
            self._mark_stage(result, "caption", "failed")
            raise
        finally:
            self.vision_processor.unload_model()

        if captions_data is None:
            captions_data = []

        try:
            with stage_timer("index", stage_metrics):
                deleted_count = 0
                if reset_index:
                    deleted_count = self.vector_indexer.delete_video_data(video_name)
                    logger.info("Deleted %d previous records for '%s'", deleted_count, video_name)

                trans_count = self.vector_indexer.index_transcriptions(
                    transcription_data,
                    video_source_info=video_source_info,
                )
                cap_count = self.vector_indexer.index_captions(
                    captions_data,
                    video_source_info=video_source_info,
                )
                multi_count = self.vector_indexer.index_multimodal_documents(
                    transcription_data=transcription_data,
                    captions_data=captions_data,
                    video_source_info=video_source_info,
                )
                self._mark_stage(result, "index", "done")
        except Exception:
            self._mark_stage(result, "index", "failed")
            raise

        data_summary = {
            "transcript_segment_count": len(transcription_data.get("segments", [])),
            "frame_caption_count": len(captions_data),
            "indexed_transcription_records": trans_count,
            "indexed_caption_records": cap_count,
            "indexed_multimodal_records": multi_count,
            "indexed_total_records": trans_count + cap_count + multi_count,
            "deleted_previous_records": deleted_count if reset_index else 0,
        }

        merged_output = {
            "video_name": video_name,
            "video_path": video_path,
            "audio_path": audio_path,
            "frames_dir": frames_dir,
            "video_source_info": video_source_info,
            "transcription": transcription_data,
            "captions": captions_data,
            "indexing_summary": data_summary,
            "runtime_metadata": runtime_metadata,
            "stage_metrics_sec": stage_metrics,
            "output_schema_notes": {
                "video_name": "Tên file video nguồn",
                "audio_path": "Đường dẫn file audio đã trích xuất",
                "frame_name": "Tên frame ảnh",
                "timestamp": "Mốc thời gian theo giây",
                "content_type": "Loại tài liệu index",
                "source_modality": "Nguồn dữ liệu: audio/image/audio+image",
                "model_name": "Tên model dùng để sinh dữ liệu",
                "pipeline_version": "Phiên bản pipeline",
                "source_platform": "Nền tảng gốc của video",
                "source_url": "Link nguồn gốc của video",
                "video_title": "Tiêu đề video trong catalog",
                "video_description": "Mô tả video trong catalog",
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
    ):
        return self.search_engine.search(
            query=query,
            top_k=top_k,
            content_type=content_type,
            video_name=video_name,
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

    if args.query:
        results = pipeline.search(
            args.query,
            top_k=args.top_k,
            content_type=args.content_type,
            video_name=args.video_name,
        )
        print(results)


if __name__ == "__main__":
    main()