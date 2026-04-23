from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from main_pipeline import MediaDataPipeline
from src.ingest.youtube_ingestor import YouTubeIngestor
from src.utils import (
    cleanup_video_artifacts,
    get_config,
    get_data_path,
    get_video_catalog_entry,
    load_video_catalog,
    looks_like_pytest_temp_path,
    resolve_video_path_from_catalog_entry,
    sanitize_filename_component,
    sanitize_video_catalog,
    save_video_catalog,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Media Semantic Search API")

ALLOWED_CONTENT_TYPES = {"transcription", "segment_chunk", "caption", "multimodal"}
ALLOWED_SEARCH_MODES = {"auto", "action", "visual", "topic", "audio"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
    "application/octet-stream",
}

_pipeline: Optional[MediaDataPipeline] = None
_youtube_ingestor: Optional[YouTubeIngestor] = None


def get_pipeline() -> MediaDataPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MediaDataPipeline(get_config())
    return _pipeline


def get_youtube_ingestor() -> YouTubeIngestor:
    global _youtube_ingestor
    if _youtube_ingestor is None:
        _youtube_ingestor = YouTubeIngestor(get_config())
    return _youtube_ingestor


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    content_type: Optional[str] = None
    video_name: Optional[str] = None
    search_mode: str = Field(default="auto")


class ProcessVideoRequest(BaseModel):
    video_path: str
    reset_index: bool = False


class IngestYouTubeRequest(BaseModel):
    video_url: str
    reset_index: bool = False


class VideoCleanupRequest(BaseModel):
    delete_raw: bool = False
    delete_audio: bool = False
    delete_frames: bool = False
    delete_interim_json: bool = False
    delete_processed: bool = False
    keep_catalog: bool = True


class ReindexVideoRequest(BaseModel):
    reset_index: bool = True


def _format_search_result(result: dict) -> dict:
    similarity_score = result.get("similarity_score")
    if similarity_score is None and "relevance" in result:
        similarity_score = result.get("relevance")

    return {
        "document": result.get("document"),
        "display_text": result.get("display_text"),
        "display_caption": result.get("display_caption"),
        "nearby_speech_context": result.get("nearby_speech_context"),
        "group_size": result.get("group_size"),
        "event_time_range": result.get("event_time_range"),
        "metadata": result.get("metadata"),
        "distance": result.get("distance"),
        "similarity_score": similarity_score,
        "score_type": result.get("score_type", "legacy_or_unspecified"),
        "fusion_score": result.get("fusion_score"),
        "query_type": result.get("query_type"),
        "search_mode": result.get("search_mode"),
        "matched_signals": result.get("matched_signals", []),
        "ranking_explanation": result.get("ranking_explanation", ""),
        "score_breakdown": result.get("score_breakdown", {}),
    }


def _resolve_video_path_from_catalog_or_raw(video_name: str, entry: dict) -> Path:
    local_video_path = str(entry.get("local_video_path", "")).strip()

    if local_video_path:
        candidate = get_data_path(local_video_path)
        if candidate.exists():
            return candidate

        raw_path_obj = Path(local_video_path)
        if raw_path_obj.is_absolute() and raw_path_obj.exists():
            return raw_path_obj

    fallback = get_data_path(get_config()["paths"].get("raw_dir", "data/raw")) / video_name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not resolve video path for '{video_name}'. "
        f"Catalog path='{local_video_path}', raw fallback='{fallback}'"
    )


def _make_non_overwriting_upload_path(upload_dir: Path, original_name: str, file_bytes: bytes) -> Path:
    safe_name = Path(original_name).name
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix.lower()

    candidate = upload_dir / safe_name
    if not candidate.exists():
        return candidate

    digest = hashlib.md5(file_bytes).hexdigest()[:8]
    sanitized_stem = sanitize_filename_component(stem, fallback="uploaded_video")
    new_name = f"{sanitized_stem}_{digest}{suffix}"
    return upload_dir / new_name


@app.get("/")
def root():
    return {
        "message": "Media Semantic Search API is running",
        "endpoints": [
            "/search",
            "/process-video",
            "/upload-video",
            "/ingest-youtube",
            "/videos",
            "/videos/inventory",
            "/videos/{video_name}",
            "/videos/{video_name}/cleanup",
            "/videos/{video_name}/reindex",
            "/catalog/sanitize",
            "/health",
            "/stats",
        ],
    }


@app.get("/health")
def health():
    dependency_status = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
    }

    config = get_config()
    paths_cfg = config.get("paths", {})

    path_status = {
        "raw_dir_exists": get_data_path(paths_cfg.get("raw_dir", "data/raw")).exists(),
        "vector_db_dir_exists": get_data_path(paths_cfg.get("vector_db_dir", "data/vector_db")).exists(),
        "video_catalog_exists": get_data_path(paths_cfg.get("video_catalog_path", "data/video_catalog.json")).exists(),
    }

    try:
        pipeline = get_pipeline()
        stats = pipeline.vector_indexer.get_stats()
        vector_db_ok = "error" not in stats
    except Exception as e:
        stats = {"error": str(e)}
        vector_db_ok = False

    ok = all(dependency_status.values()) and all(path_status.values()) and vector_db_ok

    return {
        "status": "ok" if ok else "degraded",
        "dependencies": dependency_status,
        "paths": path_status,
        "vector_db": stats,
    }


@app.get("/stats")
def stats():
    try:
        pipeline = get_pipeline()
        return pipeline.vector_indexer.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get stats: {e}") from e


@app.get("/videos")
def list_videos():
    try:
        pipeline = get_pipeline()
        videos = pipeline.vector_indexer.list_videos()
        return {
            "total_videos": len(videos),
            "videos": videos,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list videos: {e}") from e


@app.get("/videos/inventory")
def all_videos_inventory():
    try:
        pipeline = get_pipeline()
        return pipeline.vector_indexer.get_all_videos_inventory()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get video inventory: {e}") from e


@app.get("/videos/{video_name}")
def get_video_inventory(video_name: str):
    safe_video_name = video_name.strip()
    if not safe_video_name:
        raise HTTPException(status_code=400, detail="video_name must not be empty")

    try:
        pipeline = get_pipeline()
        inventory = pipeline.vector_indexer.get_video_inventory(safe_video_name)
        if not inventory.get("exists"):
            raise HTTPException(status_code=404, detail=f"Video not found in index: {safe_video_name}")
        return inventory
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get video inventory: {e}") from e


@app.delete("/videos/{video_name}")
def delete_video(video_name: str):
    safe_video_name = video_name.strip()
    if not safe_video_name:
        raise HTTPException(status_code=400, detail="video_name must not be empty")

    try:
        pipeline = get_pipeline()
        deleted_count = pipeline.vector_indexer.delete_video_data(safe_video_name)
        return {
            "video_name": safe_video_name,
            "deleted_records": deleted_count,
            "message": (
                f"Deleted {deleted_count} indexed records for '{safe_video_name}'"
                if deleted_count > 0
                else f"No indexed records found for '{safe_video_name}'"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete video data: {e}") from e


@app.post("/videos/{video_name}/cleanup")
def cleanup_video(video_name: str, request: VideoCleanupRequest):
    safe_video_name = video_name.strip()
    if not safe_video_name:
        raise HTTPException(status_code=400, detail="video_name must not be empty")

    try:
        result = cleanup_video_artifacts(
            safe_video_name,
            delete_raw=request.delete_raw,
            delete_audio=request.delete_audio,
            delete_frames=request.delete_frames,
            delete_interim_json=request.delete_interim_json,
            delete_processed=request.delete_processed,
            keep_catalog=request.keep_catalog,
            config=get_config(),
        )
        return {
            "message": f"Cleanup completed for '{safe_video_name}'",
            "cleanup_result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clean up artifacts: {e}") from e


@app.post("/videos/{video_name}/reindex")
def reindex_video(video_name: str, request: ReindexVideoRequest):
    safe_video_name = video_name.strip()
    if not safe_video_name:
        raise HTTPException(status_code=400, detail="video_name must not be empty")

    try:
        entry = get_video_catalog_entry(safe_video_name, force_reload=True, config=get_config())
        if not entry:
            raise HTTPException(status_code=404, detail=f"Video not found in catalog: {safe_video_name}")

        try:
            resolved_path = resolve_video_path_from_catalog_entry(
                safe_video_name,
                entry,
                config=get_config(),
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        source_platform = str(entry.get("source_platform", "")).strip() or "local"
        source_url = str(entry.get("source_url", "")).strip()
        title = str(entry.get("title", safe_video_name)).strip() or safe_video_name
        description = str(entry.get("description", "")).strip()
        thumbnail_url = str(entry.get("thumbnail_url", "")).strip()
        tags = entry.get("tags") or []

        pipeline = get_pipeline()
        result = pipeline.process_video(
            str(resolved_path),
            reset_index=request.reset_index,
            source_metadata={
                "source_platform": source_platform,
                "source_url": source_url,
                "video_title": title,
                "video_description": description,
                "thumbnail_url": thumbnail_url,
                "video_tags": tags,
                "ingest_method": str(entry.get("ingest_method", "local_file")).strip() or "local_file",
                "has_audio": entry.get("has_audio"),
                "video_type": str(entry.get("video_type", "")).strip(),
                "estimated_content_style": str(entry.get("estimated_content_style", "")).strip(),
                "recommended_search_mode": str(entry.get("recommended_search_mode", "")).strip(),
                "duration_sec": entry.get("duration_sec"),
            },
        )
        return {
            "message": f"Re-indexed '{safe_video_name}' successfully",
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not re-index video: {e}") from e


@app.post("/catalog/sanitize")
def sanitize_catalog():
    try:
        config = get_config()
        raw_dir = get_data_path(config["paths"].get("raw_dir", "data/raw"))
        catalog = load_video_catalog(force_reload=True, config=config)

        cleaned = []
        removed = []

        for item in catalog:
            if not isinstance(item, dict):
                continue

            video_name = str(item.get("video_name", "")).strip()
            local_video_path = str(item.get("local_video_path", "")).strip()

            should_remove = False
            if local_video_path and looks_like_pytest_temp_path(local_video_path):
                fallback = raw_dir / video_name
                if not fallback.exists():
                    should_remove = True

            if should_remove:
                removed.append(
                    {
                        "video_name": video_name,
                        "local_video_path": local_video_path,
                        "reason": "stale_pytest_temp_path",
                    }
                )
                continue

            cleaned.append(item)

        save_video_catalog(cleaned, config=config)

        return {
            "message": "Catalog sanitize completed",
            "removed_count": len(removed),
            "removed": removed,
            "remaining_count": len(cleaned),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not sanitize catalog: {e}") from e


@app.post("/search")
def search(request: SearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    if request.content_type and request.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid content_type")

    request.search_mode = (request.search_mode or "auto").strip().lower()
    if request.search_mode not in ALLOWED_SEARCH_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search_mode. Allowed: {sorted(ALLOWED_SEARCH_MODES)}",
        )

    try:
        pipeline = get_pipeline()
        results = pipeline.search(
            query=query,
            top_k=request.top_k,
            content_type=request.content_type,
            video_name=request.video_name.strip() if request.video_name else None,
            search_mode=request.search_mode,
        )

        formatted = [_format_search_result(r) for r in results]
        return {
            "search_mode": request.search_mode,
            "results": formatted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e


@app.post("/process-video")
def process_video(request: ProcessVideoRequest):
    video_path = request.video_path.strip()
    if not video_path:
        raise HTTPException(status_code=400, detail="video_path must not be empty")

    try:
        pipeline = get_pipeline()
        result = pipeline.process_video(video_path, reset_index=request.reset_index)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}") from e


@app.post("/upload-video")
def upload_video(
    file: UploadFile = File(...),
    reset_index: bool = Query(default=False),
):
    safe_name = Path(file.filename or "uploaded_video.mp4").name
    suffix = Path(safe_name).suffix.lower()

    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {suffix}. Allowed: {sorted(ALLOWED_VIDEO_SUFFIXES)}",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {content_type}",
        )

    config = get_config()
    upload_dir = Path(get_data_path(config["paths"].get("raw_dir", "data/raw")))
    upload_dir.mkdir(parents=True, exist_ok=True)

    try:
        file_bytes = file.file.read()
        save_path = _make_non_overwriting_upload_path(upload_dir, safe_name, file_bytes)

        with save_path.open("wb") as f:
            f.write(file_bytes)

        pipeline = get_pipeline()
        result = pipeline.process_video(str(save_path), reset_index=reset_index)

        return {
            "uploaded_path": str(save_path),
            "result": result,
            "message": f"Uploaded and processed '{save_path.name}'.",
        }
    except HTTPException:
        raise
    except Exception as e:
        try:
            if "save_path" in locals() and save_path.exists():
                save_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Upload/process failed: {e}") from e
    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.post("/ingest-youtube")
def ingest_youtube(request: IngestYouTubeRequest):
    raw_url = request.video_url.strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="video_url must not be empty")

    try:
        logger.info("Received YouTube ingest request: %s", raw_url)

        ingestor = get_youtube_ingestor()
        ingest_result = ingestor.ingest(raw_url)

        logger.info(
            "YouTube ingest metadata ready: %s -> %s",
            ingest_result["source_url"],
            ingest_result["video_name"],
        )

        pipeline = get_pipeline()
        process_result = pipeline.process_video(
            ingest_result["video_path"],
            reset_index=request.reset_index,
            source_metadata=ingest_result,
        )

        return {
            "ingest_result": ingest_result,
            "result": process_result,
            "message": f"Ingested and processed YouTube video: {ingest_result['video_name']}",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("YouTube ingest failed")
        raise HTTPException(status_code=500, detail=f"YouTube ingest failed: {e}") from e