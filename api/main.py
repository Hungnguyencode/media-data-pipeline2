from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from main_pipeline import MediaDataPipeline
from src.utils import get_config, get_data_path, setup_logging

setup_logging()

app = FastAPI(title="Media Semantic Search API")

ALLOWED_CONTENT_TYPES = {"transcription", "segment_chunk", "caption", "multimodal"}
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


def get_pipeline() -> MediaDataPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MediaDataPipeline(get_config())
    return _pipeline


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    content_type: Optional[str] = None
    video_name: Optional[str] = None


class ProcessVideoRequest(BaseModel):
    video_path: str
    reset_index: bool = False


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
    }


@app.get("/")
def root():
    return {
        "message": "Media Semantic Search API is running",
        "endpoints": [
            "/search",
            "/process-video",
            "/upload-video",
            "/stats",
            "/videos",
            "/videos/inventory",
            "/videos/{video_name}",
            "/health",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.post("/search")
def search(request: SearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    if request.content_type and request.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid content_type")

    try:
        pipeline = get_pipeline()
        results = pipeline.search(
            query=query,
            top_k=request.top_k,
            content_type=request.content_type,
            video_name=request.video_name.strip() if request.video_name else None,
        )

        formatted = [_format_search_result(r) for r in results]
        return {"results": formatted}
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

    save_path = upload_dir / safe_name
    if save_path.exists():
        stem = save_path.stem
        ext = save_path.suffix
        counter = 1
        while save_path.exists():
            save_path = upload_dir / f"{stem}_{counter}{ext}"
            counter += 1

    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        pipeline = get_pipeline()
        result = pipeline.process_video(str(save_path), reset_index=reset_index)
        return {
            "uploaded_path": str(save_path),
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Upload/process failed: {e}") from e
    finally:
        try:
            file.file.close()
        except Exception:
            pass