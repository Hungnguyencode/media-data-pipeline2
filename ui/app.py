from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st
import yaml


def load_api_base() -> str:
    env_api_base = os.getenv("API_BASE")
    if env_api_base:
        return env_api_base.rstrip("/")

    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            api_base = config.get("api", {}).get("base_url")
            if api_base:
                return str(api_base).rstrip("/")
        except Exception:
            pass

    return "http://127.0.0.1:8000"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_video_file_path(video_name: Optional[str]) -> Optional[Path]:
    if not video_name:
        return None

    config_path = get_project_root() / "configs" / "config.yaml"
    raw_dir = get_project_root() / "data" / "raw"

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            raw_dir = get_project_root() / config.get("paths", {}).get("raw_dir", "data/raw")
        except Exception:
            pass

    candidate = raw_dir / video_name
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def fetch_json_with_retry(
    method: str,
    url: str,
    retries: int = 3,
    delay: float = 1.0,
    **kwargs,
):
    last_error = None
    for _ in range(retries):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            time.sleep(delay)
    raise last_error


def fetch_videos():
    try:
        data = fetch_json_with_retry(
            "GET",
            f"{API_BASE}/videos",
            timeout=30,
        )
        return data.get("videos", [])
    except Exception:
        return []


def fetch_all_inventory():
    try:
        return fetch_json_with_retry(
            "GET",
            f"{API_BASE}/videos/inventory",
            timeout=60,
        )
    except Exception as e:
        return {"error": str(e), "total_videos": 0, "videos": []}


def fetch_video_inventory(video_name: str):
    try:
        return fetch_json_with_retry(
            "GET",
            f"{API_BASE}/videos/{video_name}",
            timeout=30,
        )
    except Exception as e:
        return {"error": str(e)}


def set_query_sample(sample_text: str):
    st.session_state["query_input"] = sample_text


def show_video_preview(video_name: Optional[str], title: str = "Video preview"):
    video_path = get_video_file_path(video_name)

    if not video_path:
        st.info("Chưa tìm thấy file video để preview.")
        return

    st.markdown(f"### {title}")
    st.caption(f"File: {video_path.name}")

    try:
        with open(video_path, "rb") as f:
            st.video(f.read())
    except Exception as e:
        st.warning(f"Không thể hiển thị video preview: {e}")


def shorten_text(text: str, max_chars: int = 260) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def parse_video_tags(value: object) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]

    raw = str(value).strip()
    if not raw:
        return []

    if "|" in raw:
        return [part.strip() for part in raw.split("|") if part.strip()]

    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]

    return [raw]


def show_source_info_block(meta: dict):
    source_platform = meta.get("source_platform")
    source_url = meta.get("source_url")
    video_title = meta.get("video_title") or meta.get("title")
    video_description = meta.get("video_description") or meta.get("description")
    video_tags = parse_video_tags(meta.get("video_tags") or meta.get("tags"))
    thumbnail_url = meta.get("thumbnail_url")

    if video_title:
        st.write("Title:", video_title)
    if source_platform:
        st.write("Source platform:", source_platform)
    if video_tags:
        st.write("Tags:")
        for tag in video_tags:
            st.markdown(f"- {tag}")
    if video_description:
        st.caption(shorten_text(video_description, max_chars=220))
    if thumbnail_url:
        st.write("Thumbnail:", thumbnail_url)
    if source_url:
        st.markdown(f"[Mở link nguồn]({source_url})")


def detect_video_style(source_info: Dict[str, Any]) -> str:
    text = " ".join(
        [
            str(source_info.get("video_title", "") or ""),
            str(source_info.get("video_description", "") or ""),
            " ".join(parse_video_tags(source_info.get("video_tags"))),
        ]
    ).lower()

    if any(token in text for token in ["ted", "talk", "speech", "motivation", "lecture", "presentation"]):
        return "talk"
    if any(token in text for token in ["cook", "egg", "recipe", "kitchen", "food"]):
        return "action"
    if any(token in text for token in ["wildlife", "nature", "animal", "forest", "ocean"]):
        return "visual"
    if any(token in text for token in ["music", "official video", "cinematic", "vlog", "film", "short film"]):
        return "cinematic_music"
    return "generic"


def get_suggested_queries(source_info: Dict[str, Any]) -> list[str]:
    style = detect_video_style(source_info)
    if style == "talk":
        return ["self motivation", "winning strategy", "motivational speech"]
    if style == "action":
        return ["crack egg", "separate egg", "egg yolk"]
    if style == "visual":
        return ["bird flying", "wildlife diversity", "forest animals"]
    if style == "cinematic_music":
        return ["summer sky", "car on road", "beach scene"]
    return ["main topic", "important scene", "key moment"]


def get_recommended_search_mode(source_info: Dict[str, Any]) -> str:
    style = detect_video_style(source_info)
    if style == "talk":
        return "Talk mode"
    if style == "action":
        return "Action mode"
    if style == "visual":
        return "Visual mode"
    if style == "cinematic_music":
        return "Visual mode"
    return "Manual"


def derive_search_preset_defaults(preset_name: str) -> tuple[Optional[str], str]:
    mapping = {
        "Manual": (None, "Tất cả"),
        "Talk mode": ("segment_chunk", "Đoạn transcript"),
        "Action mode": ("caption", "Caption ảnh"),
        "Visual mode": ("caption", "Caption ảnh"),
        "Audio mode": ("transcription", "Toàn bộ transcript"),
    }
    return mapping.get(preset_name, (None, "Tất cả"))


def dominant_modality(source_modality_counts: Dict[str, int]) -> str:
    if not source_modality_counts:
        return "Unknown"
    return max(source_modality_counts.items(), key=lambda item: item[1])[0]


def searchable_types_label(content_type_counts: Dict[str, int]) -> str:
    enabled = [k for k, v in content_type_counts.items() if int(v or 0) > 0]
    return ", ".join(enabled) if enabled else "None"


def format_seconds_to_hhmmss(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    try:
        total = max(0, int(float(seconds)))
    except Exception:
        return "N/A"
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def limitation_hints_for_result(result: Dict[str, Any]) -> list[str]:
    hints: list[str] = []
    meta = result.get("metadata", {}) or {}
    source_info = {
        "video_title": meta.get("video_title", ""),
        "video_description": meta.get("video_description", ""),
        "video_tags": meta.get("video_tags", ""),
    }

    content_type = str(meta.get("content_type", "") or "")
    score_type = str(result.get("score_type", "") or "")
    similarity_score = result.get("similarity_score")
    style = detect_video_style(source_info)

    if content_type == "caption":
        hints.append("Caption is auto-generated and may be approximate.")

    if similarity_score is not None:
        try:
            if float(similarity_score) < 0.35:
                hints.append("Low-confidence result.")
        except Exception:
            pass

    if style == "cinematic_music":
        hints.append("Transcript may be less reliable for this video type.")

    if style == "talk":
        hints.append("Topic/speech queries are usually more reliable for this video type.")

    if "legacy" in score_type.lower():
        hints.append("Legacy scoring path was used for this result.")

    return hints


def limitation_hints_for_video(source_info: Dict[str, Any]) -> list[str]:
    style = detect_video_style(source_info)
    if style == "cinematic_music":
        return [
            "Video này thiên về cinematic/music nên transcript có thể kém đáng tin hơn caption/image.",
            "Nên ưu tiên caption hoặc multimodal khi tìm kiếm.",
        ]
    if style == "talk":
        return [
            "Video này phù hợp với semantic search theo topic, idea và speech context.",
            "Các truy vấn dựa trên speech/topic thường đáng tin hơn truy vấn object thuần hình ảnh.",
        ]
    if style == "action":
        return [
            "Video này phù hợp với caption-based queries về hành động, vật thể và thao tác.",
        ]
    return []


def call_reindex(video_name: str):
    response = requests.post(
        f"{API_BASE}/videos/{video_name}/reindex",
        json={"reset_index": True},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def call_cleanup(
    video_name: str,
    *,
    delete_raw: bool = False,
    delete_audio: bool = False,
    delete_frames: bool = False,
    delete_interim_json: bool = False,
    delete_processed: bool = False,
    keep_catalog: bool = True,
):
    response = requests.post(
        f"{API_BASE}/videos/{video_name}/cleanup",
        json={
            "delete_raw": delete_raw,
            "delete_audio": delete_audio,
            "delete_frames": delete_frames,
            "delete_interim_json": delete_interim_json,
            "delete_processed": delete_processed,
            "keep_catalog": keep_catalog,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


API_BASE = load_api_base()

SAMPLE_QUERIES = [
    "human connection",
    "feeling connected and loved",
    "what makes people happier",
    "relationships",
    "happiness",
    "hạnh phúc đến từ đâu",
    "kết nối con người",
    "mối quan hệ với người khác",
    "crack egg",
    "separate egg",
    "egg yolk",
    "egg white",
    "tách trứng",
    "lòng đỏ trứng",
    "lòng trắng trứng",
]

CONTENT_TYPE_OPTIONS = [
    "Tất cả",
    "Toàn bộ transcript",
    "Đoạn transcript",
    "Caption ảnh",
    "Tài liệu đa phương thức",
]

CONTENT_TYPE_MAP = {
    "Tất cả": None,
    "Toàn bộ transcript": "transcription",
    "Đoạn transcript": "segment_chunk",
    "Caption ảnh": "caption",
    "Tài liệu đa phương thức": "multimodal",
}

if "last_processed_result" not in st.session_state:
    st.session_state["last_processed_result"] = None

if "last_processed_video_name" not in st.session_state:
    st.session_state["last_processed_video_name"] = None

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

if "search_preset" not in st.session_state:
    st.session_state["search_preset"] = "Manual"

st.set_page_config(page_title="Media Semantic Search", layout="wide")
st.title("Media Semantic Search")

with st.sidebar:
    st.markdown("### Backend")
    st.code(API_BASE)
    st.caption(
        "Score hiển thị là similarity proxy = 1 - distance, không phải xác suất. "
        "Caption là mô tả tự động tham khảo, không phải ground truth."
    )

    if st.button("Làm mới danh sách video"):
        st.rerun()

    st.divider()
    st.markdown("### Query mẫu")
    for sample in SAMPLE_QUERIES:
        if st.button(sample, key=f"sample_{sample}"):
            set_query_sample(sample)
            st.rerun()

videos = fetch_videos()
video_options = ["Tất cả video"] + videos

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Search", "Upload & Process", "Process by Path", "Process by YouTube URL", "Video Inventory"]
)

with tab1:
    st.subheader("Semantic Search")
    st.info(
        "Gợi ý sử dụng: truy vấn chủ đề/ý nghĩa nên ưu tiên 'Đoạn transcript' hoặc 'Tài liệu đa phương thức'. "
        "Truy vấn thiên về khung cảnh/vật thể/hành động nên ưu tiên 'Caption ảnh' hoặc 'Tài liệu đa phương thức'."
    )

    preset_col, hint_col = st.columns([1, 2])
    with preset_col:
        preset_name = st.selectbox(
            "Search preset",
            options=["Manual", "Talk mode", "Action mode", "Visual mode", "Audio mode"],
            index=["Manual", "Talk mode", "Action mode", "Visual mode", "Audio mode"].index(
                st.session_state.get("search_preset", "Manual")
            ),
        )
        st.session_state["search_preset"] = preset_name

    with hint_col:
        if preset_name != "Manual":
            st.caption(f"Preset đang dùng: **{preset_name}**")

    query = st.text_input("Nhập câu truy vấn", key="query_input")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    use_reranker = st.checkbox("Dùng Cross-Encoder Reranking (chậm hơn, chính xác hơn)", value=True)
    _, preset_label = derive_search_preset_defaults(preset_name)

    col_a, col_b = st.columns(2)
    with col_a:
        default_index = CONTENT_TYPE_OPTIONS.index(preset_label)
        content_type_label = st.selectbox(
            "Lọc theo loại nội dung",
            options=CONTENT_TYPE_OPTIONS,
            index=default_index,
        )
        content_type = CONTENT_TYPE_MAP[content_type_label]

    with col_b:
        selected_video = st.selectbox(
            "Lọc theo video",
            options=video_options,
            index=0,
        )
        custom_video_name = st.text_input(
            "Hoặc nhập tên video thủ công (tùy chọn)",
            value="",
        )

    chosen_video_for_preview = None
    if custom_video_name.strip():
        chosen_video_for_preview = custom_video_name.strip()
    elif selected_video != "Tất cả video":
        chosen_video_for_preview = selected_video

    last_video_name = st.session_state.get("last_processed_video_name")
    if last_video_name:
        show_video_preview(last_video_name, title="Video vừa xử lý gần nhất")

    if chosen_video_for_preview and chosen_video_for_preview != last_video_name:
        show_video_preview(chosen_video_for_preview, title="Video đang được chọn để tìm kiếm")

    if st.button("Search"):
        if not query.strip():
            st.warning("Vui lòng nhập truy vấn.")
        else:
            try:
                chosen_video = None
                if custom_video_name.strip():
                    chosen_video = custom_video_name.strip()
                elif selected_video != "Tất cả video":
                    chosen_video = selected_video

                payload = {
                    "query": query.strip(),
                    "top_k": top_k,
                    "content_type": content_type,
                    "video_name": chosen_video,
                }

                response = requests.post(
                    f"{API_BASE}/search",
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if not results:
                    st.info("Không tìm thấy kết quả.")
                else:
                    st.success(f"Tìm thấy {len(results)} kết quả.")
                    for i, result in enumerate(results, start=1):
                        meta = result.get("metadata", {}) or {}
                        st.markdown(f"### Đoạn video phù hợp {i}")

                        display_text = result.get("display_text") or result.get("document", "") or ""
                        auto_caption = result.get("display_caption") or result.get("document", "") or ""
                        nearby_speech = result.get("nearby_speech_context") or ""

                        st.markdown("**Matched frame description**")
                        st.write(shorten_text(display_text, max_chars=280))

                        if auto_caption:
                            st.caption(f"Auto-caption: {auto_caption}")
                        if nearby_speech:
                            st.caption(f"Nearby speech context: {shorten_text(nearby_speech, max_chars=220)}")

                        hints = limitation_hints_for_result(result)
                        for hint in hints:
                            if "Low-confidence" in hint:
                                st.warning(hint)
                            else:
                                st.caption(hint)

                        timestamp_str = meta.get("timestamp_str") or meta.get("timestamp")
                        start_time_str = meta.get("start_time_str") or meta.get("start_time")
                        end_time_str = meta.get("end_time_str") or meta.get("end_time")

                        event_range = result.get("event_time_range") or {}
                        event_start = event_range.get("start")
                        event_end = event_range.get("end")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.write("Video:", meta.get("video_name", ""))
                            st.write("Loại:", meta.get("content_type", ""))
                            st.write("Similarity proxy:", result.get("similarity_score"))
                            st.write("Distance:", result.get("distance"))
                        with c2:
                            st.write("Score type:", result.get("score_type"))
                            st.write("Mốc frame tốt nhất:", timestamp_str)
                            st.write("Khoảng event gần đúng:", f"{event_start} -> {event_end}")
                            st.write("Khoảng thời gian tài liệu:", f"{start_time_str} -> {end_time_str}")
                        with c3:
                            st.write("Modality:", meta.get("source_modality", ""))
                            st.write("Model:", meta.get("model_name", ""))
                            st.write("Language:", meta.get("document_language", ""))
                            st.write("Nearby matched frames grouped:", result.get("group_size", 1))

                        st.markdown("**Nguồn video**")
                        show_source_info_block(meta)
                        st.markdown("---")
            except Exception as e:
                st.error(f"Search failed: {e}")

with tab2:
    st.subheader("Upload & Process")

    if st.session_state.get("last_processed_video_name"):
        show_video_preview(
            st.session_state.get("last_processed_video_name"),
            title="Video gần nhất trong phiên làm việc",
        )

    uploaded_file = st.file_uploader(
        "Chọn file video để upload",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )

    reset_index_upload = st.checkbox(
        "Xóa dữ liệu cũ của video này trước khi index lại",
        value=True,
        key="upload_reset",
    )

    if st.button("Upload & Process", disabled=uploaded_file is None):
        if uploaded_file is None:
            st.warning("Vui lòng chọn video.")
        else:
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                response = requests.post(
                    f"{API_BASE}/upload-video",
                    params={"reset_index": str(reset_index_upload).lower()},
                    files=files,
                    timeout=3600,
                )
                response.raise_for_status()
                result = response.json()

                st.session_state["last_processed_result"] = result.get("result")
                if result.get("result"):
                    st.session_state["last_processed_video_name"] = result["result"].get("video_name")

                st.success(result.get("message", "Upload và xử lý thành công"))
                st.json(result)

                source_info = (result.get("result") or {}).get("video_source_info") or {}
                if source_info:
                    st.markdown("### Nguồn video")
                    show_source_info_block(source_info)

                video_name = (result.get("result") or {}).get("video_name")
                if video_name:
                    show_video_preview(video_name, title="Video vừa upload và xử lý")
            except Exception as e:
                st.error(f"Lỗi khi upload/process video: {e}")

with tab3:
    st.subheader("Process video by backend path")

    if st.session_state.get("last_processed_video_name"):
        show_video_preview(
            st.session_state.get("last_processed_video_name"),
            title="Video gần nhất trong phiên làm việc",
        )

    video_path = st.text_input("Đường dẫn video trên máy chạy backend")
    reset_index_path = st.checkbox(
        "Xóa dữ liệu cũ của video này trước khi index lại",
        value=True,
        key="path_reset",
    )

    if st.button("Process Video by Path"):
        if not video_path.strip():
            st.warning("Vui lòng nhập đường dẫn video.")
        else:
            try:
                response = requests.post(
                    f"{API_BASE}/process-video",
                    json={"video_path": video_path.strip(), "reset_index": reset_index_path},
                    timeout=3600,
                )
                response.raise_for_status()
                result = response.json()

                st.session_state["last_processed_result"] = result
                st.session_state["last_processed_video_name"] = result.get("video_name")

                st.success("Xử lý video thành công")
                st.json(result)

                source_info = result.get("video_source_info") or {}
                if source_info:
                    st.markdown("### Nguồn video")
                    show_source_info_block(source_info)

                if result.get("video_name"):
                    show_video_preview(result.get("video_name"), title="Video vừa xử lý")
            except Exception as e:
                st.error(f"Lỗi khi xử lý video: {e}")

with tab4:
    st.subheader("Process by YouTube URL")

    if st.session_state.get("last_processed_video_name"):
        show_video_preview(
            st.session_state.get("last_processed_video_name"),
            title="Video gần nhất trong phiên làm việc",
        )

    youtube_url = st.text_input("Nhập YouTube URL")
    reset_index_youtube = st.checkbox(
        "Xóa dữ liệu cũ của video này trước khi index lại",
        value=True,
        key="youtube_reset",
    )

    if st.button("Download & Process"):
        if not youtube_url.strip():
            st.warning("Vui lòng nhập YouTube URL.")
        else:
            try:
                response = requests.post(
                    f"{API_BASE}/ingest-youtube",
                    json={"video_url": youtube_url.strip(), "reset_index": reset_index_youtube},
                    timeout=7200,
                )
                response.raise_for_status()
                payload = response.json()

                process_result = payload.get("result") or {}
                ingest_result = payload.get("ingest_result") or {}

                st.session_state["last_processed_result"] = process_result
                st.session_state["last_processed_video_name"] = process_result.get("video_name")

                st.success(payload.get("message", "YouTube ingest thành công"))
                st.json(payload)

                if ingest_result:
                    st.markdown("### Nguồn video")
                    show_source_info_block(ingest_result)

                if process_result.get("video_name"):
                    show_video_preview(process_result.get("video_name"), title="Video vừa ingest và xử lý")
            except Exception as e:
                st.error(f"YouTube ingest failed: {e}")

with tab5:
    st.subheader("Video Inventory")

    if st.session_state.get("last_processed_video_name"):
        show_video_preview(
            st.session_state.get("last_processed_video_name"),
            title="Video gần nhất trong phiên làm việc",
        )

    inventory = fetch_all_inventory()
    if inventory.get("error"):
        st.warning("Backend chưa sẵn sàng. Vui lòng đợi vài giây rồi tải lại trang.")
        st.caption(str(inventory["error"]))
    else:
        st.write("Tổng số video đã index:", inventory.get("total_videos", 0))

        videos_inventory = inventory.get("videos", [])
        if not videos_inventory:
            st.info("Hiện chưa có video nào trong kho dữ liệu vector.")
        else:
            selected_inventory_video = st.selectbox(
                "Chọn video để xem chi tiết",
                options=[item.get("video_name", "") for item in videos_inventory],
                index=0,
                key="inventory_select",
            )

            selected_item = next(
                (item for item in videos_inventory if item.get("video_name") == selected_inventory_video),
                None,
            )

            if selected_item:
                source_info = selected_item.get("source_info", {}) or {}
                hints = limitation_hints_for_video(source_info)
                time_range = selected_item.get("time_range", {}) or {}
                start_val = time_range.get("min_start_time")
                end_val = time_range.get("max_end_time")

                st.markdown("### Video summary card")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**Title**")
                    st.write(source_info.get("video_title") or selected_item.get("video_name"))
                    st.markdown("**Platform**")
                    st.write(source_info.get("source_platform") or "unknown")
                    st.markdown("**Indexed time range**")
                    st.write(f"{format_seconds_to_hhmmss(start_val)} -> {format_seconds_to_hhmmss(end_val)}")

                with c2:
                    st.metric("Total indexed records", int(selected_item.get("total_records", 0) or 0))
                    st.metric("Dominant modality", dominant_modality(selected_item.get("source_modality_counts", {})))

                with c3:
                    st.markdown("**Recommended search mode**")
                    st.write(get_recommended_search_mode(source_info))
                    st.markdown("**Searchable types**")
                    st.write(searchable_types_label(selected_item.get("content_type_counts", {})))

                st.write("Suggested queries:")
                for q in get_suggested_queries(source_info):
                    st.markdown(f"- {q}")

                if hints:
                    for hint in hints:
                        st.info(hint)

                if source_info:
                    st.markdown("### Nguồn video")
                    show_source_info_block(source_info)

                show_video_preview(selected_item.get("video_name"), title="Video preview from inventory")

                st.markdown("### Video actions")
                a1, a2, a3, a4 = st.columns(4)

                with a1:
                    if st.button("Re-index selected video", key=f"reindex_{selected_item['video_name']}"):
                        try:
                            data = call_reindex(selected_item["video_name"])
                            st.success(data.get("message", "Re-index thành công."))
                            st.json(data)
                        except Exception as e:
                            st.error(f"Re-index failed: {e}")

                with a2:
                    if st.button("Delete index only", key=f"delete_index_{selected_item['video_name']}"):
                        try:
                            response = requests.delete(
                                f"{API_BASE}/videos/{selected_item['video_name']}",
                                timeout=120,
                            )
                            response.raise_for_status()
                            data = response.json()

                            if st.session_state.get("last_processed_video_name") == selected_item["video_name"]:
                                st.session_state["last_processed_video_name"] = None
                                st.session_state["last_processed_result"] = None

                            st.success(data.get("message", "Delete index thành công."))
                            st.json(data)
                        except Exception as e:
                            st.error(f"Delete index failed: {e}")

                with a3:
                    if st.button("Delete local file only", key=f"delete_local_file_{selected_item['video_name']}"):
                        try:
                            data = call_cleanup(
                                selected_item["video_name"],
                                delete_raw=True,
                                keep_catalog=True,
                            )
                            st.success(data.get("message", "Delete local file thành công."))
                            st.json(data)
                        except Exception as e:
                            st.error(f"Delete local file failed: {e}")

                with a4:
                    if st.button("Delete index + local artifacts", key=f"delete_all_{selected_item['video_name']}"):
                        try:
                            delete_resp = requests.delete(
                                f"{API_BASE}/videos/{selected_item['video_name']}",
                                timeout=120,
                            )
                            delete_resp.raise_for_status()

                            cleanup_data = call_cleanup(
                                selected_item["video_name"],
                                delete_raw=True,
                                delete_audio=True,
                                delete_frames=True,
                                delete_interim_json=True,
                                delete_processed=True,
                                keep_catalog=True,
                            )

                            if st.session_state.get("last_processed_video_name") == selected_item["video_name"]:
                                st.session_state["last_processed_video_name"] = None
                                st.session_state["last_processed_result"] = None

                            st.success("Deleted index + local artifacts.")
                            st.json(
                                {
                                    "delete_index_result": delete_resp.json(),
                                    "cleanup_result": cleanup_data,
                                }
                            )
                        except Exception as e:
                            st.error(f"Delete index + local artifacts failed: {e}")

                st.markdown("### Advanced cleanup")
                adv1, adv2, adv3, adv4, adv5 = st.columns(5)
                delete_raw = adv1.checkbox("Delete raw", key=f"adv_raw_{selected_item['video_name']}")
                delete_audio = adv2.checkbox("Delete audio", key=f"adv_audio_{selected_item['video_name']}")
                delete_frames = adv3.checkbox("Delete frames", key=f"adv_frames_{selected_item['video_name']}")
                delete_interim_json = adv4.checkbox("Delete interim json", key=f"adv_interim_{selected_item['video_name']}")
                delete_processed = adv5.checkbox("Delete processed", key=f"adv_processed_{selected_item['video_name']}")
                keep_catalog = st.checkbox(
                    "Keep catalog entry",
                    value=True,
                    key=f"adv_keep_catalog_{selected_item['video_name']}",
                )

                if st.button("Run advanced cleanup", key=f"run_adv_cleanup_{selected_item['video_name']}"):
                    try:
                        data = call_cleanup(
                            selected_item["video_name"],
                            delete_raw=delete_raw,
                            delete_audio=delete_audio,
                            delete_frames=delete_frames,
                            delete_interim_json=delete_interim_json,
                            delete_processed=delete_processed,
                            keep_catalog=keep_catalog,
                        )
                        st.success(data.get("message", "Advanced cleanup hoàn tất."))
                        st.json(data)
                    except Exception as e:
                        st.error(f"Advanced cleanup failed: {e}")

                st.markdown("### Inventory JSON")
                st.json(selected_item)