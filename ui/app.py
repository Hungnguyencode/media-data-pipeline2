from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

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

if "last_processed_result" not in st.session_state:
    st.session_state["last_processed_result"] = None

if "last_processed_video_name" not in st.session_state:
    st.session_state["last_processed_video_name"] = None

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

st.set_page_config(page_title="Media Semantic Search", layout="wide")
st.title("Media Semantic Search")

with st.sidebar:
    st.markdown("### Backend")
    st.code(API_BASE)
    st.caption(
        "Lưu ý: Score hiển thị là similarity proxy = 1 - distance, không phải xác suất. "
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["Search", "Upload & Process", "Process by Path", "Video Inventory"]
)

with tab1:
    st.subheader("Semantic Search")

    st.info(
        "Gợi ý sử dụng: truy vấn chủ đề/ý nghĩa nên ưu tiên 'Đoạn transcript' hoặc 'Tài liệu đa phương thức'. "
        "Truy vấn thiên về khung cảnh/vật thể/hành động nên ưu tiên 'Caption ảnh' hoặc 'Tài liệu đa phương thức'."
    )

    if st.session_state.get("last_processed_video_name"):
        show_video_preview(
            st.session_state.get("last_processed_video_name"),
            title="Video vừa xử lý gần nhất",
        )

    query = st.text_input("Nhập câu truy vấn", key="query_input")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)

    col_a, col_b = st.columns(2)
    with col_a:
        content_type_label = st.selectbox(
            "Lọc theo loại nội dung",
            options=[
                "Tất cả",
                "Toàn bộ transcript",
                "Đoạn transcript",
                "Caption ảnh",
                "Tài liệu đa phương thức",
            ],
            index=0,
        )

        content_type_map = {
            "Tất cả": None,
            "Toàn bộ transcript": "transcription",
            "Đoạn transcript": "segment_chunk",
            "Caption ảnh": "caption",
            "Tài liệu đa phương thức": "multimodal",
        }
        content_type = content_type_map[content_type_label]

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

    if chosen_video_for_preview:
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
                        raw_doc_text = result.get("document", "") or ""
                        nearby_speech = result.get("nearby_speech_context") or ""

                        st.markdown("**Matched frame description**")
                        st.write(shorten_text(display_text, max_chars=280))

                        if auto_caption:
                            st.caption(f"Auto-caption: {auto_caption}")

                        if nearby_speech:
                            st.caption(
                                f"Nearby speech context: {shorten_text(nearby_speech, max_chars=220)}"
                            )

                        timestamp_str = meta.get("timestamp_str") or meta.get("timestamp")
                        start_time_str = meta.get("start_time_str") or meta.get("start_time")
                        end_time_str = meta.get("end_time_str") or meta.get("end_time")

                        event_range = result.get("event_time_range") or {}
                        event_start = event_range.get("start")
                        event_end = event_range.get("end")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("Video:", meta.get("video_name"))
                        with col2:
                            st.write("Loại:", meta.get("content_type"))
                        with col3:
                            score = result.get("similarity_score")
                            if score is not None:
                                st.write("Similarity proxy:", f"{score:.4f}")
                            else:
                                st.write("Similarity proxy:", "N/A")

                        if result.get("distance") is not None:
                            st.caption(f"Distance: {result['distance']:.6f}")

                        if result.get("score_type"):
                            st.caption(f"Score type: {result['score_type']}")

                        if timestamp_str is not None:
                            st.markdown(f"**Mốc frame tốt nhất:** `{timestamp_str}`")

                        if event_start is not None and event_end is not None:
                            st.markdown(
                                f"**Khoảng event gần đúng:** `{event_start:.2f}s -> {event_end:.2f}s`"
                            )

                        if start_time_str is not None and end_time_str is not None:
                            st.markdown(f"**Khoảng thời gian tài liệu:** `{start_time_str} -> {end_time_str}`")
                            st.caption(
                                "Đối chiếu video: tua video đến đúng khoảng thời gian này để xem nội dung tương ứng."
                            )

                        if meta.get("frame_name"):
                            st.write("Frame:", meta.get("frame_name"))

                        if meta.get("source_modality"):
                            st.write("Modality:", meta.get("source_modality"))

                        if meta.get("model_name"):
                            st.write("Model:", meta.get("model_name"))

                        if meta.get("document_language"):
                            st.write("Language:", meta.get("document_language"))

                        if result.get("group_size") is not None:
                            st.write("Nearby matched frames grouped:", result.get("group_size"))

                        with st.expander("Xem toàn bộ nội dung gốc"):
                            st.write(raw_doc_text)

                        st.divider()

            except Exception as e:
                st.error(f"Lỗi khi search: {e}")

with tab2:
    st.subheader("Upload video and process")

    if st.session_state.get("last_processed_video_name"):
        show_video_preview(
            st.session_state.get("last_processed_video_name"),
            title="Video gần nhất trong phiên làm việc",
        )

    uploaded_file = st.file_uploader(
        "Chọn file video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )
    reset_index_upload = st.checkbox(
        "Xóa dữ liệu cũ của video này trước khi index lại",
        value=True,
        key="upload_reset",
    )

    if st.button("Upload & Process"):
        if uploaded_file is None:
            st.warning("Vui lòng chọn file video.")
        else:
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "video/mp4")
                }
                response = requests.post(
                    f"{API_BASE}/upload-video",
                    files=files,
                    params={"reset_index": str(reset_index_upload).lower()},
                    timeout=3600,
                )
                response.raise_for_status()
                result = response.json()

                uploaded_path = result.get("uploaded_path")
                process_result = result.get("result", {})
                video_name = process_result.get("video_name") or (Path(uploaded_path).name if uploaded_path else None)

                st.session_state["last_processed_result"] = result
                st.session_state["last_processed_video_name"] = video_name

                st.success("Upload và xử lý video thành công")
                st.json(result)

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
    reset_index = st.checkbox(
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
                    json={"video_path": video_path.strip(), "reset_index": reset_index},
                    timeout=3600,
                )
                response.raise_for_status()
                result = response.json()

                st.session_state["last_processed_result"] = result
                st.session_state["last_processed_video_name"] = result.get("video_name")

                st.success("Xử lý video thành công")
                st.json(result)

                if result.get("video_name"):
                    show_video_preview(result.get("video_name"), title="Video vừa xử lý")
            except Exception as e:
                st.error(f"Lỗi khi xử lý video: {e}")

with tab4:
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
            for item in videos_inventory:
                with st.expander(f"{item.get('video_name')} | {item.get('total_records', 0)} records"):
                    st.json(item)

    st.divider()
    st.markdown("### Xóa dữ liệu của một video khỏi index")

    delete_video_name = st.selectbox(
        "Chọn video cần xóa",
        options=videos if videos else ["Không có video"],
        index=0,
        key="delete_video_select",
    )

    if st.button("Xóa video khỏi index"):
        if not videos:
            st.warning("Không có video nào để xóa.")
        else:
            try:
                response = requests.delete(
                    f"{API_BASE}/videos/{delete_video_name}",
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

                if st.session_state.get("last_processed_video_name") == delete_video_name:
                    st.session_state["last_processed_video_name"] = None
                    st.session_state["last_processed_result"] = None

                st.success(result.get("message", "Đã xóa dữ liệu video"))
                st.json(result)
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi xóa dữ liệu video: {e}")

    st.divider()
    st.markdown("### Xem nhanh chi tiết 1 video")

    inspect_video_name = st.selectbox(
        "Chọn video để xem chi tiết",
        options=videos if videos else ["Không có video"],
        index=0,
        key="inspect_video_select",
    )

    if st.button("Xem chi tiết video"):
        if not videos:
            st.warning("Không có video nào để xem.")
        else:
            detail = fetch_video_inventory(inspect_video_name)
            if detail.get("error"):
                st.error(f"Lỗi: {detail['error']}")
            else:
                st.json(detail)
                show_video_preview(inspect_video_name, title="Video đang xem chi tiết")