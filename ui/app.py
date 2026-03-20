from __future__ import annotations

import os
from pathlib import Path

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


def fetch_videos():
    try:
        response = requests.get(f"{API_BASE}/videos", timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("videos", [])
    except Exception:
        return []


def fetch_all_inventory():
    try:
        response = requests.get(f"{API_BASE}/videos/inventory", timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "total_videos": 0, "videos": []}


def fetch_video_inventory(video_name: str):
    try:
        response = requests.get(f"{API_BASE}/videos/{video_name}", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


API_BASE = load_api_base()

st.set_page_config(page_title="Media Semantic Search", layout="wide")
st.title("Media Semantic Search")

with st.sidebar:
    st.markdown("### Backend")
    st.code(API_BASE)
    st.caption("Lưu ý: Score hiển thị là similarity proxy = 1 - distance, không phải xác suất.")
    if st.button("Làm mới danh sách video"):
        st.rerun()

videos = fetch_videos()
video_options = ["Tất cả video"] + videos

tab1, tab2, tab3, tab4 = st.tabs(
    ["Search", "Upload & Process", "Process by Path", "Video Inventory"]
)

with tab1:
    st.subheader("Semantic Search")
    query = st.text_input("Nhập câu truy vấn")
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
                    for i, result in enumerate(results, start=1):
                        meta = result.get("metadata", {})
                        st.markdown(f"### Kết quả {i}")
                        st.write(result.get("document", ""))

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

                        if meta.get("timestamp") is not None:
                            st.write("Timestamp:", meta.get("timestamp_str") or meta.get("timestamp"))

                        if meta.get("start_time_str") and meta.get("end_time_str"):
                            st.write(
                                "Khoảng thời gian:",
                                f"{meta.get('start_time_str')} -> {meta.get('end_time_str')}",
                            )

                        if meta.get("frame_name"):
                            st.write("Frame:", meta.get("frame_name"))

                        if meta.get("source_modality"):
                            st.write("Modality:", meta.get("source_modality"))

                        if meta.get("model_name"):
                            st.write("Model:", meta.get("model_name"))

                        if meta.get("document_language"):
                            st.write("Language:", meta.get("document_language"))

                        st.divider()

            except Exception as e:
                st.error(f"Lỗi khi search: {e}")

with tab2:
    st.subheader("Upload video and process")
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
                st.success("Upload và xử lý video thành công")
                st.json(result)
            except Exception as e:
                st.error(f"Lỗi khi upload/process video: {e}")

with tab3:
    st.subheader("Process video by backend path")
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
                st.success("Xử lý video thành công")
                st.json(result)
            except Exception as e:
                st.error(f"Lỗi khi xử lý video: {e}")

with tab4:
    st.subheader("Video Inventory")

    inventory = fetch_all_inventory()
    if inventory.get("error"):
        st.error(f"Không lấy được inventory: {inventory['error']}")
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