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


API_BASE = load_api_base()

st.set_page_config(page_title="Media Semantic Search", layout="wide")
st.title("Media Semantic Search")

with st.sidebar:
    st.markdown("### Backend")
    st.code(API_BASE)

tab1, tab2, tab3 = st.tabs(["Search", "Upload & Process", "Process by Path"])

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
        video_name = st.text_input("Lọc theo tên video (tùy chọn)")

    if st.button("Search"):
        if not query.strip():
            st.warning("Vui lòng nhập truy vấn.")
        else:
            try:
                payload = {
                    "query": query.strip(),
                    "top_k": top_k,
                    "content_type": content_type,
                    "video_name": video_name.strip() or None,
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
                            if result.get("relevance") is not None:
                                st.write("Relevance:", f"{result['relevance']:.4f}")
                            else:
                                st.write("Relevance:", "N/A")

                        if result.get("score_type"):
                            st.caption(f"Score type: {result['score_type']}")

                        if meta.get("timestamp") is not None:
                            st.write("Timestamp:", meta.get("timestamp_str") or meta.get("timestamp"))

                        if meta.get("frame_name"):
                            st.write("Frame:", meta.get("frame_name"))

                        if meta.get("source_modality"):
                            st.write("Modality:", meta.get("source_modality"))

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
        key="upload_reset"
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