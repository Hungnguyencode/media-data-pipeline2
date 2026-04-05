# Media Semantic Search cho Kho Dữ Liệu Video

## 1. Giới thiệu

Đây là đồ án theo định hướng **Kỹ thuật dữ liệu (Data Engineering)** với mục tiêu xây dựng một **pipeline đa phương thức cho dữ liệu video**, phục vụ các tác vụ:

- ingest video từ file local hoặc **YouTube URL**,
- trích xuất audio và frame,
- chuyển giọng nói thành văn bản,
- mô tả nội dung hình ảnh theo frame,
- lập chỉ mục vector cho dữ liệu video,
- tìm kiếm ngữ nghĩa trên kho video,
- quản lý inventory và metadata của video đã index.

Phiên bản hiện tại là **v2.2.0**, tập trung vào tính **ổn định khi demo**, **metadata rõ ràng**, và **retrieval đa phương thức** đủ mạnh cho bài báo cáo/đồ án.

Hệ thống kết hợp các thành phần chính:

- **Whisper** cho speech-to-text,
- **BLIP** cho image captioning,
- **OpenCLIP** cho text-image matching,
- **SentenceTransformers** cho text embedding,
- **ChromaDB** cho lưu trữ vector,
- **FastAPI** cho backend API,
- **Streamlit** cho giao diện demo.

---

## 2. Mục tiêu của hệ thống

Hệ thống hướng tới một pipeline video semantic search hoàn chỉnh theo các bước:

1. ingest video,
2. extract audio và frame,
3. transform thành transcript / caption / multimodal documents,
4. index vector vào ChromaDB,
5. phục vụ tìm kiếm qua API và UI,
6. quản lý inventory video trong kho dữ liệu.

Bản hiện tại ưu tiên:

- **YouTube-only ingest** cho nguồn online,
- không mở rộng sang multi-platform,
- không triển khai job queue / orchestration phức tạp,
- đủ rõ để demo và phân tích trong luận văn.

---

## 3. Những nâng cấp chính của phiên bản 2.2.0

So với bản pipeline cơ bản, phiên bản hiện tại đã được nâng cấp theo các hướng sau:

### 3.1. Ingest và quản lý nguồn video
- hỗ trợ ingest từ **YouTube URL**,
- canonicalize URL YouTube về dạng chuẩn,
- tự lấy metadata nguồn bằng `yt-dlp`,
- lưu/cập nhật catalog trong `data/video_catalog.json`,
- hỗ trợ re-index dựa trên catalog,
- hỗ trợ cleanup artifact theo từng video.

### 3.2. Pipeline ổn định hơn
- phát hiện video **không có audio**,
- nếu không có audio, hệ thống vẫn chạy theo nhánh **visual-only**,
- `WhisperProcessor` có **fallback từ CUDA sang CPU**,
- sampling frame theo thời gian với cấu hình cân bằng cho demo.

### 3.3. Metadata giàu hơn
Metadata không chỉ dừng ở `video_name`, mà còn có thêm:

- `source_platform`
- `source_url`
- `video_title`
- `video_description`
- `thumbnail_url`
- `video_tags`
- `local_video_path`
- `ingest_method`
- `has_audio`
- `video_type`
- `estimated_content_style`
- `recommended_search_mode`
- `duration_sec`

### 3.4. Retrieval tốt hơn
- semantic text retrieval,
- CLIP text-image retrieval,
- hybrid fusion,
- soft rerank theo query/action/style,
- event grouping,
- nearby speech context,
- hỗ trợ query `multimodal` nhưng vẫn tận dụng được visual signal từ caption/CLIP.

### 3.5. Đánh giá và test tốt hơn
- có `benchmark_cases.json` cho evaluation,
- có `run_eval.py` để chấm tự động,
- có test cho API, extract, retrieval, transform, vector indexing và main pipeline.

---

## 4. Chức năng chính

### 4.1. Xử lý video
- nhận video từ đường dẫn local,
- upload video qua UI/API,
- ingest video từ **YouTube URL**,
- tách audio từ video bằng FFmpeg,
- trích xuất frame theo thời gian.

### 4.2. Trích xuất thông tin
- chuyển giọng nói thành văn bản bằng Whisper,
- sinh caption mô tả nội dung frame bằng BLIP,
- sinh image embedding bằng OpenCLIP,
- tạo multimodal documents bằng cách kết hợp speech và visual context gần nhau theo thời gian.

### 4.3. Lập chỉ mục vector
Hệ thống index 4 loại document:

- `transcription`
- `segment_chunk`
- `caption`
- `multimodal`

Text documents được lưu trong **text collection**, còn CLIP embedding của frame được lưu trong **clip collection**.

### 4.4. Tìm kiếm ngữ nghĩa
Hệ thống hỗ trợ:

- semantic search toàn kho video,
- lọc theo `content_type`,
- lọc theo `video_name`,
- hybrid retrieval giữa text embedding và CLIP,
- soft rerank,
- event grouping để gom các kết quả gần nhau thành một cụm dễ đọc hơn.

### 4.5. Quản lý kho video
- liệt kê video đã index,
- xem inventory từng video,
- xóa dữ liệu index của một video,
- cleanup artifact,
- re-index video dựa trên catalog metadata.

---

## 5. Kiến trúc hệ thống

### 5.1. Extract layer
- `AudioExtractor`
- `FrameExtractor`

### 5.2. Transform layer
- `WhisperProcessor`
- `VisionProcessor`

### 5.3. Index layer
- `VectorIndexer`

### 5.4. Retrieval layer
- `SearchEngine`

### 5.5. Ingest layer
- `YouTubeIngestor`
- `LocalFileIngestor`

### 5.6. Serving layer
- `FastAPI`
- `Streamlit`

### 5.7. Catalog / metadata layer
- `data/video_catalog.json`
- các utility trong `src/utils.py` để load / save / upsert catalog entry

---

## 6. Công nghệ sử dụng

- Python
- FastAPI
- Streamlit
- ChromaDB
- SentenceTransformers
- Whisper
- BLIP
- OpenCLIP
- FFmpeg / FFprobe
- yt-dlp
- PyTest

---

## 7. Cấu trúc thư mục

```text
project/
├── api/
│   └── main.py
├── configs/
│   ├── config.yaml
│   └── logging.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   │   ├── audio/
│   │   ├── frames/
│   │   ├── transcripts/
│   │   └── captions/
│   ├── processed/
│   ├── vector_db/
│   └── video_catalog.json
├── evaluation/
│   ├── benchmark_cases.json
│   └── run_eval.py
├── logs/
├── src/
│   ├── extract/
│   ├── indexing/
│   ├── ingest/
│   ├── retrieval/
│   ├── transform/
│   └── utils.py
├── tests/
├── ui/
│   └── app.py
├── main_pipeline.py
├── README.md
└── run_demo.md
```

---

## 8. Cấu hình hiện tại

File cấu hình chính: `configs/config.yaml`

### 8.1. Sampling frame

```yaml
video:
  frame_sampling_fps: 1.0
  max_frames: 240
  max_frame_width: 960
  max_frame_height: 540
```

Ý nghĩa thực tế:

- `1.0 fps`: lấy khoảng 1 frame mỗi giây,
- `240 frames`: giới hạn tối đa 240 frame mỗi video,
- đây là cấu hình cân bằng giữa:
  - temporal coverage,
  - độ nhẹ khi demo,
  - và chất lượng truy xuất cho video dài vừa phải.

### 8.2. Mô hình
- Whisper: `base`
- BLIP: `Salesforce/blip-image-captioning-base`
- CLIP: `ViT-B-32` pretrained `openai`
- Text embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### 8.3. Retrieval
Một số tham số chính:

- `hybrid_search_alpha: 0.35`
- `clip_search_beta: 0.65`
- `hybrid_candidate_multiplier: 3`
- `caption_merge_window_sec: 3.0`
- `caption_dedup_min_gap_sec: 2.0`

---

## 9. Luồng xử lý dữ liệu

### Bước 1: Nhận video
Video đi vào hệ thống qua:
- upload từ UI/API,
- process theo đường dẫn file local,
- hoặc ingest từ **YouTube URL**.

### Bước 2: Chuẩn hóa metadata nguồn
Hệ thống chuẩn hóa `source_metadata` cho pipeline, gồm:
- thông tin nền tảng nguồn,
- đường dẫn local,
- title / description / tags,
- kiểu video,
- style nội dung,
- mode search gợi ý.

### Bước 3: Extract
- kiểm tra audio stream,
- tách audio nếu có,
- trích xuất frame theo thời gian.

### Bước 4: Transform
- Whisper sinh transcript,
- BLIP sinh caption,
- CLIP sinh image embedding,
- hệ thống tạo thêm multimodal documents từ speech + visual context.

### Bước 5: Index
- text documents được index vào text collection,
- image embedding của frame được index vào clip collection.

### Bước 6: Search / Inventory
- người dùng tìm kiếm semantic search,
- xem inventory video,
- xem metadata nguồn,
- cleanup / re-index khi cần.

---

## 10. Các loại dữ liệu được index

### 10.1. `transcription`
Toàn bộ transcript của video.

### 10.2. `segment_chunk`
Các đoạn transcript được ghép theo cửa sổ segment.

### 10.3. `caption`
Caption theo từng frame.

### 10.4. `multimodal`
Document kết hợp:
- phần speech,
- phần visual gần theo thời gian.

---

## 11. Metadata chính

Một record trong vector database có thể có các metadata như:

- `video_name`
- `content_type`
- `source_modality`
- `model_name`
- `pipeline_version`
- `timestamp`
- `timestamp_str`
- `start_time`
- `start_time_str`
- `end_time`
- `end_time_str`
- `frame_name`
- `image_path`
- `document_language`

Ngoài ra còn có metadata nguồn video:

- `source_platform`
- `source_url`
- `video_title`
- `video_description`
- `thumbnail_url`
- `video_tags`
- `local_video_path`
- `created_at`
- `ingested_at`
- `ingest_method`
- `has_audio`
- `video_type`
- `estimated_content_style`
- `recommended_search_mode`
- `duration_sec`

---

## 12. API chính

### `GET /`
Kiểm tra API đang chạy và liệt kê endpoint chính.

### `GET /health`
Health check.

### `GET /stats`
Xem thống kê vector database.

### `GET /videos`
Lấy danh sách video đã index.

### `GET /videos/inventory`
Lấy inventory toàn bộ video.

### `GET /videos/{video_name}`
Lấy inventory chi tiết của một video.

### `DELETE /videos/{video_name}`
Xóa dữ liệu index của một video.

### `POST /videos/{video_name}/cleanup`
Xóa artifact của video theo tùy chọn.

### `POST /videos/{video_name}/reindex`
Re-index video dựa trên catalog entry.

### `POST /search`
Semantic search.

Ví dụ:

```json
{
  "query": "crack egg",
  "top_k": 5,
  "content_type": "caption",
  "video_name": "egg.mp4"
}
```

### `POST /process-video`
Xử lý video theo đường dẫn file trên máy backend.

### `POST /upload-video`
Upload video rồi xử lý toàn pipeline.

### `POST /ingest-youtube`
Ingest video từ YouTube URL rồi xử lý toàn pipeline.

Ví dụ:

```json
{
  "video_url": "https://www.youtube.com/watch?v=rLXcLBfDwvE",
  "reset_index": true
}
```

---

## 13. Giao diện người dùng

UI Streamlit gồm các tab chính:

### 13.1. Search
- nhập query,
- chọn top_k,
- lọc theo loại nội dung,
- lọc theo video,
- xem timestamp, event range, caption, nearby speech context,
- xem metadata nguồn video.

### 13.2. Upload & Process
- upload file video,
- xử lý video,
- chọn `reset_index`,
- xem kết quả pipeline.

### 13.3. Process by Path
- xử lý video local theo đường dẫn tuyệt đối hoặc tương đối.

### 13.4. Process by YouTube URL
- nhập YouTube URL,
- tải video về `data/raw`,
- cập nhật source metadata,
- chạy pipeline,
- xem ingest result và pipeline result.

### 13.5. Video Inventory
- xem danh sách video đã index,
- xem inventory chi tiết,
- xem metadata nguồn,
- xóa dữ liệu index theo video.

---

## 14. Cài đặt môi trường

### 14.1. Tạo virtual environment
```bash
python -m venv venv
```

### 14.2. Kích hoạt virtual environment
Windows:
```bash
venv\Scripts\activate
```

### 14.3. Cài dependencies
```bash
pip install -r requirements.txt
```

### 14.4. Cài PyTorch theo môi trường máy
Ví dụ với CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 14.5. Cài FFmpeg / FFprobe
Đảm bảo cả `ffmpeg` và `ffprobe` chạy được từ terminal.

---

## 15. Cách chạy hệ thống

### Chạy API
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Chạy UI
```bash
streamlit run ui/app.py
```

### Chạy test
```bash
pytest -v
```

### Chạy evaluation benchmark
```bash
python evaluation/run_eval.py
```

---

## 16. Evaluation benchmark

Hệ thống hiện có:

- `evaluation/benchmark_cases.json`
- `evaluation/run_eval.py`

Trong đó:

- `benchmark_cases.json` là **bộ test case khai báo thủ công**, không tự sinh tự động,
- `run_eval.py` đọc file này, gọi `pipeline.search(...)`, rồi chấm hit/matched rank.

Điều này phù hợp với scope đồ án vì:
- bộ query nhỏ,
- có chủ đích,
- dễ giải thích ground truth trong báo cáo.

---

## 17. Luồng demo gợi ý

1. chạy FastAPI,
2. chạy Streamlit,
3. kiểm tra `data/video_catalog.json`,
4. ingest một video bằng **Process by YouTube URL** hoặc process video local,
5. chờ pipeline hoàn tất,
6. vào tab **Search** để thử query,
7. kiểm tra **Video Inventory** để xem thống kê và metadata,
8. nếu cần, chạy `evaluation/run_eval.py` để trình bày phần đánh giá.

---

## 18. Một số lưu ý thực tế

### 18.1. Trade-off của frame sampling
Cấu hình `1.0 fps, 240 frames` là một mức cân bằng, nhưng không thể tối ưu đồng thời cho mọi loại video:

- video action rất ngắn có thể thích sampling dày hơn,
- video dài nhiều cảnh có thể thích coverage dài hơn.

### 18.2. Caption không phải ground truth
Caption từ BLIP là mô tả tự động tham khảo. Khi demo, nên xem:

- matched frame,
- timestamp,
- event range,
- nearby speech context,

là các tín hiệu quan trọng hơn chỉ riêng câu caption.

### 18.3. Video nhạc / cinematic
Với video chỉ có nhạc hoặc cinematic montage, transcript có thể kém đáng tin hơn. Trong các case này, nên ưu tiên:

- `caption`
- `multimodal`

hơn là chỉ dựa vào `transcription`.

### 18.4. YouTube ingest chưa phải production-grade
Flow YouTube hiện usable cho demo, nhưng vẫn phụ thuộc:

- yt-dlp,
- format availability,
- ffmpeg / ffprobe,
- thay đổi phía YouTube.

### 18.5. Re-index sau khi đổi config hoặc metadata
Nếu thay đổi config hoặc cập nhật metadata, nên process lại video với `reset_index=True`.

---

## 19. Điểm mạnh của hệ thống

- Có pipeline rõ từ ingest đến search.
- Kết hợp cả audio và image.
- Có hybrid retrieval giữa text và CLIP.
- Có event grouping và speech context.
- Có inventory và metadata nguồn video.
- Có YouTube ingest cho Bản 2.
- Có API và UI để demo.
- Có test cho các thành phần chính.
- Có benchmark evaluation cơ bản.

---

## 20. Giới hạn hiện tại

- Chưa hướng tới production-scale.
- Caption vẫn có thể khái quát hoặc chưa đủ fine-grained.
- Một số object nhỏ hoặc khó phân biệt vẫn có thể bị caption nhầm.
- Với video cinematic / music video, transcript có thể kém đáng tin hơn talk/TED.
- Catalog hiện vẫn ở mức JSON prototype, chưa phải metadata store quy mô lớn.
- Hiện tại mới hỗ trợ ingest **YouTube-only**.

---

## 21. Hướng phát triển

- batch ingest YouTube từ danh sách URL,
- hỗ trợ thêm nền tảng khác ngoài YouTube,
- mở rộng evaluation với nhiều query hơn,
- mở rộng quản lý catalog,
- bổ sung dashboard monitoring,
- hỗ trợ orchestration trong tương lai nếu cần.

---

## 22. Kết luận

Hệ thống hiện tại đã xây dựng được một **multimodal semantic search pipeline cho video** với các thành phần chính:

- speech-to-text,
- image captioning,
- CLIP image retrieval,
- vector indexing,
- semantic search,
- video inventory,
- metadata catalog,
- và ingest video từ YouTube URL.

Đây là một prototype đủ rõ về mặt kỹ thuật để phục vụ demo đồ án theo định hướng **Kỹ thuật dữ liệu**, đồng thời đủ trực quan để trình bày semantic search trên video qua API và UI.
