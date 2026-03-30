# Media Semantic Search for Video Data Warehouse

## 1. Giới thiệu

Đây là đồ án theo định hướng **Kỹ thuật dữ liệu (Data Engineering)** với mục tiêu xây dựng một **data pipeline đa phương thức cho dữ liệu video**, phục vụ:

- trích xuất giọng nói từ video,
- mô tả nội dung hình ảnh theo frame,
- lập chỉ mục vector,
- tìm kiếm ngữ nghĩa trên kho dữ liệu video,
- quản lý inventory video đã được index,
- và hỗ trợ ingest video từ **YouTube URL**.

Phiên bản hiện tại của hệ thống hoạt động theo hướng **multimodal semantic search**, kết hợp:

- **Whisper** để speech-to-text,
- **BLIP** để sinh caption từ frame ảnh,
- **OpenCLIP** để biểu diễn hình ảnh và tăng chất lượng đối sánh text-image,
- **SentenceTransformers** để embedding cho text documents,
- **ChromaDB** để lưu trữ vector và truy xuất ngữ nghĩa,
- **FastAPI** để cung cấp API,
- **Streamlit** để làm giao diện demo.

Ngoài phần semantic search, hệ thống còn có lớp **video catalog metadata** để gắn kết video local hoặc video ingest từ YouTube với metadata nguồn như nền tảng, link gốc, tiêu đề, mô tả, thumbnail và tags.

---

## 2. Mục tiêu đề tài

Hệ thống hướng tới một pipeline hoàn chỉnh cho video semantic search, gồm các bước:

1. ingest video,
2. extract audio và frame,
3. transform thành transcript / caption / multimodal documents,
4. index vector vào ChromaDB,
5. phục vụ tìm kiếm qua API và UI,
6. quản lý inventory video trong kho dữ liệu.

---

## 3. Chức năng chính

### 3.1. Xử lý video
- nhận video từ đường dẫn local,
- upload video qua UI/API,
- ingest video từ **YouTube URL**,
- tách audio từ video bằng FFmpeg,
- trích xuất frame theo thời gian.

### 3.2. Trích xuất thông tin
- chuyển giọng nói thành văn bản bằng Whisper,
- sinh caption mô tả nội dung frame bằng BLIP,
- encode frame bằng OpenCLIP image embedding,
- tạo multimodal documents bằng cách kết hợp speech và visual context gần nhau theo thời gian.

### 3.3. Lập chỉ mục vector
Hệ thống index 4 loại document:

- `transcription`
- `segment_chunk`
- `caption`
- `multimodal`

Text documents được đưa vào **text collection**, còn CLIP image embedding của caption được đưa vào **clip collection**.

### 3.4. Tìm kiếm ngữ nghĩa
Hệ thống hỗ trợ:

- semantic search toàn kho video,
- lọc theo `content_type`,
- lọc theo `video_name`,
- hybrid retrieval giữa:
  - text embedding search,
  - CLIP text-image retrieval,
- soft rerank và event grouping để gom các frame gần nhau thành một kết quả dễ đọc hơn.

### 3.5. Quản lý kho video
- liệt kê video đã index,
- xem inventory từng video,
- xóa toàn bộ dữ liệu index của một video,
- hiển thị metadata nguồn video trong search result và inventory.

### 3.6. Auto-catalog metadata
Khi xử lý một video local mới hoặc ingest từ YouTube, hệ thống có thể **tự động tạo hoặc cập nhật entry trong `data/video_catalog.json`** với metadata như:

- `video_name`
- `local_video_path`
- `source_platform`
- `source_url`
- `title`
- `description`
- `thumbnail_url`
- `tags`
- `created_at`
- `ingested_at`

### 3.7. YouTube ingest (Bản 2)
Hệ thống hiện hỗ trợ:

- nhập **YouTube URL**,
- canonicalize về URL chuẩn dạng `watch?v=...`,
- lấy metadata bằng `yt-dlp`,
- tải video về `data/raw`,
- kiểm tra audio stream bằng `ffprobe`,
- cập nhật catalog tự động,
- đẩy video vào pipeline xử lý và index.

---

## 4. Kiến trúc hệ thống

### 4.1. Extract layer
- `AudioExtractor`
- `FrameExtractor`

### 4.2. Transform layer
- `WhisperProcessor`
- `VisionProcessor`

### 4.3. Index layer
- `VectorIndexer`

### 4.4. Retrieval layer
- `SearchEngine`

### 4.5. Ingest layer
- `YouTubeIngestor`

### 4.6. Serving layer
- `FastAPI`
- `Streamlit`

### 4.7. Catalog / metadata layer
- `data/video_catalog.json`
- các utility trong `src/utils.py` để load / save / upsert catalog entry

---

## 5. Công nghệ sử dụng

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

## 6. Cấu trúc thư mục

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

## 7. Cấu hình hiện tại

File cấu hình chính: `configs/config.yaml`

### 7.1. Sampling frame
Cấu hình hiện tại:

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
- đây là cấu hình **cân bằng** giữa:
  - action ngắn,
  - video dài vừa,
  - và tải máy khi demo.

### 7.2. Mô hình
- Whisper: `base`
- BLIP: `Salesforce/blip-image-captioning-base`
- CLIP: `ViT-B-32` pretrained `openai`
- Text embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### 7.3. Retrieval
Một số tham số chính:

- `hybrid_search_alpha: 0.35`
- `clip_search_beta: 0.65`
- `hybrid_candidate_multiplier: 3`
- `caption_merge_window_sec: 3.0`
- `caption_dedup_min_gap_sec: 2.0`

---

## 8. Luồng xử lý dữ liệu

### Bước 1: Nhận video
Video đi vào hệ thống qua:
- upload từ UI/API,
- process theo đường dẫn file local,
- hoặc ingest từ **YouTube URL**.

### Bước 2: Auto-catalog
Ngay đầu pipeline, hệ thống tự đảm bảo video có catalog entry trong `data/video_catalog.json`.

### Bước 3: Extract
- tách audio,
- kiểm tra audio stream,
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
- hoặc xóa dữ liệu index theo từng video.

---

## 9. Các loại dữ liệu được index

### 9.1. `transcription`
Toàn bộ transcript của video.

### 9.2. `segment_chunk`
Các đoạn transcript được ghép theo cửa sổ segment.

### 9.3. `caption`
Caption theo từng frame.

### 9.4. `multimodal`
Document kết hợp:
- phần speech,
- phần visual gần theo thời gian.

---

## 10. Metadata chính

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

---

## 11. API chính

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

## 12. Giao diện người dùng

UI Streamlit gồm 5 tab:

### 12.1. Search
- nhập query,
- chọn top_k,
- lọc theo loại nội dung,
- lọc theo video,
- xem preview video,
- xem timestamp, event range, caption, nearby speech context,
- xem metadata nguồn video.

### 12.2. Upload & Process
- upload file video,
- xử lý video,
- chọn `reset_index`,
- xem kết quả pipeline và preview video.

### 12.3. Process by Path
- xử lý video local theo đường dẫn tuyệt đối hoặc tương đối.

### 12.4. Process by YouTube URL
- nhập YouTube URL,
- tải video về `data/raw`,
- cập nhật source metadata,
- chạy pipeline,
- xem ingest result và pipeline result.

### 12.5. Video Inventory
- xem danh sách video đã index,
- xem inventory chi tiết,
- xem metadata nguồn,
- xóa dữ liệu index theo video.

---

## 13. Cài đặt môi trường

### 13.1. Tạo virtual environment
```bash
python -m venv venv
```

### 13.2. Kích hoạt virtual environment
Windows:
```bash
venv\Scripts\activate
```

### 13.3. Cài dependencies
```bash
pip install -r requirements.txt
```

### 13.4. Cài PyTorch theo môi trường máy
Ví dụ CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 13.5. Cài FFmpeg / FFprobe
Đảm bảo cả `ffmpeg` và `ffprobe` chạy được từ terminal.

---

## 14. Cách chạy hệ thống

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

---

## 15. Luồng demo gợi ý

1. chạy FastAPI,
2. chạy Streamlit,
3. kiểm tra `data/video_catalog.json`,
4. ingest một video bằng **Process by YouTube URL** hoặc process video local,
5. chờ pipeline hoàn tất,
6. vào tab **Search** để thử query,
7. kiểm tra **Video Inventory** để xem thống kê và metadata.

---

## 16. Một số lưu ý thực tế

### 16.1. Trade-off của frame sampling
Cấu hình `1.0 fps, 240 frames` là một mức cân bằng, nhưng không thể tối ưu đồng thời cho mọi loại video:

- video action rất ngắn có thể thích sampling dày hơn,
- video dài nhiều cảnh có thể thích coverage dài hơn.

### 16.2. Caption không phải ground truth
Caption từ BLIP là mô tả tự động tham khảo. Khi demo, nên xem:

- matched frame,
- timestamp,
- event range,
- nearby speech context,

là các tín hiệu quan trọng hơn chỉ riêng câu caption.

### 16.3. Video nhạc / cinematic
Với video chỉ có nhạc hoặc cinematic montage, Whisper vẫn có thể sinh transcript từ lyrics hoặc audio nền. Trong các case này, nên ưu tiên:

- `caption`
- `multimodal`

hơn là tin hoàn toàn vào `transcription`.

### 16.4. YouTube ingest chưa phải production-grade
Flow YouTube hiện đã usable cho demo, nhưng vẫn phụ thuộc:

- yt-dlp,
- format availability,
- ffmpeg / ffprobe,
- thay đổi phía YouTube.

### 16.5. Re-index sau khi đổi config
Nếu thay đổi config hoặc cập nhật metadata, nên process lại video với `reset_index=True`.

---

## 17. Điểm mạnh của hệ thống

- Có pipeline rõ từ ingest đến search.
- Kết hợp cả audio và image.
- Có hybrid retrieval giữa text và CLIP.
- Có event grouping và speech context.
- Có inventory và metadata nguồn video.
- Có YouTube ingest cho Bản 2.
- Có API và UI để demo.
- Có test cho các thành phần chính.
- Có auto-catalog cho video local và video ingest từ URL.

---

## 18. Giới hạn hiện tại

- Chưa hướng tới production-scale.
- Caption vẫn có thể khái quát hoặc chưa đủ fine-grained.
- Một số object nhỏ hoặc khó phân biệt vẫn có thể bị caption nhầm.
- Với video cinematic / music video, transcript có thể kém đáng tin hơn talk/TED.
- Catalog hiện vẫn ở mức JSON prototype, chưa phải metadata store quy mô lớn.
- Hiện tại mới hỗ trợ ingest **YouTube-only**, chưa mở rộng đa nền tảng.

---

## 19. Hướng phát triển

- batch ingest YouTube từ danh sách URL,
- hỗ trợ thêm nền tảng khác ngoài YouTube,
- thêm đánh giá retrieval theo metric,
- mở rộng quản lý catalog,
- bổ sung dashboard monitoring,
- hỗ trợ pipeline orchestration trong tương lai.

---

## 20. Kết luận

Hệ thống hiện tại đã xây dựng được một **multimodal semantic search pipeline cho video** với các thành phần chính:

- speech-to-text,
- image captioning,
- CLIP image retrieval,
- vector indexing,
- semantic search,
- video inventory,
- metadata catalog,
- và ingest video từ YouTube URL.

Đây là một prototype đủ rõ về mặt kỹ thuật để phục vụ demo đồ án theo định hướng **Kỹ thuật dữ liệu**, đồng thời đủ trực quan để trình diễn semantic search trên video qua API và UI.
