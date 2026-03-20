# Media Semantic Search for Video Data Warehouse

## 1. Giới thiệu

Đây là đồ án ngành thuộc chuyên ngành **Kỹ thuật dữ liệu (Data Engineering)** với đề tài:

**Xây dựng data pipeline trích xuất giọng nói và mô tả hình ảnh, lập chỉ mục vector phục vụ tìm kiếm ngữ nghĩa trong kho dữ liệu video.**

Hệ thống cho phép:
- nạp video vào pipeline xử lý,
- trích xuất âm thanh và khung hình,
- chuyển giọng nói thành văn bản,
- sinh mô tả hình ảnh từ frame,
- xây dựng vector index cho dữ liệu đa phương thức,
- tìm kiếm ngữ nghĩa trên kho dữ liệu video,
- và quản lý kho dữ liệu video ở mức cơ bản.

---

## 2. Mục tiêu đề tài

Mục tiêu của hệ thống là xây dựng một **data pipeline đa phương thức** phục vụ semantic search trên dữ liệu video.  
Pipeline tập trung vào các bước chính:

1. **Ingest video**
2. **Extract audio và image frames**
3. **Transform dữ liệu thành transcript / caption / multimodal documents**
4. **Index vector vào ChromaDB**
5. **Serve kết quả qua FastAPI và Streamlit**
6. **Quản lý kho dữ liệu video ở mức cơ bản**

---

## 3. Chức năng chính

### 3.1. Xử lý video
- Nhận video đầu vào từ đường dẫn cục bộ hoặc upload qua giao diện
- Tách audio từ video
- Trích xuất frame theo khoảng thời gian

### 3.2. Trích xuất thông tin
- Chuyển giọng nói thành văn bản bằng mô hình speech-to-text
- Sinh caption mô tả nội dung hình ảnh từ frame
- Tạo các tài liệu đa phương thức kết hợp thông tin âm thanh và hình ảnh

### 3.3. Lập chỉ mục vector
- Lưu transcript, segment chunk, caption và multimodal documents vào vector database
- Dùng embedding model để biểu diễn văn bản dưới dạng vector
- Hỗ trợ tìm kiếm ngữ nghĩa theo truy vấn ngôn ngữ tự nhiên

### 3.4. Tìm kiếm ngữ nghĩa
- Tìm kiếm trên toàn bộ kho dữ liệu video
- Lọc theo loại nội dung:
  - transcription
  - segment_chunk
  - caption
  - multimodal
- Lọc theo từng video cụ thể

### 3.5. Quản lý kho dữ liệu video
- Liệt kê danh sách video đã được index
- Xem thống kê dữ liệu theo từng video
- Xóa toàn bộ dữ liệu đã index của một video khỏi vector database

---

## 4. Kiến trúc hệ thống

Hệ thống được tổ chức thành các thành phần chính:

- **Extract Layer**
  - tách audio
  - trích xuất frame

- **Transform Layer**
  - speech-to-text
  - image captioning
  - hợp nhất dữ liệu thành tài liệu có cấu trúc

- **Index Layer**
  - sinh embedding
  - lưu vector vào ChromaDB
  - quản lý inventory video trong kho dữ liệu

- **Retrieval Layer**
  - semantic search trên vector database
  - lọc theo video và loại nội dung

- **Serving Layer**
  - FastAPI cung cấp REST API
  - Streamlit cung cấp giao diện demo

---

## 5. Cấu trúc thư mục

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
│   └── vector_db/
├── logs/
├── src/
│   ├── extract/
│   │   ├── audio_extractor.py
│   │   └── frame_extractor.py
│   ├── transform/
│   │   ├── whisper_processor.py
│   │   └── vision_processor.py
│   ├── indexing/
│   │   ├── db_manager.py
│   │   └── vector_indexer.py
│   ├── retrieval/
│   │   └── search_engine.py
│   └── utils.py
├── tests/
│   ├── test_api.py
│   ├── test_extract.py
│   ├── test_main_pipeline.py
│   ├── test_retrieval.py
│   ├── test_transform.py
│   └── test_vector_indexer.py
├── ui/
│   └── app.py
├── main_pipeline.py
├── requirements.txt
├── run_demo.md
└── README.md
```

### Mô tả ngắn các thư mục chính
- `api/`: REST API phục vụ search, process, inventory và quản lý video trong index
- `configs/`: cấu hình hệ thống và cấu hình logging
- `data/raw/`: video gốc đầu vào
- `data/interim/audio/`: file audio tách từ video
- `data/interim/frames/`: frame ảnh được trích xuất
- `data/interim/transcripts/`: dữ liệu transcript trung gian
- `data/interim/captions/`: dữ liệu caption trung gian
- `data/processed/`: dữ liệu đã hợp nhất và metadata cuối pipeline
- `data/vector_db/`: vector database ChromaDB
- `logs/`: log chạy hệ thống
- `src/extract/`: các module extract
- `src/transform/`: các module transform
- `src/indexing/`: logic index và quản lý dữ liệu vector
- `src/retrieval/`: logic semantic search
- `tests/`: bộ kiểm thử tự động
- `ui/`: giao diện Streamlit demo

---

## 6. Công nghệ sử dụng

- **Python**
- **FastAPI** cho REST API
- **Streamlit** cho giao diện demo
- **ChromaDB** cho vector database
- **SentenceTransformers** cho embedding
- **Whisper** cho speech-to-text
- **BLIP / Vision model** cho image captioning
- **FFmpeg** cho xử lý audio/video
- **PyTest** cho kiểm thử

---

## 7. Luồng xử lý dữ liệu

### Bước 1: Nhận video
Video được đưa vào hệ thống qua:
- upload từ UI
- gọi API
- hoặc truyền đường dẫn file cục bộ

### Bước 2: Extract
- tách audio từ video
- trích xuất các frame tại các mốc thời gian

### Bước 3: Transform
- audio được chuyển thành transcript
- frame được sinh caption mô tả nội dung hình ảnh
- transcript và caption có thể được hợp nhất thành multimodal documents

### Bước 4: Index
- các documents được chuyển thành vector embedding
- vector và metadata được lưu vào ChromaDB

### Bước 5: Search / Inventory
- người dùng truy vấn semantic search
- hệ thống trả về document phù hợp nhất cùng metadata
- người dùng có thể xem inventory của kho video hoặc xóa dữ liệu theo từng video

---

## 8. Các loại dữ liệu được index

Hệ thống hiện hỗ trợ 4 loại document:

### 8.1. `transcription`
Toàn bộ transcript của video.

### 8.2. `segment_chunk`
Các đoạn transcript được chia theo cửa sổ thời gian hoặc nhóm segment.

### 8.3. `caption`
Mô tả hình ảnh sinh ra từ từng frame.

### 8.4. `multimodal`
Tài liệu kết hợp giữa lời nói và mô tả hình ảnh gần cùng ngữ cảnh thời gian.

---

## 9. Metadata chính

Mỗi record trong vector database có thể chứa các metadata như:
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

Những metadata này giúp:
- lọc kết quả tìm kiếm,
- truy vết dữ liệu,
- thống kê theo từng video,
- và quản lý inventory trong kho dữ liệu video.

---

## 10. API chính

### 10.1. `GET /`
Kiểm tra API đang hoạt động và liệt kê các endpoint chính.

### 10.2. `GET /health`
Kiểm tra tình trạng hệ thống.

### 10.3. `GET /stats`
Lấy thống kê tổng quan của collection vector database.

### 10.4. `GET /videos`
Lấy danh sách các video đã được index trong kho dữ liệu vector.

### 10.5. `GET /videos/inventory`
Lấy thống kê toàn bộ kho dữ liệu video theo từng video.

### 10.6. `GET /videos/{video_name}`
Lấy thống kê chi tiết của một video.

### 10.7. `DELETE /videos/{video_name}`
Xóa toàn bộ dữ liệu đã index của một video khỏi vector database.

### 10.8. `POST /search`
Thực hiện tìm kiếm ngữ nghĩa.

Ví dụ request:
```json
{
  "query": "giới thiệu về trí tuệ nhân tạo",
  "top_k": 5,
  "content_type": "transcription",
  "video_name": "demo.mp4"
}
```

### 10.9. `POST /process-video`
Xử lý video theo đường dẫn file trên máy backend.

### 10.10. `POST /upload-video`
Upload video rồi chạy toàn bộ pipeline xử lý.

---

## 11. Giao diện người dùng

Ứng dụng Streamlit hiện có 4 tab chính:

### 11.1. Search
- nhập truy vấn semantic search
- chọn top_k
- lọc theo loại nội dung
- lọc theo video bằng dropdown hoặc nhập tên thủ công

### 11.2. Upload & Process
- upload file video
- xử lý pipeline
- chọn reset index trước khi index lại

### 11.3. Process by Path
- nhập đường dẫn video trên máy chạy backend
- xử lý pipeline từ file local

### 11.4. Video Inventory
- xem danh sách video đã index
- xem thống kê theo từng video
- xóa dữ liệu của một video khỏi index
- xem nhanh inventory của từng video

---

## 12. Kiểm thử

Hệ thống có bộ test cho các thành phần chính:

- `test_api.py`: kiểm thử API
- `test_extract.py`: kiểm thử extract audio/frame
- `test_main_pipeline.py`: kiểm thử pipeline tổng
- `test_retrieval.py`: kiểm thử search
- `test_transform.py`: kiểm thử transform
- `test_vector_indexer.py`: kiểm thử vector indexer

Điều này giúp tăng độ tin cậy của hệ thống và hỗ trợ bảo trì code.

---

## 13. Cài đặt môi trường

### 13.1. Cài thư viện Python
```bash
pip install -r requirements.txt
```

### 13.2. Cài FFmpeg
Đảm bảo máy chạy đã cài **FFmpeg** và lệnh `ffmpeg` có thể dùng từ terminal.

### 13.3. Cài PyTorch phù hợp
Nếu cần, cài PyTorch theo môi trường CPU hoặc GPU đang dùng trước khi chạy các mô hình.

---

## 14. Cách chạy hệ thống

### 14.1. Chạy API
```bash
uvicorn api.main:app --reload
```

### 14.2. Chạy giao diện Streamlit
```bash
streamlit run ui/app.py
```

### 14.3. Chạy test
```bash
pytest -v
```

---

## 15. Luồng demo ngắn

Một luồng demo điển hình:
1. chạy API và Streamlit
2. upload một video hoặc xử lý video từ đường dẫn
3. đợi pipeline extract, transform và index hoàn tất
4. vào tab **Search** để thử semantic search
5. vào tab **Video Inventory** để xem video đã index, thống kê theo video và thử xóa dữ liệu một video

---

## 16. Điểm mạnh của hệ thống

- Có pipeline dữ liệu rõ ràng từ ingest đến serving
- Kết hợp cả dữ liệu âm thanh và hình ảnh
- Có vector indexing phục vụ semantic search
- Có API và UI để demo
- Có metadata phục vụ truy vết và thống kê
- Có test cho các thành phần chính
- Có khả năng quản lý kho dữ liệu video ở mức cơ bản

---

## 17. Giới hạn hiện tại

Phiên bản hiện tại tập trung vào:
- xử lý offline theo từng video hoặc từng đợt upload nhỏ,
- semantic search ở quy mô demo hoặc đồ án,
- quản lý kho video ở mức cơ bản.

Hệ thống chưa hướng tới production-scale ở thời điểm hiện tại, ví dụ:
- chưa có orchestration như Airflow hoặc Prefect
- chưa có scheduling định kỳ
- chưa tối ưu batch ingest quy mô lớn
- chưa có monitoring production đầy đủ
- chưa có dashboard quản trị chuyên sâu

---

## 18. Hướng phát triển

Trong tương lai, hệ thống có thể mở rộng theo các hướng:
- xử lý batch nhiều video tự động
- thêm lịch chạy pipeline định kỳ
- bổ sung dashboard theo dõi pipeline
- đánh giá retrieval bằng top-k metrics
- hỗ trợ chỉnh sửa hoặc đồng bộ lại dữ liệu đã index
- mở rộng thành hệ thống quản lý kho dữ liệu video ở quy mô lớn hơn

---

## 19. Kết luận

Đồ án đã xây dựng được một data pipeline đa phương thức cho dữ liệu video, bao gồm:
- trích xuất giọng nói,
- mô tả hình ảnh,
- lập chỉ mục vector,
- tìm kiếm ngữ nghĩa,
- và quản lý kho dữ liệu video ở mức cơ bản.

Hệ thống phù hợp với định hướng của chuyên ngành Kỹ thuật dữ liệu, vì tập trung vào:
- tổ chức pipeline xử lý dữ liệu,
- chuẩn hóa metadata,
- xây dựng tầng indexing,
- phục vụ khai thác dữ liệu qua semantic search,
- và bổ sung quản lý inventory dữ liệu theo từng video.
