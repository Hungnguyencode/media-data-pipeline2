# Media Semantic Search Pipeline for Video Data

Hệ thống xây dựng data pipeline để:
- trích xuất âm thanh từ video
- nhận dạng giọng nói thành văn bản
- trích xuất frame từ video và sinh mô tả ảnh
- tạo embedding vector và lập chỉ mục trong ChromaDB
- phục vụ tìm kiếm ngữ nghĩa trên kho dữ liệu video

## Mục tiêu

Project này phục vụ đề tài:

**Xây dựng data pipeline trích xuất giọng nói và mô tả hình ảnh, lập chỉ mục vector phục vụ tìm kiếm ngữ nghĩa trong kho dữ liệu video**

Hệ thống tập trung vào pipeline xử lý dữ liệu đa phương tiện theo hướng batch, phù hợp với bài toán của ngành Kỹ thuật Dữ liệu.

## Kiến trúc hệ thống

```text
Video Input
   |
   |----> AudioExtractor ----> WhisperProcessor ----> Transcript
   |
   |----> FrameExtractor ----> VisionProcessor ----> Captions
   |
   |----> VectorIndexer ----> ChromaDB
   |
   |----> SearchEngine ----> FastAPI / Streamlit UI
```

## Mô tả các thành phần chính

- **AudioExtractor**: tách audio từ video bằng `ffmpeg`
- **FrameExtractor**: trích xuất frame từ video theo `fps` cấu hình
- **WhisperProcessor**: nhận dạng giọng nói thành transcript
- **VisionProcessor**: sinh caption từ frame bằng mô hình BLIP
- **VectorIndexer**: tạo embedding và lưu vào ChromaDB
- **SearchEngine**: tìm kiếm ngữ nghĩa bằng vector similarity
- **FastAPI**: cung cấp API để process video và search
- **Streamlit UI**: giao diện demo để upload video, chạy pipeline và tìm kiếm

## Tính năng chính

- Tách audio từ video bằng `ffmpeg`
- Trích xuất frame theo tần suất cấu hình
- Chuyển giọng nói thành văn bản bằng Whisper
- Sinh caption cho frame bằng BLIP
- Tạo vector embedding bằng SentenceTransformers
- Lập chỉ mục và truy vấn bằng ChromaDB
- Hỗ trợ semantic search qua API và giao diện Streamlit
- Hỗ trợ tài liệu đa phương thức bằng cách ghép transcript và caption theo timestamp gần nhau
- Tối ưu bộ nhớ GPU bằng lazy loading và giải phóng VRAM sau từng stage
- Có fallback từ GPU sang CPU khi gặp lỗi out-of-memory

## Cấu trúc thư mục

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
├── src/
│   ├── extract/
│   │   ├── audio_extractor.py
│   │   └── frame_extractor.py
│   ├── transform/
│   │   ├── whisper_processor.py
│   │   └── vision_processor.py
│   ├── indexing/
│   │   ├── vector_indexer.py
│   │   └── db_manager.py
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
├── pytest.ini
└── README.md
```

## Yêu cầu môi trường

- Python 3.10 hoặc 3.11
- Đã cài `ffmpeg` và thêm vào `PATH`
- Khuyến nghị dùng môi trường ảo
- Có thể chạy trên GPU NVIDIA RTX 2050 hoặc RTX 3050 với cấu hình mặc định đã tối ưu
- Nếu GPU yếu hoặc thiếu VRAM, hệ thống có thể fallback sang CPU ở một số stage

## Cài đặt

### 1. Tạo môi trường ảo

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài PyTorch theo CUDA phù hợp

Nên cài `torch`, `torchvision`, `torchaudio` trước theo đúng CUDA trên máy.

Ví dụ với CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Nếu không dùng GPU, có thể cài bản CPU theo hướng dẫn từ trang chính thức của PyTorch.

### 3. Cài thư viện của project

```bash
pip install -r requirements.txt
```

Nếu `openai-whisper` lỗi khi cài trên Windows, thử:

```bash
pip install openai-whisper --no-build-isolation
```

## Chạy hệ thống

### Chạy backend API

```bash
uvicorn api.main:app --reload
```

### Chạy giao diện Streamlit

```bash
streamlit run ui/app.py
```

## Cách sử dụng

### 1. Upload video và xử lý
- Mở giao diện Streamlit
- Chọn tab `Upload & Process`
- Upload video
- Hệ thống sẽ:
  - lưu file video
  - trích xuất audio
  - trích frame
  - tạo transcript
  - tạo caption
  - index vào vector database

### 2. Xử lý video bằng đường dẫn
- Mở tab `Process by Path`
- Nhập đường dẫn video trên máy chạy backend
- Chạy pipeline trực tiếp

### 3. Tìm kiếm ngữ nghĩa
- Mở tab `Search`
- Nhập câu truy vấn
- Có thể lọc theo:
  - toàn bộ transcript
  - đoạn transcript
  - caption ảnh
  - tài liệu đa phương thức
- Nhận kết quả kèm metadata và relevance

## API chính

### `GET /`
Kiểm tra API hoạt động.

### `GET /health`
Kiểm tra trạng thái hệ thống.

### `GET /stats`
Lấy thống kê số lượng document trong vector database.

### `POST /search`
Tìm kiếm ngữ nghĩa.

Ví dụ request:

```json
{
  "query": "giới thiệu về trí tuệ nhân tạo",
  "top_k": 5,
  "content_type": "transcription",
  "video_name": "demo.mp4"
}
```

### `POST /process-video`
Chạy pipeline với đường dẫn video trên backend.

Ví dụ:

```json
{
  "video_path": "data/raw/demo.mp4",
  "reset_index": true
}
```

### `POST /upload-video`
Upload video và xử lý trực tiếp.

## Kiểm thử

Chạy toàn bộ test:

```bash
pytest
```

## Tối ưu cho RTX 2050 / RTX 3050

- Giảm `frame_sampling_fps` để giảm số frame cần caption
- Giới hạn `max_frames`
- Resize frame trước khi caption
- Lazy loading model trong pipeline
- Giải phóng VRAM sau từng stage
- Fallback từ GPU sang CPU khi BLIP hoặc Whisper gặp OOM

Nếu máy yếu hơn mong đợi, có thể chỉnh trong `configs/config.yaml`:

```yaml
video:
  frame_sampling_fps: 0.25
  max_frames: 100

models:
  whisper:
    name: "tiny"
```

## Lỗi thường gặp và cách xử lý

### 1. Không nhận `ffmpeg`
Biểu hiện:
- không extract được audio
- báo lỗi không tìm thấy ffmpeg

Cách xử lý:
- cài `ffmpeg`
- thêm `ffmpeg` vào `PATH`
- kiểm tra bằng:

```bash
ffmpeg -version
```

### 2. Lỗi `No module named 'whisper'`
Cách xử lý:

```bash
pip install openai-whisper
```

Nếu lỗi build:

```bash
pip install openai-whisper --no-build-isolation
```

### 3. Cài nhầm bản torch CPU
Biểu hiện:
- chạy được nhưng rất chậm
- `torch.cuda.is_available()` trả về `False`

Cách xử lý:
- gỡ torch cũ
- cài lại theo đúng CUDA từ trang PyTorch

### 4. GPU out of memory
Biểu hiện:
- pipeline báo lỗi CUDA OOM
- xử lý video chậm hoặc dừng giữa chừng

Cách xử lý:
- giảm `frame_sampling_fps`
- giảm `max_frames`
- đổi Whisper từ `base` sang `tiny`
- đóng các ứng dụng ngốn GPU khác

## Ghi chú về DBManager

File `src/indexing/db_manager.py` hiện được giữ lại như một utility hỗ trợ quản trị ChromaDB và truy xuất thống kê. Luồng chạy chính hiện dùng trực tiếp `VectorIndexer`, nhưng `DBManager` vẫn hữu ích cho các chức năng mở rộng như quản trị collection hoặc dashboard thống kê.

## Hướng phát triển

- Dùng Airflow hoặc Prefect để orchestration pipeline
- Tối ưu chunking transcript nâng cao hơn
- Đánh giá semantic retrieval bằng bộ truy vấn kiểm thử và metric cụ thể
- Mở rộng sang multilingual speech-to-text
- Bổ sung monitoring và metadata tracking chi tiết hơn
