# Media Semantic Search for Video Data Warehouse

## 1. Giới thiệu

Đây là đồ án thuộc chuyên ngành **Kỹ thuật dữ liệu (Data Engineering)** với đề tài:

**Xây dựng data pipeline đa phương thức để trích xuất giọng nói, mô tả hình ảnh, lập chỉ mục vector và phục vụ tìm kiếm ngữ nghĩa trên kho dữ liệu video.**

Phiên bản hiện tại của hệ thống đã được nâng cấp từ pipeline semantic search dựa chủ yếu trên transcript/caption text sang pipeline **multimodal semantic search** hoàn chỉnh hơn, kết hợp:

- **Whisper** để chuyển giọng nói thành văn bản,
- **BLIP** để sinh caption mô tả nội dung frame,
- **CLIP** để tăng khả năng đối sánh giữa truy vấn văn bản và nội dung hình ảnh,
- **SentenceTransformers** để biểu diễn ngữ nghĩa cho text documents,
- **ChromaDB** để lưu trữ và truy xuất vector.

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
- Dùng **CLIP image embedding** để biểu diễn nội dung hình ảnh
- Hỗ trợ tìm kiếm ngữ nghĩa theo truy vấn ngôn ngữ tự nhiên

### 3.4. Tìm kiếm ngữ nghĩa
- Tìm kiếm trên toàn bộ kho dữ liệu video
- Hybrid retrieval giữa:
  - text embedding search
  - CLIP text-image retrieval
- Lọc theo loại nội dung:
  - `transcription`
  - `segment_chunk`
  - `caption`
  - `multimodal`
- Lọc theo từng video cụ thể

### 3.5. Quản lý kho dữ liệu video
- Liệt kê danh sách video đã được index
- Xem thống kê dữ liệu theo từng video
- Xóa toàn bộ dữ liệu đã index của một video khỏi vector database

---

## 4. Những nâng cấp đã thực hiện

Phiên bản hiện tại đã được nâng cấp theo các hướng chính sau:

### 4.1. Nâng cấp BLIP-only thành BLIP + CLIP
Trước đây hệ thống chủ yếu dựa vào caption text sinh từ BLIP.  
Hiện tại, mỗi frame được xử lý theo 2 hướng:

- **BLIP** sinh caption mô tả nội dung hình ảnh
- **CLIP** sinh image embedding để phục vụ retrieval tốt hơn theo truy vấn văn bản

### 4.2. Tách vector database thành 2 collection
Hệ thống hiện dùng:

- `video_semantic_search_text`
- `video_semantic_search_clip`

Điều này giúp tách riêng:

- text embedding space cho transcript / multimodal / caption text
- image embedding space cho CLIP

### 4.3. Hybrid retrieval
Khi người dùng truy vấn bằng text, hệ thống sẽ:

- encode query bằng SentenceTransformer để tìm trên text collection
- encode query bằng CLIP text encoder để tìm trên clip collection
- fusion kết quả để tạo ra kết quả cuối cùng

### 4.4. Tạo multimodal documents
Hệ thống tạo thêm các tài liệu kết hợp:

- `[Speech] ...`
- `[Visual] ...`

giúp kết nối nội dung lời nói và nội dung hình ảnh gần cùng thời điểm.

### 4.5. Cải thiện temporal precision
Cấu hình sampling frame hiện được thiết lập ở mức phù hợp hơn cho video demo:

- `frame_sampling_fps = 0.75`
- `max_frames = 180`

so với cấu hình test nhẹ dùng để debug nhanh.

### 4.6. Cải thiện UI và quality of results
Để tránh caption thô của BLIP làm người dùng hiểu sai, hệ thống đã bổ sung:

- hậu xử lý caption (caption refinement)
- event grouping cho các frame gần nhau
- nearby speech context
- UI wording rõ hơn: caption là mô tả tự động tham khảo, không phải ground truth

---

## 5. Khả năng tổng quát hóa của các nâng cấp

Phần lớn các nâng cấp trên **không chỉ áp dụng cho riêng video demo hiện tại**, mà có thể dùng cho nhiều loại video demo khác, đặc biệt là:

- video hướng dẫn
- video bài giảng / thuyết trình
- tutorial
- cooking video
- DIY / hands-on demo
- video có speech + visual context rõ ràng

Tuy nhiên, hệ thống vẫn còn giới hạn trong các trường hợp:

- hành động quá ngắn
- động tác quá tinh vi
- motion blur mạnh
- caption model mô tả chưa đủ chi tiết

---

## 6. Kiến trúc hệ thống

Hệ thống được tổ chức thành các thành phần chính:

- **Extract Layer**
  - tách audio
  - trích xuất frame

- **Transform Layer**
  - speech-to-text
  - image captioning
  - CLIP image embedding
  - hợp nhất dữ liệu thành tài liệu có cấu trúc

- **Index Layer**
  - sinh embedding
  - lưu vector vào ChromaDB
  - quản lý inventory video trong kho dữ liệu

- **Retrieval Layer**
  - semantic search trên vector database
  - hybrid retrieval text + clip
  - lọc theo video và loại nội dung

- **Serving Layer**
  - FastAPI cung cấp REST API
  - Streamlit cung cấp giao diện demo

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

## 8. Công nghệ sử dụng

- **Python**
- **FastAPI** cho REST API
- **Streamlit** cho giao diện demo
- **ChromaDB** cho vector database
- **SentenceTransformers** cho embedding text
- **Whisper** cho speech-to-text
- **BLIP** cho image captioning
- **CLIP** cho text-image alignment
- **FFmpeg** cho xử lý audio/video
- **PyTest** cho kiểm thử

---

## 9. Luồng xử lý dữ liệu

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
- frame được encode thêm bằng CLIP
- transcript và caption có thể được hợp nhất thành multimodal documents

### Bước 4: Index
- các documents được chuyển thành vector embedding
- vector và metadata được lưu vào ChromaDB

### Bước 5: Search / Inventory
- người dùng truy vấn semantic search
- hệ thống trả về document phù hợp nhất cùng metadata
- người dùng có thể xem inventory của kho video hoặc xóa dữ liệu theo từng video

---

## 10. Các loại dữ liệu được index

Hệ thống hiện hỗ trợ 4 loại document:

### 10.1. `transcription`
Toàn bộ transcript của video.

### 10.2. `segment_chunk`
Các đoạn transcript được chia theo cửa sổ thời gian hoặc nhóm segment.

### 10.3. `caption`
Mô tả hình ảnh sinh ra từ từng frame.

### 10.4. `multimodal`
Tài liệu kết hợp giữa lời nói và mô tả hình ảnh gần cùng ngữ cảnh thời gian.

---

## 11. Metadata chính

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

Ngoài ra, ở phiên bản nâng cấp còn có thể xuất hiện:

- `clip_model_name`
- `embedding_source`
- các field event/grouping ở tầng retrieval và UI

Những metadata này giúp:

- lọc kết quả tìm kiếm,
- truy vết dữ liệu,
- thống kê theo từng video,
- và quản lý inventory trong kho dữ liệu video.

---

## 12. API chính

### 12.1. `GET /`
Kiểm tra API đang hoạt động và liệt kê các endpoint chính.

### 12.2. `GET /health`
Kiểm tra tình trạng hệ thống.

### 12.3. `GET /stats`
Lấy thống kê tổng quan của collection vector database.

### 12.4. `GET /videos`
Lấy danh sách các video đã được index trong kho dữ liệu vector.

### 12.5. `GET /videos/inventory`
Lấy thống kê toàn bộ kho dữ liệu video theo từng video.

### 12.6. `GET /videos/{video_name}`
Lấy thống kê chi tiết của một video.

### 12.7. `DELETE /videos/{video_name}`
Xóa toàn bộ dữ liệu đã index của một video khỏi vector database.

### 12.8. `POST /search`
Thực hiện tìm kiếm ngữ nghĩa.

Ví dụ request:

```json
{
  "query": "crack egg",
  "top_k": 5,
  "content_type": "caption",
  "video_name": "egg.mp4"
}
```

### 12.9. `POST /process-video`
Xử lý video theo đường dẫn file trên máy backend.

### 12.10. `POST /upload-video`
Upload video rồi chạy toàn bộ pipeline xử lý.

---

## 13. Giao diện người dùng

Ứng dụng Streamlit hiện có 4 tab chính:

### 13.1. Search
- nhập truy vấn semantic search
- chọn top_k
- lọc theo loại nội dung
- lọc theo video bằng dropdown hoặc nhập tên thủ công
- xem video preview
- xem matched frame description, auto-caption, speech context và metadata

### 13.2. Upload & Process
- upload file video
- xử lý pipeline
- chọn reset index trước khi index lại

### 13.3. Process by Path
- nhập đường dẫn video trên máy chạy backend
- xử lý pipeline từ file local

### 13.4. Video Inventory
- xem danh sách video đã index
- xem thống kê theo từng video
- xóa dữ liệu của một video khỏi index
- xem nhanh inventory của từng video

---

## 14. Kiểm thử

Hệ thống có bộ test cho các thành phần chính:

- `test_api.py`: kiểm thử API
- `test_extract.py`: kiểm thử extract audio/frame
- `test_main_pipeline.py`: kiểm thử pipeline tổng
- `test_retrieval.py`: kiểm thử search
- `test_transform.py`: kiểm thử transform
- `test_vector_indexer.py`: kiểm thử vector indexer

Trạng thái hiện tại:
- **33 tests passed**

Điều này giúp tăng độ tin cậy của hệ thống và hỗ trợ bảo trì code.

---

## 15. Cài đặt môi trường

### 15.1. Cài thư viện Python
```bash
pip install -r requirements.txt
```

### 15.2. Cài FFmpeg
Đảm bảo máy chạy đã cài **FFmpeg** và lệnh `ffmpeg` có thể dùng từ terminal.

### 15.3. Cài PyTorch phù hợp
Nếu cần, cài PyTorch theo môi trường CPU hoặc GPU đang dùng trước khi chạy các mô hình.

Ví dụ với CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 16. Cách chạy hệ thống

### 16.1. Chạy API
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 16.2. Chạy giao diện Streamlit
```bash
streamlit run ui/app.py
```

### 16.3. Chạy test
```bash
pytest -v
```

---

## 17. Luồng demo ngắn

Một luồng demo điển hình:

1. chạy API và Streamlit
2. upload một video hoặc xử lý video từ đường dẫn
3. đợi pipeline extract, transform và index hoàn tất
4. vào tab **Search** để thử semantic search
5. vào tab **Video Inventory** để xem video đã index, thống kê theo video và thử xóa dữ liệu một video

---

## 18. Điểm mạnh của hệ thống

- Có pipeline dữ liệu rõ ràng từ ingest đến serving
- Kết hợp cả dữ liệu âm thanh và hình ảnh
- Có BLIP + CLIP cho retrieval tốt hơn trên nội dung hình ảnh
- Có vector indexing phục vụ semantic search
- Có API và UI để demo
- Có metadata phục vụ truy vết và thống kê
- Có test cho các thành phần chính
- Có khả năng quản lý kho dữ liệu video ở mức cơ bản
- Có nâng cấp thực tế để cải thiện caption và presentation layer

---

## 19. Giới hạn hiện tại

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

Ngoài ra, một số giới hạn kỹ thuật hiện tại gồm:

- caption BLIP đôi lúc vẫn còn mô tả khái quát
- các hành động rất ngắn hoặc rất tinh vi vẫn khó truy xuất tuyệt đối chính xác
- hệ thống hiện phù hợp nhất với demo/prototype/local processing

---

## 20. Hướng phát triển

Trong tương lai, hệ thống có thể mở rộng theo các hướng:

- xử lý batch nhiều video tự động
- thêm lịch chạy pipeline định kỳ
- bổ sung dashboard theo dõi pipeline
- đánh giá retrieval bằng top-k metrics
- hỗ trợ chỉnh sửa hoặc đồng bộ lại dữ liệu đã index
- mở rộng thành hệ thống quản lý kho dữ liệu video ở quy mô lớn hơn
- thay thế hoặc mở rộng model captioning nếu có phần cứng mạnh hơn

---

## 21. Kết luận

Đồ án đã xây dựng được một data pipeline đa phương thức cho dữ liệu video, bao gồm:

- trích xuất giọng nói,
- mô tả hình ảnh,
- lập chỉ mục vector,
- tìm kiếm ngữ nghĩa,
- và quản lý kho dữ liệu video ở mức cơ bản.

Phiên bản nâng cấp hiện tại đã mở rộng từ pipeline text-heavy sang hệ thống **multimodal semantic search** hoàn chỉnh hơn, sử dụng **Whisper + BLIP + CLIP + ChromaDB**, phù hợp với định hướng của chuyên ngành Kỹ thuật dữ liệu vì tập trung vào:

- tổ chức pipeline xử lý dữ liệu,
- chuẩn hóa metadata,
- xây dựng tầng indexing,
- phục vụ khai thác dữ liệu qua semantic search,
- và bổ sung quản lý inventory dữ liệu theo từng video.
