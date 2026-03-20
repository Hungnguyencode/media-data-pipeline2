# Xây dựng data pipeline trích xuất giọng nói và mô tả hình ảnh, lập chỉ mục vector phục vụ tìm kiếm ngữ nghĩa trong kho dữ liệu video

## 1. Giới thiệu đề tài

Đây là đồ án ngành thuộc chuyên ngành **Kỹ thuật Dữ liệu (Data Engineering)** với mục tiêu xây dựng một hệ thống xử lý dữ liệu video theo hướng đa phương thức. Hệ thống thực hiện trích xuất thông tin từ **âm thanh** và **hình ảnh** trong video, sau đó chuẩn hóa dữ liệu, tạo **vector embedding**, lập chỉ mục trong **vector database**, và hỗ trợ **tìm kiếm ngữ nghĩa** thông qua API và giao diện người dùng.

Đề tài tập trung vào bài toán:

- trích xuất lời nói trong video thành văn bản
- trích xuất mô tả ngữ nghĩa từ các khung hình
- hợp nhất dữ liệu đa nguồn thành tài liệu có cấu trúc
- lập chỉ mục vector để phục vụ semantic search trong kho dữ liệu video

Bài toán này phù hợp với định hướng **Kỹ thuật Dữ liệu** vì bao gồm đầy đủ các bước cốt lõi của một pipeline dữ liệu:

- **ingest** dữ liệu video đầu vào
- **extract** dữ liệu từ audio và image
- **transform** dữ liệu thành dạng có cấu trúc
- **manage metadata** phục vụ truy vết và khai thác
- **index** dữ liệu bằng vector database
- **retrieve** dữ liệu bằng tìm kiếm ngữ nghĩa

---

## 2. Mục tiêu của hệ thống

Hệ thống được xây dựng nhằm đáp ứng các mục tiêu sau:

1. Tự động xử lý video đầu vào và tách thành các nguồn dữ liệu trung gian gồm:
   - audio
   - frame ảnh
   - transcript
   - caption ảnh

2. Tổ chức dữ liệu theo pipeline rõ ràng, có lưu trữ dữ liệu tạm và dữ liệu đã xử lý.

3. Sinh vector embedding cho nhiều loại tài liệu:
   - toàn bộ transcript
   - các đoạn transcript
   - caption của frame
   - tài liệu đa phương thức kết hợp lời nói và hình ảnh

4. Lập chỉ mục vector trong ChromaDB để truy hồi dữ liệu theo ngữ nghĩa.

5. Cung cấp giao diện API và UI để:
   - nạp video
   - xử lý pipeline
   - tìm kiếm ngữ nghĩa
   - xem thống kê dữ liệu đã index

---

## 3. Phạm vi và đầu vào của hệ thống

### 3.1. Đầu vào
Hệ thống nhận đầu vào là các file video số, ví dụ:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

### 3.2. Đầu ra
Hệ thống tạo ra các đầu ra chính:

- file audio được tách từ video
- bộ frame ảnh được lấy mẫu từ video
- transcript tiếng nói từ audio
- caption mô tả ảnh từ các frame
- dữ liệu tổng hợp dạng JSON
- vector index trong ChromaDB
- kết quả truy hồi semantic search qua API/UI

### 3.3. Giới hạn hiện tại
Phiên bản hiện tại tập trung vào pipeline xử lý offline cho từng video hoặc từng đợt upload nhỏ. Hệ thống chưa hướng tới xử lý phân tán quy mô lớn hoặc streaming realtime.

---

## 4. Kiến trúc tổng thể

### 4.1. Luồng xử lý dữ liệu

```text
Raw Video
   ↓
Extract Audio + Extract Frames
   ↓
Speech-to-Text (Whisper) + Image Captioning (BLIP)
   ↓
Transform / Normalize / Merge Metadata
   ↓
Embedding (Sentence Transformers)
   ↓
Vector Indexing (ChromaDB)
   ↓
Semantic Search (FastAPI + Streamlit)
```

### 4.2. Các lớp chức năng chính

- **Extract layer**
  - tách audio từ video
  - lấy mẫu frame từ video

- **Transform layer**
  - chuyển audio thành transcript bằng Whisper
  - sinh caption cho frame bằng BLIP
  - chuẩn hóa dữ liệu và metadata

- **Indexing layer**
  - tạo embedding cho tài liệu
  - lưu vector vào ChromaDB
  - quản lý collection và thống kê dữ liệu

- **Retrieval layer**
  - nhận truy vấn
  - sinh embedding cho query
  - tìm kiếm các tài liệu gần nhất trong vector DB

- **Serving layer**
  - FastAPI cho backend
  - Streamlit cho UI demo

---

## 5. Cấu trúc thư mục dự án

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

---

## 6. Công nghệ sử dụng

### 6.1. Xử lý video và ảnh
- `ffmpeg`
- `opencv-python`
- `Pillow`

### 6.2. Speech-to-text
- `openai-whisper`

### 6.3. Image captioning
- `transformers`
- `Salesforce/blip-image-captioning-base`

### 6.4. Embedding
- `sentence-transformers`
- `paraphrase-multilingual-MiniLM-L12-v2`

### 6.5. Vector database
- `chromadb`

### 6.6. API và UI
- `FastAPI`
- `Uvicorn`
- `Streamlit`

### 6.7. Testing và cấu hình
- `pytest`
- `PyYAML`

---

## 7. Cài đặt môi trường

### 7.1. Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường:

**Windows**
```powershell
venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 7.2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Lưu ý: project chưa tự cài `torch`, `torchvision`, `torchaudio` trong `requirements.txt`. Người dùng cần cài PyTorch riêng theo môi trường CPU/GPU đang sử dụng.

### 7.3. Cài PyTorch
Do PyTorch phụ thuộc CPU/GPU và CUDA, nên cài riêng theo môi trường máy.

Ví dụ bản CPU:

```bash
pip install torch torchvision torchaudio
```

Nếu dùng GPU CUDA, cài theo hướng dẫn chính thức của PyTorch tương ứng với phiên bản CUDA.

### 7.4. Cài đặt FFmpeg
Hệ thống cần `ffmpeg` để trích xuất audio và xử lý video.

Kiểm tra:

```bash
ffmpeg -version
```

Nếu chưa có, cài `ffmpeg` vào máy và bảo đảm lệnh này chạy được từ terminal.

---

## 8. Cấu hình hệ thống

File cấu hình chính: `configs/config.yaml`

Ví dụ các nhóm cấu hình chính:

- đường dẫn lưu dữ liệu
- cấu hình xử lý video
- model Whisper
- model BLIP
- embedding model
- pipeline version
- collection name trong ChromaDB
- API base URL

Một số tham số quan trọng:

- `frame_sampling_fps`: tần suất lấy mẫu frame
- `max_frames`: giới hạn số frame tối đa
- `embedding.name`: model dùng để vector hóa
- `vector_db.collection_name`: tên collection
- `pipeline.max_top_k`: số kết quả tối đa khi search

Ví dụ cấu hình API:

```yaml
api:
  base_url: "http://127.0.0.1:8000"
```

---

## 9. Quy trình xử lý dữ liệu

### 9.1. Bước 1 - Trích xuất audio
Từ video đầu vào, hệ thống tách phần âm thanh và lưu vào thư mục `data/interim/audio`.

### 9.2. Bước 2 - Trích xuất frame
Video được lấy mẫu theo FPS cấu hình, sau đó lưu frame ảnh vào `data/interim/frames`.

### 9.3. Bước 3 - Speech-to-text
File audio được đưa qua mô hình Whisper để tạo:

- `full_text`
- danh sách `segments`
- ngôn ngữ
- metadata xử lý

### 9.4. Bước 4 - Image captioning
Mỗi frame được xử lý bằng mô hình BLIP để sinh caption mô tả nội dung ảnh.

### 9.5. Bước 5 - Chuẩn hóa và hợp nhất dữ liệu
Dữ liệu transcript và caption được chuẩn hóa thành các tài liệu có thể index. Ngoài transcript và caption độc lập, hệ thống còn tạo ra **multimodal documents** bằng cách ghép lời nói và mô tả hình ảnh gần cùng thời điểm.

### 9.6. Bước 6 - Sinh embedding
Các tài liệu văn bản được chuyển thành vector embedding bằng mô hình đa ngôn ngữ.

### 9.7. Bước 7 - Lập chỉ mục vector
Embedding và metadata được lưu vào ChromaDB để phục vụ truy hồi ngữ nghĩa.

### 9.8. Bước 8 - Semantic search
Khi người dùng nhập câu truy vấn, hệ thống:

- sinh embedding cho query
- truy vấn vector DB
- trả về các tài liệu gần nhất kèm metadata

---

## 10. Schema dữ liệu đầu ra

Các bản ghi trong pipeline và vector index có thể chứa những trường sau:

| Trường | Ý nghĩa |
|---|---|
| `video_name` | Tên file video nguồn |
| `video_path` | Đường dẫn video đầu vào |
| `audio_path` | Đường dẫn file audio đã trích xuất |
| `frame_name` | Tên file frame |
| `image_path` | Đường dẫn ảnh frame |
| `timestamp` | Mốc thời gian theo giây |
| `timestamp_str` | Mốc thời gian dạng HH:MM:SS |
| `start_time` | Thời điểm bắt đầu của đoạn dữ liệu |
| `end_time` | Thời điểm kết thúc của đoạn dữ liệu |
| `content_type` | Loại dữ liệu: `transcription`, `segment_chunk`, `caption`, `multimodal` |
| `source_modality` | Nguồn dữ liệu: `audio`, `image`, `audio+image` |
| `model_name` | Model được dùng để sinh dữ liệu |
| `document_language` | Ngôn ngữ chính của bản ghi |
| `pipeline_version` | Phiên bản pipeline |
| `distance` | Khoảng cách vector trả về từ ChromaDB |
| `similarity_score` | Giá trị xấp xỉ độ tương đồng, tính từ `1 - distance` |

---

## 11. Các loại dữ liệu được index

### 11.1. Toàn bộ transcript
Đây là bản transcript tổng hợp toàn bộ nội dung lời nói của video.

### 11.2. Segment chunk
Transcript được chia theo nhóm segment để tăng độ chi tiết khi truy hồi.

### 11.3. Caption
Mỗi frame ảnh sẽ có một caption tương ứng và được index riêng.

### 11.4. Multimodal document
Dữ liệu lời nói và mô tả hình ảnh được kết hợp trong cùng một tài liệu khi chúng gần nhau theo trục thời gian. Loại tài liệu này giúp cải thiện semantic retrieval cho các truy vấn chứa cả yếu tố nội dung nói và nội dung hình ảnh.

---

## 12. Hướng dẫn chạy hệ thống

### 12.1. Chạy pipeline qua CLI

Xử lý một video:

```bash
python main_pipeline.py --video data/raw/demo.mp4 --reset-index
```

Tìm kiếm ngữ nghĩa:

```bash
python main_pipeline.py --query "đoạn nói về trí tuệ nhân tạo" --top-k 5
```

Tìm kiếm có filter:

```bash
python main_pipeline.py --query "cảnh có slide trên màn hình" --top-k 5 --content-type multimodal
```

### 12.2. Chạy API

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### 12.3. Chạy UI

```bash
streamlit run ui/app.py
```

### 12.4. Chạy test

```bash
pytest -v
```

### 12.5. Chạy nhanh trên Windows

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
streamlit run ui/app.py
pytest -v
```

---

## 13. API chính

### 13.1. `GET /`
Kiểm tra API đang chạy.

### 13.2. `GET /health`
Kiểm tra trạng thái hệ thống.

### 13.3. `GET /stats`
Lấy thống kê collection hiện tại trong vector DB.

### 13.4. `POST /process-video`
Xử lý video bằng đường dẫn từ backend.

Ví dụ request:

```json
{
  "video_path": "data/raw/demo.mp4",
  "reset_index": true
}
```

### 13.5. `POST /upload-video`
Upload video trực tiếp và xử lý pipeline.

### 13.6. `POST /search`
Tìm kiếm ngữ nghĩa.

Ví dụ request:

```json
{
  "query": "đoạn nói về trí tuệ nhân tạo",
  "top_k": 5,
  "content_type": "multimodal",
  "video_name": "demo.mp4"
}
```

Ví dụ response rút gọn:

```json
{
  "results": [
    {
      "document": "Nội dung lời nói và mô tả hình ảnh...",
      "metadata": {
        "video_name": "demo.mp4",
        "content_type": "multimodal",
        "source_modality": "audio+image"
      },
      "distance": 0.132,
      "similarity_score": 0.868,
      "score_type": "similarity_proxy_from_distance"
    }
  ]
}
```

---

## 14. Giao diện người dùng

Giao diện Streamlit gồm 3 tab chính:

1. **Search**
   - nhập câu truy vấn
   - chọn top-k
   - lọc theo loại nội dung
   - lọc theo tên video

2. **Upload & Process**
   - upload video từ máy người dùng
   - xử lý và index ngay sau khi upload

3. **Process by Path**
   - xử lý video theo đường dẫn có sẵn trên máy backend

---

## 15. Logging và metadata

Hệ thống lưu log trong thư mục `logs/` và lưu metadata chạy trong `data/processed`.

Một số metadata quan trọng:

- checksum MD5 của video
- kích thước video
- trạng thái từng stage
- thời gian chạy từng stage
- số lượng transcript segment
- số lượng caption
- số lượng document được index

Các thông tin này hữu ích cho:
- theo dõi pipeline
- đánh giá hiệu năng
- truy vết lỗi
- báo cáo đồ án

---

## 16. Đánh giá hệ thống

Để phần báo cáo hoàn chỉnh hơn, nên đánh giá trên các tiêu chí sau:

### 16.1. Đánh giá hiệu năng
- thời gian xử lý video 1 phút, 5 phút, 10 phút
- thời gian của từng stage:
  - extract audio
  - extract frames
  - transcribe
  - caption
  - index
- số lượng frame được xử lý
- số lượng document sinh ra

### 16.2. Đánh giá chất lượng truy hồi
Nên chuẩn bị bộ truy vấn mẫu và đánh giá top-k theo mức độ phù hợp.

Ví dụ truy vấn tiếng Việt:
- đoạn nói về trí tuệ nhân tạo
- phần trình bày giới thiệu bài giảng
- cảnh có người đang đứng thuyết trình
- video có slide trên màn hình
- nội dung nói về dữ liệu lớn

Ví dụ truy vấn tiếng Anh:
- a person presenting in front of a screen
- introduction to artificial intelligence
- a lecture slide on the screen
- classroom presentation scene

Các chỉ số có thể dùng:
- top-1 đúng / sai
- top-3 có kết quả phù hợp hay không
- mức độ phù hợp định tính theo từng truy vấn

---

## 17. Ưu điểm của hệ thống

- Có pipeline xử lý dữ liệu rõ ràng, tách lớp hợp lý
- Kết hợp dữ liệu từ cả âm thanh và hình ảnh
- Có metadata phục vụ truy vết và đánh giá
- Hỗ trợ vector indexing và semantic retrieval
- Có cả backend API và giao diện demo
- Có thể mở rộng để xử lý nhiều video trong kho dữ liệu

---

## 18. Hạn chế hiện tại

### 18.1. Không đồng nhất ngôn ngữ hoàn toàn
- Whisper được cấu hình nhận dạng tiếng Việt
- BLIP chủ yếu sinh caption bằng tiếng Anh

Do đó, dữ liệu trong hệ thống có thể mang tính song ngữ. Dù embedding model hiện tại là đa ngôn ngữ, chất lượng truy hồi vẫn có thể giảm trong một số trường hợp.

### 18.2. Similarity score chỉ là giá trị xấp xỉ
`similarity_score` hiện được tính từ công thức `1 - distance`, nên chỉ mang tính chất **proxy** để hiển thị. Đây không phải xác suất hay độ tin cậy tuyệt đối.

### 18.3. Chưa tối ưu cho dữ liệu lớn
Hệ thống hiện phù hợp với quy mô đồ án và demo. Chưa triển khai:
- orchestration với Airflow/Prefect
- xử lý song song số lượng lớn video
- batch scheduling ở quy mô production
- monitoring chuyên sâu

### 18.4. Caption ảnh còn phụ thuộc model nền
Caption sinh bởi BLIP có thể chưa phản ánh đầy đủ ngữ cảnh chuyên ngành hoặc nội dung đặc thù.

---

## 19. Hướng phát triển

Trong tương lai, hệ thống có thể mở rộng theo các hướng sau:

- bổ sung orchestrator như Airflow hoặc Prefect
- hỗ trợ pipeline theo batch hoặc theo lịch
- cải thiện đánh giá chất lượng retrieval bằng bộ ground truth
- dùng mô hình caption tốt hơn hoặc hỗ trợ caption tiếng Việt
- bổ sung filter theo thời gian, theo video, theo loại dữ liệu
- hỗ trợ xóa/chỉnh sửa dữ liệu đã index
- xây dựng dashboard theo dõi pipeline và thống kê retrieval

---

## 20. Kết luận

Đề tài đã xây dựng được một pipeline dữ liệu video đa phương thức theo hướng **Kỹ thuật Dữ liệu**, trong đó hệ thống có khả năng:

- trích xuất dữ liệu âm thanh và hình ảnh từ video
- chuyển đổi dữ liệu thô thành dữ liệu có cấu trúc
- tổ chức metadata phục vụ truy vết
- sinh vector embedding và lưu trữ trong vector database
- hỗ trợ semantic search trên kho dữ liệu video

Kết quả đạt được cho thấy hệ thống có tính ứng dụng tốt trong các bài toán khai thác nội dung video theo ngữ nghĩa, đồng thời tạo nền tảng để mở rộng sang các hướng xử lý dữ liệu video quy mô lớn hơn trong tương lai.

---

## 21. Tác giả và thông tin đề tài

- **Môn học**: Đồ án ngành
- **Ngành**: Kỹ thuật dữ liệu (Data Engineering)
- **Tên đề tài**: Xây dựng data pipeline trích xuất giọng nói và mô tả hình ảnh, lập chỉ mục vector phục vụ tìm kiếm ngữ nghĩa trong kho dữ liệu video
