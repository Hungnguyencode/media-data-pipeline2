# Run Demo Guide

## 1. Chuẩn bị môi trường

### 1.1. Tạo virtual environment
```bash
python -m venv venv
```

### 1.2. Kích hoạt môi trường
Windows:
```bash
venv\Scripts\activate
```

### 1.3. Cài dependencies
```bash
pip install -r requirements.txt
```

### 1.4. Cài PyTorch theo môi trường máy
Ví dụ với CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1.5. Cài FFmpeg
Đảm bảo lệnh `ffmpeg` chạy được trong terminal.

---

## 2. Chạy backend API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

API:
- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

> Khi demo chính thức, nên chạy không có `--reload` để tránh reload ngoài ý muốn.

---

## 3. Chạy Streamlit UI

Mở terminal khác:

```bash
streamlit run ui/app.py
```

UI:
- `http://localhost:8501`

---

## 4. Chạy test trước demo

```bash
pytest -v
```

Mục tiêu:
- đảm bảo test pass trước khi demo,
- xác nhận API, pipeline, retrieval và transform đang ổn.

---

## 5. Kiểm tra cấu hình demo

File cấu hình chính: `configs/config.yaml`

Cấu hình hiện tại nên dùng:

```yaml
video:
  frame_sampling_fps: 1.0
  max_frames: 240
  max_frame_width: 960
  max_frame_height: 540
```

Đây là cấu hình cân bằng để demo nhiều loại video khác nhau.

### Ý nghĩa
- `1.0 fps`: lấy khoảng 1 frame mỗi giây
- `240 frames`: cho coverage tốt hơn video dài vừa phải
- phù hợp để dùng chung cho talk, cooking, wildlife, tutorial ở mức demo

> Nếu thay đổi config sampling, nhớ restart backend và process lại video.

---

## 6. Video catalog metadata

Hệ thống dùng file:

```text
data/video_catalog.json
```

để lưu metadata nguồn video.

Một entry điển hình:

```json
{
  "video_name": "egg.mp4",
  "local_video_path": "data/raw/egg.mp4",
  "source_platform": "youtube",
  "source_url": "https://www.youtube.com/watch?v=zG9qOl0pCjU&t=183s",
  "title": "5 Minutes, 2 Eggs! Quick and Easy Fluffy Souffle Omelette Recipe!",
  "description": "Cooking tutorial demonstrating how to crack and separate eggs for baking.",
  "thumbnail_url": "",
  "tags": ["cooking", "egg", "tutorial"],
  "created_at": "2026-03-27T00:00:00",
  "ingested_at": "2026-03-27T00:00:00"
}
```

### Lưu ý quan trọng
- `video_name` phải khớp tên file thật.
- Nếu video local mới chưa có entry, pipeline hiện tại có thể **tự tạo catalog entry cơ bản**.
- Nếu cậu sửa metadata nguồn bằng tay, nên **process lại video với reset index** để metadata mới đi vào vector DB.

---

## 7. Hai cách đưa video vào hệ thống

### Cách 1: Process by Path
Đây là cách ổn định nhất khi demo.

Luồng:
1. mở tab **Process by Path**
2. nhập đường dẫn video trên máy backend
3. bật `reset_index`
4. bấm **Process Video by Path**

Ví dụ:

```text
C:\Users\Admin\media-data-pipeline2\data\raw\egg.mp4
```

### Cách 2: Upload & Process
Luồng:
1. mở tab **Upload & Process**
2. chọn file video
3. bật `reset_index` nếu cần
4. bấm **Upload & Process**

### Khác biệt thực tế
- `Process by Path` thường dễ kiểm soát tên file hơn.
- `Upload & Process` hiện lưu file vào `data/raw/` theo đúng tên file upload.
- Nếu upload trùng tên file đã có, backend sẽ **ghi đè file cũ** rồi process lại.

---

## 8. Các loại video phù hợp để demo

Những nhóm video khá hợp với pipeline hiện tại:

- talk / TED / presentation
- cooking
- tutorial / how-to
- DIY / hands-on
- wildlife / nature
- fitness / yoga
- product review / unboxing

### Gợi ý thực tế
- `egg.mp4` để demo action ngắn
- `ted_happier.mp4` để demo semantic topic retrieval
- video wildlife / tutorial khác để test độ đa dạng

---

## 9. Luồng demo khuyến nghị

### Bước 1
Chạy API.

### Bước 2
Chạy Streamlit.

### Bước 3
Kiểm tra `data/video_catalog.json`.

### Bước 4
Process video bằng **Process by Path** hoặc **Upload & Process**.

### Bước 5
Đợi pipeline hoàn tất.

### Bước 6
Vào tab **Video Inventory** để xác nhận:
- video đã được index,
- record counts đã có,
- metadata nguồn hiển thị đúng.

### Bước 7
Vào tab **Search** để demo 2–3 query đẹp nhất.

---

## 10. Những gì cần kiểm tra sau khi process xong

### 10.1. Dữ liệu trung gian
- `data/interim/audio/`
- `data/interim/frames/`
- `data/interim/transcripts/`
- `data/interim/captions/`

### 10.2. Dữ liệu đã xử lý
- `data/processed/`

### 10.3. Vector database
- `data/vector_db/`

### 10.4. Inventory
- vào tab **Video Inventory**
- kiểm tra:
  - total videos
  - total records
  - content type counts
  - source modality counts
  - language
  - source info
  - time range

---

## 11. Query gợi ý để demo

### 11.1. Với video talk / TED
- `human connection`
- `feeling connected and loved`
- `what makes people happier`
- `relationships`
- `happiness`
- `kết nối con người`
- `mối quan hệ với người khác`
- `hạnh phúc đến từ đâu`

### 11.2. Với video trứng / cooking
- `crack egg`
- `separate egg`
- `egg yolk`
- `egg white`
- `tách trứng`
- `lòng đỏ trứng`
- `lòng trắng trứng`

### 11.3. Với wildlife / nature
- `bird flying`
- `fish swimming`
- `animals hunting`
- `wildlife diversity`
- `forest animals`
- `underwater animals`

### 11.4. Với tutorial / action video
- `cutting`
- `mixing ingredients`
- `holding object`
- `person using tools`
- `fold paper`

---

## 12. Chọn content type khi search

### Query chủ đề / ý nghĩa
Ưu tiên:
- `segment_chunk`
- `multimodal`

### Query thiên về hình ảnh / hành động / cảnh vật
Ưu tiên:
- `caption`
- `multimodal`

### Query thiên về lời nói
Ưu tiên:
- `transcription`
- `segment_chunk`

> Nếu chưa chắc, để `Tất cả` rồi quan sát loại kết quả trả về.

---

## 13. Cách đọc kết quả khi demo

Mỗi kết quả thường có:
- `Matched frame description`
- `Auto-caption`
- `Nearby speech context`
- `Similarity proxy`
- `Distance`
- `Score type`
- `timestamp`
- `event time range`
- `content_type`
- `source_modality`
- `source info`

### Giải thích ngắn gọn khi trình bày
- **Matched frame description**: mô tả frame phù hợp nhất
- **Auto-caption**: caption tự động, chỉ mang tính tham khảo
- **Nearby speech context**: speech gần mốc thời gian đó
- **Similarity proxy**: điểm tương đối từ kết quả retrieval
- **Event range**: khoảng thời gian gần đúng của cụm frame phù hợp

---

## 14. Những điểm nên nhấn mạnh khi demo

- Hệ thống không chỉ dựa vào transcript.
- Mỗi video được xử lý theo cả **audio** và **image**.
- Search đang dùng **hybrid retrieval**:
  - text embedding
  - CLIP text-image
- Hệ thống có **event grouping** để gom các frame gần nhau thành một kết quả dễ hiểu hơn.
- Kết quả còn gắn với **metadata nguồn video** để hỗ trợ truy vết.

---

## 15. Nếu kết quả chưa hoàn hảo

Có thể giải thích như sau:

- Caption là mô tả tự động nên có thể khái quát.
- Timestamp và matched frame là bằng chứng trực quan quan trọng hơn.
- Với video khác nhau, sampling frame luôn là trade-off giữa:
  - temporal precision,
  - và temporal coverage.
- Mục tiêu hiện tại là prototype semantic search, chưa phải production-grade retrieval cho mọi loại video.

---

## 16. Lưu ý kỹ thuật trước demo

- Lần đầu tải model có thể chậm nếu model chưa cache.
- Nếu thay config, nhớ restart backend.
- Nếu đổi metadata catalog, nên re-index video.
- Không nên đổi model lớn hơn ngay trước demo.
- Không nên sửa thêm code retrieval ngay sát giờ demo nếu hệ thống hiện đã chạy ổn.

---

## 17. Checklist trước khi demo

- [ ] Tạo và activate virtual environment
- [ ] `pip install -r requirements.txt`
- [ ] `ffmpeg` chạy được trong terminal
- [ ] `pytest -v`
- [ ] `uvicorn api.main:app --host 127.0.0.1 --port 8000`
- [ ] `streamlit run ui/app.py`
- [ ] kiểm tra `configs/config.yaml`
- [ ] kiểm tra `data/video_catalog.json`
- [ ] process lại video cần demo
- [ ] kiểm tra inventory
- [ ] thử trước 2–3 query đẹp nhất cho từng video

---

## 18. Luồng demo ngắn gọn nên dùng

1. chạy API  
2. chạy Streamlit  
3. process video  
4. mở Inventory để xác nhận video đã vào kho  
5. search bằng 2–3 query đẹp nhất  
6. giải thích:
   - matched frame,
   - timestamp,
   - nearby speech context,
   - hybrid retrieval,
   - source metadata

---

## 19. Query ưu tiên nếu cần demo nhanh

### Video trứng
- `crack egg`
- `separate egg`
- `egg yolk`

### Video TED
- `human connection`
- `happiness`
- `relationships`

### Video wildlife
- `bird flying`
- `fish swimming`
- `wildlife diversity`

---

## 20. Tóm tắt

Bản hiện tại phù hợp nhất với demo theo hướng:

- ingest video,
- extract audio và frame,
- tạo transcript + caption + multimodal documents,
- index vector,
- semantic search,
- inventory video,
- và hiển thị metadata nguồn video.

Khi demo, nên tập trung vào:
- pipeline rõ ràng,
- hybrid retrieval,
- timestamp / event grouping,
- và việc kết nối semantic search với kho video có metadata nguồn.
