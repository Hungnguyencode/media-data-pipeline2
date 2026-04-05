# Hướng Dẫn Chạy Demo Hệ Thống

## 1. Mục đích của tài liệu này

Tài liệu này hướng dẫn cách chuẩn bị môi trường, chạy hệ thống, và trình bày demo cho phiên bản **Media Semantic Search v2.2.0**.

Mục tiêu khi demo là thể hiện được các điểm sau:

- ingest video từ local hoặc **YouTube URL**,
- pipeline xử lý audio + frame,
- semantic search đa phương thức,
- inventory video đã index,
- metadata nguồn video,
- event grouping và nearby speech context.

---

## 2. Chuẩn bị môi trường

### 2.1. Tạo virtual environment
```bash
python -m venv venv
```

### 2.2. Kích hoạt môi trường
Windows:
```bash
venv\Scripts\activate
```

### 2.3. Cài dependencies
```bash
pip install -r requirements.txt
```

### 2.4. Cài PyTorch theo môi trường máy
Ví dụ với CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.5. Cài FFmpeg / FFprobe
Đảm bảo cả `ffmpeg` và `ffprobe` chạy được trong terminal.

### 2.6. Cache model trước nếu có thể
Nếu đã từng chạy:
- Whisper
- BLIP
- OpenCLIP
- SentenceTransformers

thì nên giữ cache model sẵn để lúc demo không phải tải lại.

---

## 3. Chạy backend API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Địa chỉ:
- API: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

Lưu ý:
- khi demo chính thức, nên chạy **không có `--reload`** để tránh reload ngoài ý muốn.

---

## 4. Chạy Streamlit UI

Mở terminal khác:

```bash
streamlit run ui/app.py
```

Địa chỉ:
- UI: `http://localhost:8501`

---

## 5. Chạy test trước demo

```bash
pytest -v
```

Mục tiêu:
- đảm bảo test pass trước khi demo,
- xác nhận API, pipeline, retrieval, ingest và transform đang ổn.

Nếu bạn đang dùng đúng bản hiện tại, kỳ vọng là toàn bộ test pass.

---

## 6. Chạy evaluation trước khi demo

```bash
python evaluation/run_eval.py
```

Điều này hữu ích nếu bạn muốn trình bày thêm phần đánh giá.

Lưu ý:
- `evaluation/benchmark_cases.json` là **bộ benchmark nhập thủ công**,
- `run_eval.py` chỉ đọc file đó và chấm tự động,
- benchmark này **không tự sinh case mới**.

---

## 7. Kiểm tra cấu hình demo

File cấu hình chính: `configs/config.yaml`

Cấu hình hiện tại nên dùng:

```yaml
video:
  frame_sampling_fps: 1.0
  max_frames: 240
  max_frame_width: 960
  max_frame_height: 540
```

### Ý nghĩa
- `1.0 fps`: lấy khoảng 1 frame mỗi giây,
- `240 frames`: cho coverage tốt hơn với video dài vừa phải,
- phù hợp để demo nhiều loại video khác nhau.

Nếu thay đổi config sampling:
- nhớ restart backend,
- và process lại video nếu cần.

---

## 8. Video catalog metadata

Hệ thống dùng file:

```text
data/video_catalog.json
```

để lưu metadata nguồn video.

Một entry thường có:

```json
{
  "video_name": "egg.mp4",
  "local_video_path": "data/raw/egg.mp4",
  "source_platform": "youtube",
  "source_url": "https://www.youtube.com/watch?v=...",
  "title": "...",
  "description": "...",
  "thumbnail_url": "...",
  "tags": ["..."],
  "created_at": "2026-03-27T00:00:00",
  "ingested_at": "2026-03-29T15:44:59"
}
```

### Lưu ý quan trọng
- `video_name` phải khớp tên file thật.
- Nếu là video local mới, pipeline có thể tự tạo catalog entry cơ bản.
- Nếu ingest bằng YouTube URL, hệ thống sẽ tự cập nhật metadata nguồn.
- Nếu sửa metadata bằng tay, nên **re-index video** để metadata mới đi vào vector DB.

---

## 9. Ba cách đưa video vào hệ thống

### Cách 1: Process by YouTube URL
Đây là điểm mới quan trọng của Bản 2.

Luồng:
1. mở tab **Process by YouTube URL**,
2. nhập YouTube URL,
3. bật `reset_index` nếu muốn làm mới index,
4. bấm **Download & Process**.

Hệ thống sẽ:
- canonicalize URL,
- tải video về `data/raw`,
- lấy metadata nguồn,
- cập nhật `video_catalog.json`,
- chạy pipeline xử lý và index.

### Cách 2: Process by Path
Đây là cách ổn định nhất nếu video đã có sẵn local.

Luồng:
1. mở tab **Process by Path**,
2. nhập đường dẫn video trên máy backend,
3. bật `reset_index` nếu cần,
4. bấm **Process Video by Path**.

Ví dụ:

```text
C:\Users\Admin\media-data-pipeline2\data\raw\egg.mp4
```

### Cách 3: Upload & Process
Luồng:
1. mở tab **Upload & Process**,
2. chọn file video,
3. bật `reset_index` nếu cần,
4. bấm **Upload & Process**.

### Khác biệt thực tế
- `Process by YouTube URL`: tiện nhất để thể hiện nâng cấp của Bản 2.
- `Process by Path`: ổn định nhất nếu video đã có sẵn.
- `Upload & Process`: tiện khi muốn nạp nhanh file từ máy demo.

---

## 10. Các loại video phù hợp để demo

Những nhóm video hợp với pipeline hiện tại:

- talk / TED / presentation,
- cooking,
- tutorial / how-to,
- wildlife / nature,
- cinematic vlog / short film,
- music/cinematic clip để minh họa visual-heavy retrieval.

### Gợi ý thực tế
- `egg.mp4` để demo action ngắn,
- `ted_happier.mp4` để demo semantic topic retrieval,
- một video TED khác để demo YouTube ingest,
- wildlife / cinematic video để test visual search.

---

## 11. Luồng demo khuyến nghị cho Bản 2

### Demo 1: chứng minh ingest YouTube
1. chạy API,
2. chạy Streamlit,
3. vào tab **Process by YouTube URL**,
4. nhập 1 YouTube URL sạch,
5. bấm **Download & Process**,
6. chỉ ra:
   - video được tải về `data/raw`,
   - metadata nguồn được cập nhật,
   - pipeline chạy xong,
   - video xuất hiện trong inventory.

### Demo 2: chứng minh semantic search
1. vào tab **Search**,
2. chọn video vừa ingest,
3. thử 2–3 query đẹp nhất,
4. giải thích:
   - matched frame,
   - timestamp,
   - event range,
   - nearby speech context,
   - score type,
   - source metadata.

### Demo 3: chứng minh inventory
1. vào tab **Video Inventory**,
2. mở chi tiết video,
3. cho thấy:
   - total records,
   - content type counts,
   - source modality counts,
   - language,
   - source info,
   - time range.

---

## 12. Những gì cần kiểm tra sau khi process xong

### 12.1. Dữ liệu trung gian
- `data/interim/audio/`
- `data/interim/frames/`
- `data/interim/transcripts/`
- `data/interim/captions/`

### 12.2. Dữ liệu đã xử lý
- `data/processed/`

### 12.3. Vector database
- `data/vector_db/`

### 12.4. Inventory
Vào tab **Video Inventory** hoặc dùng API để kiểm tra:
- total videos,
- total records,
- content type counts,
- source modality counts,
- languages,
- source info,
- time range.

---

## 13. Query gợi ý để demo

### 13.1. Với video talk / TED
- `human connection`
- `relationships`
- `happiness`
- `self motivation`
- `winning strategy`
- `motivational speech`

### 13.2. Với video cooking / trứng
- `crack egg`
- `separate egg`
- `egg yolk`
- `egg white`
- `tách trứng`
- `lòng đỏ trứng`
- `lòng trắng trứng`

### 13.3. Với wildlife / nature
- `bird flying`
- `forest animals`
- `wildlife diversity`
- `fish swimming`

### 13.4. Với cinematic / vlog
- `summer sky`
- `car on road`
- `beach scene`
- `person walking outdoors`

---

## 14. Cách chọn content type khi search

### Query chủ đề / ý nghĩa
Ưu tiên:
- `segment_chunk`
- `multimodal`

Ví dụ:
- `self motivation`
- `winning strategy`
- `happiness`
- `relationships`

### Query thiên về hình ảnh / hành động / cảnh vật
Ưu tiên:
- `caption`
- `multimodal`

Ví dụ:
- `crack egg`
- `bird flying`
- `car on road`

### Query thiên về lời nói
Ưu tiên:
- `transcription`
- `segment_chunk`

Nếu chưa chắc, có thể để tất cả rồi quan sát loại kết quả trả về.

---

## 15. Cách đọc kết quả khi demo

Mỗi kết quả thường có:
- `display_text` hoặc `display_caption`,
- `nearby_speech_context`,
- `similarity_score`,
- `distance`,
- `score_type`,
- `timestamp` hoặc `event time range`,
- `content_type`,
- `source_modality`,
- `source info`.

### Giải thích ngắn khi trình bày
- **Caption/matched text**: mô tả nội dung phù hợp nhất,
- **Nearby speech context**: lời nói gần vị trí đó,
- **Similarity score**: điểm tương đối từ retrieval,
- **Event range**: khoảng thời gian gần đúng của cụm kết quả,
- **Source metadata**: thông tin nguồn gốc video.

---

## 16. Những điểm nên nhấn mạnh khi demo

- Hệ thống không chỉ dựa vào transcript.
- Mỗi video được xử lý theo cả **audio** và **image**.
- Search đang dùng **hybrid retrieval**:
  - text embedding,
  - CLIP text-image,
  - soft rerank.
- Hệ thống có **event grouping** để gom các frame gần nhau thành một kết quả dễ hiểu hơn.
- Kết quả gắn với **metadata nguồn video** để hỗ trợ truy vết.
- Bản 2 đã hỗ trợ **YouTube ingest**, giảm thao tác thủ công khi đưa video vào hệ thống.

---

## 17. Nếu kết quả chưa hoàn hảo

Bạn có thể giải thích như sau:

- Caption là mô tả tự động nên có thể khái quát.
- Timestamp và matched frame là bằng chứng trực quan quan trọng hơn.
- Với video khác nhau, sampling frame luôn là trade-off giữa:
  - temporal precision,
  - temporal coverage.
- Với video cinematic / music video, transcript có thể kém đáng tin hơn talk/TED.
- Mục tiêu hiện tại là **prototype semantic search**, chưa phải production-grade retrieval cho mọi loại video.

---

## 18. Lưu ý kỹ thuật trước demo

- Lần đầu tải model có thể chậm nếu model chưa cache.
- Nếu thay config, nhớ restart backend.
- Nếu đổi metadata catalog, nên re-index video.
- Không nên đổi model lớn hơn ngay sát giờ demo.
- Không nên sửa thêm retrieval ngay sát giờ demo nếu hệ thống hiện đã ổn.
- Với YouTube ingest, nên dùng URL sạch dạng:

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

---

## 19. Checklist trước khi demo

- [ ] Tạo và activate virtual environment
- [ ] `pip install -r requirements.txt`
- [ ] `ffmpeg` và `ffprobe` chạy được trong terminal
- [ ] `pytest -v`
- [ ] `python evaluation/run_eval.py`
- [ ] `uvicorn api.main:app --host 127.0.0.1 --port 8000`
- [ ] `streamlit run ui/app.py`
- [ ] kiểm tra `configs/config.yaml`
- [ ] kiểm tra `data/video_catalog.json`
- [ ] thử ingest một video YouTube
- [ ] process lại video cần demo
- [ ] kiểm tra inventory
- [ ] thử trước 2–3 query đẹp nhất cho từng video

---

## 20. Luồng demo ngắn gọn nên dùng

1. chạy API,
2. chạy Streamlit,
3. ingest video từ YouTube URL,
4. mở Inventory để xác nhận video đã vào kho,
5. search bằng 2–3 query đẹp nhất,
6. giải thích:
   - matched frame,
   - timestamp,
   - nearby speech context,
   - hybrid retrieval,
   - source metadata.

---

## 21. Query ưu tiên nếu cần demo nhanh

### Video TED / motivation
- `self motivation`
- `winning strategy`
- `motivational speech`

### Video trứng
- `crack egg`
- `separate egg`
- `egg yolk`

### Video wildlife
- `bird flying`
- `wildlife diversity`

### Video cinematic
- `summer sky`
- `car on road`
- `beach scene`

---

## 22. Tóm tắt

Bản hiện tại phù hợp nhất với demo theo hướng:

- ingest video local hoặc YouTube,
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
- YouTube ingest của Bản 2,
- và việc kết nối semantic search với kho video có metadata nguồn.
