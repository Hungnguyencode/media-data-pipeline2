# Hướng Dẫn Chạy Demo Hệ Thống

## 1. Mục đích của tài liệu này

Tài liệu này hướng dẫn cách chuẩn bị môi trường, chạy hệ thống, và trình bày demo cho phiên bản **Media Semantic Search v2.2.0**.

Mục tiêu khi demo là thể hiện được các điểm sau:

- ingest video từ local hoặc **YouTube URL**
- pipeline xử lý audio + frame
- semantic search đa phương thức
- inventory video đã index
- metadata nguồn video
- event grouping và nearby speech context
- search mode theo loại truy vấn
- evaluation benchmark

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
- khi demo chính thức, nên chạy **không có `--reload`**
- nếu muốn dọn catalog trước demo, có thể gọi `POST /catalog/sanitize`

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

Kỳ vọng hiện tại:
- toàn bộ test pass

---

## 6. Chạy evaluation trước khi demo

```bash
python evaluation/run_eval.py --reranker=off
```

Điều này hữu ích nếu bạn muốn trình bày thêm phần đánh giá.

Lưu ý:
- `evaluation/benchmark_cases.json` là bộ benchmark nhập thủ công
- `run_eval.py` đọc file đó và chấm tự động
- benchmark này không tự sinh case mới
- report được ghi ra `evaluation/latest_report.json`

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

Nếu thay đổi config sampling:
- nhớ restart backend
- và process lại video nếu cần

Ngoài ra nên kiểm tra retrieval thresholds:

```yaml
retrieval:
  min_score_thresholds:
    action: 0.25
    visual: 0.30
    topic: 0.18
    audio: 0.12
    generic: 0.15
```

---

## 8. Video catalog metadata

Hệ thống dùng file:

```text
data/video_catalog.json
```

để lưu metadata nguồn video.

### Lưu ý quan trọng
- `video_name` phải khớp tên file thật
- nếu là video local mới, pipeline có thể tự tạo catalog entry cơ bản
- nếu ingest bằng YouTube URL, hệ thống sẽ tự cập nhật metadata nguồn
- nếu sửa metadata bằng tay, nên **re-index video** để metadata mới đi vào vector DB
- trước khi push GitHub hoặc demo chính thức, nên dọn các entry pytest/temp path nếu có

---

## 9. Ba cách đưa video vào hệ thống

### Cách 1: Process by YouTube URL
Đây là điểm mới quan trọng của Bản 2.

Luồng:
1. mở tab **Process by YouTube URL**
2. nhập YouTube URL
3. bật `reset_index` nếu muốn làm mới index
4. bấm **Download & Process**

Hệ thống sẽ:
- canonicalize URL
- tải video về `data/raw`
- lấy metadata nguồn
- cập nhật `video_catalog.json`
- chạy pipeline xử lý và index

### Cách 2: Process by Path
Đây là cách ổn định nhất nếu video đã có sẵn local.

Luồng:
1. mở tab **Process by Path**
2. nhập đường dẫn video trên máy backend
3. bật `reset_index` nếu cần
4. bấm **Process Video by Path**

### Cách 3: Upload & Process
Luồng:
1. mở tab **Upload & Process**
2. chọn file video
3. bật `reset_index` nếu cần
4. bấm **Upload & Process**

### Khác biệt thực tế
- `Process by YouTube URL`: tiện nhất để thể hiện nâng cấp của Bản 2
- `Process by Path`: ổn định nhất nếu video đã có sẵn
- `Upload & Process`: tiện khi muốn nạp nhanh file từ máy demo

---

## 10. Các loại video phù hợp để demo

Những nhóm video hợp với pipeline hiện tại:

- talk / TED / presentation
- cooking
- tutorial / how-to
- wildlife / nature
- cinematic vlog / short film
- music/cinematic clip để minh họa visual-heavy retrieval

### Gợi ý thực tế
- `egg.mp4` để demo action ngắn
- video talk/TED để demo topic retrieval
- một video YouTube khác để demo ingest
- video cinematic để test visual search

---

## 11. Luồng demo khuyến nghị

### Demo 1: chứng minh ingest YouTube
1. chạy API
2. chạy Streamlit
3. vào tab **Process by YouTube URL**
4. nhập 1 YouTube URL sạch
5. bấm **Download & Process**
6. chỉ ra:
   - video được tải về `data/raw`
   - metadata nguồn được cập nhật
   - pipeline chạy xong
   - video xuất hiện trong inventory

### Demo 2: chứng minh semantic search
1. vào tab **Search**
2. chọn video vừa ingest hoặc `egg.mp4`
3. thử 2–3 query đẹp nhất
4. giải thích:
   - matched frame
   - timestamp
   - event range
   - nearby speech context
   - score type
   - matched signals
   - ranking explanation
   - source metadata

### Demo 3: chứng minh inventory
1. vào tab **Video Inventory**
2. mở chi tiết video
3. cho thấy:
   - total records
   - content type counts
   - source modality counts
   - language
   - source info
   - time range

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
- total videos
- total records
- content type counts
- source modality counts
- languages
- source info
- time range

---

## 13. Query gợi ý để demo

### 13.1. Với video talk / TED
- `human connection`
- `relationships`
- `happiness`
- `self motivation`
- `winning strategy`
- `motivational speech`

### 13.2. Với video trứng
- `crack egg`
- `separate egg yolk`
- `egg in bowl`

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

## 14. Cách dùng `search_mode`

Hệ thống hiện hỗ trợ:

- `auto`
- `action`
- `visual`
- `topic`
- `audio`

### Gợi ý dùng nhanh
- video trứng / cooking: `action` hoặc `visual`
- TED / talk: `topic`
- query thiên lời nói: `audio`
- chưa chắc thì để `auto`

### Ví dụ request
```json
{
  "query": "crack egg",
  "top_k": 5,
  "content_type": "caption",
  "video_name": "egg.mp4",
  "search_mode": "action"
}
```

---

## 15. Cách chọn content type khi search

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

Nếu chưa chắc, có thể để tất cả rồi quan sát loại kết quả trả về.

---

## 16. Cách đọc kết quả khi demo

Mỗi kết quả thường có:
- `display_text` hoặc `display_caption`
- `nearby_speech_context`
- `similarity_score`
- `distance`
- `score_type`
- `timestamp` hoặc `event time range`
- `content_type`
- `source_modality`
- `search_mode`
- `matched_signals`
- `ranking_explanation`
- `source info`

### Giải thích ngắn khi trình bày
- **Caption/matched text**: mô tả nội dung phù hợp nhất
- **Nearby speech context**: lời nói gần vị trí đó
- **Event range**: khoảng thời gian gần đúng của cụm kết quả
- **Matched signals**: hệ thống match theo action/object/style gì
- **Ranking explanation**: lý do kết quả được đẩy lên hoặc bị phạt
- **Source metadata**: thông tin nguồn gốc video

---

## 17. Những điểm nên nhấn mạnh khi demo

- Hệ thống không chỉ dựa vào transcript
- Mỗi video được xử lý theo cả **audio** và **image**
- Search đang dùng **hybrid retrieval**
- Có **search_mode** để ép kiểu truy vấn khi cần
- Có **event grouping** để gom các frame gần nhau thành một kết quả dễ hiểu hơn
- Kết quả gắn với **metadata nguồn video**
- Bản 2 đã hỗ trợ **YouTube ingest**
- Có benchmark và report tự sinh

---

## 18. Nếu kết quả chưa hoàn hảo

Bạn có thể giải thích như sau:

- Caption là mô tả tự động nên có thể khái quát
- Timestamp và matched frame là bằng chứng trực quan quan trọng hơn
- Với video khác nhau, sampling frame luôn là trade-off giữa temporal precision và temporal coverage
- Với video cinematic / music video, transcript có thể kém đáng tin hơn talk/TED
- Mục tiêu hiện tại là **prototype semantic search**, chưa phải production-grade retrieval cho mọi loại video

---

## 19. Kết quả benchmark hiện tại nên nhớ khi demo

Lần chạy gần nhất với:

```bash
python evaluation/run_eval.py --reranker=off
```

cho kết quả:

- `3/9`
- `action: 2/2`
- `visual: 1/1`
- `hallucination_negative: 0/6`

Query demo đẹp nhất với `egg.mp4` hiện tại:
- `crack egg`
- `separate egg yolk`
- `egg in bowl`

---

## 20. Lưu ý kỹ thuật trước demo

- Lần đầu tải model có thể chậm nếu model chưa cache
- Nếu thay config, nhớ restart backend
- Nếu đổi metadata catalog, nên re-index video
- Không nên đổi model lớn hơn ngay sát giờ demo
- Không nên sửa thêm retrieval ngay sát giờ demo nếu hệ thống hiện đã ổn
- Với YouTube ingest, nên dùng URL sạch dạng:

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

---

## 21. Checklist trước khi demo

- [ ] Tạo và activate virtual environment
- [ ] `pip install -r requirements.txt`
- [ ] `ffmpeg` và `ffprobe` chạy được trong terminal
- [ ] `pytest -v`
- [ ] `python evaluation/run_eval.py --reranker=off`
- [ ] `uvicorn api.main:app --host 127.0.0.1 --port 8000`
- [ ] `streamlit run ui/app.py`
- [ ] kiểm tra `configs/config.yaml`
- [ ] kiểm tra `data/video_catalog.json`
- [ ] nếu catalog có path test/temp, chạy `POST /catalog/sanitize`
- [ ] thử ingest một video YouTube
- [ ] process lại video cần demo
- [ ] kiểm tra inventory
- [ ] thử trước 2–3 query đẹp nhất cho từng video

---

## 22. Luồng demo ngắn gọn nên dùng

1. chạy API
2. chạy Streamlit
3. ingest video từ YouTube URL
4. mở Inventory để xác nhận video đã vào kho
5. search bằng 2–3 query đẹp nhất
6. giải thích:
   - matched frame
   - timestamp
   - nearby speech context
   - hybrid retrieval
   - search mode
   - ranking explanation
   - source metadata

---

## 23. Tóm tắt

Bản hiện tại phù hợp nhất với demo theo hướng:

- ingest video local hoặc YouTube
- extract audio và frame
- tạo transcript + caption + multimodal documents
- index vector
- semantic search
- inventory video
- metadata nguồn video
- evaluation benchmark
- retrieval explanation

Khi demo, nên tập trung vào:
- pipeline rõ ràng
- hybrid retrieval
- timestamp / event grouping
- YouTube ingest của Bản 2
- search mode
- và việc kết nối semantic search với kho video có metadata nguồn
