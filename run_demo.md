# Run Demo Guide

## 1. Chuẩn bị môi trường

### Tạo và kích hoạt virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Cài dependencies
```bash
pip install -r requirements.txt
```

### Cài PyTorch theo môi trường máy
Ví dụ với máy có CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Cài FFmpeg
Đảm bảo máy chạy đã cài **FFmpeg** và lệnh `ffmpeg` có thể chạy từ terminal.

---

## 2. Chạy backend API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

API chạy tại:
- http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

> Khuyến nghị: không dùng `--reload` khi demo để tránh reload ngoài ý muốn và đỡ phát sinh lỗi vặt.

---

## 3. Chạy Streamlit UI

Mở terminal khác:

```bash
streamlit run ui/app.py
```

UI chạy tại:
- http://localhost:8501

---

## 4. Chạy test trước demo

```bash
pytest -v
```

Kết quả mong đợi:
- **33 passed**

---

## 5. Chuẩn bị video catalog metadata

Phiên bản hiện tại có thêm lớp metadata nguồn video thông qua file:

```text
data/video_catalog.json
```

Bạn nên kiểm tra trước khi demo:

- `video_name` phải khớp đúng tên file video
- `source_platform` nên là `youtube`, `facebook`, `tiktok` hoặc `local`
- `source_url` nên là link thật của video gốc
- `title` nên là tiêu đề video
- `description` nên là mô tả ngắn
- `tags` nên là các từ khóa ngắn

Ví dụ:

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

> Lưu ý: sau khi sửa `video_catalog.json`, cần **process lại video với reset index** để metadata mới được đưa vào vector DB.

---

## 6. Video demo hiện tại

Video gợi ý:
- `egg.mp4`
- `egg_1.mp4`
- hoặc video demo khác đặt trong `data/raw/`

Video cooking với hành động rõ như:
- crack egg
- separate egg
- egg yolk / egg white

là nhóm video rất phù hợp để minh họa phần BLIP + CLIP retrieval.

---

## 7. Luồng demo khuyến nghị

### Cách 1: Process by Path
Đây là cách ổn định nhất khi demo vì giữ đúng tên file và khớp tốt với catalog metadata.

- Mở tab **Process by Path**
- Nhập đường dẫn video, ví dụ:
  ```text
  C:\Users\Admin\media-data-pipeline2\data\raw\egg.mp4
  ```
- Bật **reset index**
- Bấm **Process Video by Path**
- Chờ pipeline chạy xong

### Cách 2: Upload & Process
- Mở tab **Upload & Process**
- Chọn file video
- Bấm **Upload & Process**
- Chờ pipeline chạy xong

> Lưu ý:
> - Nếu upload lại cùng một file nhiều lần, backend có thể đổi tên thành `egg_1.mp4`, `egg_2.mp4`, ...
> - Nếu tên file thay đổi mà catalog chưa có entry tương ứng, metadata nguồn có thể rơi về mặc định kiểu `local`.
> - Vì vậy khi demo phần catalog metadata, nên ưu tiên **Process by Path**.

---

## 8. Cấu hình demo nên dùng

Cấu hình đã được kiểm nghiệm ổn cho demo hiện tại:

```yaml
video:
  frame_sampling_fps: 0.75
  max_frames: 180
```

### Ý nghĩa
- `0.75 fps` giúp lấy frame dày hơn, cải thiện độ chính xác thời gian
- `180 frames` đủ để demo tốt hơn so với cấu hình test nhẹ

### Cấu hình test nhanh khi debug
Nếu chỉ muốn kiểm tra pipeline nhanh:

```yaml
video:
  frame_sampling_fps: 0.3
  max_frames: 60
```

---

## 9. Kiểm tra kết quả pipeline

Sau khi xử lý xong, hệ thống sẽ sinh ra:

### Dữ liệu trung gian
- `data/interim/audio/`
- `data/interim/frames/`
- `data/interim/transcripts/`
- `data/interim/captions/`

### Dữ liệu đã xử lý
- `data/processed/`

### Vector database
- `data/vector_db/`

### Catalog metadata
- metadata nguồn được đọc từ `data/video_catalog.json`
- metadata này được gắn vào:
  - `video_source_info` trong kết quả pipeline
  - metadata của các records trong vector DB
  - inventory
  - search result

---

## 10. Kiểm tra inventory

Trong UI:
- mở tab **Video Inventory**
- kiểm tra:
  - tổng số video đã index
  - số lượng records theo từng loại
  - time range
  - ngôn ngữ
  - source modality
  - source platform
  - source URL
  - video title
  - description
  - tags

Ví dụ với một video cooking đã xử lý thành công, inventory có thể hiển thị:
- `transcription`
- `segment_chunk`
- `caption`
- `multimodal`
- metadata nguồn video từ YouTube

---

## 11. Truy vấn demo gợi ý

### Query tiếng Anh cho video cooking
- `crack egg`
- `separate egg`
- `egg yolk`
- `egg white`

### Query tiếng Việt cho video cooking
- `tách trứng`
- `lòng đỏ trứng`
- `lòng trắng trứng`

### Query tổng quát hơn
- `person cooking`
- `mixing ingredients`
- `bowl on table`

### Query cho video speech / talk
- `human connection`
- `feeling connected and loved`
- `what makes people happier`
- `relationships`
- `happiness`
- `kết nối con người`
- `mối quan hệ với người khác`
- `hạnh phúc đến từ đâu`

---

## 12. Gợi ý chọn loại nội dung khi search

- Query chủ đề / ý nghĩa:
  - `segment_chunk`
  - `multimodal`

- Query thiên hình ảnh / khung cảnh / hành động:
  - `caption`
  - `multimodal`

- Muốn tìm chi tiết lời nói:
  - `transcription`
  - `segment_chunk`

---

## 13. Kết quả mong đợi khi demo

- process video thành công
- inventory hiển thị video đã index
- search trả về:
  - matched frame description
  - auto-caption
  - timestamp
  - content type
  - modality
  - similarity proxy
  - nearby speech context (nếu có)
  - source platform
  - source URL
  - video title
  - description
  - tags

Trong các video có hành động ngắn như `crack egg`, kết quả tốt thường rơi vào các frame gần nhau, ví dụ quanh `0:04–0:05`.

---

## 14. Cách giải thích kết quả khi demo

### Điều nên nhấn mạnh
- Hệ thống không chỉ dùng transcript mà còn dùng cả hình ảnh
- Mỗi frame được xử lý bằng:
  - BLIP captioning
  - CLIP image embedding
- Search là **hybrid retrieval**
- Kết quả hiển thị là frame phù hợp nhất hoặc nhóm frame gần nhau
- Kết quả còn gắn với **nguồn video cụ thể** thông qua catalog metadata

### Nếu caption chưa hoàn hảo
Có thể giải thích:
- caption là mô tả tự động tham khảo
- matched frame và timestamp mới là bằng chứng trực quan quan trọng hơn
- BLIP có thể mô tả hơi khái quát, nhưng retrieval vẫn tìm đúng ngữ cảnh

### Nếu bị hỏi về catalog metadata
Có thể giải thích:
- catalog hiện là lớp metadata prototype của kho video
- mục tiêu là liên kết kết quả semantic search với nguồn video cụ thể
- với quy mô lớn hơn, hệ thống có thể mở rộng sang cơ chế sinh catalog cơ bản tự động hoặc ingest metadata bán tự động

---

## 15. Lưu ý kỹ thuật

- Lần đầu tải CLIP model có thể chậm nếu mạng yếu
- Sau khi model đã cache thành công, các lần chạy sau sẽ ổn định hơn nhiều
- Hệ thống hiện phù hợp cho demo/prototype/local processing, chưa nhắm tới production-scale
- Với phần cứng như RTX 2050 4GB, nên giữ:
  - **BLIP image-captioning-base**
  - **CLIP ViT-B-32**
  - không nên nâng sang model caption lớn hơn lúc demo
- Nếu sửa `video_catalog.json`, nhớ **re-index video** để metadata mới được cập nhật vào kết quả

---

## 16. Checklist trước khi demo

- [ ] `pip install -r requirements.txt`
- [ ] `ffmpeg` chạy được trong terminal
- [ ] `pytest -v` pass
- [ ] `uvicorn api.main:app --host 127.0.0.1 --port 8000`
- [ ] `streamlit run ui/app.py`
- [ ] video demo đã có trong `data/raw/`
- [ ] `data/video_catalog.json` đã có metadata đúng cho video demo
- [ ] đã process lại video sau khi sửa catalog
- [ ] đã thử trước vài query đẹp nhất

---

## 17. Query nên ưu tiên khi demo bản hiện tại

Nếu demo video trứng:
- `crack egg`
- `separate egg`
- `egg yolk`

Trong đó `crack egg` là query đang cho kết quả đẹp và trực quan nhất trong bản hiện tại.

---

## 18. Tóm tắt luồng demo ngắn

1. chạy API
2. chạy Streamlit
3. kiểm tra `video_catalog.json`
4. process video bằng **Process by Path**
5. đợi pipeline xử lý hoàn tất
6. kiểm tra inventory
7. search bằng 2–3 query đẹp nhất
8. giải thích kết quả retrieval, timestamp, matched frame và metadata nguồn video
