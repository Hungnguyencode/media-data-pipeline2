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

## 5. Video demo hiện tại

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

## 6. Luồng demo khuyến nghị

### Cách 1: Upload & Process
- Mở tab **Upload & Process**
- Chọn file video
- Bấm **Upload & Process**
- Chờ pipeline chạy xong

### Cách 2: Process by Path
- Mở tab **Process by Path**
- Nhập đường dẫn video
- Bấm **Process Video by Path**

> Nếu muốn làm sạch dữ liệu cũ trước khi chạy lại video, bật tùy chọn **reset index**.

---

## 7. Cấu hình demo nên dùng

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

## 8. Kiểm tra kết quả pipeline

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

---

## 9. Kiểm tra inventory

Trong UI:
- mở tab **Video Inventory**
- kiểm tra:
  - tổng số video đã index
  - số lượng records theo từng loại
  - time range
  - ngôn ngữ
  - source modality

Ví dụ với một video cooking đã xử lý thành công, inventory có thể hiển thị:
- `transcription`
- `segment_chunk`
- `caption`
- `multimodal`

---

## 10. Truy vấn demo gợi ý

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

## 11. Gợi ý chọn loại nội dung khi search

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

## 12. Kết quả mong đợi khi demo

- upload/process thành công
- inventory hiển thị video đã index
- search trả về:
  - matched frame description
  - auto-caption
  - timestamp
  - content type
  - modality
  - similarity proxy
  - nearby speech context (nếu có)

Trong các video có hành động ngắn như `crack egg`, kết quả tốt thường rơi vào các frame gần nhau, ví dụ quanh `0:04–0:05`.

---

## 13. Cách giải thích kết quả khi demo

### Điều nên nhấn mạnh
- Hệ thống không chỉ dùng transcript mà còn dùng cả hình ảnh
- Mỗi frame được xử lý bằng:
  - BLIP captioning
  - CLIP image embedding
- Search là **hybrid retrieval**
- Kết quả hiển thị là frame phù hợp nhất hoặc nhóm frame gần nhau

### Nếu caption chưa hoàn hảo
Có thể giải thích:
- caption là mô tả tự động tham khảo
- matched frame và timestamp mới là bằng chứng trực quan quan trọng hơn
- BLIP có thể mô tả hơi khái quát, nhưng retrieval vẫn tìm đúng ngữ cảnh

---

## 14. Lưu ý kỹ thuật

- Lần đầu tải CLIP model có thể chậm nếu mạng yếu
- Sau khi model đã cache thành công, các lần chạy sau sẽ ổn định hơn nhiều
- Hệ thống hiện phù hợp cho demo/prototype/local processing, chưa nhắm tới production-scale
- Với phần cứng như RTX 2050 4GB, nên giữ:
  - **BLIP image-captioning-base**
  - **CLIP ViT-B-32**
  - không nên nâng sang model caption lớn hơn lúc demo

---

## 15. Checklist trước khi demo

- [ ] `pip install -r requirements.txt`
- [ ] `ffmpeg` chạy được trong terminal
- [ ] `pytest -v` pass
- [ ] `uvicorn api.main:app --host 127.0.0.1 --port 8000`
- [ ] `streamlit run ui/app.py`
- [ ] video demo đã có trong `data/raw/` hoặc sẵn sàng upload
- [ ] đã xử lý xong ít nhất 1 video mẫu
- [ ] đã thử trước vài query đẹp nhất

---

## 16. Query nên ưu tiên khi demo bản hiện tại

Nếu demo video trứng:
- `crack egg`
- `separate egg`
- `egg yolk`

Trong đó `crack egg` là query đang cho kết quả đẹp và trực quan nhất trong bản hiện tại.

---

## 17. Tóm tắt luồng demo ngắn

1. chạy API
2. chạy Streamlit
3. upload video hoặc process by path
4. đợi pipeline xử lý hoàn tất
5. kiểm tra inventory
6. search bằng 2–3 query đẹp nhất
7. giải thích kết quả retrieval, timestamp và matched frame
