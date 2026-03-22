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

## 2. Chạy backend API

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

API chạy tại:
- http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

## 3. Chạy Streamlit UI

Mở terminal khác:

```bash
streamlit run ui/app.py
```

UI chạy tại:
- http://localhost:8501

## 4. Video demo hiện tại

Video gợi ý:
- `ted_happier.mp4`
- TED Talk: *1 thing you can do today to be happier*

## 5. Luồng demo khuyến nghị

### Cách 1: Process by Path
- Mở tab **Process by Path**
- Nhập đường dẫn video
- Bấm **Process Video by Path**

### Cách 2: Upload & Process
- Mở tab **Upload & Process**
- Chọn file video
- Bấm **Upload & Process**

## 6. Kiểm tra kết quả pipeline

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

## 7. Kiểm tra inventory

Trong UI:
- mở tab **Video Inventory**
- kiểm tra:
  - tổng số video đã index
  - số lượng records theo từng loại
  - time range
  - ngôn ngữ
  - source modality

## 8. Truy vấn demo gợi ý

### Truy vấn tiếng Anh
- `human connection`
- `feeling connected and loved`
- `what makes people happier`
- `relationships`
- `happiness`

### Truy vấn tiếng Việt
- `kết nối con người`
- `mối quan hệ với người khác`
- `điều gì khiến con người hạnh phúc hơn`
- `cảm giác được kết nối và yêu thương`
- `hạnh phúc đến từ đâu`

## 9. Gợi ý chọn loại nội dung khi search

- Query chủ đề / ý nghĩa:
  - `segment_chunk`
  - `multimodal`

- Query thiên hình ảnh / khung cảnh:
  - `caption`
  - `multimodal`

## 10. Kết quả mong đợi khi demo

- upload/process thành công
- inventory hiển thị video đã index
- search trả về:
  - nội dung liên quan
  - timestamp
  - start/end time
  - content type
  - modality
  - similarity proxy

## 11. Lưu ý

- Sau khi thêm dedup caption, số lượng caption và multimodal record có thể giảm nhẹ.
- Đây là hành vi đúng mong đợi, giúp giảm dữ liệu trùng và làm kết quả search gọn hơn.
- Hệ thống hiện phù hợp cho demo/prototype/local processing, chưa nhắm tới production-scale.
