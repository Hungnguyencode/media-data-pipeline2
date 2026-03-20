# Run Demo

## 1. Tạo môi trường ảo
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

## 2. Cài dependencies
```powershell
pip install -r requirements.txt
```

## 3. Cài PyTorch riêng
```powershell
pip install torch torchvision torchaudio
```

## 4. Kiểm tra FFmpeg
```powershell
ffmpeg -version
```

## 5. Chạy backend API
```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## 6. Chạy UI
```powershell
streamlit run ui/app.py
```

## 7. Chạy test
```powershell
pytest -v
```

## 8. Luồng demo đề xuất
- Mở giao diện Streamlit
- Upload một video mẫu
- Chờ pipeline xử lý xong
- Thử các truy vấn:
  - đoạn nói về trí tuệ nhân tạo
  - cảnh có người đang thuyết trình
  - video có slide trên màn hình
  - introduction to artificial intelligence

## 9. Ghi chú
- Project không sử dụng Makefile; các lệnh được chạy trực tiếp trên Windows.
- Nếu máy không có GPU, vẫn có thể chạy bằng CPU nhưng sẽ chậm hơn.
- Nếu gặp lỗi model, kiểm tra lại bước cài PyTorch và FFmpeg trước.
