# Hệ thống nhận diện cảm xúc khuôn mặt

Dự án sử dụng MobileNetV2 để nhận diện 7 loại cảm xúc cơ bản trên khuôn mặt người: Angry, Disgust, Fear, Happy, Neutral, Sad, và Surprise.

## Tính năng

- Nhận diện cảm xúc từ webcam hoặc video
- Hiển thị xác suất cho từng loại cảm xúc
- Hỗ trợ nhận diện nhiều khuôn mặt trong cùng một frame
- Hiển thị FPS để theo dõi hiệu suất
- Giao diện trực quan với khung khuôn mặt và nhãn cảm xúc

## Yêu cầu hệ thống

- Python 3.8+
- Webcam hoặc camera
- GPU (khuyến nghị) để tăng tốc độ xử lý

## Cài đặt

1. Clone repository:
```bash
git clone [repository-url]
cd emotion-tracking
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Sử dụng

1. Chạy chương trình:
```bash
python mobileNetProject/run.py
```

2. Điều khiển:
- Nhấn 'q' để thoát chương trình
- Đảm bảo khuôn mặt được chiếu sáng tốt để tăng độ chính xác
- Giữ khoảng cách vừa phải với camera (không quá gần hoặc quá xa)

## Cấu trúc dự án

```
emotion-tracking/
├── mobileNetProject/
│   ├── run.py              # Chương trình chính để nhận diện cảm xúc
│   ├── train.py            # Script huấn luyện model
│   └── emotion_mobilenetv2_finetuned.h5  # Model đã được huấn luyện
├── requirements.txt        # Danh sách các thư viện cần thiết
└── README.md              # Tài liệu hướng dẫn
```

## Các thư viện chính

- TensorFlow: Xử lý deep learning
- OpenCV: Xử lý ảnh và video
- NumPy: Xử lý dữ liệu số
- Keras: API deep learning

## Tối ưu hóa

- Sử dụng MobileNetV2 làm backbone để tăng tốc độ xử lý
- Tối ưu hóa quá trình tiền xử lý ảnh
- Xử lý đa luồng cho việc nhận diện nhiều khuôn mặt
- Hiển thị FPS để theo dõi hiệu suất

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request để đóng góp vào dự án.

## Giấy phép

MIT License
