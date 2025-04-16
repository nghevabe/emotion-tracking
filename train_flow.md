# Luồng code trong train.py

## 1. Import thư viện
- Import các thư viện cần thiết: tensorflow, keras, os
- Sử dụng MobileNetV2 làm model base
- Sử dụng ImageDataGenerator để tăng cường dữ liệu

## 2. Chuẩn bị dữ liệu
- Định nghĩa đường dẫn dataset: `datasets/FER2013`
- Tạo ImageDataGenerator với các tham số:
  - Rescale: chuẩn hóa giá trị pixel về khoảng [0,1]
  - Rotation range: xoay ảnh ngẫu nhiên trong khoảng ±30 độ
  - Zoom range: phóng to/thu nhỏ ảnh ngẫu nhiên
  - Horizontal flip: lật ảnh ngang
  - Validation split: chia 80% train, 20% validation

## 3. Load dữ liệu
- Load dữ liệu training và validation từ thư mục
- Kích thước ảnh đầu vào: 224x224
- Batch size: 32
- Chế độ phân loại: categorical

## 4. Xây dựng model
- Load MobileNetV2 pre-trained trên ImageNet
- Mở khóa 30 layer cuối để fine-tune
- Thêm các lớp mới:
  - GlobalAveragePooling2D
  - Dense(256) với activation ReLU
  - Dropout(0.5)
  - Dense(128) với activation ReLU
  - Dense(7) với activation Softmax (7 lớp cảm xúc)

## 5. Compile và Train
- Compile model với:
  - Optimizer: Adam với learning rate 0.0001
  - Loss function: categorical_crossentropy
  - Metrics: accuracy
- Train model trong 20 epochs
- Batch size: 32

## 6. Lưu model
- Lưu model đã train vào file `emotion_mobilenetv2_finetuned.h5` 