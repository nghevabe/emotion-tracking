# Change Log - Cải tiến Model Emotion Detection

## 1. Tăng cường dữ liệu (Data Augmentation)
- Thêm các kỹ thuật tăng cường dữ liệu mới:
  - `width_shift_range=0.2`: dịch chuyển ảnh ngang ngẫu nhiên
  - `height_shift_range=0.2`: dịch chuyển ảnh dọc ngẫu nhiên
  - `shear_range=0.2`: biến dạng ảnh
  - `brightness_range=[0.8, 1.2]`: điều chỉnh độ sáng
  - `fill_mode='nearest'`: xử lý các pixel bị thiếu

## 2. Cải thiện cấu trúc Model
- Tăng số units trong các layer Dense:
  - Layer 1: 512 units (tăng từ 256)
  - Layer 2: 256 units (tăng từ 128)
- Thêm BatchNormalization sau mỗi layer Dense
- Điều chỉnh tỷ lệ Dropout:
  - Layer 1: 0.5 (giữ nguyên)
  - Layer 2: 0.3 (giảm từ 0.5)

## 3. Tối ưu hóa Training Process
- Thêm các Callbacks quan trọng:
  - `ModelCheckpoint`: lưu model tốt nhất dựa trên val_accuracy
  - `EarlyStopping`: dừng training sớm nếu không cải thiện (patience=10)
  - `ReduceLROnPlateau`: giảm learning rate khi loss không giảm
  - `TensorBoard`: theo dõi quá trình training
- Tăng số epochs từ 20 lên 50
- Thêm metric `top_k_categorical_accuracy`

## 4. Tối ưu hóa Optimizer
- Cấu hình chi tiết Adam optimizer:
  - `learning_rate=0.0001`
  - `beta_1=0.9`
  - `beta_2=0.999`
  - `epsilon=1e-07`
  - `decay=1e-6` (weight decay để tránh overfitting)

## Mục tiêu cải tiến
- Tăng tính đa dạng của dữ liệu training
- Cải thiện khả năng học của model
- Tránh overfitting tốt hơn
- Tăng độ chính xác của model
- Theo dõi và lưu trữ kết quả training tốt hơn 