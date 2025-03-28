import cv2
import numpy as np
import tensorflow as tf

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("emotion_mobilenetv2_finetuned.h5")

# Nhãn cảm xúc
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load bộ phát hiện khuôn mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Mở webcam hoặc video (sử dụng 0 cho camera mặc định, hoặc đặt đường dẫn video)
video_path = 0  # Để chạy webcam, đổi thành "video.mp4" nếu dùng file video
cap = cv2.VideoCapture(video_path)


def preprocess_frame(frame, x, y, w, h):
    """
    Tiền xử lý ảnh khuôn mặt trước khi đưa vào model
    - Cắt khuôn mặt từ ảnh màu gốc
    - Resize về 224x224
    - Chuẩn hóa pixel về [0,1]
    - Thêm batch dimension
    """
    face = frame[y:y + h, x:x + w]  # Cắt vùng khuôn mặt từ ảnh gốc
    face = cv2.resize(face, (224, 224))  # Resize về 224x224
    face = face / 255.0  # Chuẩn hóa pixel về [0,1]
    face = np.expand_dims(face, axis=0)  # Thêm batch dimension
    return face


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Thoát nếu hết video hoặc lỗi

    # Chuyển ảnh sang grayscale để nhận diện khuôn mặt (nhưng vẫn giữ ảnh màu)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Hiển thị ảnh khuôn mặt trước khi dự đoán
        cv2.imshow("Face Detected", frame[y:y + h, x:x + w])

        # Tiền xử lý ảnh
        face_input = preprocess_frame(frame, x, y, w, h)

        # Dự đoán cảm xúc
        predictions = model.predict(face_input)[0]  # Lấy vector xác suất
        predicted_emotion = emotion_labels[np.argmax(predictions)]  # Lấy nhãn có xác suất cao nhất

        # Vẽ khung khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Hiển thị xác suất từng cảm xúc
        for i, (emotion, prob) in enumerate(zip(emotion_labels, predictions)):
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hiển thị video với nhận diện cảm xúc
    cv2.imshow("Emotion Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
