import cv2
import numpy as np
import time
import datetime
import tensorflow as tf

import firebase_admin
from firebase_admin import db, credentials

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("emotion_mobilenetv2_finetuned.h5")

# Nhãn cảm xúc
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Nhãn cảm xúc tiêu cực
emotion_negative = ["Disgust", "Fear", "Sad", "Angry"]

# Nhãn cảm xúc tiêu cực
emotion_positive = ["Happy"]

json_path = r'C:\Users\Public\cred.json'

cred = credentials.Certificate(json_path)
obj = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-server-edaa9-default-rtdb.firebaseio.com'
})

# db.reference('fsb_emotion_detech/emotion_negative', app=obj).listen(listener)

# Load bộ phát hiện khuôn mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Mở webcam hoặc video (sử dụng 0 cho camera mặc định, hoặc đặt đường dẫn video)
video_path = 0  # Để chạy webcam, đổi thành "video.mp4" nếu dùng file video
is_negative = 0
cap = cv2.VideoCapture(video_path)

limit_time = 5


negative_time_counter = 0
positive_time_counter = 0

negative_time_report = 0
positive_time_report = 0

lst_time_negative = []
lst_time_positive = []
start_time = ""
end_time = ""


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


def report_negative():
    # Thống kê Nagative
    if negative_time_counter > limit_time:
        time_stamp = str(time.time()).split(".")[0]
        date_value = datetime.datetime.now().strftime("%Y-%m-%d")
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = start_time + " - " + end_time

        print("Time Stamp: " + time_stamp)
        print("Time Negative: " + report)

        address = 'fsb_emotion_detech/alert_system/report/negative/' + date_value
        node = db.reference(address)
        node.update({
            time_stamp: report  # gửi tín hiệu lên server Firebase
        })
    # Thống kê Nagative


def report_positive():
    # Thống kê Positive
    if positive_time_counter > limit_time:
        time_stamp = str(time.time()).split(".")[0]
        date_value = datetime.datetime.now().strftime("%Y-%m-%d")
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = start_time + " - " + end_time

        print("Time Stamp: " + time_stamp)
        print("Time Positive: " + report)

        address = 'fsb_emotion_detech/alert_system/report/positive/' + date_value
        node = db.reference(address)
        node.update({
            time_stamp: report  # gửi tín hiệu lên server Firebase
        })
    # Thống kê Positive


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Thoát nếu hết video hoặc lỗi

    # Chuyển ảnh sang grayscale để nhận diện khuôn mặt (nhưng vẫn giữ ảnh màu)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Tiền xử lý ảnh
        face_input = preprocess_frame(frame, x, y, w, h)

        # Dự đoán cảm xúc
        predictions = model.predict(face_input)[0]  # Lấy vector xác suất
        predicted_emotion = emotion_labels[np.argmax(predictions)]  # Lấy nhãn có xác suất cao nhất

        # Vẽ khung khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Xử lý nhận diện Cảm Xúc Tiêu Cực
        if predicted_emotion in emotion_negative:
            time.sleep(1)
            negative_time_counter += 1
            print("negative_time_counter Counter: " + str(negative_time_counter))

            # Nếu phát hiện ra Cảm Xúc Tiêu Cực kéo dài trong 5 giây
            if negative_time_counter == limit_time:
                start_time = predicted_emotion + ": " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("Negative Emotion")
                node = db.reference('fsb_emotion_detech/alert_system')
                node.update({
                    'negative': '1' # gửi tín hiệu lên server Firebase
                })
            # Nếu phát hiện ra Cảm Xúc Tiêu Cực kéo dài trong 5 giây

            positive_time_counter = 0

        # Xử lý nhận diện Cảm Xúc Tích Cực
        if predicted_emotion in emotion_positive:
            time.sleep(1)
            positive_time_counter += 1

            # Nếu phát hiện ra Cảm Xúc Tích Cực kéo dài trong 5 giây
            if positive_time_counter == limit_time:
                start_time = predicted_emotion + ": " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("Positive Emotion")
                node = db.reference('fsb_emotion_detech/alert_system')
                node.update({
                    'positive': '1' # gửi tín hiệu lên server Firebase
                })
             # Nếu phát hiện ra Cảm Xúc Tích Cực kéo dài trong 5 giây

            negative_time_counter = 0

        if predicted_emotion == "Neutral":
            report_negative()
            report_positive()
            positive_time_counter = 0
            negative_time_counter = 0


    # Hiển thị video với nhận diện cảm xúc
    cv2.imshow("Emotion Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
