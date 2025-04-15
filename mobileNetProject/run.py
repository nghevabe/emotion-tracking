import cv2
import numpy as np
import tensorflow as tf
import time
from typing import Tuple, List

class EmotionDetector:
    def __init__(self, model_path: str = "emotion_mobilenetv2_finetuned.h5"):
        """Khởi tạo detector với model và các tham số cần thiết"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.fps = 0
            self.frame_count = 0
            self.start_time = time.time()
        except Exception as e:
            print(f"Lỗi khi khởi tạo EmotionDetector: {str(e)}")
            raise

    def preprocess_frame(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Tiền xử lý ảnh khuôn mặt"""
        try:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            return np.expand_dims(face, axis=0)
        except Exception as e:
            print(f"Lỗi khi tiền xử lý frame: {str(e)}")
            return None

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Phát hiện khuôn mặt trong frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Giảm scaleFactor để tăng độ chính xác
            minNeighbors=5,
            minSize=(50, 50)
        )

    def predict_emotion(self, face_input: np.ndarray) -> Tuple[str, np.ndarray]:
        """Dự đoán cảm xúc từ ảnh khuôn mặt"""
        if face_input is None:
            return None, None
        predictions = self.model.predict(face_input, verbose=0)[0]
        predicted_emotion = self.emotion_labels[np.argmax(predictions)]
        return predicted_emotion, predictions

    def draw_results(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                    predictions: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Vẽ kết quả lên frame"""
        for (x, y, w, h), (emotion, probs) in zip(faces, predictions):
            if emotion and probs is not None:
                # Vẽ khung khuôn mặt
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Hiển thị xác suất
                for i, (label, prob) in enumerate(zip(self.emotion_labels, probs)):
                    text = f"{label}: {prob:.2f}"
                    cv2.putText(frame, text, (10, 30 + i * 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Hiển thị FPS
        self.frame_count += 1
        if time.time() - self.start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()
        
        cv2.putText(frame, f"FPS: {self.fps}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

def main():
    try:
        detector = EmotionDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Không thể mở camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera")
                break

            # Phát hiện khuôn mặt
            faces = detector.detect_faces(frame)
            
            # Xử lý từng khuôn mặt
            predictions = []
            for (x, y, w, h) in faces:
                face_input = detector.preprocess_frame(frame, x, y, w, h)
                emotion, probs = detector.predict_emotion(face_input)
                predictions.append((emotion, probs))

            # Vẽ kết quả
            frame = detector.draw_results(frame, faces, predictions)
            
            # Hiển thị
            cv2.imshow("Emotion Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Lỗi trong quá trình chạy: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
