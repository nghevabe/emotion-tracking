import tensorflow as tf
import os

from keras import Model
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Định nghĩa đường dẫn dataset
from keras.src.optimizers import Adam

dataset_path = "datasets/FER2013"

# Tăng cường dữ liệu để tránh overfitting
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

# Load dữ liệu từ thư mục
train_data = train_datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Load MobileNetV2 (pre-trained trên ImageNet)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Mở khóa 30 layer cuối để fine-tune
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Thêm các lớp đầu ra mới
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)  # ✅ Phải gán lại x sau mỗi lớp
x = Dropout(0.5)(x)  # Thêm dropout để tránh overfitting
x = Dense(128, activation="relu")(x)
output = Dense(7, activation="softmax")(x)  # Kết nối đầu ra

# Xây dựng model hoàn chỉnh
model = Model(inputs=base_model.input, outputs=output)

# Compile model với learning rate thấp để fine-tune
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    batch_size=32
)

# Lưu model sau khi train
model.save("emotion_mobilenetv2_finetuned.h5")