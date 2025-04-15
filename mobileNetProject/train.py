import tensorflow as tf
import os
import kagglehub

from keras import Model
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Định nghĩa đường dẫn dataset
from keras.src.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard


dataset_path = kagglehub.dataset_download("msambare/fer2013")

#dataset_path = "datasets/FER2013"

# Tăng cường dữ liệu để tránh overfitting
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
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
x = Dense(512, activation="relu")(x)  # Tăng số units
x = BatchNormalization()(x)  # Thêm BatchNormalization
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(7, activation="softmax")(x)

# Xây dựng model hoàn chỉnh
model = Model(inputs=base_model.input, outputs=output)

# Compile model với learning rate thấp và weight decay
optimizer = Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    decay=1e-6
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy", "top_k_categorical_accuracy"]
)

# Thêm các callbacks
callbacks = [
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# Train model với callbacks
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,  # Tăng số epochs
    batch_size=32,
    callbacks=callbacks
)

# Lưu model sau khi train
model.save("emotion_mobilenetv2_finetuned.h5")