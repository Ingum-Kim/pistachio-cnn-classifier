import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import time

print("="*60)
print("피스타치오 CNN 분류 모델 학습 시작")
print("="*60)

# 1. 데이터 로딩 및 전처리
print("\n[1단계] 데이터 로딩 중...")

# 데이터셋 경로
DATA_DIR = os.path.join('Pistachio_Image_Dataset', 'Pistachio_Image_Dataset')
KIRMIZI_DIR = os.path.join(DATA_DIR, 'Kirmizi_Pistachio')
SIIRT_DIR = os.path.join(DATA_DIR, 'Siirt_Pistachio')

# 이미지 크기 설정 (Chapter 10-2 권장)
IMG_SIZE = 120

# 데이터 로딩 함수
def load_images_from_folder(folder, label):
    """폴더에서 이미지를 로딩하고 라벨과 함께 반환"""
    images = []
    labels = []

    image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

    for img_file in image_files:
        try:
            img_path = os.path.join(folder, img_file)
            img = Image.open(img_path)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img)

            # RGB 이미지인지 확인 (일부 이미지가 그레이스케일일 수 있음)
            if len(img_array.shape) == 2:
                # 그레이스케일을 RGB로 변환
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[2] == 4:
                # RGBA를 RGB로 변환
                img_array = img_array[:, :, :3]

            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"이미지 로딩 실패: {img_file} - {e}")
            continue

    return images, labels

# Kirmizi 클래스 로딩 (라벨: 0)
print(f"Kirmizi 클래스 로딩 중... ({KIRMIZI_DIR})")
kirmizi_images, kirmizi_labels = load_images_from_folder(KIRMIZI_DIR, 0)
print(f"  → Kirmizi 이미지 수: {len(kirmizi_images)}")

# Siirt 클래스 로딩 (라벨: 1)
print(f"Siirt 클래스 로딩 중... ({SIIRT_DIR})")
siirt_images, siirt_labels = load_images_from_folder(SIIRT_DIR, 1)
print(f"  → Siirt 이미지 수: {len(siirt_images)}")

# 전체 데이터 합치기
X = np.array(kirmizi_images + siirt_images, dtype='float32')
y = np.array(kirmizi_labels + siirt_labels, dtype='int32')

print(f"\n총 데이터 수: {len(X)}")
print(f"데이터 shape: {X.shape}")

# 정규화 (0~1 범위로 스케일링)
X = X / 255.0
print("데이터 정규화 완료 (0~1 범위)")

# One-hot encoding
y_categorical = to_categorical(y, num_classes=2)
print(f"라벨 인코딩 완료: {y_categorical.shape}")


# 2. 데이터 분할 (Train:Test = 7:3)

print("\n[2단계] 데이터 분할 중...")
print("※ 프로젝트 요구사항: random_state=123, test_size=0.3")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical,
    test_size=0.3,
    random_state=123,  # 필수 요구사항!
    stratify=y  # 클래스 비율 유지
)

print(f"Train 데이터: {X_train.shape[0]}개")
print(f"Test 데이터: {X_test.shape[0]}개")
print(f"Train 비율: {X_train.shape[0]/len(X)*100:.1f}%")
print(f"Test 비율: {X_test.shape[0]/len(X)*100:.1f}%")


# 3. CNN 모델 구축 (전이학습 사용 안 함!)

print("\n[3단계] CNN 모델 구축 중...")
print("※ 전이학습(Transfer Learning) 사용 금지 - 직접 CNN 구축")

model = Sequential([
    # Conv Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    # Conv Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    # Conv Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Conv Block 4
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    Dropout(0.3),

    # Output Layer
    Dense(2, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n모델 구조:")
model.summary()


# 4. 이미지 증강 (Image Augmentation)

print("\n[4단계] 이미지 증강 설정 중...")
print("※ Chapter 10-2 참고: 성능 향상을 위한 데이터 증강")

# Train 데이터 증강
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation 데이터는 증강 안 함
val_datagen = ImageDataGenerator()

print("이미지 증강 설정 완료")


# 5. 콜백 함수 설정

print("\n[5단계] 학습 콜백 설정 중...")

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=40,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        patience=15,
        factor=0.3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_pistachio_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("콜백 설정 완료")


# 6. 모델 학습

print("\n[6단계] 모델 학습 시작...")
print("="*60)

# 학습 설정
EPOCHS = 100
BATCH_SIZE = 16

# 클래스 가중치 계산
class_weight = {0: 1.0, 1: len(kirmizi_images) / len(siirt_images)}

# 학습 시작 시간 기록
start_time = time.time()

# 모델 학습
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=val_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# 학습 시간 계산
training_time = time.time() - start_time
print(f"\n학습 완료! 소요 시간: {training_time/60:.2f}분")


# 7. 모델 평가 (프로젝트 필수 요구사항)

print("\n[7단계] 모델 평가")
print("="*60)

# Train accuracy
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f"Train Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

# Test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

print("="*60)


# 8. 학습곡선 그래프 저장

print("\n[8단계] 학습곡선 그래프 생성 중...")

plt.figure(figsize=(14, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("학습곡선 그래프 저장 완료: training_history.png")


# 9. 최종 모델 저장

print("\n[9단계] 최종 모델 저장 중...")

model.save('pistachio_cnn_model.h5')
print("모델 저장 완료: pistachio_cnn_model.h5")
print("※ Part 2 UI에서 이 모델을 로드하여 사용합니다.")


# 10. 결과 요약

print("\n" + "="*60)
print("학습 결과 요약")
print("="*60)
print(f"총 에폭: {len(history.history['loss'])}")
print(f"학습 시간: {training_time/60:.2f}분")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Accuracy 차이: {abs(train_acc - test_acc)*100:.2f}%")
print("="*60)
