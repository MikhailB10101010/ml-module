import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import csv
import json
import argparse
import math


def calculate_squat_features(keypoints):
    """Вычисление признаков для анализа приседаний"""
    features = []

    # Проверка наличия всех необходимых точек
    if len(keypoints) < 17:
        return None

    # Функция для вычисления угла между тремя точками
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # предотвращаем ошибки из-за нестабильности
        return np.degrees(angle)

    # Таз: среднее между правым и левым бедром (точки 11 и 12)
    hip_center = (keypoints[11] + keypoints[12]) / 2

    # 1. Угол правого колена (между бедром, колено, лодыжка)
    right_knee = calculate_angle(keypoints[11], keypoints[13], keypoints[15])

    # 2. Угол левого колена
    left_knee = calculate_angle(keypoints[12], keypoints[14], keypoints[16])

    # 3. Угол правого бедра (между тазом, бедро, колено)
    right_hip = calculate_angle(hip_center, keypoints[11], keypoints[13])

    # 4. Угол левого бедра
    left_hip = calculate_angle(hip_center, keypoints[12], keypoints[14])

    # 5. Расстояние между коленями
    dist_knees = np.linalg.norm(keypoints[13] - keypoints[14])

    # 6. Расстояние между ступнями
    dist_feet = np.linalg.norm(keypoints[15] - keypoints[16])

    # 7. Глубина приседа (разница Y между тазом и средней лодыжкой)
    ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
    depth = hip_center[1] - ankle_y

    # 8. Отклонение коленей от вертикали (для правой ноги)
    knee_deviation = abs(keypoints[13][0] - keypoints[11][0])

    # Нормализуем глубину относительно роста (примерное значение)
    if keypoints[11][1] > 0 and keypoints[12][1] > 0:
        height_estimate = max(keypoints[11][1], keypoints[12][1]) - min(keypoints[15][1], keypoints[16][1])
        if height_estimate > 0:
            depth = depth / height_estimate

    features = [
        right_knee, left_knee, right_hip, left_hip,
        dist_knees, dist_feet, depth, knee_deviation
    ]

    return features


def load_dataset(dataset_path):
    """Загрузка данных из датасета с поддержкой новой структуры папок"""
    X = []
    y = []

    # Загрузка правильных приседаний
    correct_dir = os.path.join(dataset_path, 'train', 'correct')
    for folder in sorted(os.listdir(correct_dir)):
        folder_path = os.path.join(correct_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        video_file = os.path.join(folder_path, "video.mp4")
        if os.path.exists(video_file):
            print(f"Обработка правильного видео: {video_file}")
            features, labels = process_video(video_file, 1)
            X.extend(features)
            y.extend(labels)

    # Загрузка неправильных приседаний
    incorrect_dir = os.path.join(dataset_path, 'train', 'incorrect')
    for folder in sorted(os.listdir(incorrect_dir)):
        folder_path = os.path.join(incorrect_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        video_file = os.path.join(folder_path, "video.mp4")
        if os.path.exists(video_file):
            print(f"Обработка неправильного видео: {video_file}")
            features, labels = process_video(video_file, 0)
            X.extend(features)
            y.extend(labels)

    # Валидационные данные
    val_X = []
    val_y = []

    correct_val_dir = os.path.join(dataset_path, 'val', 'correct')
    for folder in sorted(os.listdir(correct_val_dir)):
        folder_path = os.path.join(correct_val_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        video_file = os.path.join(folder_path, "video.mp4")
        if os.path.exists(video_file):
            print(f"Обработка валидационного правильного видео: {video_file}")
            features, labels = process_video(video_file, 1)
            val_X.extend(features)
            val_y.extend(labels)

    incorrect_val_dir = os.path.join(dataset_path, 'val', 'incorrect')
    for folder in sorted(os.listdir(incorrect_val_dir)):
        folder_path = os.path.join(incorrect_val_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        video_file = os.path.join(folder_path, "video.mp4")
        if os.path.exists(video_file):
            print(f"Обработка валидационного неправильного видео: {video_file}")
            features, labels = process_video(video_file, 0)
            val_X.extend(features)
            val_y.extend(labels)

    return np.array(X), np.array(y), np.array(val_X), np.array(val_y)


def process_video(video_path, label):
    """Обработка видео файла для извлечения признаков"""
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(video_path)
    features_list = []
    labels_list = []

    sequence_length = 30
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра
        results = model(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy()

        if len(keypoints) > 0:
            kp = keypoints[0]
            features = calculate_squat_features(kp)
            if features is not None:
                sequence.append(features)

                # Если набрали достаточную последовательность
                if len(sequence) >= sequence_length:
                    features_array = np.array(sequence[-sequence_length:])
                    features_list.append(features_array)
                    labels_list.append(label)

    cap.release()
    return features_list, labels_list


def build_lstm_model(sequence_length, num_features):
    """Создание LSTM модели"""
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    # Путь к датасету
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')

    # Проверка наличия датасета
    if not os.path.exists(dataset_path):
        print("❌ Датасет не найден. Создайте структуру датасета в папке 'dataset'")
        print("Структура должна быть:")
        print("dataset/")
        print("├── train/")
        print("│   ├── correct/")
        print("│   └── incorrect/")
        print("└── val/")
        print("    ├── correct/")
        print("    └── incorrect/")
        return

    print("Загрузка данных...")
    X_train, y_train, X_val, y_val = load_dataset(dataset_path)

    if len(X_train) == 0 or len(X_val) == 0:
        print("❌ Нет данных для обучения. Пожалуйста, добавьте видео в датасет.")
        return

    print(f"Данные загружены: {len(X_train)} обучающих примеров, {len(X_val)} валидационных примеров")

    # Нормализация данных
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    X_val_flat = X_val.reshape(-1, X_val.shape[2])

    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)
    std[std == 0] = 1  # Избегаем деления на ноль

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Сохранение mean и std для использования в основном приложении
    np.save('features_mean.npy', mean)
    np.save('features_std.npy', std)

    # Построение модели
    sequence_length = X_train.shape[1]
    num_features = X_train.shape[2]
    model = build_lstm_model(sequence_length, num_features)

    # Обучение модели
    print("Обучение модели...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop]
    )

    # Оценка модели
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"Точность на валидационных данных: {test_acc:.4f}")

    # Вывод графика обучения
    try:
        plt.figure(figsize=(12, 4))

        # График точности
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Тренировочная точность')
        plt.plot(history.history['val_accuracy'], label='Валидационная точность')
        plt.title('Точность')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()

        # График потерь
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Тренировочная потеря')
        plt.plot(history.history['val_loss'], label='Валидационная потеря')
        plt.title('Потеря')
        plt.xlabel('Эпоха')
        plt.ylabel('Потеря')
        plt.legend()

        # Сохранение графика
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("✅ График обучения сохранен в 'training_history.png'")

    except Exception as e:
        print(f"❌ Не удалось сохранить график: {e}")


if __name__ == "__main__":
    main()