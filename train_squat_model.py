import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json

# Параметры обучения
SEQUENCE_LENGTH = 30
BATCH_SIZE = 8
EPOCHS = 50


def calculate_squat_features_v11(keypoints):
    """Вычисление признаков для анализа приседаний с использованием YOLOv11 индексов"""
    features = []

    # Проверка наличия всех необходимых точек (17 точек в YOLOv11)
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
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    try:
        # Индексы точек в YOLOv11:
        # 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
        # 5: Left Shoulder, 6: Right Shoulder
        # 7: Left Elbow, 8: Right Elbow
        # 9: Left Wrist, 10: Right Wrist
        # 11: Left Hip, 12: Right Hip
        # 13: Left Knee, 14: Right Knee
        # 15: Left Ankle, 16: Right Ankle

        # Таз: среднее между правым и левым бедром
        hip_center = (keypoints[11] + keypoints[12]) / 2

        # 1. Угол правого колена (бедро-колено-ступня)
        right_knee = calculate_angle(keypoints[11], keypoints[13], keypoints[15])

        # 2. Угол левого колена (бедро-колено-ступня)
        left_knee = calculate_angle(keypoints[12], keypoints[14], keypoints[16])

        # 3. Угол правого бедра (плечо-бедро-колено)
        right_hip = calculate_angle(keypoints[6], keypoints[11], keypoints[13])

        # 4. Угол левого бедра (плечо-бедро-колено)
        left_hip = calculate_angle(keypoints[5], keypoints[12], keypoints[14])

        # 5. Расстояние между коленями
        dist_knees = np.linalg.norm(keypoints[13] - keypoints[14])

        # 6. Расстояние между ступнями
        dist_feet = np.linalg.norm(keypoints[15] - keypoints[16])

        # 7. Глубина приседа (высота таза относительно ступней)
        ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
        depth = hip_center[1] - ankle_y

        # 8. Отклонение коленей от вертикали
        knee_deviation = abs(keypoints[13][0] - keypoints[11][0])  # правая нога
        knee_deviation_left = abs(keypoints[14][0] - keypoints[12][0])  # левая нога

        # 9. Расстояние между запястьями
        wrist_distance = np.linalg.norm(keypoints[9] - keypoints[10])

        # 10. Высота запястий относительно плеч
        shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
        wrist_y = (keypoints[9][1] + keypoints[10][1]) / 2
        wrist_shoulder_diff = wrist_y - shoulder_y

        # 11. Расстояние от корпуса до запястий
        shoulder_center = (keypoints[5] + keypoints[6]) / 2
        wrist_to_body_dist = np.linalg.norm(keypoints[9] - shoulder_center)

        # 12. Угол между руками и телом
        right_arm_vector = keypoints[10] - keypoints[8]  # правая рука
        left_arm_vector = keypoints[9] - keypoints[7]  # левая рука
        body_vector = hip_center - shoulder_center
        right_arm_angle = np.arccos(np.clip(np.dot(right_arm_vector, body_vector) /
                                            (np.linalg.norm(right_arm_vector) * np.linalg.norm(body_vector)), -1.0,
                                            1.0))
        left_arm_angle = np.arccos(np.clip(np.dot(left_arm_vector, body_vector) /
                                           (np.linalg.norm(left_arm_vector) * np.linalg.norm(body_vector)), -1.0, 1.0))
        arm_body_angle = (right_arm_angle + left_arm_angle) / 2

        # Нормализация глубины относительно роста
        if keypoints[11][1] > 0 and keypoints[12][1] > 0:
            height_estimate = max(keypoints[11][1], keypoints[12][1]) - min(keypoints[15][1], keypoints[16][1])
            if height_estimate > 0:
                depth = depth / height_estimate

        # --- НОВЫЕ ПРИЗНАКИ СИММЕТРИИ ---

        # 13. Разница в углах колен (симметрия)
        knee_angle_diff = abs(right_knee - left_knee)

        # 14. Разница в углах бедер (симметрия)
        hip_angle_diff = abs(right_hip - left_hip)

        # 15. Разница в отклонении колен от вертикали (симметрия)
        knee_deviation_diff = abs(knee_deviation - knee_deviation_left)

        # 16. Разница в высоте запястий (симметрия)
        wrist_height_diff = abs(keypoints[9][1] - keypoints[10][1])

        # 17. Разница в расстоянии от корпуса до запястий (симметрия)
        wrist_to_body_dist_diff = abs(wrist_to_body_dist - np.linalg.norm(keypoints[10] - shoulder_center))

        # 18. Средняя симметрия (вспомогательный признак)
        avg_symmetry = (knee_angle_diff + hip_angle_diff + knee_deviation_diff + wrist_height_diff) / 4

        # Сбор всех признаков (19 штук)
        features = [
            right_knee, left_knee, right_hip, left_hip,
            dist_knees, dist_feet, depth, knee_deviation,
            wrist_distance, wrist_shoulder_diff, wrist_to_body_dist, arm_body_angle,
            knee_deviation_left,
            knee_angle_diff, hip_angle_diff, knee_deviation_diff, wrist_height_diff, wrist_to_body_dist_diff, avg_symmetry
        ]

        return features

    except Exception as e:
        print(f"Ошибка при вычислении признаков: {e}")
        return None


def process_video_file(video_file, label):
    """Обработка видео файла для извлечения признаков"""
    # Используем YOLO напрямую для получения keypoints
    model = YOLO('yolo11l-pose.pt')
    cap = cv2.VideoCapture(video_file)
    features_list = []
    labels_list = []
    sequence = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        keypoints = results[0].keypoints.xy.cpu().numpy()

        if len(keypoints) > 0 and len(keypoints[0]) >= 17:
            kp = keypoints[0]
            features = calculate_squat_features_v11(kp)
            if features is not None:
                sequence.append(features)

                # Если набрали достаточную последовательность
                if len(sequence) >= SEQUENCE_LENGTH:
                    features_array = np.array(sequence[-SEQUENCE_LENGTH:])
                    features_list.append(features_array)
                    labels_list.append(label)

        frame_count += 1

    cap.release()
    print(f"Обработано {frame_count} кадров, извлечено {len(features_list)} последовательностей")
    return features_list, labels_list


def load_unlabeled_dataset(dataset_path):
    """Загрузка не размеченных данных для обучения"""
    X = []
    y = []

    # Правильные приседания
    good_squats_dir = os.path.join(dataset_path, 'good_squats')
    if os.path.exists(good_squats_dir):
        for video_file in os.listdir(good_squats_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(good_squats_dir, video_file)
                print(f"Обработка правильного приседания: {video_path}")
                features, labels = process_video_file(video_path, 1)  # метка 1 = правильный
                if len(features) > 0:
                    X.extend(features)
                    y.extend(labels)
                    print(f"  Извлечено {len(features)} последовательностей")

    # Неправильные приседания
    bad_squats_dir = os.path.join(dataset_path, 'bad_squats')
    if os.path.exists(bad_squats_dir):
        for video_file in os.listdir(bad_squats_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(bad_squats_dir, video_file)
                print(f"Обработка неправильного приседания: {video_path}")
                features, labels = process_video_file(video_path, 0)  # метка 0 = неправильный
                if len(features) > 0:
                    X.extend(features)
                    y.extend(labels)
                    print(f"  Извлечено {len(features)} последовательностей")

    # Другие упражнения (считаем их неправильными для классификации приседаний)
    other_exercises_dir = os.path.join(dataset_path, 'other_exercises')
    if os.path.exists(other_exercises_dir):
        for video_file in os.listdir(other_exercises_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(other_exercises_dir, video_file)
                print(f"Обработка другого упражнения: {video_path}")
                features, labels = process_video_file(video_path, 0)  # метка 0 = не приседание
                if len(features) > 0:
                    X.extend(features)
                    y.extend(labels)
                    print(f"  Извлечено {len(features)} последовательностей")

    return np.array(X), np.array(y)


def build_improved_model(sequence_length, num_features):
    """Создание улучшенной LSTM модели (один выход — sigmoid)"""
    model = Sequential([
        # Первый слой Bidirectional LSTM
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features))),
        Dropout(0.3),
        BatchNormalization(),

        # Второй слой Bidirectional LSTM
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        BatchNormalization(),

        # Третий слой Bidirectional LSTM
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        BatchNormalization(),

        # Полносвязные слои
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Один выход — вероятность правильности
    ])

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
        print("Структура:")
        print("dataset/")
        print("├── good_squats/     # Правильные приседания")
        print("├── bad_squats/      # Неправильные приседания")
        print("└── other_exercises/ # Другие упражнения")
        return

    print("Загрузка данных...")

    # Загрузка данных
    X_train, y_train = load_unlabeled_dataset(dataset_path)

    if len(X_train) == 0:
        print("❌ Нет данных для обучения. Пожалуйста, добавьте видео в датасет.")
        print("Убедитесь, что видео содержат приседания и имеют длительность не менее 30 кадров.")
        return

    print(f"Данные загружены: {len(X_train)} обучающих примеров")

    # Нормализация данных
    X_train_flat = X_train.reshape(-1, X_train.shape[2])

    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)
    std[std == 0] = 1  # Избегаем деления на ноль

    X_train = (X_train - mean) / std

    # Разделение на train/val
    split_idx = int(len(X_train) * 0.8)
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]

    # Сохранение mean и std для использования в основном приложении
    np.save('features_mean.npy', mean)
    np.save('features_std.npy', std)

    # Построение модели
    sequence_length = X_train.shape[1]
    num_features = X_train.shape[2]
    model = build_improved_model(sequence_length, num_features)

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

    # Обучение
    print("Обучение модели...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Оценка
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Точность на валидационных данных: {test_acc:.4f}")


    # График обучения
    try:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Тренировочная точность')
        plt.plot(history.history['val_accuracy'], label='Валидационная точность')
        plt.title('Точность')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Тренировочная потеря')
        plt.plot(history.history['val_loss'], label='Валидационная потеря')
        plt.title('Потеря')
        plt.xlabel('Эпоха')
        plt.ylabel('Потеря')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        print("✅ График обучения сохранен в 'training_history.png'")

    except Exception as e:
        print(f"❌ Не удалось сохранить график: {e}")

    # Сохранение модели
    try:
        model.save('squat_model.h5')
        print("✅ Модель сохранена в 'squat_model.h5'")
    except Exception as e:
        print(f"❌ Ошибка при сохранении модели: {e}")

    # Информация о модели
    model_info = {
        'sequence_length': sequence_length,
        'num_features': num_features,
        'accuracy': float(test_acc),
        'loss': float(test_loss)
    }

    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("✅ Информация о модели сохранена в 'model_info.json'")


if __name__ == "__main__":
    main()