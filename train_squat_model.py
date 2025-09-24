import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json

# Параметры
SEQUENCE_LENGTH = 30
BATCH_SIZE = 8
EPOCHS = 100

ERROR_CLASSES = ["hands", "legs_width", "deep", "simmetry", "other_exercise"]
N_ERROR_CLASSES = len(ERROR_CLASSES)

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

        # Сбор всех признаков
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


def process_video_file(video_file, label, error_labels_list):
    """Обработка видео файла для извлечения признаков и меток ошибок"""
    model = YOLO('yolo11l-pose.pt')
    cap = cv2.VideoCapture(video_file)
    features_list = []
    correctness_labels = []
    error_labels = []
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

                if len(sequence) >= SEQUENCE_LENGTH:
                    features_array = np.array(sequence[-SEQUENCE_LENGTH:])
                    features_list.append(features_array)
                    correctness_labels.append(label)
                    error_labels.append(error_labels_list)

        frame_count += 1

    cap.release()
    print(f"Обработано {frame_count} кадров, извлечено {len(features_list)} последовательностей")
    return features_list, correctness_labels, error_labels


def load_labeled_dataset(dataset_path):
    """Загрузка размеченных данных с метками ошибок"""
    X = []
    y_correctness = []
    y_errors = []

    # Загрузка правильных приседаний
    correct_dir = os.path.join(dataset_path, 'train', 'correct')
    if os.path.exists(correct_dir):
        for folder in sorted(os.listdir(correct_dir)):
            folder_path = os.path.join(correct_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            video_file = os.path.join(folder_path, "video.mp4")
            if os.path.exists(video_file):
                print(f"Обработка правильного видео: {video_file}")
                features, correctness, errors = process_video_file(video_file, 1, [0]*N_ERROR_CLASSES)
                if len(features) > 0:
                    X.extend(features)
                    y_correctness.extend(correctness)
                    y_errors.extend(errors)
                    print(f"  Извлечено {len(features)} последовательностей")

    # Загрузка неправильных приседаний
    incorrect_dir = os.path.join(dataset_path, 'train', 'incorrect')
    if os.path.exists(incorrect_dir):
        for folder in sorted(os.listdir(incorrect_dir)):
            folder_path = os.path.join(incorrect_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            video_file = os.path.join(folder_path, "video.mp4")
            if os.path.exists(video_file):
                # Определяем метки ошибок из файлов .txt
                error_vector = [0] * N_ERROR_CLASSES
                for idx, err_name in enumerate(ERROR_CLASSES):
                    err_file = os.path.join(folder_path, f"{err_name}.txt")
                    if os.path.exists(err_file):
                        error_vector[idx] = 1

                print(f"Обработка неправильного видео: {video_file} -> {error_vector}")
                features, correctness, errors = process_video_file(video_file, 0, error_vector)
                if len(features) > 0:
                    X.extend(features)
                    y_correctness.extend(correctness)
                    y_errors.extend(errors)
                    print(f"  Извлечено {len(features)} последовательностей")

    return np.array(X), np.array(y_correctness), np.array(y_errors)


def build_multi_output_model(sequence_length, num_features):
    """Создание модели с двумя выходами: correctness и errors"""
    inputs = tf.keras.Input(shape=(sequence_length, num_features))

    # Базовая структура LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    # Выход 1: основной прогноз (правильность)
    correctness = Dense(1, activation='sigmoid', name='correctness')(x)

    # Выход 2: ошибки (многоклассовая классификация)
    errors = Dense(N_ERROR_CLASSES, activation='sigmoid', name='errors')(x)  # sigmoid, т.к. могут быть несколько активных

    model = Model(inputs=inputs, outputs=[correctness, errors])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'correctness': 'binary_crossentropy',
            'errors': 'binary_crossentropy'  # т.к. несколько меток могут быть активны
        },
        loss_weights={
            'correctness': 1.0,
            'errors': 0.5
        },
        metrics={
            'correctness': 'accuracy',
            'errors': 'accuracy'
        }
    )
    return model


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')

    if not os.path.exists(dataset_path):
        print("❌ Датасет не найден. Создайте структуру датасета в папке 'dataset'")
        return

    print("Загрузка данных...")
    X, y_correctness, y_errors = load_labeled_dataset(dataset_path)

    if len(X) == 0:
        print("❌ Нет данных для обучения.")
        return

    print(f"Данные загружены: {len(X)} обучающих примеров")

    # Нормализация
    X_flat = X.reshape(-1, X.shape[2])
    mean = np.mean(X_flat, axis=0)
    std = np.std(X_flat, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std

    # Сохранение mean и std
    np.save('features_mean.npy', mean)
    np.save('features_std.npy', std)

    # Разделение на обучение и валидацию
    split_idx = int(len(X) * 0.8)
    X_val = X[split_idx:]
    y_corr_val = y_correctness[split_idx:]
    y_err_val = y_errors[split_idx:]
    X = X[:split_idx]
    y_correctness = y_correctness[:split_idx]
    y_errors = y_errors[:split_idx]

    # Построение модели
    sequence_length = X.shape[1]
    num_features = X.shape[2]
    model = build_multi_output_model(sequence_length, num_features)

    # Callbacks
    early_stop = EarlyStopping(monitor='val_correctness_accuracy', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_correctness_loss', factor=0.5, patience=5, min_lr=1e-7)

    # Обучение
    print("Обучение модели...")
    history = model.fit(
        X, {'correctness': y_correctness, 'errors': y_errors},
        validation_data=(X_val, {'correctness': y_corr_val, 'errors': y_err_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Сохранение модели
    model.save('squat_model_multi.h5')
    print("✅ Модель сохранена в 'squat_model_multi.h5'")

    # Сохранение информации о модели
    model_info = {
        'sequence_length': sequence_length,
        'num_features': num_features,
        'error_classes': ERROR_CLASSES
    }
    with open('model_info_multi.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("✅ Информация о модели сохранена в 'model_info_multi.json'")

    # График обучения
    try:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history.history['correctness_accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_correctness_accuracy'], label='Val Accuracy')
        plt.title('Correctness Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history.history['errors_accuracy'], label='Train Errors Acc')
        plt.plot(history.history['val_errors_accuracy'], label='Val Errors Acc')
        plt.title('Errors Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(history.history['loss'], label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history_multi.png')
        print("✅ График обучения сохранен в 'training_history_multi.png'")
    except Exception as e:
        print(f"❌ Не удалось сохранить график: {e}")


if __name__ == "__main__":
    main()