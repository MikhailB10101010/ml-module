# server_api.py
import os
import cv2
import numpy as np
from ultralytics import solutions, YOLO
import torch
from tensorflow.keras.models import load_model
from flask import Flask, request, send_file, jsonify
import tempfile
import uuid
import logging
import threading


# --- Настройки Flask ---
app = Flask(__name__)
# максимальный размер загружаемого файла
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# --- Пути к моделям ---
LSTM_MODEL_PATH = 'squat_model.h5'
FEATURES_MEAN_PATH = 'features_mean.npy'
FEATURES_STD_PATH = 'features_std.npy'
YOLO_MODEL_NAME = 'yolo11l-pose.pt'  # Убедитесь, что модель доступна

# --- Глобальное хранение последнего обработанного видео ---
# Переменная для хранения пути
last_video_path = None
# Блокировка, чтобы безопасно читать/писать last_video_path из разных потоков
last_video_lock = threading.Lock()


class SquatAnalyzerServer:
    """Класс, отвечающий за анализ видео с упражнениями."""

    def __init__(self):
        """Инициализация анализатора приседаний для сервера."""
        print("Загрузка моделей...")
        # Загрузка обученной LSTM модели
        if os.path.exists(LSTM_MODEL_PATH):
            self.lstm_model = load_model(LSTM_MODEL_PATH)
            print("✅ LSTM модель загружена успешно")
        else:
            self.lstm_model = None
            print(f"⚠️  LSTM модель не найдена по пути: {LSTM_MODEL_PATH}")

        # Загрузка параметров нормализации признаков
        if os.path.exists(FEATURES_MEAN_PATH) and os.path.exists(FEATURES_STD_PATH):
            self.features_mean = np.load(FEATURES_MEAN_PATH)
            self.features_std = np.load(FEATURES_STD_PATH)
            print("✅ Параметры нормализации загружены")
        else:
            self.features_mean = None
            self.features_std = None
            print(f"⚠️  Параметры нормализации не найдены по пути: {FEATURES_MEAN_PATH} или {FEATURES_STD_PATH}")

        # Загрузка YOLO модели для определения точек позы
        try:
            self.yolo_model = YOLO(YOLO_MODEL_NAME)
            print("✅ YOLO модель загружена успешно")
        except Exception as e:
            print(f"❌ Ошибка загрузки YOLO модели: {e}")
            self.yolo_model = None
            # Если YOLO не загружена, анализ не будет работать

        # Инициализация AIGym для подсчёта приседаний
        # AIGym может вести себя нестабильно в серверной среде
        try:
            self.gym = solutions.AIGym(
                model=YOLO_MODEL_NAME,
                kpts=[11, 13, 15],  # бедро, колено, ступня
                up_angle=145.0,     # угол "вверху" (прямые ноги)
                down_angle=90.0,    # угол "внизу" (присед)
                show=False,
                line_width=2,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("✅ AIGym инициализирован")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации AIGym: {e}")
            self.gym = None

        # Для LSTM анализ нужно накопить последовательность кадров
        self.sequence = []
        self.SEQUENCE_LENGTH = 30
        self.frame_count = 0
        self.squat_counter = 0
        self.last_counter = 0

        # Названия признаков, чтобы понимать, что LSTM "считает неправильным"
        self.feature_names = [
            "right_knee", "left_knee", "right_hip", "left_hip",
            "dist_knees", "dist_feet", "depth", "knee_deviation",
            "wrist_distance", "wrist_shoulder_diff", "wrist_to_body_dist", "arm_body_angle",
            "knee_deviation_left",
            "knee_angle_diff", "hip_angle_diff", "knee_deviation_diff", "wrist_height_diff", "wrist_to_body_dist_diff", "avg_symmetry"
        ]

    def calculate_squat_features_v11(self, keypoints):
        """
        Вычисление признаков для анализа приседаний по точкам YOLOv11.

        Индексы точек в YOLOv11 Pose:
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder
            7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist
            11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee
            15: Left Ankle, 16: Right Ankle
        """
        # Проверим, есть ли у нас все 17 точек позы
        if len(keypoints) < 17:
            return None

        # Вспомогательная функция для вычисления угла между тремя точками
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
            # Центр таза (примерная позиция корпуса)
            hip_center = (keypoints[11] + keypoints[12]) / 2

            # Углы суставов
            right_knee_angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15])  # правое колено
            left_knee_angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])  # левое колено
            right_hip_angle = calculate_angle(keypoints[6], keypoints[11], keypoints[13])  # правое бедро
            left_hip_angle = calculate_angle(keypoints[5], keypoints[12], keypoints[14])  # левое бедро

            # Расстояния
            dist_knees = np.linalg.norm(keypoints[13] - keypoints[14])  # между коленями
            dist_feet = np.linalg.norm(keypoints[15] - keypoints[16])  # между ступнями

            # Глубина приседа (относительно средней точки ступней)
            ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
            depth = hip_center[1] - ankle_y

            # Отклонение коленей от вертикальной линии (ось тела)
            knee_deviation_right = abs(keypoints[13][0] - keypoints[11][0])
            knee_deviation_left = abs(keypoints[14][0] - keypoints[12][0])

            # Расстояние между запястьями
            wrist_distance = np.linalg.norm(keypoints[9] - keypoints[10])

            # Высота запястий относительно плеч
            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            wrist_y = (keypoints[9][1] + keypoints[10][1]) / 2
            wrist_shoulder_diff = wrist_y - shoulder_y

            # Расстояние от центра плеч до запястья
            shoulder_center = (keypoints[5] + keypoints[6]) / 2
            wrist_to_body_dist = np.linalg.norm(keypoints[9] - shoulder_center)

            # Угол между рукой и телом (упрощённо)
            right_arm_vector = keypoints[10] - keypoints[8]  # правая рука
            left_arm_vector = keypoints[9] - keypoints[7]  # левая рука
            body_vector = hip_center - shoulder_center
            right_arm_angle = np.arccos(np.clip(np.dot(right_arm_vector, body_vector) /
                                                (np.linalg.norm(right_arm_vector) * np.linalg.norm(body_vector)), -1.0, 1.0))
            left_arm_angle = np.arccos(np.clip(np.dot(left_arm_vector, body_vector) /
                                               (np.linalg.norm(left_arm_vector) * np.linalg.norm(body_vector)), -1.0, 1.0))
            arm_body_angle = (right_arm_angle + left_arm_angle) / 2


            # Нормализация глубины относительно роста (грубая оценка)
            if keypoints[11][1] > 0 and keypoints[12][1] > 0:
                height_estimate = max(keypoints[11][1], keypoints[12][1]) - min(keypoints[15][1], keypoints[16][1])
                if height_estimate > 0:
                    depth = depth / height_estimate

            # --- Признаки симметрии ---
            knee_angle_diff = abs(right_knee_angle - left_knee_angle)
            hip_angle_diff = abs(right_hip_angle - left_hip_angle)
            knee_deviation_diff = abs(knee_deviation_right - knee_deviation_left)
            wrist_height_diff = abs(keypoints[9][1] - keypoints[10][1])
            wrist_to_body_dist_diff = abs(wrist_to_body_dist - np.linalg.norm(keypoints[10] - shoulder_center))
            avg_symmetry = (knee_angle_diff + hip_angle_diff + knee_deviation_diff + wrist_height_diff) / 4

            # Собираем все признаки в список
            features = [
                right_knee_angle, left_knee_angle, right_hip_angle, left_hip_angle,
                dist_knees, dist_feet, depth, knee_deviation_right,
                wrist_distance, wrist_shoulder_diff, wrist_to_body_dist, arm_body_angle,
                knee_deviation_left,
                knee_angle_diff, hip_angle_diff, knee_deviation_diff, wrist_height_diff, wrist_to_body_dist_diff, avg_symmetry
            ]
            return features

        except Exception as e:
            print(f"Ошибка при вычислении признаков: {e}")
            return None

    def analyze_frame_with_lstm(self, keypoints):
        """
        Анализ кадра с помощью LSTM. Пытаемся понять, правильно ли выполнено упражнение.

        Возвращает:
            tuple: (float, str) - Прогноз LSTM (0.0 - 1.0), причина (или None).
        """
        # Проверяем, всё ли готово для LSTM анализа
        if self.lstm_model is None or self.features_mean is None or self.features_std is None:
            return None, None

        # Вычисляем признаки для текущего кадра
        features = self.calculate_squat_features_v11(keypoints)
        if features is None:
            return None, None

        # Добавляем признаки в последовательность
        self.sequence.append(features)

        # Как только у нас набралось нужное количество кадров (SEQUENCE_LENGTH)...
        if len(self.sequence) >= self.SEQUENCE_LENGTH:
            # Берём последние SEQUENCE_LENGTH кадров
            sequence_array = np.array(self.sequence[-self.SEQUENCE_LENGTH:])
            # Нормализуем их так же, как обучающую выборку
            sequence_normalized = (sequence_array - self.features_mean) / self.features_std

            # Прогоняем через LSTM модель
            prediction = self.lstm_model.predict(np.expand_dims(sequence_normalized, axis=0), verbose=0)[0][0]

            # Пытаемся понять, *почему* LSTM дал такой прогноз
            # Сравниваем текущие признаки со средними значениями обучающей выборки
            current_norm = (np.array(features) - self.features_mean) / self.features_std
            deviations = np.abs(current_norm)

            # Находим признак с наибольшим отклонением
            max_dev_index = np.argmax(deviations)
            reason = self.feature_names[max_dev_index]

            # Если отклонение по признаку симметрии велико, укажем это как причину
            symmetry_feature_idx = 18  # avg_symmetry
            if deviations[symmetry_feature_idx] > 1.5:  # порог можно настроить
                reason = "symmetry"
            return prediction, reason

        # Если последовательность ещё не готова, возвращаем None
        return None, None

    def process_video(self, input_video_path, output_video_path):
        """
        Обработка всего видеофайла целиком.
        Args:
            input_video_path (str): Путь к входному видеофайлу.
            output_video_path (str): Путь, куда сохранить обработанное видео.
        Returns:
            bool: True, если обработка прошла успешно, иначе False.
        """
        if not self.yolo_model:
            print("❌ YOLO модель не загружена, обработка невозможна.")
            return False

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видео: {input_video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"❌ Не удалось создать VideoWriter для: {output_video_path}")
            cap.release()
            return False

        frame_num = 0
        print(f"Начинаем обработку видео: {total_frames} кадров")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count = frame_num
            frame_num += 1

            # --- Анализ кадра ---
            # Получаем точки позы с помощью YOLO
            yolo_results = self.yolo_model(frame, verbose=False)
            keypoints = yolo_results[0].keypoints.xy.cpu().numpy()

            # Аппаратно накладываем разметку AIGym (если доступна)
            if self.gym:
                try:
                    gym_results = self.gym(frame)
                    annotated_frame = gym_results.plot()
                except Exception as e:
                    print(f"Ошибка в AIGym для кадра {frame_num}: {e}")
                    annotated_frame = frame.copy()  # просто копируем кадр, если AIGym сломался
            else:
                annotated_frame = frame.copy()

            lstm_prediction = None
            reason = None

            # Получаем ключевые точки для LSTM анализа
            if len(keypoints) > 0 and len(keypoints[0]) >= 17:
                kp = keypoints[0]

                # Запускаем LSTM анализ
                lstm_prediction, reason = self.analyze_frame_with_lstm(kp)

                # Обновляем счётчик приседаний из AIGym (если AIGym доступен)
                if self.gym:
                    try:
                        if hasattr(self.gym, 'count') and self.gym.count != self.last_counter:
                            self.squat_counter = self.gym.count
                            self.last_counter = self.gym.count
                    except Exception as e:
                        print(f"Ошибка при получении счётчика AIGym для кадра {frame_num}: {e}")

            # Добавляем результат LSTM на кадр
            if lstm_prediction is not None:
                pred_text = f"LSTM: {lstm_prediction:.2f} ({'pravilno' if lstm_prediction > 0.5 else 'ne pravilno'})"
                if reason:
                    pred_text += f"; {reason}"
                color = (0, 255, 0) if lstm_prediction > 0.5 else (0, 0, 255)
                cv2.putText(
                    annotated_frame,
                    pred_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                # Показываем углы коленей на кадре (для наглядности)
                def calculate_angle(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    ba = a - b
                    bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    return np.degrees(angle)

                if len(kp) >= 17:
                    left_knee_angle = calculate_angle(kp[12], kp[14], kp[16])
                    right_knee_angle = calculate_angle(kp[11], kp[13], kp[15])

                    cv2.putText(annotated_frame, f"{left_knee_angle:.2f}",
                                (int(kp[14][0]), int(kp[14][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, f"{right_knee_angle:.2f}",
                                (int(kp[13][0]), int(kp[13][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Показываем номер текущего кадра
            cv2.putText(
                annotated_frame,
                f"frame: {self.frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Записываем обработанный кадр в выходное видео
            out.write(annotated_frame)

            # Показываем прогресс каждые 100 кадров
            if frame_num % 100 == 0:
                print(f"Обработано кадров: {frame_num}/{total_frames}")
        # Закрываем видеофайлы
        cap.release()
        out.release()
        print(f"Видео обработано и сохранено: {output_video_path}")
        print(f"Всего приседаний посчитано: {self.squat_counter}")
        return True


# --- Инициализация анализатора ---
analyzer = SquatAnalyzerServer()


@app.route('/process_video', methods=['POST'])
def upload_and_process():
    """
    Принимает видео, обрабатывает, возвращает результат.
    Ожидает POST-запрос с файлом 'file'.
    """
    if request.method == 'POST':
        # Проверяем, есть ли файл в запросе
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Проверяем расширение файла
        if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Создаём временные файлы для входа и выхода
            input_fd, input_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
            output_fd, output_path = tempfile.mkstemp(suffix='.mp4')  # Выход всегда mp4

            try:
                print(f"Получен файл: {file.filename}, сохранён во временный файл: {input_path}")
                file.save(input_path)

                # Запускаем обработку
                success = analyzer.process_video(input_path, output_path)

                if success:
                    # Сбрасываем внутренние счётчики анализатора
                    analyzer.squat_counter = 0
                    analyzer.last_counter = 0
                    analyzer.sequence = []  # Очищаем накопленную последовательность LSTM
                    analyzer.frame_count = 0

                    # Обновляем путь к последнему обработанному видео (безопасно для потоков)
                    with last_video_lock:
                        global last_video_path
                        if last_video_path:  # Удаляем предыдущий файл, чтобы не накапливать мусор
                            try:
                                os.remove(last_video_path)
                                print(f"Удалён предыдущий файл последнего видео: {last_video_path}")
                            except OSError as e:
                                print(f"Ошибка при удалении предыдущего файла: {e}")
                        last_video_path = output_path  # Сохраняем путь к новому файлу
                    print(f"Обработка завершена успешно. Файл сохранён как последний: {output_path}")
                    # Отправляем клиенту готовое видео
                    return send_file(output_path, as_attachment=True, download_name=f"processed_{file.filename}")
                else:
                    print("Ошибка при обработке видео.")
                    return jsonify({'error': 'Error processing video'}), 500

            except Exception as e:
                print(f"Ошибка при обработке запроса: {e}")
                return jsonify({'error': 'Internal server error'}), 500
            finally:
                # Удаляем временный входной файл после обработки или ошибки
                try:
                    os.close(input_fd)
                    os.remove(input_path)
                except OSError:
                    pass
                # Выходной файл (output_path) сразу НЕ удаляем, он нужен для отправки клиенту и для хранения как 'last_video'. Он будет удалён позже, при перезаписи.
        else:
            return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, or MKV.'}), 400

    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/last_video', methods=['GET'])
def get_last_video():
    """
    Отправляет последнее обработанное видео.
    Возвращает GET-запрос, который отправляет файл последнего обработанного видео.
    """
    if request.method == 'GET':
        # Безопасно читаем глобальную переменную last_video_path
        with last_video_lock:
            global last_video_path
            # Проверяем, есть ли файл и существует ли он
            if last_video_path and os.path.exists(last_video_path):
                print(f"Отправка последнего обработанного видео: {last_video_path}")
                # Отправляем файл клиенту
                return send_file(last_video_path, as_attachment=True, download_name="last_processed_video.mp4")
            else:
                print("Последнее видео не найдено.")
                # Если файла нет, возвращаем ошибку 404
                return jsonify({'error': 'No last video available'}), 404


@app.route('/health', methods=['GET'])
def health_check():
    """
    Проверяет, запущен ли сервер и загружены ли модели.

    Возвращает JSON со статусом.
    """
    # Проверяем, загружены ли основные модели
    models_ok = analyzer.yolo_model is not None and analyzer.lstm_model is not None
    return jsonify({'status': 'healthy', 'models_loaded': models_ok})

if __name__ == '__main__':
    print("Запуск сервера анализа видео...")
    # port=5000 - стандартный порт для Flask
    app.run(host='0.0.0.0', port=5000, debug=False)