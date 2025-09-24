import cv2
import numpy as np
from ultralytics import solutions, YOLO
import torch
import json
import os
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading


class SquatAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор приседаний с YOLOv11 — СИММЕТРИЯ + ГЛУБИНА")
        self.root.geometry("1200x800")

        # Инициализация анализатора
        self.analyzer = SquatAnalyzer()

        # Переменные
        self.cap = None
        self.video_writer = None
        self.is_playing = False
        self.is_camera = False
        self.current_frame = None

        # Для визуализации симметрии
        self.left_symmetry_values = []
        self.right_symmetry_values = []
        self.max_history = 100  # количество точек в графике

        # Создание интерфейса
        self.create_widgets()

        # Запуск обновления кадров
        self.update_frame()

    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Панель управления
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Кнопки управления
        ttk.Button(control_frame, text="Открыть видео", command=self.open_video).pack(side=tk.LEFT, padx=5)
        self.camera_button = ttk.Button(control_frame, text="Видео с камеры", command=self.open_camera)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        self.play_button = ttk.Button(control_frame, text="▶ Воспроизвести", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⏹ Остановить", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сохранить результат", command=self.save_result).pack(side=tk.LEFT, padx=5)

        # Информационная панель
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.frame_label = ttk.Label(info_frame, text="Кадр: 0")
        self.frame_label.pack(side=tk.LEFT)

        self.prediction_label = ttk.Label(info_frame, text="Предсказание LSTM: -")
        self.prediction_label.pack(side=tk.LEFT, padx=20)

        self.counter_label = ttk.Label(info_frame, text="Счетчик приседаний: 0")
        self.counter_label.pack(side=tk.LEFT, padx=20)

        # Метка симметрии
        self.symmetry_label = ttk.Label(info_frame, text="✅ Симметрично", foreground="green")
        self.symmetry_label.pack(side=tk.LEFT, padx=20)

        # Видео панель
        video_frame = ttk.Frame(self.root)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Холст для видео
        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Графики симметрии
        graph_frame = ttk.Frame(self.root)
        graph_frame.pack(fill=tk.X, padx=10, pady=5)

        # Левый график
        self.left_canvas = tk.Canvas(graph_frame, width=50, height=200, bg="white")
        self.left_canvas.pack(side=tk.LEFT, padx=5)

        # Правый график
        self.right_canvas = tk.Canvas(graph_frame, width=50, height=200, bg="white")
        self.right_canvas.pack(side=tk.RIGHT, padx=5)

        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_video(self):
        """Открытие видео файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if file_path:
            self.stop_video()
            self.is_camera = False
            self.analyzer.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)

            if not self.cap.isOpened():
                self.status_var.set("❌ Не удалось открыть видео")
                return

            self.status_var.set(f"Видео загружено: {os.path.basename(file_path)}")
            self.is_playing = True
            self.play_button.config(text="⏸ Пауза")

    def open_camera(self):
        """Открытие веб-камеры"""
        self.stop_video()
        self.is_camera = True
        self.cap = cv2.VideoCapture(0)  # 0 - первая камера по умолчанию

        if not self.cap.isOpened():
            self.status_var.set("❌ Не удалось открыть веб-камеру")
            return

        self.status_var.set("Веб-камера активна")
        self.is_playing = True
        self.play_button.config(text="⏸ Пауза")

    def toggle_play(self):
        """Переключение воспроизведения/паузы"""
        if self.cap is None:
            return

        self.is_playing = not self.is_playing
        self.play_button.config(text="⏸ Пауза" if self.is_playing else "▶ Воспроизвести")

    def stop_video(self):
        """Остановка видео или камеры"""
        self.is_playing = False
        self.is_camera = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.current_frame = None
        self.status_var.set("Видео/камера остановлены")
        self.frame_label.config(text="Кадр: 0")
        self.prediction_label.config(text="Предсказание LSTM: -")
        self.counter_label.config(text="Счетчик приседаний: 0")
        self.symmetry_label.config(text="✅ Симметрично", foreground="green")
        self.left_symmetry_values.clear()
        self.right_symmetry_values.clear()

    def save_result(self):
        """Сохранение результата"""
        if self.cap is None:
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
        )

        if file_path:
            self.analyzer.output_path = file_path
            self.status_var.set(f"Результат будет сохранен в: {file_path}")

    def update_frame(self):
        """Обновление кадра"""
        if self.is_playing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Анализ кадра
                annotated_frame, lstm_prediction, reason, squat_counter = self.analyzer.process_frame(frame)
                self.current_frame = annotated_frame
                self.analyzer.frame_count += 1

                # Обновление информации
                self.frame_label.config(text=f"Кадр: {self.analyzer.frame_count}")
                self.counter_label.config(text=f"Счетчик приседаний: {squat_counter}")

                if lstm_prediction is not None:
                    pred_text = f"{'Правильный' if lstm_prediction > 0.5 else 'Неправильный'} ({lstm_prediction:.2f})"
                    if reason:
                        pred_text += f"; {reason}"
                    self.prediction_label.config(text=f"Предсказание LSTM: {pred_text}")
                    self.prediction_label.config(foreground="green" if lstm_prediction > 0.5 else "red")

                    # Обновляем визуализацию симметрии
                    self.update_symmetry_graphs(frame)

                # Сохранение кадра если нужно
                if hasattr(self.analyzer, 'output_path') and self.analyzer.output_path and not self.is_camera:
                    if self.video_writer is None:
                        h, w = annotated_frame.shape[:2]
                        self.video_writer = cv2.VideoWriter(
                            self.analyzer.output_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30,  # fps
                            (w, h)
                        )
                    self.video_writer.write(annotated_frame)
            else:
                if not self.is_camera:  # Только для видео файлов
                    self.stop_video()

        # Отображение кадра
        if self.current_frame is not None:
            self.display_frame(self.current_frame)

        # Планируем следующее обновление
        self.root.after(30, self.update_frame)  # ~33 fps

    def display_frame(self, frame):
        """Отображение кадра в интерфейсе"""
        # Конвертация BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Преобразование в PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Масштабирование под размер холста
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Сохраняем пропорции
            img_width, img_height = pil_image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Конвертация в PhotoImage
        photo = ImageTk.PhotoImage(pil_image)

        # Отображение на холсте
        self.video_canvas.delete("all")
        x = (self.video_canvas.winfo_width() - pil_image.width) // 2
        y = (self.video_canvas.winfo_height() - pil_image.height) // 2
        self.video_canvas.create_image(max(0, x), max(0, y), anchor=tk.NW, image=photo)
        self.video_canvas.image = photo  # Сохраняем ссылку

    def update_symmetry_graphs(self, frame):
        """Обновление графиков симметрии"""
        # Получаем keypoints через YOLO
        results = self.analyzer.yolo_model(frame, verbose=False)
        keypoints = results[0].keypoints.xy.cpu().numpy()

        if len(keypoints) < 1 or len(keypoints[0]) < 17:
            return

        kp = keypoints[0]

        # Функция для вычисления угла
        def calculate_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)

        # Углы колен
        left_knee_angle = calculate_angle(kp[12], kp[14], kp[16])  # левое колено
        right_knee_angle = calculate_angle(kp[11], kp[13], kp[15])  # правое колено

        # Нормализуем в диапазон 0-50 для графика
        left_val = min(max(left_knee_angle - 90, 0), 50)  # от 90 до 140 градусов
        right_val = min(max(right_knee_angle - 90, 0), 50)

        # Добавляем в историю
        self.left_symmetry_values.append(left_val)
        self.right_symmetry_values.append(right_val)

        if len(self.left_symmetry_values) > self.max_history:
            self.left_symmetry_values.pop(0)
            self.right_symmetry_values.pop(0)

        # Отрисовка графиков
        self.draw_symmetry_graph(self.left_canvas, self.left_symmetry_values, "Левая нога")
        self.draw_symmetry_graph(self.right_canvas, self.right_symmetry_values, "Правая нога")

        # Обновляем метку симметрии
        diff = abs(left_knee_angle - right_knee_angle)
        if diff < 10:
            self.symmetry_label.config(text="✅ Симметрично", foreground="green")
        else:
            self.symmetry_label.config(text="❌ Несимметрично", foreground="red")

    def draw_symmetry_graph(self, canvas, values, title):
        """Рисует график симметрии"""
        canvas.delete("all")

        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        # Ось Y: от 0 до 50
        # Ось X: по истории значений
        step = w / max(len(values), 1)

        # Рисуем сетку
        for i in range(0, 51, 10):
            y = h - (i / 50) * h
            canvas.create_line(0, y, w, y, fill="lightgray", dash=(2, 2))

        # Рисуем точки
        points = []
        for i, val in enumerate(values):
            x = i * step
            y = h - (val / 50) * h
            points.append((x, y))

        if len(points) > 1:
            canvas.create_line(points, fill="blue", width=2)

        # Последняя точка — красный круг
        if points:
            last_x, last_y = points[-1]
            canvas.create_oval(last_x - 3, last_y - 3, last_x + 3, last_y + 3, fill="red")

        # Подпись
        canvas.create_text(w//2, 10, text=title, fill="black", font=("Arial", 8))

    def on_closing(self):
        """Обработка закрытия приложения"""
        self.stop_video()
        self.root.destroy()


class SquatAnalyzer:
    def __init__(self):
        """Инициализация анализатора приседаний"""
        # Загрузка обученной модели
        if os.path.exists('squat_model.h5'):
            self.lstm_model = load_model('squat_model.h5')
            print("✅ LSTM модель загружена успешно")
        else:
            self.lstm_model = None
            print("⚠️  LSTM модель не найдена")

        # Загрузка параметров нормализации
        if os.path.exists('features_mean.npy') and os.path.exists('features_std.npy'):
            self.features_mean = np.load('features_mean.npy')
            self.features_std = np.load('features_std.npy')
            print("✅ Параметры нормализации загружены")
        else:
            self.features_mean = None
            self.features_std = None
            print("⚠️  Параметры нормализации не найдены")

        # Инициализация YOLO напрямую для получения keypoints
        self.yolo_model = YOLO('yolo11l-pose.pt')

        # Инициализация AIGym для анализа приседаний
        self.gym = solutions.AIGym(
            model="yolo11l-pose.pt",
            kpts=[11, 13, 15],  # ключевые точки: бедро, колено, ступня
            up_angle=145.0,  # угол "вверху"
            down_angle=90.0,  # угол "внизу"
            show=False,
            line_width=2,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Инициализация последовательности для LSTM
        self.sequence = []
        self.SEQUENCE_LENGTH = 30
        self.frame_count = 0
        self.video_path = None
        self.output_path = None
        self.squat_counter = 0
        self.last_counter = 0

        # Названия признаков (для объяснения)
        self.feature_names = [
            "right_knee", "left_knee", "right_hip", "left_hip",
            "dist_knees", "dist_feet", "depth", "knee_deviation",
            "wrist_distance", "wrist_shoulder_diff", "wrist_to_body_dist", "arm_body_angle",
            "knee_deviation_left",
            "knee_angle_diff", "hip_angle_diff", "knee_deviation_diff", "wrist_height_diff", "wrist_to_body_dist_diff", "avg_symmetry"
        ]

    def calculate_squat_features_v11(self, keypoints):
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
                                               (np.linalg.norm(left_arm_vector) * np.linalg.norm(body_vector)), -1.0,
                                               1.0))
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

    def analyze_frame_with_lstm(self, keypoints):
        """Анализ кадра с использованием LSTM модели + объяснение причин"""
        if self.lstm_model is None or self.features_mean is None or self.features_std is None:
            return None, None

        features = self.calculate_squat_features_v11(keypoints)
        if features is None:
            return None, None

        self.sequence.append(features)

        if len(self.sequence) >= self.SEQUENCE_LENGTH:
            sequence_array = np.array(self.sequence[-self.SEQUENCE_LENGTH:])
            sequence_normalized = (sequence_array - self.features_mean) / self.features_std

            prediction = self.lstm_model.predict(np.expand_dims(sequence_normalized, axis=0), verbose=0)[0][0]

            # Добавляем объяснение: какие признаки наиболее "неправильные"
            # Сравниваем текущие признаки со средними значениями
            current_norm = (np.array(features) - self.features_mean) / self.features_std
            deviations = np.abs(current_norm)

            # Находим индекс признака с максимальным отклонением
            max_dev_index = np.argmax(deviations)

            reason = self.feature_names[max_dev_index]

            # проверяем симметрию: если симметрия плохая — добавляем это как причину
            symmetry_feature_idx = 18  # avg_symmetry
            if deviations[symmetry_feature_idx] > 1.5:  # порог — можно настроить
                reason = "symmetry"

            return prediction, reason

        return None, None

    def process_frame(self, frame):
        """Обработка одного кадра"""
        # Используем YOLO напрямую для получения keypoints
        yolo_results = self.yolo_model(frame, verbose=False)
        keypoints = yolo_results[0].keypoints.xy.cpu().numpy()

        # Используем AIGym для получения аннотированного кадра и счетчика
        gym_results = self.gym(frame)

        lstm_prediction = None
        reason = None
        current_counter = self.squat_counter

        # Получаем ключевые точки для LSTM анализа
        if len(keypoints) > 0 and len(keypoints[0]) >= 17:
            kp = keypoints[0]

            # Анализ с LSTM моделью
            lstm_prediction, reason = self.analyze_frame_with_lstm(kp)

            # Проверяем счетчик приседаний из AIGym
            try:
                if hasattr(self.gym, 'count') and self.gym.count != self.last_counter:
                    self.squat_counter = self.gym.count
                    self.last_counter = self.gym.count
                elif hasattr(gym_results, 'count') and gym_results.count != self.last_counter:
                    self.squat_counter = gym_results.count
                    self.last_counter = gym_results.count
            except:
                pass

        # Получаем аннотированный кадр
        try:
            if hasattr(gym_results, 'plot'):
                annotated_frame = gym_results.plot()
            else:
                annotated_frame = frame.copy()
        except:
            annotated_frame = frame.copy()

        # Добавляем информацию о LSTM предсказании на кадр
        if lstm_prediction is not None:
            prediction_text = f"LSTM: {lstm_prediction:.2f} ({'pravilno' if lstm_prediction > 0.5 else 'ne pravilno'})"
            if reason:
                prediction_text += f"; {reason}"
            color = (0, 255, 0) if lstm_prediction > 0.5 else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                prediction_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Добавляем углы коленей
            def calculate_angle(a, b, c):
                a = np.array(a)
                b = np.array(b)
                c = np.array(c)
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                return np.degrees(angle)

            left_knee_angle = calculate_angle(kp[12], kp[14], kp[16])
            right_knee_angle = calculate_angle(kp[11], kp[13], kp[15])

            # Выводим углы на кадр
            cv2.putText(annotated_frame, f"{left_knee_angle:.2f}",
                        (int(kp[14][0]), int(kp[14][1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"{right_knee_angle:.2f}",
                        (int(kp[13][0]), int(kp[13][1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Добавляем номер кадра
        cv2.putText(
            annotated_frame,
            f"frame: {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return annotated_frame, lstm_prediction, reason, self.squat_counter


def main():
    """Основная функция"""
    root = tk.Tk()
    app = SquatAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()