import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO
import cv2
import threading
import time
import queue
import os
import csv
import numpy as np
import json
import subprocess
import sys
from datetime import datetime
import uuid
import tensorflow as tf
from tensorflow.keras.models import load_model
import math


class YOLOApp:
    def __init__(self, root, config=None):
        self.root = root
        self.root.title("YOLOv8 Pose Tracker - Приседания")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        # Загрузка конфигурации
        self.config = config or self.load_config()

        # Инициализация модели на основе конфигурации
        model_paths = {
            'nano': 'yolov8n-pose.pt',
            'medium': 'yolov8m-pose.pt',
            'large': 'yolov8l-pose.pt'
        }

        model_path = model_paths.get(self.config['model_size'], 'yolov8n-pose.pt')
        self.model = YOLO(model_path)

        # Установка устройства
        device = 'cuda' if self.config['use_gpu'] and self.check_cuda_available() else 'cpu'
        self.model.to(device)
        print(f"Используется устройство: {device}")

        # Переменные состояния
        self.is_recording = False
        self.is_processing_video = False
        self.start_time = None
        self.video_writer = None
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.csv_file = None
        self.csv_writer = None
        self.squat_model = None
        self.features_mean = None
        self.features_std = None
        self.sequence_length = 30
        self.feature_sequence = []

        # Путь к CSV-файлу
        self.csv_path = 'keypoints.csv'

        # Проверка наличия модели для анализа приседаний
        self.check_squat_model()

        # Создание интерфейса
        self.create_widgets()

        # Запуск фонового потока для обновления таймера
        self.update_timer()

    def load_config(self):
        """Загрузка конфигурации из файла"""
        default_config = {
            'model_size': 'nano',
            'resolution': [640, 480],
            'fps': 20,
            'use_gpu': True
        }

        if os.path.exists('config.json'):
            try:
                with open('config.json', 'r') as f:
                    return json.load(f)
            except:
                pass
        return default_config

    def check_cuda_available(self):
        """Проверка доступности CUDA"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def check_squat_model(self):
        """Проверка наличия модели анализа приседаний"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'squat_model.h5')
            mean_path = os.path.join(os.path.dirname(__file__), 'features_mean.npy')
            std_path = os.path.join(os.path.dirname(__file__), 'features_std.npy')

            if os.path.exists(model_path) and os.path.exists(mean_path) and os.path.exists(std_path):
                self.squat_model = load_model(model_path)
                self.features_mean = np.load(mean_path)
                self.features_std = np.load(std_path)
                self.squat_status = "Готово"
                self.squat_color = "green"
            else:
                self.squat_status = "Модель не загружена"
                self.squat_color = "red"
                print("⚠️ Модель анализа приседаний не найдена. Для работы необходима предобученная модель.")
        except Exception as e:
            self.squat_status = f"Ошибка: {str(e)}"
            self.squat_color = "red"
            print(f"❌ Ошибка загрузки модели: {e}")

    def create_widgets(self):
        # Заголовок
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=10)
        title_label = ttk.Label(title_frame, text="Анализ приседаний", font=('Arial', 16, 'bold'))
        title_label.pack()

        # Информация о модели
        model_info = ttk.Frame(self.root)
        model_info.pack(pady=5)
        ttk.Label(model_info, text=f"Модель YOLO: {self.config['model_size']} | ", font=('Arial', 10)).pack(
            side=tk.LEFT)
        ttk.Label(model_info, text=f"Статус анализа приседаний: {self.squat_status}",
                  foreground=self.squat_color, font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        # Контейнер для кнопок
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Кнопка запуска/остановки записи
        self.record_btn = ttk.Button(button_frame, text="▶ Начать запись",
                                     command=self.toggle_recording, width=20)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка обработки готового видео
        self.process_video_btn = ttk.Button(button_frame, text="🎬 Обработать видео",
                                            command=self.process_existing_video, width=20)
        self.process_video_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка обучения модели
        self.train_btn = ttk.Button(button_frame, text="📚 Обучить модель",
                                    command=self.train_model, width=20)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        # Таймер
        self.timer_label = ttk.Label(button_frame, text="00:00:00",
                                     font=('Arial', 14, 'bold'), foreground="blue")
        self.timer_label.pack(side=tk.LEFT, padx=10)

        # Метка для отображения видео
        self.video_label = ttk.Label(self.root, background="black", relief="solid", borderwidth=1)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Индикатор записи
        self.status_label = ttk.Label(self.root, text="Готово", font=('Arial', 10), foreground="gray")
        self.status_label.pack(side=tk.BOTTOM, pady=5)

        # Информация о конфигурации
        config_text = f"Разрешение: {self.config['resolution'][0]}x{self.config['resolution'][1]} | FPS: {self.config['fps']}"
        config_label = ttk.Label(self.root, text=config_text, font=('Arial', 9), foreground="gray")
        config_label.pack(side=tk.BOTTOM, pady=2)

        # Информация о работе
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(info_frame, text="Инструкция:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        info_text = "1. Нажмите 'Обучить модель', чтобы подготовить модель анализа приседаний\n" \
                    "2. Используйте 'Начать запись' для записи видео с камерой\n" \
                    "3. Используйте 'Обработать видео' для анализа готового видеофайла\n" \
                    "4. Для сбора данных используйте скрипт collect_data.py"
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, wraplength=850).pack(anchor=tk.W)

    def toggle_recording(self):
        """Переключение состояния записи"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Запуск записи видео и CSV"""
        if self.is_processing_video:
            self.show_error("Сначала остановите обработку видео!")
            return

        self.is_recording = True
        self.start_time = time.time()
        self.record_btn.config(text="■ Остановить запись")
        self.process_video_btn.config(state='disabled')
        self.train_btn.config(state='disabled')
        self.status_label.config(text="Запись ведётся...", foreground="red")

        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Не удалось открыть камеру!")
            return

        # Установка разрешения камеры
        width, height = self.config['resolution']
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Получаем размеры кадра
        ret, frame = self.cap.read()
        if not ret:
            self.show_error("Не удалось получить кадр с камеры!")
            return

        # Создаем уникальные имена файлов
        video_filename, csv_filename = self.generate_unique_filenames()

        # Создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_filename, fourcc, self.config['fps'],
            (frame.shape[1], frame.shape[0])
        )

        # Создаем CSV-файл
        self.csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        # Заголовки CSV: frame_time, point_0_x, point_0_y, ..., point_16_x, point_16_y
        headers = ['timestamp'] + [f'point_{i}_{coord}' for i in range(17) for coord in ['x', 'y']]
        self.csv_writer.writerow(headers)

        # Запуск потока для обработки видео
        self.process_thread = threading.Thread(
            target=self.process_camera_video,
            daemon=True
        )
        self.process_thread.start()

    def stop_recording(self):
        """Остановка записи видео и CSV"""
        self.is_recording = False
        self.record_btn.config(text="▶ Начать запись")
        self.process_video_btn.config(state='normal')
        self.train_btn.config(state='normal')
        self.status_label.config(text="Запись остановлена", foreground="gray")

        # Освобождение ресурсов
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()

    def process_existing_video(self):
        """Обработка готового видео файла"""
        if self.is_recording:
            self.show_error("Сначала остановите запись!")
            return

        # Открытие диалога выбора файла
        file_path = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=[
                ("Видео файлы", "*.mp4 *.avi *.mov *.mkv"),
                ("Все файлы", "*.*")
            ]
        )

        if not file_path:
            return

        # Проверка существования файла
        if not os.path.exists(file_path):
            self.show_error("Выбранный файл не существует!")
            return

        self.is_processing_video = True
        self.process_video_btn.config(text="⏹ Остановить обработку")
        self.record_btn.config(state='disabled')
        self.train_btn.config(state='disabled')
        self.status_label.config(text="Обработка видео...", foreground="orange")

        # Запуск потока для обработки видео
        self.process_thread = threading.Thread(
            target=self.process_video_file,
            args=(file_path,),
            daemon=True
        )
        self.process_thread.start()

    def stop_video_processing(self):
        """Остановка обработки видео"""
        self.is_processing_video = False
        self.process_video_btn.config(text="🎬 Обработать видео")
        self.record_btn.config(state='normal')
        self.train_btn.config(state='normal')
        self.status_label.config(text="Обработка остановлена", foreground="gray")

        # Освобождение ресурсов
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()

    def process_camera_video(self):
        """Основной цикл обработки видео с камеры"""
        while self.is_recording:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Обработка кадра
            device = 'cuda' if self.config['use_gpu'] and self.check_cuda_available() else 'cpu'
            results = self.model(frame, device=device)
            annotated_frame = results[0].plot()

            # Запись в видеофайл
            self.video_writer.write(annotated_frame)

            # Получаем ключевые точки
            keypoints = results[0].keypoints.xy.cpu().numpy()  # shape: [num_people, 17, 2]
            timestamp = time.time() - self.start_time

            # Запись в CSV: только первая обнаруженная персона
            if len(keypoints) > 0:
                row = [f"{timestamp:.3f}"]  # Время в секундах с точностью 0.001
                for x, y in keypoints[0]:  # Первая персона, 17 точек
                    row.extend([f"{x:.3f}", f"{y:.3f}"])
                self.csv_writer.writerow(row)

            # Обработка признаков для приседаний
            if len(keypoints) > 0 and self.squat_model is not None:
                kp = keypoints[0]  # Первая персона
                features = self.calculate_squat_features(kp)

                if features is not None:
                    self.feature_sequence.append(features)

                    # Если набрали достаточную последовательность
                    if len(self.feature_sequence) >= self.sequence_length:
                        # Нормализуем данные
                        features_array = np.array(self.feature_sequence[-self.sequence_length:])
                        features_normalized = (features_array - self.features_mean) / self.features_std

                        # Добавляем размерность для batch
                        features_normalized = np.expand_dims(features_normalized, axis=0)

                        # Предсказание
                        prediction = self.squat_model.predict(features_normalized)
                        is_correct = prediction[0][0] > 0.5

                        # Добавляем текст на кадр
                        text = "✅ Правильно" if is_correct else "❌ Неправильно"
                        color = (0, 255, 0) if is_correct else (0, 0, 255)
                        annotated_frame = cv2.putText(
                            annotated_frame,
                            text,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            2
                        )

            # Передача кадра в очередь для отображения
            try:
                self.frame_queue.put_nowait(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            except queue.Full:
                pass

        # Очистка после завершения
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()

    def process_video_file(self, video_path):
        """Обработка готового видео файла"""
        try:
            # Инициализация видео файла
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.show_error("Не удалось открыть видео файл!")
                self.stop_video_processing()
                return

            # Получаем параметры исходного видео
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Создаем уникальные имена файлов для результата
            video_filename, csv_filename = self.generate_unique_filenames(prefix="processed_")

            # Получаем размеры первого кадра
            ret, first_frame = self.cap.read()
            if not ret:
                self.show_error("Не удалось получить кадр из видео!")
                self.stop_video_processing()
                return

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Возвращаемся к началу

            # Создаем VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_filename, fourcc, fps,
                (first_frame.shape[1], first_frame.shape[0])
            )

            # Создаем CSV-файл
            self.csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            headers = ['frame_number'] + [f'point_{i}_{coord}' for i in range(17) for coord in ['x', 'y']]
            self.csv_writer.writerow(headers)

            frame_count = 0
            start_time = time.time()

            # Обработка кадров
            while self.is_processing_video:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1

                # Обработка кадра
                device = 'cuda' if self.config['use_gpu'] and self.check_cuda_available() else 'cpu'
                results = self.model(frame, device=device)
                annotated_frame = results[0].plot()

                # Запись в видеофайл
                self.video_writer.write(annotated_frame)

                # Получаем ключевые точки
                keypoints = results[0].keypoints.xy.cpu().numpy()

                # Запись в CSV: только первая обнаруженная персона
                if len(keypoints) > 0:
                    row = [frame_count]
                    for x, y in keypoints[0]:  # Первая персона, 17 точек
                        row.extend([f"{x:.3f}", f"{y:.3f}"])
                    self.csv_writer.writerow(row)

                # Обработка признаков для приседаний
                if len(keypoints) > 0 and self.squat_model is not None:
                    kp = keypoints[0]  # Первая персона
                    features = self.calculate_squat_features(kp)

                    if features is not None:
                        self.feature_sequence.append(features)

                        # Если набрали достаточную последовательность
                        if len(self.feature_sequence) >= self.sequence_length:
                            # Нормализуем данные
                            features_array = np.array(self.feature_sequence[-self.sequence_length:])
                            features_normalized = (features_array - self.features_mean) / self.features_std

                            # Добавляем размерность для batch
                            features_normalized = np.expand_dims(features_normalized, axis=0)

                            # Предсказание
                            prediction = self.squat_model.predict(features_normalized)
                            is_correct = prediction[0][0] > 0.5

                            # Добавляем текст на кадр
                            text = "✅ Правильно" if is_correct else "❌ Неправильно"
                            color = (0, 255, 0) if is_correct else (0, 0, 255)
                            annotated_frame = cv2.putText(
                                annotated_frame,
                                text,
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                color,
                                2
                            )

                # Передача кадра в очередь для отображения
                try:
                    self.frame_queue.put_nowait(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                except queue.Full:
                    pass

                # Обновление статуса
                if frame_count % 30 == 0:  # Обновляем каждые 30 кадров
                    elapsed_time = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    self.status_label.config(text=f"Обработка: {progress:.1f}%")

            # Завершение обработки
            self.stop_video_processing()

            if frame_count > 0:
                processing_time = time.time() - start_time
                print(f"Обработка завершена за {processing_time:.2f} секунд")
                print(f"Обработано {frame_count} кадров")
                print(f"Результаты сохранены в:")
                print(f"  Видео: {video_filename}")
                print(f"  CSV: {csv_filename}")

        except Exception as e:
            self.show_error(f"Ошибка при обработке видео: {str(e)}")
            self.stop_video_processing()

    def calculate_squat_features(self, keypoints):
        """Вычисление признаков для анализа приседаний"""
        # keypoints: numpy array shape [17, 2] (координаты точек)
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
            angle = np.arccos(cosine_angle)
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

    def generate_unique_filenames(self, prefix=""):
        """Генерация уникальных имен файлов"""
        # Создаем папку на рабочем столе
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        records_folder = os.path.join(desktop_path, "Записи работы модели")

        if not os.path.exists(records_folder):
            os.makedirs(records_folder)

        # Генерируем уникальный номер
        base_name = f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        video_filename = os.path.join(records_folder, f"{base_name}.mp4")
        csv_filename = os.path.join(records_folder, f"{base_name}.csv")

        # Проверяем существование файлов и добавляем номер
        counter = 1
        original_video = video_filename
        original_csv = csv_filename

        while os.path.exists(video_filename) or os.path.exists(csv_filename):
            name_without_ext = os.path.splitext(os.path.basename(original_video))[0]
            video_filename = os.path.join(records_folder, f"{name_without_ext}_{counter}.mp4")
            csv_filename = os.path.join(records_folder, f"{name_without_ext}_{counter}.csv")
            counter += 1

        return video_filename, csv_filename

    def update_timer(self):
        """Обновление таймера и отображения кадров"""
        if self.is_recording:
            elapsed = int(time.time() - self.start_time)
            hours, rem = divmod(elapsed, 3600)
            mins, secs = divmod(rem, 60)
            self.timer_label.config(text=f"{hours:02}:{mins:02}:{secs:02}")

        # Попытка получить новый кадр из очереди
        try:
            frame = self.frame_queue.get_nowait()
            self.display_frame(frame)
        except queue.Empty:
            pass

        # Повторный вызов через 33 мс (~30 FPS)
        self.root.after(33, self.update_timer)

    def display_frame(self, frame):
        """Отображение кадра в Label с использованием PIL"""
        # Масштабируем кадр
        img_resized = self.resize_frame(frame)

        # Создаём объект Image из массива
        pil_image = Image.fromarray(img_resized)

        # Конвертируем в PhotoImage для Tkinter
        photo = ImageTk.PhotoImage(image=pil_image)

        # Обновляем метку
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Сохраняем ссылку, чтобы избежать сборки мусора

    def resize_frame(self, frame):
        """Масштабирование кадра под размер окна"""
        width = self.video_label.winfo_width()
        height = self.video_label.winfo_height()

        if width <= 0 or height <= 0:
            return frame  # Не масштабируем, если размеры не известны

        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_width = min(width, int(height * aspect_ratio))
        new_height = min(height, int(width / aspect_ratio))

        return cv2.resize(frame, (new_width, new_height))

    def show_error(self, message):
        """Показ сообщения об ошибке"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Ошибка")
        ttk.Label(error_window, text=message, padding=20).pack()
        ttk.Button(error_window, text="OK", command=error_window.destroy).pack(pady=10)

    def train_model(self):
        """Запуск обучения модели"""
        if self.is_recording or self.is_processing_video:
            self.show_error("Сначала остановите запись или обработку видео!")
            return

        # Запускаем скрипт обучения в отдельном процессе
        try:
            subprocess.Popen([sys.executable, 'train_squat_model.py'])
            self.show_info("Запущено обучение модели. Пожалуйста, подождите...")
        except Exception as e:
            self.show_error(f"Ошибка при запуске обучения: {str(e)}")

    def show_info(self, message):
        """Показ информационного сообщения"""
        info_window = tk.Toplevel(self.root)
        info_window.title("Информация")
        ttk.Label(info_window, text=message, padding=20).pack()
        ttk.Button(info_window, text="OK", command=info_window.destroy).pack(pady=10)


def check_models():
    """Проверка наличия моделей"""
    models = ['yolov8n-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt']
    missing_models = []

    for model in models:
        if not os.path.exists(model):
            missing_models.append(model)

    if missing_models:
        print("❌ Отсутствуют модели:")
        for model in missing_models:
            print(f"  - {model}")
        print("Скачайте модели с https://github.com/ultralytics/ultralytics/releases")
        return False
    return True


def run_configuration():
    """Запуск скрипта конфигурации"""
    try:
        # Проверяем, существует ли config.json
        if not os.path.exists('config.json'):
            # Создаем конфиг по умолчанию
            default_config = {
                'model_size': 'nano',
                'resolution': [640, 480],
                'fps': 20,
                'use_gpu': True
            }
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=4)
            print("Создан файл конфигурации config.json")
        return True
    except Exception as e:
        print(f"Ошибка при конфигурации: {e}")
        return False


if __name__ == "__main__":
    # Проверка наличия моделей
    if not check_models():
        exit(1)

    # Запуск конфигурации
    print("Запуск конфигурации...")
    if not run_configuration():
        print("Продолжаем с настройками по умолчанию...")

    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()