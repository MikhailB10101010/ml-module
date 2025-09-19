import tkinter as tk
from tkinter import ttk, filedialog
from ultralytics import YOLO
import cv2
import threading
import time
import queue
import os
import csv
from PIL import Image, ImageTk
import json
import subprocess
import sys
from datetime import datetime
import uuid


class YOLOApp:
    def __init__(self, root, config=None):
        self.root = root
        self.root.title("YOLOv8 Pose Tracker - CSV Export")
        self.root.geometry("900x700")

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

        # Путь к CSV-файлу
        self.csv_path = 'keypoints.csv'

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

    def create_widgets(self):
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

        # Таймер
        self.timer_label = ttk.Label(button_frame, text="00:00:00",
                                     font=('Arial', 14, 'bold'), foreground="blue")
        self.timer_label.pack(side=tk.LEFT, padx=10)

        # Метка для отображения видео
        self.video_label = ttk.Label(self.root, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Индикатор записи
        self.status_label = ttk.Label(self.root, text="Готово", font=('Arial', 10), foreground="gray")
        self.status_label.pack(side=tk.BOTTOM, pady=5)

        # Информация о конфигурации
        config_text = f"Модель: {self.config['model_size']} | Разрешение: {self.config['resolution'][0]}x{self.config['resolution'][1]} | FPS: {self.config['fps']}"
        config_label = ttk.Label(self.root, text=config_text, font=('Arial', 9), foreground="gray")
        config_label.pack(side=tk.BOTTOM, pady=2)

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
        # Проверяем, существует ли test.py
        if os.path.exists('test.py'):
            result = subprocess.run([sys.executable, 'test.py'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                print("Ошибка при запуске test.py:", result.stderr)
                return False
        else:
            print("Файл test.py не найден")
            return False
    except Exception as e:
        print(f"Ошибка при запуске конфигурации: {e}")
        return False


if __name__ == "__main__":
    # Проверка наличия моделей
    if not check_models():
        exit(1)

    # Запуск конфигурации
    print("Запуск тестирования конфигурации...")
    if not run_configuration():
        print("Продолжаем с настройками по умолчанию...")

    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()