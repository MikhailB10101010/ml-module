import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import cv2
import threading
import time
import queue
import os
from PIL import Image, ImageTk
import torch

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Pose Tracker (Nano)")
        self.root.geometry("800x600")

        # Проверка доступности CUDA
        if torch.cuda.is_available():
            print("CUDA доступна. Используется GPU")
            self.device = 'cuda'
        else:
            print("CUDA недоступна. Используется CPU (производительность может быть низкой)")
            self.device = 'cpu'

        # Загрузка модели Nano (менее требовательная к ресурсам)
        try:
            self.model = YOLO('yolov8n-pose.pt')
            print("Модель YOLOv8n-pose загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("Попробуйте скачать модель вручную с https://github.com/ultralytics/ultralytics/releases")
            exit(1)

        # Переменные состояния
        self.is_recording = False
        self.start_time = None
        self.video_writer = None
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)

        # Создание интерфейса
        self.create_widgets()

        # Запуск фонового потока для обновления таймера
        self.update_timer()

    def create_widgets(self):
        # Контейнер для кнопок
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Кнопка запуска/остановки
        self.record_btn = ttk.Button(button_frame, text="▶ Начать запись",
                                    command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Таймер
        self.timer_label = ttk.Label(button_frame, text="00:00:00",
                                    font=('Arial', 12))
        self.timer_label.pack(side=tk.LEFT, padx=10)

        # Метка для отображения видео
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def toggle_recording(self):
        """Переключение состояния записи"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Запуск записи видео"""
        self.is_recording = True
        self.start_time = time.time()
        self.record_btn.config(text="■ Остановить запись")

        # Инициализация камеры с оптимизированными параметрами
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Не удалось открыть камеру!")
            return

        # Установка меньшего разрешения для RTX 1650 (640x480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Получаем размеры кадра
        ret, frame = self.cap.read()
        if not ret:
            self.show_error("Не удалось получить кадр с камеры!")
            return

        # Создаем VideoWriter с уменьшенным FPS (15) для экономии ресурсов
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            'output_pose.mp4', fourcc, 15.0,
            (frame.shape[1], frame.shape[0])
        )

        # Запуск потока для обработки видео
        self.process_thread = threading.Thread(
            target=self.process_video,
            daemon=True
        )
        self.process_thread.start()

    def stop_recording(self):
        """Остановка записи видео"""
        self.is_recording = False
        self.record_btn.config(text="▶ Начать запись")

        # Освобождение ресурсов
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        print("Запись завершена. Файл сохранен как output_pose.mp4")

    def process_video(self):
        """Основной цикл обработки видео в отдельном потоке"""
        while self.is_recording:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Обработка кадра с использованием GPU (если доступен)
            results = self.model(frame, device=self.device)
            annotated_frame = results[0].plot()

            # Запись в файл (преобразуем RGB в BGR для OpenCV)
            self.video_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            # Передача кадра в очередь для отображения (RGB)
            try:
                self.frame_queue.put_nowait(annotated_frame)
            except queue.Full:
                pass

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

        # Конвертируем RGB (OpenCV) в RGB (PIL)
        pil_image = Image.fromarray(img_resized)
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

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()