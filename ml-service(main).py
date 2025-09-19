import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import cv2
import threading
import time
import queue
import os
import torch
from PIL import Image, ImageTk

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Pose Tracker")
        self.root.geometry("800x600")

        # Проверка CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Инициализация модели
        self.model = YOLO('yolov8n-pose.pt').to(self.device)

        # Переменные состояния
        self.is_recording = False
        self.start_time = None
        self.video_writer = None
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.recording_number = 1  # Номер текущего видео

        # Создание интерфейса
        self.create_widgets()

        # Автоматическое открытие камеры
        self.start_camera()

        # Запуск обновления интерфейса
        self.update_timer()

    def create_widgets(self):
        # Контейнер для кнопок
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Кнопка записи
        self.record_btn = ttk.Button(button_frame, text="▶ Начать запись",
                                    command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Таймер
        self.timer_label = ttk.Label(button_frame, text="00:00:00",
                                    font=('Arial', 12))
        self.timer_label.pack(side=tk.LEFT, padx=10)

        # Метка для видео
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def start_camera(self):
        """Инициализация камеры и запуск потока"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Не удалось открыть камеру!")
            return

        # Установка разрешения
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Запуск потока обработки кадров
        self.process_thread = threading.Thread(target=self.process_video, daemon=True)
        self.process_thread.start()

    def process_video(self):
        """Цикл обработки кадров"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Обработка кадра
            results = self.model(frame, device=self.device)
            annotated_frame = results[0].plot()

            # Запись в файл, если активна
            if self.is_recording:
                try:
                    self.video_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"Ошибка записи: {e}")

            # Отправка кадра в очередь для отображения
            try:
                self.frame_queue.put_nowait(annotated_frame)
            except queue.Full:
                pass

    def toggle_recording(self):
        """Переключение записи"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Начало записи видео"""
        self.is_recording = True
        self.start_time = time.time()
        self.record_btn.config(text="■ Остановить запись")

        # Создаем папку для сохранения, если её нет
        save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Recordings")
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла
        filename = f"output_{self.recording_number}.mp4"
        filepath = os.path.join(save_dir, filename)

        # Получаем размеры кадра
        ret, frame = self.cap.read()
        if not ret:
            self.show_error("Не удалось получить кадр с камеры!")
            return

        # Создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            filepath, fourcc, 15.0,
            (frame.shape[1], frame.shape[0])
        )

        self.recording_number += 1  # Увеличиваем номер для следующей записи

    def stop_recording(self):
        """Остановка записи"""
        self.is_recording = False
        self.record_btn.config(text="▶ Начать запись")

        if hasattr(self, 'video_writer'):
            self.video_writer.release()
            print(f"Запись завершена. Файл сохранен в: {self.video_writer.filename}")
            self.video_writer = None

    def update_timer(self):
        """Обновление таймера и отображения кадров"""
        if self.is_recording:
            elapsed = int(time.time() - self.start_time)
            hours, rem = divmod(elapsed, 3600)
            mins, secs = divmod(rem, 60)
            self.timer_label.config(text=f"{hours:02}:{mins:02}:{secs:02}")

        # Обновление кадра
        try:
            frame = self.frame_queue.get_nowait()
            self.display_frame(frame)
        except queue.Empty:
            pass

        self.root.after(33, self.update_timer)

    def display_frame(self, frame):
        """Отображение кадра в интерфейсе"""
        img_resized = self.resize_frame(frame)
        pil_image = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(image=pil_image)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def resize_frame(self, frame):
        """Масштабирование кадра под размер окна"""
        width = self.video_label.winfo_width()
        height = self.video_label.winfo_height()

        if width <= 0 or height <= 0:
            return frame

        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_width = min(width, int(height * aspect_ratio))
        new_height = min(height, int(width / aspect_ratio))

        return cv2.resize(frame, (new_width, new_height))

    def show_error(self, message):
        """Вывод ошибок"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Ошибка")
        ttk.Label(error_window, text=message, padding=20).pack()
        ttk.Button(error_window, text="OK", command=error_window.destroy).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()