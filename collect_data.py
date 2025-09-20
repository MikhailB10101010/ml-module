import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import numpy as np
import csv
import json
from datetime import datetime
import threading
import queue
import time
import uuid
from ultralytics import YOLO
from PIL import Image, ImageTk  # Добавлено для отображения кадров


class DataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Сбор данных для анализа приседаний")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        # Переменные состояния
        self.is_recording = False
        self.start_time = None
        self.video_writer = None
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.csv_file = None
        self.csv_writer = None
        self.current_label = "correct"  # Текущая метка: correct или incorrect

        # Создание интерфейса
        self.create_widgets()

        # Запуск фонового потока для обновления таймера
        self.update_timer()

        # Загрузка модели YOLO для отображения ключевых точек
        self.model = YOLO('yolov8l-pose.pt')

    def create_widgets(self):
        # Заголовок
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=10)
        title_label = ttk.Label(title_frame, text="Сбор данных для анализа приседаний", font=('Arial', 16, 'bold'))
        title_label.pack()

        # Контейнер для кнопок
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Кнопка выбора метки
        self.correct_btn = ttk.Button(button_frame, text="✅ Правильное приседание",
                                      command=lambda: self.set_label("correct"), width=20)
        self.correct_btn.pack(side=tk.LEFT, padx=5)

        self.incorrect_btn = ttk.Button(button_frame, text="❌ Неправильное приседание",
                                        command=lambda: self.set_label("incorrect"), width=20)
        self.incorrect_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка запуска/остановки записи
        self.record_btn = ttk.Button(button_frame, text="▶ Начать запись",
                                     command=self.toggle_recording, width=20)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка загрузки видео
        self.load_btn = ttk.Button(button_frame, text="📂 Загрузить видео",
                                   command=self.load_video_file, width=20)
        self.load_btn.pack(side=tk.LEFT, padx=5)

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

        # Информация о текущей метке
        self.label_info = ttk.Label(self.root, text="Текущая метка: правильное приседание",
                                    font=('Arial', 10, 'bold'), foreground="green")
        self.label_info.pack(side=tk.BOTTOM, pady=5)

    def set_label(self, label_type):
        """Установка текущей метки"""
        self.current_label = label_type
        if label_type == "correct":
            self.label_info.config(text="Текущая метка: правильное приседание", foreground="green")
            self.correct_btn.config(state='disabled')
            self.incorrect_btn.config(state='normal')
        else:
            self.label_info.config(text="Текущая метка: неправильное приседание", foreground="red")
            self.correct_btn.config(state='normal')
            self.incorrect_btn.config(state='disabled')

    def toggle_recording(self):
        """Переключение состояния записи"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Запуск записи видео и CSV"""
        self.is_recording = True
        self.start_time = time.time()
        self.record_btn.config(text="■ Остановить запись")
        self.status_label.config(text="Запись ведётся...", foreground="red")

        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Не удалось открыть камеру!")
            return

        # Установка разрешения камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Получаем размеры кадра
        ret, frame = self.cap.read()
        if not ret:
            self.show_error("Не удалось получить кадр с камеры!")
            return

        # Создаем уникальные имена файлов с учетом метки
        video_filename, csv_filename = self.generate_unique_filenames(self.current_label)

        # Создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_filename, fourcc, 20,
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
        self.status_label.config(text="Запись остановлена", foreground="gray")

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

            # Обработка кадра YOLO
            results = self.model(frame)
            annotated_frame = results[0].plot()

            # Запись в видеофайл
            self.video_writer.write(annotated_frame)

            # Получаем ключевые точки
            keypoints = results[0].keypoints.xy.cpu().numpy()
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

    def load_video_file(self):
        """Загрузка и обработка готового видеофайла"""
        video_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not video_path:
            return  # Пользователь отменил выбор

        # Убедимся, что выбрана метка
        if not messagebox.askyesno(
                "Подтверждение метки",
                f"Обработать видео как '{'правильное' if self.current_label == 'correct' else 'неправильное'}' приседание?"):
            return

        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.show_error("Не удалось открыть выбранный видеофайл!")
            return

        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Генерируем имена файлов
        video_filename, csv_filename = self.generate_unique_filenames(self.current_label)

        # Создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        # Создаем CSV-файл
        csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        headers = ['timestamp'] + [f'point_{i}_{coord}' for i in range(17) for coord in ['x', 'y']]
        csv_writer.writerow(headers)

        frame_count = 0
        self.status_label.config(text="Обработка видео...", foreground="orange")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Обработка YOLO
            results = self.model(frame)
            annotated_frame = results[0].plot()
            video_writer.write(annotated_frame)

            # Ключевые точки
            keypoints = results[0].keypoints.xy.cpu().numpy()
            timestamp = frame_count / fps

            if len(keypoints) > 0:
                row = [f"{timestamp:.3f}"]
                for x, y in keypoints[0]:
                    row.extend([f"{x:.3f}", f"{y:.3f}"])
                csv_writer.writerow(row)

            # Обновляем интерфейс каждые 30 кадров
            if frame_count % 30 == 0:
                self.display_frame(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                self.root.update_idletasks()

            frame_count += 1

        # Закрываем ресурсы
        cap.release()
        video_writer.release()
        csv_file.close()

        self.status_label.config(text="Видео обработано и сохранено!", foreground="green")
        messagebox.showinfo("Готово", f"Видео и данные успешно сохранены в папку:\n{os.path.dirname(video_filename)}")

    def generate_unique_filenames(self, label):
        """Генерация уникальных имен файлов с учетом метки (correct/incorrect)"""
        # Создаем папку на рабочем столе
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        base_records_folder = os.path.join(desktop_path, "Собранные данные")
        records_folder = os.path.join(base_records_folder, label)  # Подпапка correct или incorrect

        if not os.path.exists(records_folder):
            os.makedirs(records_folder)

        # Генерируем уникальное имя
        base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollector(root)
    root.mainloop()