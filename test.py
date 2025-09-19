import cv2
import torch
import time
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from ultralytics import YOLO


class HardwareTester:
    def __init__(self):
        self.results = {
            'gpu_available': False,
            'gpu_name': '',
            'cuda_version': '',
            'cpu_count': 0,
            'ram_gb': 0,
            'camera_available': False,
            'camera_resolution': (0, 0),
            'max_fps': 0
        }

    def test_gpu(self):
        """Тестирование GPU"""
        self.results['gpu_available'] = torch.cuda.is_available()
        if self.results['gpu_available']:
            self.results['gpu_name'] = torch.cuda.get_device_name(0)
            self.results['cuda_version'] = torch.version.cuda
        return self.results['gpu_available']

    def test_cpu_ram(self):
        """Тестирование CPU и RAM"""
        import psutil
        import multiprocessing

        self.results['cpu_count'] = multiprocessing.cpu_count()
        self.results['ram_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    def test_camera(self):
        """Тестирование камеры"""
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            self.results['camera_available'] = True
            # Проверяем максимальное разрешение
            resolutions = [
                (1920, 1080),  # Full HD
                (1280, 720),  # HD
                (1024, 768),  # XGA
                (800, 600),  # SVGA
                (640, 480)  # VGA
            ]

            for width, height in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if actual_width == width and actual_height == height:
                    self.results['camera_resolution'] = (width, height)
                    break

            # Тест FPS
            self.results['max_fps'] = self._test_camera_fps(cap)

        cap.release()

    def _test_camera_fps(self, cap):
        """Тестирование FPS камеры"""
        frames = 30
        start_time = time.time()

        for _ in range(frames):
            ret, _ = cap.read()
            if not ret:
                return 0

        end_time = time.time()
        fps = frames / (end_time - start_time)
        return round(fps)

    def run_all_tests(self):
        """Запуск всех тестов"""
        print("Запуск тестирования системы...")
        self.test_gpu()
        self.test_cpu_ram()
        self.test_camera()
        return self.results


def recommend_configuration(test_results):
    """Рекомендация конфигурации на основе результатов теста"""
    config = {
        'model_size': 'nano',  # nano, medium, large
        'resolution': (640, 480),
        'fps': 20,
        'use_gpu': False
    }

    # Определяем размер модели
    if test_results['gpu_available']:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if gpu_memory >= 8 and test_results['ram_gb'] >= 16:
            config['model_size'] = 'large'
        elif gpu_memory >= 4 and test_results['ram_gb'] >= 8:
            config['model_size'] = 'medium'
        else:
            config['model_size'] = 'nano'
        config['use_gpu'] = True
    else:
        # На CPU используем только nano модель
        config['model_size'] = 'nano'
        config['use_gpu'] = False

    # Определяем разрешение
    cam_width, cam_height = test_results['camera_resolution']
    if config['model_size'] == 'large' and cam_width >= 1280:
        config['resolution'] = (1280, 720)
    elif config['model_size'] == 'medium' and cam_width >= 1024:
        config['resolution'] = (1024, 768)
    else:
        config['resolution'] = (640, 480)

    # Определяем FPS
    if test_results['max_fps'] >= 30:
        config['fps'] = 30
    elif test_results['max_fps'] >= 20:
        config['fps'] = 20
    else:
        config['fps'] = 15

    return config


def save_configuration(config, filename='config.json'):
    """Сохранение конфигурации в файл"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)


def load_configuration(filename='config.json'):
    """Загрузка конфигурации из файла"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


class ConfigGUI:
    def __init__(self, test_results, recommended_config):
        self.root = tk.Tk()
        self.root.title("Конфигурация YOLO Pose Tracker")
        self.root.geometry("600x500")
        self.test_results = test_results
        self.recommended_config = recommended_config
        self.selected_config = recommended_config.copy()

        self.create_widgets()

    def create_widgets(self):
        # Заголовок
        title_label = ttk.Label(self.root, text="Настройка конфигурации",
                                font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        # Результаты тестирования
        test_frame = ttk.LabelFrame(self.root, text="Результаты тестирования", padding=10)
        test_frame.pack(fill=tk.X, padx=10, pady=5)

        test_text = f"""GPU: {'Да' if self.test_results['gpu_available'] else 'Нет'}
{'GPU: ' + self.test_results['gpu_name'] if self.test_results['gpu_available'] else ''}
CPU: {self.test_results['cpu_count']} ядер
RAM: {self.test_results['ram_gb']} ГБ
Камера: {'Да' if self.test_results['camera_available'] else 'Нет'}
Разрешение: {self.test_results['camera_resolution'][0]}x{self.test_results['camera_resolution'][1]}
Макс. FPS: {self.test_results['max_fps']}"""

        test_label = ttk.Label(test_frame, text=test_text, justify=tk.LEFT)
        test_label.pack()

        # Настройки конфигурации
        config_frame = ttk.LabelFrame(self.root, text="Настройки конфигурации", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        # Размер модели
        model_frame = ttk.Frame(config_frame)
        model_frame.pack(fill=tk.X, pady=2)

        ttk.Label(model_frame, text="Размер модели:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.recommended_config['model_size'])
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                   values=['nano', 'medium', 'large'], state='readonly')
        model_combo.pack(side=tk.RIGHT)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)

        # Разрешение
        res_frame = ttk.Frame(config_frame)
        res_frame.pack(fill=tk.X, pady=2)

        ttk.Label(res_frame, text="Разрешение:").pack(side=tk.LEFT)
        self.res_var = tk.StringVar(
            value=f"{self.recommended_config['resolution'][0]}x{self.recommended_config['resolution'][1]}")
        resolutions = ['640x480', '1024x768', '1280x720', '1920x1080']
        res_combo = ttk.Combobox(res_frame, textvariable=self.res_var,
                                 values=resolutions, state='readonly')
        res_combo.pack(side=tk.RIGHT)

        # FPS
        fps_frame = ttk.Frame(config_frame)
        fps_frame.pack(fill=tk.X, pady=2)

        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.IntVar(value=self.recommended_config['fps'])
        fps_combo = ttk.Combobox(fps_frame, textvariable=self.fps_var,
                                 values=[15, 20, 25, 30], state='readonly')
        fps_combo.pack(side=tk.RIGHT)

        # Использование GPU
        gpu_frame = ttk.Frame(config_frame)
        gpu_frame.pack(fill=tk.X, pady=2)

        ttk.Label(gpu_frame, text="Использовать GPU:").pack(side=tk.LEFT)
        self.gpu_var = tk.BooleanVar(value=self.recommended_config['use_gpu'])
        gpu_check = ttk.Checkbutton(gpu_frame, variable=self.gpu_var)
        gpu_check.pack(side=tk.RIGHT)

        # Кнопки
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Применить и запустить",
                   command=self.apply_and_run).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Сохранить конфигурацию",
                   command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Выход",
                   command=self.root.destroy).pack(side=tk.LEFT, padx=5)

    def on_model_change(self, event=None):
        """Обработчик изменения размера модели"""
        model_size = self.model_var.get()
        if model_size == 'large' and self.test_results['gpu_available']:
            # Для large модели рекомендуем более высокое разрешение
            self.res_var.set('1280x720')
            self.fps_var.set(min(20, self.test_results['max_fps']))
        elif model_size == 'medium':
            self.res_var.set('1024x768')
            self.fps_var.set(min(25, self.test_results['max_fps']))
        else:
            self.res_var.set('640x480')
            self.fps_var.set(min(30, self.test_results['max_fps']))

    def get_selected_config(self):
        """Получение выбранной конфигурации"""
        width, height = map(int, self.res_var.get().split('x'))
        return {
            'model_size': self.model_var.get(),
            'resolution': (width, height),
            'fps': self.fps_var.get(),
            'use_gpu': self.gpu_var.get()
        }

    def apply_and_run(self):
        """Применить конфигурацию и запустить основной скрипт"""
        self.selected_config = self.get_selected_config()
        save_configuration(self.selected_config)
        self.root.destroy()
        return True

    def save_config(self):
        """Сохранить конфигурацию"""
        config = self.get_selected_config()
        save_configuration(config)
        messagebox.showinfo("Сохранено", "Конфигурация сохранена в config.json")

    def run(self):
        """Запуск GUI"""
        self.root.mainloop()
        return self.selected_config


def run_test_and_config():
    """Основная функция для запуска тестирования и настройки"""
    # Запуск тестирования
    tester = HardwareTester()
    results = tester.run_all_tests()

    # Рекомендация конфигурации
    recommended_config = recommend_configuration(results)

    # Создание GUI для настройки
    gui = ConfigGUI(results, recommended_config)
    gui.run()

    return gui.selected_config


if __name__ == "__main__":
    config = run_test_and_config()
    print("Выбранная конфигурация:", config)