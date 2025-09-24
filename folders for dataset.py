import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

def organize_gui():
    # Пути к папкам
    dataset_path = "dataset"
    correct_input_path = os.path.join(dataset_path, "train", "correct")
    incorrect_input_path = os.path.join(dataset_path, "train", "incorrect")
    other_input_path = os.path.join(dataset_path, "train", "other_exercises")

    # Создаём структуру, если не существует
    os.makedirs(correct_input_path, exist_ok=True)
    os.makedirs(incorrect_input_path, exist_ok=True)
    os.makedirs(other_input_path, exist_ok=True)

    # Классы ошибок
    error_classes = ["hands", "legs_width", "deep", "simmetry", "other_exercise"]

    print("Обработка папки 'correct'...")
    process_folder_gui(correct_input_path, error_classes, prefix="good")

    print("Обработка папки 'incorrect'...")
    process_folder_gui(incorrect_input_path, error_classes, prefix="bad")

    print("Обработка папки 'other_exercises'...")
    process_folder_gui(other_input_path, error_classes, prefix="other")

    print(f"✅ Готово! Видео организованы в нужную структуру.")


def process_folder_gui(input_dir, error_classes, prefix=""):
    if not os.path.exists(input_dir):
        print(f"❌ Папка {input_dir} не найдена.")
        return

    # Получаем список видео
    video_files = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isfile(item_path) and item.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(item)

    # Сортируем для предсказуемости
    video_files.sort()

    # Обрабатываем каждое видео
    for idx, video_name in enumerate(video_files, 1):
        folder_name = f"{idx:03d}_{prefix}"
        folder_path = os.path.join(input_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Перемещаем видео
        src_video = os.path.join(input_dir, video_name)
        dst_video = os.path.join(folder_path, "video.mp4")
        shutil.move(src_video, dst_video)

        # Если это правильный — просто создаем папку без меток
        if prefix == "good":
            print(f"  Создана папка: {folder_name} (без меток, т.к. правильный)")
        else:
            # Показываем GUI с превью видео и выбором ошибок
            selected_errors = select_errors_with_preview(dst_video, error_classes)
            # Создаём только выбранные метки
            for err in selected_errors:
                label_file = os.path.join(folder_path, f"{err}.txt")
                with open(label_file, 'w') as f:
                    pass  # пустой файл

            print(f"  Создана папка: {folder_name} с метками: {', '.join(selected_errors) if selected_errors else 'нет'}")


def select_errors_with_preview(video_path, error_classes):
    """Открывает GUI с превью видео и чекбоксами для выбора ошибок"""
    root = tk.Tk()
    root.title(f"Выберите ошибки для видео: {os.path.basename(video_path)}")
    root.geometry("800x600")

    # Получаем первый кадр видео
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        frame = None

    # Фрейм для превью
    preview_frame = tk.Frame(root)
    preview_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Заголовок
    label = tk.Label(preview_frame, text=f"Видео: {os.path.basename(video_path)}\nВыберите типы ошибок:", font=("Arial", 12))
    label.pack(pady=5)

    # Canvas для изображения
    canvas = tk.Canvas(preview_frame, width=400, height=300, bg="black")
    canvas.pack(pady=5)

    if frame is not None:
        # Конвертируем кадр в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Масштабируем под размер canvas
        pil_image.thumbnail((400, 300), Image.Resampling.LANCZOS)

        # Конвертируем в PhotoImage
        photo = ImageTk.PhotoImage(pil_image)

        # Отображаем
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # Сохраняем ссылку

    # Фрейм для чекбоксов
    checkbox_frame = tk.Frame(root)
    checkbox_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

    selected_vars = []
    checkboxes = []

    # Заголовок чекбоксов
    label = tk.Label(checkbox_frame, text="Типы ошибок:", font=("Arial", 12))
    label.pack(anchor="w", pady=5)

    # Чекбоксы
    for err in error_classes:
        var = tk.BooleanVar()
        cb = tk.Checkbutton(checkbox_frame, text=err, variable=var, font=("Arial", 10))
        cb.pack(anchor="w", padx=5, pady=2)
        selected_vars.append(var)
        checkboxes.append(cb)

    # Кнопка OK
    result = []
    def on_ok():
        nonlocal result
        result = [error_classes[i] for i, var in enumerate(selected_vars) if var.get()]
        root.destroy()

    btn = tk.Button(checkbox_frame, text="OK", command=on_ok, font=("Arial", 12), bg="lightgreen")
    btn.pack(pady=20)

    # Центрируем окно
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"800x600+{x}+{y}")

    root.mainloop()

    return result


if __name__ == "__main__":
    organize_gui()