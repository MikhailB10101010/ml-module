import os
import tempfile
import threading
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES

last_video_path = None
# Блокировка, чтобы безопасно читать/писать last_video_path из разных потоков
last_video_lock = threading.Lock()

router = APIRouter()

@router.get("/health")
async def health(request: Request):
    analyzer = request.app.state.squat_analyzer_service
    
    if analyzer is None:
        return JSONResponse(
          status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
          content={"status": "unavailable", "error": "Service not initialized"}
        )
    
    models_ok = analyzer.yolo_model is not None and analyzer.lstm_model is not None
    
    return JSONResponse(
      status_code=status.HTTP_200_OK,
      content={'status': 'healthy', 'models_loaded': models_ok}
    )

@router.post("/process-video")
def upload_and_process(request: Request):
    analyzer = request.app.state.squat_analyzer_service

    # Проверяем, есть ли файл в запросе
    if 'file' not in request.files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'No file part in the request'}
        )

    file = request.files['file']
    if file.filename == '':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'No file selected'}
        )
    
    # Проверяем расширение и mimeType файла
    file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if file_extension not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Недопустимый формат видео. Разрешены: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    expected_mime_type = ALLOWED_MIME_TYPES[file_extension]
    if file.content_type != expected_mime_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Несоответствие типа файла. Для {file_extension} ожидается {expected_mime_type}"
        )

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
            return FileResponse(
                path=output_path,
                filename=f"processed_{file.filename}",
                media_type='video/mp4'
            )
        
        else:
            print("Ошибка при обработке видео.")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={'error': 'Error processing video'}
            )

    except Exception as e:
        print(f"Ошибка при обработке запроса: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': 'Internal server error'}
        )
    finally:
        # Удаляем временный входной файл после обработки или ошибки
        try:
            os.close(input_fd)
            os.remove(input_path)
        except OSError:
            pass
        # Выходной файл (output_path) сразу НЕ удаляем, он нужен для отправки клиенту и для хранения как 'last_video'. Он будет удалён позже, при перезаписи.

@router.get("/last-video")
async def get_last_video():
    with last_video_lock:
        global last_video_path
        # Проверяем, есть ли файл и существует ли он
        if last_video_path and os.path.exists(last_video_path):
            print(f"Отправка последнего обработанного видео: {last_video_path}")
            # Отправляем файл клиенту
            return FileResponse(
                path=last_video_path,
                filename="last_processed_video.mp4",
                media_type='video/mp4',
                headers={'Content-Disposition': 'attachment; filename="last_processed_video.mp4"'}
            )
        else:
            print("Последнее видео не найдено.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Not found'
            )