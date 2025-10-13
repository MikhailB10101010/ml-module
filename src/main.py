from contextlib import asynccontextmanager
from fastapi import FastAPI
from api import main_router
from services import SquatAnalyzerService

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Инициализация сервисов...")

    squat_analyzer_service = SquatAnalyzerService()
    app.state.squat_analyzer_service = squat_analyzer_service

    yield
    print("👋 Сервисы остановлены")

app = FastAPI(lifespan=lifespan)
app.include_router(main_router)