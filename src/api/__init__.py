from fastapi import APIRouter
from .base import router as base_router

main_router = APIRouter(prefix="/api", tags=["API"])

main_router.include_router(base_router)