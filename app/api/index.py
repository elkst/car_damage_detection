# api/index.py - точка входа для Vercel
import sys
import os

# Добавляем папку app в путь Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Импортируем ваше приложение из app/main.py
try:
    from main import app  # Импорт из папки app/

    print("✓ Успешно импортировано из app/main.py")
except ImportError as e:
    print(f"✗ Ошибка импорта из app/main.py: {e}")
    # Создаем минимальное приложение как fallback
    from fastapi import FastAPI

    app = FastAPI()


    @app.get("/")
    async def root():
        return {"message": "Приложение загружено, но основной импорт не удался"}