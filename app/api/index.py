# api/index.py
import sys
import os

# Добавляем путь к корню проекта
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    # Пробуем импортировать из main.py в корне
    from main import app

    print("✓ Импортировано из main.py")
except ImportError:
    try:
        # Пробуем импортировать из app/main.py
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
        from main import app

        print("✓ Импортировано из app/main.py")
    except ImportError as e:
        print(f"✗ Ошибка импорта: {e}")
        # Создаём минимальное приложение
        from fastapi import FastAPI

        app = FastAPI()


        @app.get("/")
        async def root():
            return {
                "status": "running",
                "message": "FastAPI работает на Vercel"
            }