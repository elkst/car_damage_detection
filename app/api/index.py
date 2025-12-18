# api/index.py
import sys
import os

# КРИТИЧЕСКИ ВАЖНО: импортируем из ПАПКИ app
from app.main import app  # ← ЗДЕСЬ ИЗМЕНЕНИЕ!

print("✅ FastAPI app загружен")