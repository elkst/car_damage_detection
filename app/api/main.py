# Импорт необходимых библиотек
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
from datetime import datetime
import uvicorn
from PIL import Image, ImageDraw, ImageFont

# Создаем веб-приложение FastAPI с названием
app = FastAPI(title="Car Damage Detector")

# Определяем пути к основным папкам
BASE_DIR = Path(__file__).parent  # Папка где находится этот файл
STATIC_DIR = BASE_DIR / "static"  # Папка для картинок, CSS, JS
TEMPLATES_DIR = BASE_DIR / "templates"  # Папка для HTML шаблонов

# Создаем папки если их нет
(STATIC_DIR / "uploads").mkdir(exist_ok=True)  # Для загруженных фото
(STATIC_DIR / "results").mkdir(exist_ok=True)  # Для обработанных фото

# Настраиваем FastAPI для работы с файлами и шаблонами
app.mount("../static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Список типов повреждений которые может найти модель
# Каждому номеру соответствует название повреждения на английском
CLASSES = {
    0: 'Front-windscreen-damage',  # Повреждение лобового стекла
    1: 'Headlight-damage',  # Повреждение фары
    2: 'Rear-windscreen-Damage',  # Повреждение заднего стекла
    3: 'Runningboard-Damage',  # Повреждение порогов
    4: 'Sidemirror-Damage',  # Повреждение зеркала
    5: 'Taillight-Damage',  # Повреждение заднего фонаря
    6: 'bonnet-dent',  # Вмятина капота
    7: 'boot-dent',  # Вмятина багажника
    8: 'doorouter-dent',  # Вмятина двери
    9: 'fender-dent',  # Вмятина крыла
    10: 'front-bumper-dent',  # Вмятина переднего бампера
    11: 'quaterpanel-dent',  # Вмятина боковины
    12: 'rear-bumper-dent',  # Вмятина заднего бампера
    13: 'roof-dent'  # Вмятина крыши
}

# Перевод названий повреждений на русский язык
# Для отображения в интерфейсе
RUSSIAN_NAMES = {
    'Front-windscreen-damage': 'Переднее стекло',
    'Headlight-damage': 'Передняя фара',
    'Rear-windscreen-Damage': 'Заднее стекло',
    'Runningboard-Damage': 'Пороги',
    'Sidemirror-Damage': 'Боковое зеркало',
    'Taillight-Damage': 'Задняя фара',
    'bonnet-dent': 'Вмятина капота',
    'boot-dent': 'Вмятина багажника',
    'doorouter-dent': 'Вмятина двери',
    'fender-dent': 'Вмятина крыла',
    'front-bumper-dent': 'Вмятина переднего бампера',
    'quaterpanel-dent': 'Вмятина боковины',
    'rear-bumper-dent': 'Вмятина заднего бампера',
    'roof-dent': 'Вмятина крыши'
}

# Разные цвета для разных типов повреждений
# Каждый цвет для своего класса повреждений
COLORS = [
    (41, 128, 185), (230, 126, 34), (155, 89, 182), (46, 204, 113),
    (231, 76, 60), (241, 196, 15), (52, 152, 219), (149, 165, 166),
    (243, 156, 18), (22, 160, 133), (192, 57, 43), (142, 68, 173),
    (39, 174, 96), (211, 84, 0)
]

# Загружаем модель для распознавания повреждений
# Модель обучена на 14 типах повреждений
try:
    from ultralytics import YOLO  # Библиотека для работы с YOLO

    # Путь к файлу с обученной моделью
    model_path = BASE_DIR / "best.pt"

    # Загружаем модель
    model = YOLO(str(model_path))
    print("✅ Модель загружена успешно")
except Exception as e:
    # Если не получилось загрузить модель
    print(f"❌ Ошибка при загрузке модели: {e}")
    model = None  # Устанавливаем модель в None


def draw_boxes_pil(image, detections, show_area=False):
    # Функция рисует прямоугольники вокруг повреждений на фото

    # Если повреждений нет, возвращаем фото как есть
    if not detections:
        return image

    # Создаем инструмент для рисования на фото
    draw = ImageDraw.Draw(image)

    # Используем стандартный шрифт (он всегда есть)
    font = ImageFont.load_default()
    font_height = 12  # Высота шрифта примерно 12 пикселей

    # Обрабатываем каждое найденное повреждение
    for det in detections:
        # Координаты прямоугольника: лево, верх, право, низ
        x1, y1, x2, y2 = det['bbox']

        # Номер типа повреждения (от 0 до 13)
        class_id = det['class_id']

        # Уверенность модели от 0 до 1
        confidence = det['confidence']

        # Площадь повреждения в пикселях
        area = det.get('area', 0)

        # Получаем название повреждения
        class_name = CLASSES.get(class_id, f"Class_{class_id}")
        russian_name = RUSSIAN_NAMES.get(class_name, class_name)

        # Выбираем цвет для этого типа повреждения
        color = COLORS[class_id % len(COLORS)]

        # Рисуем прямоугольник вокруг повреждения
        # outline - цвет рамки, width - толщина линии
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Формируем текст который покажем над прямоугольником
        if show_area and area > 0:
            # Показываем площадь повреждения
            if area > 1000000:  # Если больше миллиона
                area_text = f"{area / 1000000:.1f}M"  # В миллионах
            elif area > 1000:  # Если больше тысячи
                area_text = f"{area / 1000:.1f}K"  # В тысячах
            else:
                area_text = f"{area}"  # Просто число
            label = f"{russian_name}: {area_text}px²"
        else:
            # Показываем процент уверенности
            label = f"{russian_name}: {int(confidence * 100)}%"

        # Вычисляем где разместить текст
        # Текст будет выше прямоугольника
        text_y = max(y1 - font_height - 6, 0)

        # Примерно вычисляем ширину текста
        text_width = len(label) * 8 + 10

        # Не даем тексту выйти за правый край фото
        text_x2 = min(x1 + text_width, image.width)

        # Рисуем цветной фон для текста
        draw.rectangle([x1, text_y, text_x2, y1], fill=color)

        # Рисуем белый текст на цветном фоне
        draw.text((x1 + 2, text_y), label, fill=(255, 255, 255), font=font)

    # Возвращаем фото с нарисованными прямоугольниками
    return image


def get_damage_severity(damage_percentage):
    # Функция определяет насколько серьезны повреждения
    # На основе процента поврежденной площади

    if damage_percentage == 0:
        return "Нет повреждений"
    elif damage_percentage < 5:
        return "Незначительные повреждения"
    elif damage_percentage < 15:
        return "Средние повреждения"
    elif damage_percentage < 30:
        return "Серьезные повреждения"
    else:
        return "Критические повреждения"


def format_area(area):
    # Функция красиво форматирует площадь
    # Делает большие числа читаемыми

    if area > 1000000:
        return f"{area / 1000000:.2f} млн px²"
    elif area > 1000:
        return f"{area / 1000:.1f} тыс px²"
    else:
        return f"{area} px²"


@app.get("/")
async def home(request: Request):
    # Главная страница сайта
    # Показывает HTML шаблон с формой загрузки фото

    return templates.TemplateResponse("index.html", {
        "request": request,  # Данные запроса
        "result": None,  # Пока нет результата анализа
        "classes": CLASSES,  # Список типов повреждений
        "russian_names": RUSSIAN_NAMES  # Перевод на русский
    })


@app.post("/detect/")
async def detect_car_damage(request: Request, file: UploadFile = File(...), show_area: bool = False):
    # Обрабатывает загруженное фото автомобиля
    # Находит и отмечает повреждения

    # Проверяем загружена ли модель
    if not model:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"success": False, "error": "Модель не загружена"}
        })

    # Создаем уникальное имя файла с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_filename = f"{timestamp}_{file.filename}"

    # Пути где сохранить оригинал и результат
    upload_path = STATIC_DIR / "uploads" / upload_filename
    result_path = STATIC_DIR / "results" / upload_filename

    # Сохраняем загруженное фото
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Открываем фото для обработки
        img_pil = Image.open(upload_path)

        # Если фото не в RGB формате, конвертируем
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        # Получаем размеры фото
        width, height = img_pil.size

        # Сохраняем оригинал в правильном формате
        img_pil.save(upload_path, "JPEG")

        # Запускаем модель для поиска повреждений
        results = model.predict(
            source=str(upload_path),  # Путь к фото
            conf=0.25,  # Минимальная уверенность 25%
            imgsz=320,  # Размер фото для обработки
            save=False,  # Не сохранять автоматически
            verbose=False  # Не выводить подробности
        )

        # Подготовка данных о найденных повреждениях
        detection_data = []  # Список всех повреждений
        class_counts = {}  # Сколько каждого типа найдено
        damage_classes_found = set()  # Какие типы повреждений найдены

        # Для расчета площадей
        total_damage_area = 0  # Общая площадь всех повреждений
        image_area = width * height  # Общая площадь всего фото

        # Обрабатываем результаты работы модели
        for result in results:
            if hasattr(result, 'boxes') and result.boxes:
                # Обрабатываем каждый найденный объект
                for box in result.boxes:
                    # Координаты прямоугольника
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Тип повреждения
                    class_id = int(box.cls[0])

                    # Уверенность модели
                    confidence = float(box.conf[0])

                    # Получаем названия повреждения
                    class_name = CLASSES.get(class_id, f"Class_{class_id}")
                    russian_name = RUSSIAN_NAMES.get(class_name, class_name)

                    # Рассчитываем площадь этого повреждения
                    damage_width = x2 - x1
                    damage_height = y2 - y1
                    damage_area = damage_width * damage_height
                    total_damage_area += damage_area

                    # Обновляем счетчики
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    damage_classes_found.add(class_name)

                    # Сохраняем информацию о повреждении
                    detection_data.append({
                        "class": class_name,  # Английское название
                        "russian_name": russian_name,  # Русское название
                        "class_id": class_id,  # Номер типа
                        "confidence": round(confidence, 3),  # Уверенность
                        "confidence_percent": int(confidence * 100),  # В процентах
                        "bbox": [x1, y1, x2, y2],  # Координаты
                        "width": damage_width,  # Ширина
                        "height": damage_height,  # Высота
                        "area": damage_area,  # Площадь
                        "area_formatted": format_area(damage_area),  # Красиво
                        "area_percentage": round((damage_area / image_area) * 100, 2)  # Процент от фото
                    })

        # Рассчитываем общую статистику
        damage_percentage = round((total_damage_area / image_area) * 100, 2) if image_area > 0 else 0
        avg_damage_area = round(total_damage_area / len(detection_data), 2) if detection_data else 0

        # Находим самое большое и самое маленькое повреждение
        max_damage = max(detection_data, key=lambda x: x['area'], default=None)
        min_damage = min(detection_data, key=lambda x: x['area'], default=None)

        # Рисуем прямоугольники на копии фото
        annotated_img = img_pil.copy()
        if detection_data:
            annotated_img = draw_boxes_pil(annotated_img, detection_data, show_area=show_area)

        # Сохраняем обработанное фото
        annotated_img.save(result_path, "JPEG", quality=95)

        # Формируем данные для отображения на сайте
        result_data = {
            "success": True,  # Успешно обработано
            "filename": file.filename,  # Имя исходного файла
            "upload_filename": upload_filename,  # Уникальное имя
            "upload_path": f"/static/uploads/{upload_filename}",  # Путь к оригиналу
            "result_path": f"/static/results/{upload_filename}",  # Путь к результату
            "detections": detection_data,  # Список всех повреждений
            "statistics": {  # Общая статистика
                "total_detections": len(detection_data),  # Сколько всего найдено
                "damage_types_found": len(damage_classes_found),  # Сколько разных типов
                "class_counts": class_counts,  # Сколько каждого типа
                "detection_time": timestamp,  # Время обработки
                "image_size": f"{width}x{height}",  # Размеры фото
                "image_area": image_area,  # Площадь всего фото
                "image_area_formatted": format_area(image_area),  # Красиво
                "total_damage_area": total_damage_area,  # Общая площадь повреждений
                "total_damage_area_formatted": format_area(total_damage_area),  # Красиво
                "damage_percentage": damage_percentage,  # Процент повреждений
                "avg_damage_area": avg_damage_area,  # Средняя площадь повреждения
                "avg_damage_area_formatted": format_area(avg_damage_area) if avg_damage_area > 0 else "0 px²",
                # Красиво
                "damage_severity": get_damage_severity(damage_percentage),  # Оценка серьезности
                "max_damage": {  # Самое большое повреждение
                    "class": max_damage['russian_name'] if max_damage else None,
                    "area": max_damage['area'] if max_damage else 0,
                    "area_formatted": format_area(max_damage['area']) if max_damage else "0 px²"
                } if max_damage else None,
                "min_damage": {  # Самое маленькое повреждение
                    "class": min_damage['russian_name'] if min_damage else None,
                    "area": min_damage['area'] if min_damage else 0,
                    "area_formatted": format_area(min_damage['area']) if min_damage else "0 px²"
                } if min_damage else None
            }
        }

        # Показываем результат на сайте
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result_data,
            "classes": CLASSES,
            "russian_names": RUSSIAN_NAMES,
            "show_area": show_area  # Сохраняем настройку показа площади
        })

    except Exception as e:
        # Если произошла ошибка
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"success": False, "error": str(e)}
        })


@app.post("/api/v1/detect")
async def api_detect(file: UploadFile = File(...)):
    # API версия для программистов
    # Возвращает данные в JSON формате

    if not model:
        return JSONResponse({"success": False, "error": "Model not loaded"}, status_code=500)

    try:
        import io

        # Читаем загруженное фото
        contents = await file.read()

        # Открываем фото для обработки
        image_bytes = io.BytesIO(contents)
        img = Image.open(image_bytes)

        # Конвертируем если нужно
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Получаем размеры
        width, height = img.size
        image_area = width * height

        # Подготавливаем фото для модели
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # Запускаем модель
        results = model.predict(
            source=img_byte_arr,
            conf=0.25,
            imgsz=320,
            save=False,
            verbose=False
        )

        # Собираем данные о повреждениях
        detections = []
        class_counts = {}
        total_damage_area = 0

        for result in results:
            if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    # Координаты
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Названия
                    class_name = CLASSES.get(class_id, f"class_{class_id}")
                    russian_name = RUSSIAN_NAMES.get(class_name, class_name)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    # Площадь
                    damage_area = (x2 - x1) * (y2 - y1)
                    total_damage_area += damage_area

                    # Сохраняем данные
                    detections.append({
                        "class": class_name,
                        "russian_name": russian_name,
                        "confidence": round(confidence, 4),
                        "bbox": [x1, y1, x2, y2],
                        "area": damage_area,
                        "area_percentage": round((damage_area / image_area) * 100, 4)
                    })

        # Общая статистика
        damage_percentage = round((total_damage_area / image_area) * 100, 2) if image_area > 0 else 0

        # Возвращаем JSON ответ
        return JSONResponse({
            "success": True,
            "detections": detections,
            "statistics": {
                "total": len(detections),
                "classes": class_counts,
                "image_area": image_area,
                "total_damage_area": total_damage_area,
                "damage_percentage": damage_percentage,
                "damage_severity": get_damage_severity(damage_percentage)
            }
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/v1/classes")
async def get_classes():
    # API для получения списка типов повреждений

    return JSONResponse({
        "success": True,
        "classes": CLASSES,
        "russian_names": RUSSIAN_NAMES,
        "total": len(CLASSES)
    })


@app.get("/api/v1/health")
async def health_check():
    # API для проверки работы сервиса

    return JSONResponse({
        "success": True,
        "model_loaded": model is not None,
        "status": "healthy" if model else "model_not_loaded",
        "timestamp": datetime.now().isoformat()
    })


#if __name__ == "__main__":
#    # Запуск сервера
#    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

