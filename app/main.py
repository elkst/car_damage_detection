from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
from datetime import datetime
import uvicorn
from PIL import Image, ImageDraw, ImageFont


app = FastAPI(title="Car Damage Detector")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

(STATIC_DIR / "uploads").mkdir(exist_ok=True)
(STATIC_DIR / "results").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# КЛАССЫ
CLASSES = {
    0: 'Front-windscreen-damage',
    1: 'Headlight-damage',
    2: 'Rear-windscreen-Damage',
    3: 'Runningboard-Damage',
    4: 'Sidemirror-Damage',
    5: 'Taillight-Damage',
    6: 'bonnet-dent',
    7: 'boot-dent',
    8: 'doorouter-dent',
    9: 'fender-dent',
    10: 'front-bumper-dent',
    11: 'quaterpanel-dent',
    12: 'rear-bumper-dent',
    13: 'roof-dent'
}

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

COLORS = [
    (41, 128, 185), (230, 126, 34), (155, 89, 182), (46, 204, 113),
    (231, 76, 60), (241, 196, 15), (52, 152, 219), (149, 165, 166),
    (243, 156, 18), (22, 160, 133), (192, 57, 43), (142, 68, 173),
    (39, 174, 96), (211, 84, 0)
]

# Загрузка модели
try:
    from ultralytics import YOLO

    model_path = BASE_DIR / "best.pt"
    model = YOLO(str(model_path))
    print("✅ Модель загружена")
except Exception as e:
    print(f"❌ Ошибка модели: {e}")
    model = None


def draw_boxes_pil(image, detections, show_area=False):
    """Рисует боксы через PIL с опцией показа площади"""
    if not detections:
        return image

    draw = ImageDraw.Draw(image)

    # Используем шрифт по умолчанию - он есть всегда
    font = ImageFont.load_default()
    # Для default шрифта берем приблизительную высоту
    font_height = 12

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class_id']
        confidence = det['confidence']
        area = det.get('area', 0)

        # Получаем русское название класса
        class_name = CLASSES.get(class_id, f"Class_{class_id}")
        russian_name = RUSSIAN_NAMES.get(class_name, class_name)

        color = COLORS[class_id % len(COLORS)]

        # Рисуем прямоугольник (основной бокс)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Формируем текст в зависимости от настроек
        if show_area and area > 0:
            # Форматируем площадь для удобочитаемости
            if area > 1000000:
                area_text = f"{area / 1000000:.1f}M"
            elif area > 1000:
                area_text = f"{area / 1000:.1f}K"
            else:
                area_text = f"{area}"
            label = f"{russian_name}: {area_text}px²"
        else:
            # Обычный текст с процентом уверенности
            label = f"{russian_name}: {int(confidence * 100)}%"

        # Позиция для подложки - выше основного бокса
        text_y = max(y1 - font_height - 6, 0)  # Не выходим за верхнюю границу

        # Грубая оценка ширины текста (с запасом)
        text_width = len(label) * 8 + 10  # 8px на символ + 10px запаса

        # Не даем подложке выйти за правую границу изображения
        text_x2 = min(x1 + text_width, image.width)

        # Рисуем подложку (прямоугольник для текста)
        draw.rectangle([x1, text_y, text_x2, y1], fill=color)

        # Рисуем текст
        draw.text((x1 + 2, text_y), label, fill=(255, 255, 255), font=font)

    return image


def get_damage_severity(damage_percentage):
    """Определяет уровень серьезности повреждений"""
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
    """Форматирует площадь для удобочитаемости"""
    if area > 1000000:
        return f"{area / 1000000:.2f} млн px²"
    elif area > 1000:
        return f"{area / 1000:.1f} тыс px²"
    else:
        return f"{area} px²"


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "classes": CLASSES,
        "russian_names": RUSSIAN_NAMES
    })


@app.post("/detect/")
async def detect_car_damage(request: Request, file: UploadFile = File(...), show_area: bool = False):
    """Обработка изображения с опцией показа площади"""
    if not model:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"success": False, "error": "Модель не загружена"}
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_filename = f"{timestamp}_{file.filename}"
    upload_path = STATIC_DIR / "uploads" / upload_filename
    result_path = STATIC_DIR / "results" / upload_filename

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Открываем через PIL
        img_pil = Image.open(upload_path)

        # Конвертируем в RGB если нужно (например, если PNG с прозрачностью)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        width, height = img_pil.size

        # Сохраняем оригинал
        img_pil.save(upload_path, "JPEG")

        # Детекция
        results = model.predict(
            source=str(upload_path),
            conf=0.25,
            imgsz=320,
            save=False,
            verbose=False
        )

        detection_data = []
        class_counts = {}
        damage_classes_found = set()

        # Переменные для расчета площади
        total_damage_area = 0
        image_area = width * height

        for result in results:
            if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = CLASSES.get(class_id, f"Class_{class_id}")
                    russian_name = RUSSIAN_NAMES.get(class_name, class_name)

                    # Рассчитываем площадь текущего повреждения
                    damage_width = x2 - x1
                    damage_height = y2 - y1
                    damage_area = damage_width * damage_height
                    total_damage_area += damage_area

                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    damage_classes_found.add(class_name)

                    detection_data.append({
                        "class": class_name,
                        "russian_name": russian_name,
                        "class_id": class_id,
                        "confidence": round(confidence, 3),
                        "confidence_percent": int(confidence * 100),
                        "bbox": [x1, y1, x2, y2],
                        "width": damage_width,
                        "height": damage_height,
                        "area": damage_area,
                        "area_formatted": format_area(damage_area),
                        "area_percentage": round((damage_area / image_area) * 100, 2)
                    })

        # Рассчитываем общую статистику по площади
        damage_percentage = round((total_damage_area / image_area) * 100, 2) if image_area > 0 else 0
        avg_damage_area = round(total_damage_area / len(detection_data), 2) if detection_data else 0

        # Находим самое большое повреждение
        max_damage = max(detection_data, key=lambda x: x['area'], default=None)
        min_damage = min(detection_data, key=lambda x: x['area'], default=None)

        # Рисуем боксы с учетом опции show_area
        annotated_img = img_pil.copy()
        if detection_data:
            annotated_img = draw_boxes_pil(annotated_img, detection_data, show_area=show_area)

        # Сохраняем результат
        annotated_img.save(result_path, "JPEG", quality=95)

        # Формируем результат
        result_data = {
            "success": True,
            "filename": file.filename,
            "upload_filename": upload_filename,
            "upload_path": f"/static/uploads/{upload_filename}",
            "result_path": f"/static/results/{upload_filename}",
            "detections": detection_data,
            "statistics": {
                "total_detections": len(detection_data),
                "damage_types_found": len(damage_classes_found),
                "class_counts": class_counts,
                "detection_time": timestamp,
                "image_size": f"{width}x{height}",
                "image_area": image_area,
                "image_area_formatted": format_area(image_area),
                "total_damage_area": total_damage_area,
                "total_damage_area_formatted": format_area(total_damage_area),
                "damage_percentage": damage_percentage,
                "avg_damage_area": avg_damage_area,
                "avg_damage_area_formatted": format_area(avg_damage_area) if avg_damage_area > 0 else "0 px²",
                "damage_severity": get_damage_severity(damage_percentage),
                "max_damage": {
                    "class": max_damage['russian_name'] if max_damage else None,
                    "area": max_damage['area'] if max_damage else 0,
                    "area_formatted": format_area(max_damage['area']) if max_damage else "0 px²"
                } if max_damage else None,
                "min_damage": {
                    "class": min_damage['russian_name'] if min_damage else None,
                    "area": min_damage['area'] if min_damage else 0,
                    "area_formatted": format_area(min_damage['area']) if min_damage else "0 px²"
                } if min_damage else None
            }
        }

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result_data,
            "classes": CLASSES,
            "russian_names": RUSSIAN_NAMES,
            "show_area": show_area
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"success": False, "error": str(e)}
        })


@app.post("/api/v1/detect")
async def api_detect(file: UploadFile = File(...)):
    if not model:
        return JSONResponse({"success": False, "error": "Model not loaded"}, status_code=500)

    try:
        import io
        contents = await file.read()

        # Открываем изображение для предварительной обработки
        image_bytes = io.BytesIO(contents)
        img = Image.open(image_bytes)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        image_area = width * height

        # Сохраняем обратно в bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        results = model.predict(
            source=img_byte_arr,
            conf=0.25,
            imgsz=320,
            save=False,
            verbose=False
        )

        detections = []
        class_counts = {}
        total_damage_area = 0

        for result in results:
            if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = CLASSES.get(class_id, f"class_{class_id}")
                    russian_name = RUSSIAN_NAMES.get(class_name, class_name)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    # Расчет площади повреждения
                    damage_area = (x2 - x1) * (y2 - y1)
                    total_damage_area += damage_area

                    detections.append({
                        "class": class_name,
                        "russian_name": russian_name,
                        "confidence": round(confidence, 4),
                        "bbox": [x1, y1, x2, y2],
                        "area": damage_area,
                        "area_percentage": round((damage_area / image_area) * 100, 4)
                    })

        damage_percentage = round((total_damage_area / image_area) * 100, 2) if image_area > 0 else 0

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
    return JSONResponse({
        "success": True,
        "classes": CLASSES,
        "russian_names": RUSSIAN_NAMES,
        "total": len(CLASSES)
    })


@app.get("/api/v1/health")
async def health_check():
    return JSONResponse({
        "success": True,
        "model_loaded": model is not None,
        "status": "healthy" if model else "model_not_loaded",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)