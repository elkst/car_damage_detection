from ultralytics import YOLO
import torch


def main():
    # Проверяем доступность GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Загружаем предобученную модель YOLOv8m
    model = YOLO('yolov8m.pt')

    # Параметры обучения
    training_config = {
        'data': 'data.yaml',
        'epochs': 20,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'workers': 4,
        'save': True,
        'save_period': 10,
        'project': 'runs/train',
        'name': 'yolov8m_car_damage',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,  # начальная скорость обучения
        'lrf': 0.01,  # финальная скорость обучения
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'fliplr': 0.5,  # горизонтальный флип
        'flipud': 0.0,  # вертикальный флип (можно 0.1 если нужно)
        'mosaic': 1.0,
        'mixup': 0.15,  # увеличение данных
        'copy_paste': 0.3,  # увеличение данных
        'degrees': 10.0,  # поворот
        'translate': 0.1,  # смещение
        'scale': 0.5,  # масштабирование
        'shear': 2.0,  # сдвиг
        'perspective': 0.0,
        'hsv_h': 0.015,  # HSV augmentation
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    }

    # Тренировка модели
    results = model.train(**training_config)

    # Валидация на тестовом наборе
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")

    # Сохраняем лучшую модель
    best_model_path = 'best_car_damage_model.pt'
    print(f"Training completed. Best model saved.")


if __name__ == '__main__':
    main()
