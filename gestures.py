"""Распознавание жестов: извлечение признаков из landmark'ов MediaPipe и предсказание модели."""

import numpy as np
from joblib import load
from numpy.linalg import norm

GESTURES = {
    0: "fist",       # кулак
    1: "open_hand",  # открытая ладонь
    2: "pointing",   # указательный палец
    3: "thumbs_up",  # большой палец вверх
}

MODEL_PATH = "model.pkl"


def extract_features(hand_landmarks):
    """Превращает 21 landmark руки в нормализованный вектор из 63 признаков.

    Логика должна совпадать с collect_data.py, иначе модель получит данные
    в другом масштабе.
    """
    cords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Центрируем координаты относительно запястья (landmark 0)
    cords = cords - cords[0]

    # Нормализуем по максимальному расстоянию от запястья
    distances = norm(cords, axis=1)
    max_distance = np.max(distances)
    if max_distance != 0:
        cords = cords / max_distance

    return cords.flatten()


class GestureClassifier:
    """Обёртка над обученной моделью (model.pkl)."""

    def __init__(self, model_path=MODEL_PATH):
        self.model = load(model_path)

    def predict(self, hand_landmarks):
        """Возвращает название жеста для landmark'ов руки."""
        features = extract_features(hand_landmarks)
        label = int(self.model.predict([features])[0])
        return GESTURES.get(label, str(label))

    def predict_proba(self, hand_landmarks):
        """Возвращает уверенность модели в предсказанном жесте (0..1)."""
        features = extract_features(hand_landmarks)
        probabilities = self.model.predict_proba([features])[0]
        return float(np.max(probabilities))
