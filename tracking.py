"""Отслеживание руки через MediaPipe Hands."""

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

INDEX_FINGER_TIP = 8


class HandTracker:
    """Обёртка над MediaPipe Hands: находит одну руку и её landmark'и."""

    def __init__(
        self,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ):
        self.hands = mp_hands.Hands(
            model_complexity=model_complexity,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb):
        """Возвращает landmark'и первой найденной руки или None."""
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None

    @staticmethod
    def index_tip(hand_landmarks):
        """Нормализованные координаты (x, y) кончика указательного пальца."""
        landmark = hand_landmarks.landmark[INDEX_FINGER_TIP]
        return landmark.x, landmark.y

    @staticmethod
    def draw(frame, hand_landmarks):
        """Рисует скелет руки на кадре."""
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    def close(self):
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
