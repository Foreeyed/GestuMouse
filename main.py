"""GestuMouse: управление курсором мыши жестами руки."""

import cv2

from gestures import GestureClassifier
from mouse_control import MouseController
from tracking import HandTracker

# Сколько кадров подряд жест должен держаться, чтобы сработало действие (анти-дребезг)
HOLD_FRAMES = 5

# Схема: жест -> действие
GESTURE_ACTIONS = {
    "pointing": "move",          # двигать курсор
    "fist": "click",             # одиночный клик
    "open_hand": "idle",         # ничего не делать
    "thumbs_up": "double_click", # двойной клик
}


def draw_status(frame, gesture, confidence, action):
    """Показывает текущий жест и действие в углу кадра."""
    if gesture is None:
        text = "No hand detected"
        color = (0, 0, 255)
    else:
        text = f"{gesture} ({confidence:.0%}) -> {action}"
        color = (0, 255, 0)

    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Не удалось открыть камеру.")
        return

    classifier = GestureClassifier()
    mouse = MouseController()

    current_gesture = None
    gesture_frames = 0

    try:
        with HandTracker() as tracker:
            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("Не удалось получить кадр с веб-камеры.")
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_landmarks = tracker.process(frame_rgb)

                display_gesture = None
                confidence = 0.0
                active_action = None

                if hand_landmarks is not None:
                    gesture = classifier.predict(hand_landmarks)
                    confidence = classifier.predict_proba(hand_landmarks)
                    display_gesture = gesture

                    # Анти-дребезг: считаем жест активным после HOLD_FRAMES кадров подряд
                    if gesture == current_gesture:
                        gesture_frames += 1
                    else:
                        current_gesture = gesture
                        gesture_frames = 1

                    tracker.draw(frame, hand_landmarks)

                    tip_x, tip_y = tracker.index_tip(hand_landmarks)
                    tip_px = (int(tip_x * frame.shape[1]), int(tip_y * frame.shape[0]))
                    cv2.circle(frame, tip_px, 8, (0, 255, 0), -1)

                    if gesture_frames == HOLD_FRAMES:
                        # Жест только что активировался — выполняем одноразовое действие
                        action = GESTURE_ACTIONS.get(gesture)
                        if action == "click":
                            mouse.click()
                        elif action == "double_click":
                            mouse.double_click()

                    if gesture_frames >= HOLD_FRAMES:
                        active_action = GESTURE_ACTIONS.get(gesture)
                        if active_action == "move":
                            mouse.move(tip_x, tip_y)
                else:
                    current_gesture = None
                    gesture_frames = 0
                    mouse.reset()

                draw_status(frame, display_gesture, confidence, active_action)

                cv2.imshow("GestuMouse", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
