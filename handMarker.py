import collections
import os
import time

import cv2
import mediapipe as mp

import pandas as pd

import numpy as np
from numpy.linalg import norm

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

is_recording = False
last_save_time = 0
SAVE_INTERVAL_SECONDS = 0.3

GESTURES = {
    0: "fist", # кулак
    1: "open_hand", # открытая ладонь
    2: "pointing", # указательный палец
    3: "thumbs_up" # большой палец вверх
}

current_label_id = 0

cap = cv2.VideoCapture(0)


print("\n--- ИНСТРУКЦИЯ ПО СБОРУ ДАТАСЕТА ---")
print("Переключение жестов на клавиатуре:")
for gid, name in GESTURES.items():
    print(f"  Нажмите '{gid}' -> переключить на жест [{name.upper()}]")
print("2. Нажмите 'R' на клавиатуре, чтобы начать/остановить сбор жестов 3 раза в секунду базу данных.")
print("3. Нажмите 'Q', чтобы выйти и сохранить файл.\n")



CSV_FILE = "gestures_dataset.csv"

columns = []
for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])

columns.append("class")



if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE)
    print(f"Загружен существующий файл. Длина: {len(df_existing)}")
else:
    dtypes = {col: "float64" for col in columns if col != "class"}
    dtypes["class"] = "int64"

    df_existing = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})
    print("Создан новый файл датасета с заданными типами данных.")


new_rows_list = []


#For mapping - для сглаживания движения точки на конце указательного пальца
# x_cords = collections.deque(maxlen=3)
# y_cords = collections.deque(maxlen=3)


with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Не удалось получить кадр с веб-камеры.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #roi = image_rgb[roi_y1:roi_y2, roi_x1:roi_x2]

        results = hands.process(image_rgb)
        feature_vector = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                #Создаем массив из 21 подмассивов (21 * 3, x, y, z), для последующей передачи в ИИ
                cords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                #Центрируем все координаты относительно начала кисти
                cords = cords - cords[0]
                #Считаем расстояния от начала кисти до каждой точки
                distances = norm(cords, axis=1)

                max_distance = np.max(distances)

                if max_distance != 0:
                    cords = cords / max_distance

                feature_vector = cords.flatten()

                mp_drawing.draw_landmarks(image,hand_landmarks, mp_hands.HAND_CONNECTIONS)


        current_name = GESTURES[current_label_id].upper()
        current_time = time.time()

        if is_recording and feature_vector is not None and len(feature_vector) == 63:
            if current_time - last_save_time > SAVE_INTERVAL_SECONDS:
                row_data = feature_vector.tolist() + [current_label_id]
                new_rows_list.append(row_data)

                last_save_time = current_time


        cv2.putText(image, f"ACTIVE MODE: {current_name} ({current_label_id})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(image, f"Buffered rows: {len(new_rows_list)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(image, f"Record hand: {is_recording}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Hands', image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('0'):
            current_label_id = 0
            print(f"Переключено на жест: {GESTURES[0]}")
        elif key == ord('1'):
            current_label_id = 1
            print(f"Переключено на жест: {GESTURES[1]}")
        elif key == ord('2'):
            current_label_id = 2
            print(f"Переключено на жест: {GESTURES[2]}")
        elif key == ord('3'):
            current_label_id = 3
            print(f"Переключено на жест: {GESTURES[3]}")

        if key == ord('r') or key == ord('R'):
            is_recording = not is_recording

        elif key == ord('q') or key == ord('Q'):
            break


if new_rows_list:
    df_new = pd.DataFrame(new_rows_list, columns=columns)
    df_new["class"] = df_new["class"].astype("int64")
    df_total = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_total = df_existing

df_total.to_csv(CSV_FILE, index=False)
print(f"\nПроцесс завершен. Датасет сохранен в {CSV_FILE}. \nИтого строк в файле: {len(df_total)}")


if not df_total.empty:
    print("\nСтатистика по классам в файле:")
    print(df_total["class"].value_counts().rename(index=GESTURES))

cap.release()
cv2.destroyAllWindows()