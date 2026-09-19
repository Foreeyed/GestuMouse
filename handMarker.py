import collections

import cv2
import mediapipe as mp

import numpy as np
from numpy.linalg import norm

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

roi_x1, roi_x2 = 760, 1160
roi_y1, roi_y2 = 390, 690

roi_w = roi_x2 - roi_x1
roi_h = roi_y2 - roi_y1

cap = cv2.VideoCapture(0)

#For mapping - для сглаживания движения точки на конце указательного пальца
x_cords = collections.deque(maxlen=3)
y_cords = collections.deque(maxlen=3)


with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Не удалось получить кадр с веб-камеры.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #roi = image_rgb[roi_y1:roi_y2, roi_x1:roi_x2]

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                #Создаем массив из 21 подмассивов (21 * 3, x, y, z), для последующей передачи в ИИ
                cords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                #Центрируем все координаты относительно начала кисти
                cords = cords - cords[0]
                #Считаем расстояния от начала кисти до каждой точки
                distances = norm(abs(cords), axis=1)

                max_distance = np.max(distances)
                min_distance = np.min(distances)

                #Находим
                cords = cords / max_distance

                feature_vector = cords.flatten()

                #print(f"Number of features:", len(feature_vector))

                print("Max distance:", max_distance)
                # print("Min distance:", min_distance)
                # print("Feature vector:", feature_vector)

                mp_drawing.draw_landmarks(image,hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    #Для отрисовки конца указательного пальца
                    #index_finger_tip = hand_landmarks.landmark[8]
                    #
                    # if 0 <= index_finger_tip.x <= 1 and 0 <= index_finger_tip.y <= 1:
                    #
                    #     x = int(index_finger_tip.x * image.shape[1])
                    #     y = int(index_finger_tip.y * image.shape[0])
                    #
                    #     x_cords.append(x)
                    #     y_cords.append(y)
                    #
                    #     x = int(sum(x_cords) / len(x_cords))
                    #     y = int(sum(y_cords) / len(y_cords))


                        # if roi_x1 <= x < roi_x2 and roi_y1 <= y < roi_y2:
                        #     cv2.circle(image, (x, y ), 5, (0, 255, 0), -1)

        else:
            x_cords.clear()
            y_cords.clear()

        #Отрисовка ROI
        #cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        cv2.imshow('MediaPipe Hands', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
