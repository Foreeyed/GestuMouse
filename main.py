import collections
import mediapipe as mp

import cv2
import numpy as np

camera = cv2.VideoCapture(0)

frame_count = 0
# center_x, center_y = 960, 540

roi_x1, roi_x2 = 760, 1160
roi_y1, roi_y2 = 390, 690

roi_w = roi_x2 - roi_x1
roi_h = roi_y2 - roi_y1

SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1600

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

x_cords = collections.deque(maxlen=5)
y_cords = collections.deque(maxlen=5)


def from_roi_to_screen(local_cX, local_cY, roi_w, roi_h, screen_w, screen_h):
    percent_x = local_cX / roi_w
    percent_y = local_cY / roi_h

    local_screen_x = int(percent_x * screen_w)
    local_screen_y = int(percent_y * screen_h)

    screen_x = max(0, min(local_screen_x, screen_w - 1))
    screen_y = max(0, min(local_screen_y, screen_h - 1))

    return screen_x, screen_y


if camera.isOpened():
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            max_global_cX, max_global_cY = 0, 0
            screen_x, screen_y = 0, 0

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

            full_mask_frame = cv2.bitwise_or(mask1, mask2)
            full_mask_roi = full_mask_frame[roi_y1:roi_y2, roi_x1:roi_x2]

            contours, _ = cv2.findContours(full_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_cnt = None
            max_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100 and area > max_area:
                    max_area = area
                    max_cnt = cnt

            if max_cnt is not None:
                max_M = cv2.moments(max_cnt)
                if max_M["m00"] != 0:
                    max_local_cX = int(max_M["m10"] / max_M["m00"])
                    max_local_cY = int(max_M["m01"] / max_M["m00"])

                    x, y, w, h = cv2.boundingRect(max_cnt)
                    global_x = x + roi_x1
                    global_y = y + roi_y1

                    max_global_cX = max_local_cX + roi_x1
                    max_global_cY = max_local_cY + roi_y1

                    screen_x, screen_y = from_roi_to_screen(
                        max_local_cX, max_local_cY,
                        roi_w, roi_h,
                        SCREEN_WIDTH, SCREEN_HEIGHT
                    )

                    #print("Before mapping:", screen_x, screen_y)

                    x_cords.append(screen_x), y_cords.append(screen_y)

                    screen_x = int(sum(x_cords) / len(x_cords))
                    screen_y = int(sum(y_cords) / len(y_cords))

                    #print("After mapping:", screen_x, screen_y)

                    cv2.rectangle(frame, (global_x, global_y), (global_x + w, global_y + h), (0, 255, 0), 3)
                    cv2.circle(frame, (max_global_cX, max_global_cY), 5, (0, 255, 0), -1)
                    #cv2.circle(frame, (screen_x, screen_y), 5, (0, 255, 0), -1)
            else:
                x_cords.clear()
                y_cords.clear()

            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

            #print(f"Кадр {frame_count} | Камера: ({max_global_cX}, {max_global_cY}) | Экран: ({screen_x}, {screen_y})")
            #result_frame = cv2.bitwise_and(frame, frame, mask=full_mask_frame)

            cv2.imshow('Camera', cv2.flip(frame,1))
            frame_count += 1
        else:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()