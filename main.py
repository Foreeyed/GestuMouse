import cv2
import time

import numpy as np

camera = cv2.VideoCapture(0)
# start = time.perf_counter()
# frame_count = 0
center_x, center_y = 960, 540
x1, x2 = 760, 1160
y1, y2 = 390, 690

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])



if camera.isOpened():
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
            full_mask = np.bitwise_or(mask1, mask2)
            result_hsv_frame = cv2.bitwise_and(frame, frame, mask=full_mask)
            cv2.imshow('hsv_frame', result_hsv_frame)
            # frame_count += 1




        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break




# end = time.perf_counter()
#
# elapsed = end - start
# fps = frame_count / elapsed
# time_avg = elapsed / frame_count * 1000

camera.release()
cv2.destroyAllWindows()

# print("Elapsed time:", elapsed)
# print("FPS:", fps)
# print("Average frame time:", time_avg)

