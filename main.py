import cv2

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

            full_mask_frame = cv2.bitwise_or(mask1, mask2)
            full_mask_roi = full_mask_frame[y1:y2, x1:x2]

            contours, _ = cv2.findContours(full_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)

                # Опционально: можно отфильтровать слишком мелкий шум
                if area > 100 and M["m00"] != 0:
                    local_cX = int(M["m10"] / M["m00"])
                    local_cY = int(M["m01"] / M["m00"])

                    x, y, w, h = cv2.boundingRect(cnt)
                    global_x = x + x1
                    global_y = y + y1

                    global_cX = local_cX + x1
                    global_cY = local_cY + y1

                    #print(f"Контур №{i + 1}: Площадь = {area} пикселей, X:{global_x} Y:{global_y}, Width:{w} Height:{h}")
                    cv2.rectangle(frame, (global_x, global_y), (global_x + w, global_y + h), (0, 255, 0), 3)
                    cv2.circle(frame, (global_cX, global_cY), 5, (0, 255, 0), -1)
                    #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            result_frame = cv2.bitwise_and(frame, frame, mask=full_mask_frame)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imshow('hsv_frame', result_frame)
            # frame_count += 1q




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