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

            max_cnt = None
            max_area = 0

            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                #M = cv2.moments(cnt)

                if area > max_area:
                    max_area = area
                    max_cnt = cnt


                    #print(f"Контур №{i + 1}: Площадь = {area} пикселей, X:{global_x} Y:{global_y}, Width:{w} Height:{h}")
                    # cv2.rectangle(frame, (global_x, global_y), (global_x + w, global_y + h), (0, 255, 0), 3)
                    # cv2.circle(frame, (global_cX, global_cY), 5, (0, 255, 0), -1)
                    #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            if max_cnt is not None:
                max_M = cv2.moments(max_cnt)

                max_local_cX = int(max_M["m10"] / max_M["m00"])
                max_local_cY = int(max_M["m01"] / max_M["m00"])

                x, y, w, h = cv2.boundingRect(max_cnt)
                global_x = x + x1
                global_y = y + y1

                max_global_cX = max_local_cX + x1
                max_global_cY = max_local_cY + y1





                cv2.rectangle(frame, (global_x, global_y), (global_x + w, global_y + h), (0, 255, 0), 3)
                cv2.circle(frame, (max_global_cX, max_global_cY), 5, (0, 255, 0), -1)
            # cv2.rectangle(frame, (global_x, global_y), (global_x + w, global_y + h), (0, 255, 0), 3)
            # cv2.circle(frame, (global_cX, global_cY), 5, (0, 255, 0), -1)



            result_frame = cv2.bitwise_and(frame, frame, mask=full_mask_frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imshow('hsv_frame', frame)
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