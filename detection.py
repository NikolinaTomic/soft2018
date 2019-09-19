import numpy as np
import cv2
from utils import find_best_line, preprocess_image


def detect_lines(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    res_blue = cv2.bitwise_and(image, image, mask=mask_blue)
    canny_blue = cv2.Canny(res_blue, 50, 200, None, 3)

    lines_blue = cv2.HoughLinesP(canny_blue, 1, np.pi / 180, 50, None, 50, 20)

    detected_line_points = [[(0,0), (0,0)], [(0,0), (0,0)]]
    if lines_blue is not None:
        print("[INFO]: Detected {} blue lines in the given frame.".format(len(lines_blue)))

        if len(lines_blue) < 2:
            print("[ERROR]: Detected less then 2 blue lines in the given frame.")
            exit(1)

        detected_line_points[0][0], detected_line_points[0][1] = find_best_line(lines_blue)
        cv2.line(image, (detected_line_points[0][0][0], detected_line_points[0][0][1]), (detected_line_points[0][1][0], detected_line_points[0][1][1]),(0, 0, 255), 1, cv2.LINE_AA)

    lower_green = np.array([60, 100, 100])
    upper_green = np.array([70, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    res_green = cv2.bitwise_and(image, image, mask=mask_green)
    canny_green = cv2.Canny(res_green, 50, 200, None, 3)

    lines_green = cv2.HoughLinesP(canny_green, 1, np.pi / 180, 50, None, 50, 20)

    if lines_green is not None:
        print("[INFO]: Detected {} green lines in the given frame.".format(len(lines_green)))

        if len(lines_green) < 2:
            print("[ERROR]: Detected less then 2 green lines in the given frame.")
            exit(1)

        detected_line_points[0][0], detected_line_points[0][1] = find_best_line(lines_blue)
        cv2.line(image, (detected_line_points[1][0][0], detected_line_points[1][0][1]),
                 (detected_line_points[1][1][0], detected_line_points[1][1][1]), (0, 0, 255), 1, cv2.LINE_AA)

    return detected_line_points


def detect_numbers(frame, frame_idx):
    dilated, thresh = preprocess_image(frame)
    im2, contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_numbers = []
    for contour in contours:
        [x,y, width, height] = cv2.boundingRect(contour)
        number = thresh[y:y+height, x:x+width]

        center_x = x + (width / 2)
        center_y = y + (height /2)
        center = (center_x, center_y)

        found_number = tuple(frame_idx, center, number)
        found_numbers.append(found_number)

    return found_numbers


