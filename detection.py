import numpy as np
import cv2
import math

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


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        if (h > 12 and w >= 12) or (h > 12 and w <= 4 and w >= 2):
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda x: x[1][0])

    sorted_regions = []
    sorted_regions_coords = []
    for region in regions_array:
        sorted_regions.append(region[0])
        sorted_regions_coords.append(region[1])

    return sorted_regions,sorted_regions_coords,


def search_for_detection(detections, detection):
    indices = []
    i = 0
    for d in detections:
        (X1, Y1) = detection.center
        (X2, Y2) = d.center

        distance = math.sqrt(math.pow((X1 - X2),2) + math.pow((Y1 - Y2),2))

        if distance < 20:
            indices.append(i)

        i += 1

    return indices
