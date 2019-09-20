import math
import cv2
import numpy as np


def find_best_line(detected_lines):
    max_distance = 0
    line_point_one, line_point_two = (0, 0)
    for i in range(0, len(detected_lines)):
        line = detected_lines[i][0]

        distance = calculate_point_distance(line)
        if distance > max_distance:
            line_point_one = (line[0], line[1])
            line_point_two = (line[2], line[3])

            max_distance = distance

    return line_point_one, line_point_two


def calculate_midpoint(first_point, second_point):
    return (first_point[0] + second_point[0])/2, (first_point[1] + second_point[1])/2


def calculate_point_distance(line):
    return math.sqrt( math.pow((line[2]-line[0]), 2) + math.pow((line[3]-line[1]), 2))


def preprocess_image(image):
    kernel_1 = np.ones((1, 1), np.uint8)
    kernel_2 = np.ones((2, 2), np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)

    eroded = cv2.erode(thresh, kernel_1, 1)
    dilated = cv2.dilate(eroded, kernel_2, 1)

    return dilated, thresh


def find_center(roi):
    (x, y ,w ,h) = roi
    cx = int(x+ w/2)
    cy = int(y+ h/2)
    return int(cx),int(cy)


def pnt2line(pnt, start, end):
    try:
        line_vec = vector(start, end)
        pnt_vec = vector(start, pnt)
        line_len = length(line_vec)
        line_unitvec = unit(line_vec)
        pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
        t = dot(line_unitvec, pnt_vec_scaled)
        r = 1
        if t < 0.0:
            t = 0.0
            r = -1
        elif t > 1.0:
            t = 1.0
            r = -1
        nearest = scale(line_vec, t)
        dist = distance(nearest, pnt_vec)
        nearest = add(nearest, start)
        return (dist, (int(nearest[0]), int(nearest[1])), r, False)
    except Exception as e:
        print("[ERROR]: Skipping frame due to the error - ", e)
        return (-1,-1,-1, True)


def dot(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return X - x, Y - y


def unit(v):
    x, y = v
    mag = length(v)
    return x / mag, y / mag


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    x, y = v
    return x * sc, y * sc


def add(v, w):
    x, y = v
    X, Y = w

    return x + X, y + Y
