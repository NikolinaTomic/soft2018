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


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest
# distance from pnt to the line and the coordinates of the
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.
# Malcolm Kesson 16 Dec 2012


def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)

    line_len = length(line_vec)
    line_unitvec = unit(line_vec)

    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)

    return dist


def pnt2line2(pnt, start, end):
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

    return (dist, (int(nearest[0]), int(nearest[1])), r)


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
