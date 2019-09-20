import cv2
import os
import argparse
import matplotlib.pyplot as plt

from architecture import Model
from utils import preprocess_image, find_center, pnt2line
from detection import detect_lines, select_roi, search_for_detection
import numpy as np


class Detection:
    def __init__(self, passed_green_line = False, passed_blue_line=False, label=-1, center=tuple((-1,-1)), frame_idx=-1):
        self.passed_green_line = passed_green_line
        self.passed_blue_line = passed_blue_line
        self.label = label
        self.center = center
        self.frame_idx = frame_idx
        self.history = []

    def intersecting_blue(self, blue_line_points):
        passed_blue_line_before = self.passed_blue_line
        if not passed_blue_line_before:
            print(blue_line_points[0], blue_line_points[1])
            dist_blue, pnt, r_blue, skip_frame = pnt2line(detection.center, blue_line_points[0],
                                                    blue_line_points[1])
            if skip_frame:
                return False, True

        if dist_blue < 12.0 and not passed_blue_line_before and r_blue == 1:
            self.passed_blue_line = True
            return True, False

        return False, False

    def intersecting_green(self,green_line_points):
        passed_green_line_before = self.passed_green_line
        if not passed_green_line_before:
            dist_green, pnt, r_green, skip_frame = pnt2line(detection.center, green_line_points[0],
                                                      green_line_points[1])
            if skip_frame:
                return False, True

        if dist_green < 12.0 and not passed_green_line_before and r_green == 1:
            self.passed_green_line = True
            return True, False

        return False, False


class History:
    def __init__(self, passed_green_line = False, passed_blue_line=False,  center=tuple((-1,-1)), frame_idx=-1):
        self.passed_green_line = passed_green_line
        self.passed_blue_line = passed_blue_line
        self.center = center
        self.frame_idx = frame_idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--dir', help='path to dataset directory')
    ap.add_argument('-m', '--model',help='path to pre-trained CNN model')
    args = vars(ap.parse_args())

    if args["dir"] is None:
        print("Please specify path to dataset directory ...")
        exit(1)
    else:
        directory = args["dir"]

    if args["dir"] is None:
        print("Please specify path to pre-trained CNN model ...")
        exit(1)
    else:
        model_path = args["model"]

    classes, height, width, channels = 10, 28, 28, 1
    model = Model.build_network(classes, height, width, channels)

    try:
        model.load_weights(model_path)
    except Exception as e:
        print("[ERROR]: Error occurred while trying to load model weights. Message - ", e)
        exit(1)

    num_sum = 0
    for filename in os.listdir(directory):
        if filename.endswith(".avi") and not filename.startswith("demo"):
            file_path = os.path.join(directory, filename)

            cap = cv2.VideoCapture(file_path)
            # Check if camera opened successfully
            if not cap.isOpened():
                print("[ERROR]: Error occurred while trying to open video stream or file")
                exit(1)

            print("[INFO]: Successfully opened video file - " + file_path)
            while cap.isOpened():
                frame_idx = int(cap.get(1)) # 1 - CV_CAP_PROP_POS_FRAMES

                print("[INFO]: Reading frame with IDX = {}".format(frame_idx))
                flag, frame = cap.read()
                if flag:
                    print("[INFO]: Frame with IDX = {} has been successfully read.".format(frame_idx))
                    if frame_idx == 0:
                        print("[INFO]: This is the first frame of the {} file. Trying to detect the lines".format(file_path))
                        try:
                            detect_line_points = detect_lines(frame)

                            # Display the resulting frame
                            plt.title(file_path)
                            plt.imshow(frame)
                            plt.show()
                        except Exception as e:
                            print("[ERROR]: Failed while trying to detect lines in the first frame. Error message : {}".format(e))
                            exit(1)

                print("[INFO]: Detecting numbers for frame with IDX = {}.".format(frame_idx))
                dilated, thresh = preprocess_image(frame)
                sorted_regions, sorted_regions_coords = select_roi(frame, thresh)

                detections = []
                for r, r_coords in zip(sorted_regions, sorted_regions_coords):
                    detection = Detection()
                    detection.center = find_center(r_coords)
                    detection.frame_idx = frame_idx

                    indices = search_for_detection(detections, detection)

                    if len(indices) == 0:
                        classes = model.predict_classes(np.expand_dims(np.expand_dims(r, 0), 3), 1)

                        detection.label = classes[0]
                        detections.append(detection)
                    elif len(indices) == 1:
                        hst = History()

                        hst.frame_idx = frame_idx
                        detections[indices[0]].frame_idx = frame_idx

                        hst.center = detection.center
                        detections[indices[0]].history.append(hst)

                        detections[indices[0]].center = detection.center

                for detection in detections:
                    if (frame_idx - detection.frame_idx) > 10:
                        continue

                    color = [0, 0, 255]
                    intersecting_blue, skipping_frame = detection.intersecting_blue(detect_line_points[0])
                    if skipping_frame:
                        continue
                    elif intersecting_blue:
                        num_sum += detection.label
                        color = [220, 0, 0]
                        cv2.line(frame, (detect_line_points[0][0][0], detect_line_points[0][0][1]),
                                 (detect_line_points[0][1][0], detect_line_points[0][1][1]), (0, 0, 255), 1, cv2.LINE_AA)

                    intersecting_green, skipping_frame = detection.intersecting_green(detect_line_points[1])
                    if skipping_frame:
                        continue
                    elif intersecting_green:
                        num_sum -= detection.label
                        color = [0, 220, 0]
                        cv2.line(frame, (detect_line_points[1][0][0], detect_line_points[1][0][1]),
                                 (detect_line_points[1][1][0], detect_line_points[1][1][1]), (0, 0, 255), 1,
                                 cv2.LINE_AA)


                    cv2.circle(frame, detection.center, 15, color, 2)
                    cv2.putText(frame, str(detection.label), (detection.center[0] + 12, detection.center[1] + 12),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                    for hst in detection.history:
                        if frame_idx - hst.frame_idx < 200:
                            cv2.circle(frame, hst.center, 1, (255, 255, 255), 1)

                cv2.putText(frame, "Sum: " + str(num_sum) + "Frame number:" + str(frame_idx), (15, 20), cv2.FONT_HERSHEY_COMPLEX,1,(255, 0, 0), 1)
                cv2.imshow("frame",frame)

                if cv2.waitKey(1) == 13:
                    break

            # When everything done, release the video capture object
            cap.release()
            # Closes all the frames
            cv2.destroyAllWindows()
