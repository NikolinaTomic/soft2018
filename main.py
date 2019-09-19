import cv2
import os
import argparse
import matplotlib.pyplot as plt

from architecture import Model
from detection import detect_lines, detect_numbers

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
                detected_numbers = detect_numbers(frame, frame_idx)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # When everything done, release the video capture object
            cap.release()
            # Closes all the frames
            cv2.destroyAllWindows()
