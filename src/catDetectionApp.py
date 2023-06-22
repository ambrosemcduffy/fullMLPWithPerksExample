import numpy as np
import cv2

from dataEvaluation import predict

# loading weights
with np.load("weights/epoch3000.npz") as data:
    parameters = {key: data[key] for key in data}

# import video
videoPath = "/mnt/e/codingProjects/mlFromScratch/videos/testVideo2.mp4"

# cv2 video settings
cap = cv2.VideoCapture(videoPath)
font = cv2.FONT_HERSHEY_SIMPLEX
if not cap.isOpened():
    print("Error opening Video File")

# adding a frame information to video
frameNum = 0
while cap.isOpened():
    ret, frame = cap.read()
    _, pred = predict(frame, parameters)
    print(pred)
    label = ""
    if ret:
        index = np.argmax(pred)
        if index == 0:
            label = "other"
        elif index == 1:
            label = "cat"
        elif index == 2:
            label = "dog"
        elif index == 3:
            label = "chick"
        elif index == 4:
            label = "human"
        cv2.putText(frame, "frame {} prediction: {}".format(frameNum, label), (10, frame.shape[0]-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("frame", frame)
        frameNum += 1
                # increase the frame rate by skipping frames
        for i in range(15):
            ret = cap.grab()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
