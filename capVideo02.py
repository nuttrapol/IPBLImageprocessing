import cv2
import numpy as np

cap = cv2.VideoCapture('video1.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('video frame', frame)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
