import cv2
import numpy as np
from matplotlib import pyplot as plt

def createMaskImage(hsv, hue, sat, val):
    imh, imw, channels = hsv.shape  # get image size and the number of channels
    mask = np.zeros((imh, imw, channels), np.uint8) # initialize hsv gradation image with 0

    # if hue argument is pair value enclosed in []
    hmin = hue[0]
    hmax = hue[1]

    # if sat argument is pair value enclosed in []
    smin = sat[0]
    smax = sat[1]

    #  val argument is pair value enclosed in []
    vmin = val[0]
    vmax = val[1]

    return cv2.inRange(hsv, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # read image
    cv2.imshow("Image", frame)

    # convert to HSV (Hue, Saturation, Value(Brightness))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("Hue", hsv[:, :, 0])

    rnb_red1 = createMaskImage(hsv, [161, 170], [50, 255], [50, 255])
    rnb_red2 = createMaskImage(hsv, [5, 9], [50, 255], [50, 230])
    rnb_red = rnb_red1 + rnb_red2
    #cv2.imshow('Chocolate', rnb_red)

    rnb_green = createMaskImage(hsv, [40, 65], [50, 240], [50, 240])
    #cv2.imshow('Matcha', rnb_green)

    rnb_pink1 = createMaskImage(hsv, [168, 180], [70, 100], [50, 91])
    #cv2.imshow('Strawberry', rnb_red)

    rnb_yellow = createMaskImage(hsv, [15, 30], [50, 255], [50, 160])
    cv2.imshow('Almond', rnb_yellow)




    if cv2.waitKey(1)&0xFF == ord('q'):
        break
