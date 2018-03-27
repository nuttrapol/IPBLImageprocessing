# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

# create sample mask image window function-------------------------------------------------
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

# main function-----------------------------------------------------------------------------
def main():

    # read image
    img = cv2.imread("./cylinders02.jpg")#objects.jpg")
    cv2.imshow("Image", img)

    # convert to HSV (Hue, Saturation, Value(Brightness))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Hue", hsv[:, :, 0])

    #-green
    rnb_green = createMaskImage(hsv, [30, 45], [50, 240], [50, 240])
    cv2.imshow("rainbow", rnb_green)

    # keep all windows until "ESC" button is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
