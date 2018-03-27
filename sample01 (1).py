# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

# create sample mask image window function-------------------------------------------------
def createMaskImage(hsv, hue, sat, val):
    imh, imw, channels = hsv.shape  # get image size and the number of channels
    mask = np.zeros((imh, imw, channels), np.uint8) # initialize hsv gradation image with 0

    if isinstance(hue, list): # if hue argument is pair value enclosed in []
        hmin = hue[0]
        hmax = hue[1]
    else:                     # if hue argument is single value
        hmin = hue
        hmax = hue

    if isinstance(sat, list): # if sat argument is pair value enclosed in []
        smin = sat[0]
        smax = sat[1]
    else:                     # if sat argument is single value
        smin = sat
        smax = sat

    if isinstance(val, list): #  val argument is pair value enclosed in []
        vmin = val[0]
        vmax = val[1]
    else:                    # if val argument is single value
        vmin = int(val)
        vmax = int(val)

    return cv2.inRange(hsv, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))

# main function-----------------------------------------------------------------------------
def main():

    # read image
    img = cv2.imread("./cylinders02.jpg")#objects.jpg")
    cv2.imshow("Image", img)

    # convert to HSV (Hue, Saturation, Value(Brightness))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Hue", hsv[:, :, 0])

    #detect green
    greens = createMaskImage(hsv, [55, 70], [50, 200], [50, 200])
    cv2.imshow("mask", greens)

    #detect rainbow
    rainbow_mask = np.zeros((img.shape[1],img.shape[0],1),np.uint8)

    #-red
    rnb_red1 = createMaskImage(hsv, [161, 185], [50, 220], [50, 230])
    rnb_red2 = createMaskImage(hsv, [0, 5], [50, 220], [50, 230])
    rnb_red = rnb_red1 + rnb_red2
    rainbow_mask = rainbow_mask + rnb_red
    cv2.imshow("rainbow", cv2.cvtColor(rainbow_mask, cv2.COLOR_HSV2BGR))

    #-orange
    rnb_orange = createMaskImage(hsv, [6, 18], [50, 240], [50, 240])
    rainbow_mask = rainbow_mask + rnb_orange
    #cv2.imshow("rainbow", rainbow_mask)

    #-yellow
    rnb_yellow = createMaskImage(hsv, [19, 25], [50, 240], [50, 240])
    rainbow_mask = rainbow_mask + rnb_yellow
    #cv2.imshow("rainbow", rainbow_mask)

    #-green
    rnb_green = createMaskImage(hsv, [30, 45], [50, 240], [50, 240])
    rainbow_mask = rainbow_mask + rnb_green
    #cv2.imshow("rainbow", rainbow_mask)

    #-blue
    rnb_blue = createMaskImage(hsv, [90, 110], [50, 240], [50, 240])
    rainbow_mask = rainbow_mask + rnb_blue
    #cv2.imshow("rainbow", rainbow_mask)

    #-nsvy
    rnb_navy = createMaskImage(hsv, [111, 135], [50, 240], [50, 240])
    rainbow_mask = rainbow_mask + rnb_navy
    #cv2.imshow("rainbow", rainbow_mask)

    #-purple
    rnb_purple = createMaskImage(hsv, [140, 160], [50, 200], [50, 200])
    rainbow_mask = rainbow_mask + rnb_purple
    #cv2.imshow("rainbow", rainbow_mask)

    # keep all windows until "ESC" button is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
