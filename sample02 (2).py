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
    img = cv2.imread("./pockys.jpg")#objects.jpg")
    cv2.imshow("Image", img)

    # convert to HSV (Hue, Saturation, Value(Brightness))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow("Hue", hsv[:, :, 0])

    #-green
    rnb_green = createMaskImage(hsv, [30, 45], [50, 240], [50, 240])
    #cv2.imshow("Green", rnb_green)

    #-Red
    rnb_red1 = createMaskImage(hsv, [161, 190], [50, 255], [50, 255])
    rnb_red2 = createMaskImage(hsv, [0, 5], [50, 220], [50, 230])

    rnb_red=rnb_red1 + rnb_red2
    #cv2.imshow("Red", rnb_red)

    #-Orange

    rgb_orange = createMaskImage(hsv, [6, 18], [50, 240], [50, 240])

    #cv2.imshow("Orange", rgb_orange)

    # -yellow
    rnb_yellow = createMaskImage(hsv, [19, 25], [50, 240], [50, 240])
    #cv2.imshow("Yellow", rnb_yellow)

    # -blue
    rnb_blue = createMaskImage(hsv, [90, 110], [50, 240], [50, 240])
    #cv2.imshow("Blue", rnb_blue)

    # -indigo
    rnb_navy = createMaskImage(hsv, [111, 135], [50, 240], [50, 240])
    #cv2.imshow("Indigo", rnb_navy)

    #-purple
    rnb_purple = createMaskImage(hsv, [140, 160], [50, 200], [50, 200])
    #cv2.imshow("Purple", rnb_purple)

    #-allcolor
    #rnb_purple = createMaskImage(hsv, [140, 160], [50, 200], [50, 200])
    rnb_red1 = createMaskImage(hsv, [160, 170], [50, 255], [50, 255])
    rnb_red2 = createMaskImage(hsv, [2, 5], [30, 240], [50, 230])

    rnb_orange = createMaskImage(hsv, [6, 18], [50, 240], [50, 240])
    rnb_yellow = createMaskImage(hsv, [19, 25], [50, 240], [50, 240])
    rnb_pink = createMaskImage(hsv, [170, 180], [50, 240], [50, 240])
    #rnb_blue = createMaskImage(hsv, [90, 110], [50, 240], [50, 240])
    #rnb_navy = createMaskImage(hsv, [111, 135], [50, 240], [50, 240])
    #rnb_purple = createMaskImage(hsv, [140, 160], [50, 200], [50, 200])


    rnb_all=rnb_red1+rnb_red2
    cv2.imshow("Allcolor", rnb_all)

    #-white
 #   rnb_white = createMaskImage(hsv, [0, 179], [0 ,0], [255, 255])
 #   cv2.imshow("rainbow", rnb_white)

    # keep all windows until "ESC" button is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
