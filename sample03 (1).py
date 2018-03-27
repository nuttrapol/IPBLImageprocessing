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
    img = cv2.imread("./shape.jpg")#objects.jpg")
    cv2.imshow("Image", img)

    # convert to HSV (Hue, Saturation, Value(Brightness))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Hue", hsv[:, :, 0])

    #detect green
    greens = createMaskImage(hsv, [55, 70], [50, 200], [50, 200])
    cv2.imshow("mask", greens)

    #### METHODS (You have to think the order of following methods) ####

    # METHOD2: lt and Pepper Noise Reduction using Opening process (Erosion next to Dilation) of mathematical morphology
    greens = cv2.morphologyEx(greens, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))) # using 4x4 ellipse kernel
    greens = cv2.morphologyEx(greens, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # using 9x9 ellipse kernel
    cv2.imshow("adapt opening", greens)

    # METHOD1: fill hole in the object using Closing process (Dilation next to Erosion) of mathematical morphology
    greens = cv2.morphologyEx(greens, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))) # using 9x9 ellipse kernel
    cv2.imshow("adapt closing", greens)

    # METHOD3 : Object detecting
    # edge detection and extract contours
    im, contours, hierarchy = cv2.findContours(greens, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # -- show extracted contours with gray line
    cv2.drawContours(greens, contours, -5, (128, 128, 128), 2)
    cv2.imshow("mask", greens)


    # show bounding box of extracted objects using contours
    for i in range(len(contours)):
        # -- get information of bounding rect of each contours
        posx, posy, width, height = cv2.boundingRect(contours[i])
        # -- decide "Skal" object using aspect ratio of bounding area
        if width*2.5<height and height<width*6: # --
            cv2.rectangle(img, (posx, posy), (posx + width, posy + height), (0, 0, 255), 2)
            strSize = cv2.getTextSize("Grande Cylinder", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (posx, posy-strSize[1]), (posx+strSize[0], posy), (0, 0, 255), -1)
            cv2.putText(img, "Grande Cylinder", (posx, posy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        else: # -- is not "Skal"
            cv2.rectangle(img, (posx, posy), (posx + width, posy + height), (0, 255, 0), 2)
        cv2.imshow("Image", img)

    # keep all windows until "ESC" button is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
