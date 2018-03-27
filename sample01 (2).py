# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

# initial window position
posX = 20
posY = 20

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

    # create masked image that is filled values between HSV min and HSV max with white
    mask = cv2.inRange(hsv, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))

    # put text of status values on masked image
    cv2.putText(mask, "Heu: " + str(hmin) + " - " + str(hmax), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    cv2.putText(mask, "Sat: " + str(smin) + " - " + str(smax), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    cv2.putText(mask, "Val: " + str(vmin) + " - " + str(vmax), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    return mask

# create HSV-gradation image window function-------------------------------------------------
def createGradationImage(hue, sat, val):
    cimg = np.zeros((256, 256, 3), np.uint8) # initialize hsv gradation image with 0
    posHSV = [0, 0] # to show HSV value of clicked area

    for j in range(256):
        for i in range(256):
            cimg[j,i,0] = np.uint8(hue) # Hue
            cimg[j,i,1] = i             # Saturation
            cimg[j,i,2] = j             # Value

            # show HSV value area of click position
            if j == np.uint8(val) and i == np.uint8(sat):
                posHSV = [i, j]

    # put text on Image
    cv2.putText(cimg, "> satulation", (5, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))
    cv2.putText(cimg, "|128", (128, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0,255))
    cv2.putText(cimg, "V vlue", (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0,255))
    cv2.putText(cimg, "- 128", (0, 131), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))

    # show HSV value of clicked area
    cv2.rectangle(cimg, (posHSV[0] - 2, posHSV[1] - 2), (posHSV[0] + 2, posHSV[1] + 2), (0, 255, 255), 1)

    cv2.imshow("color range", cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR))

# trackbar event function-------------------------------------------------------------------
def changeTrackbarRange(val):
    global av_s, av_v
    # update gradation image
    createGradationImage(val, av_s, av_v)

# mouse event function----------------------------------------------------------------------
def mouse_event(event, x, y, flg, prm):
    global img, cache
    global av_h, av_s, av_v

    # when mouse is moved
    if event == cv2.EVENT_MOUSEMOVE:
        # --clear image (keep mark of the latest clicked area)
        mvcache = cache.copy()

        # show mouse position
        cv2.rectangle(mvcache, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), 1)
        cv2.imshow("target image", mvcache)

    # when left button is clicked
    elif event == cv2.EVENT_LBUTTONDOWN:
        # --clear image (to target image)
        cache = img.copy()

        # average of each color components around pointing area
        av_b = np.mean(img[y - 2:y + 2, x - 2:x + 2, 0])
        av_g = np.mean(img[y - 2:y + 2, x - 2:x + 2, 1])
        av_r = np.mean(img[y - 2:y + 2, x - 2:x + 2, 2])

        print("(B,G,R) = (" + str(av_b) + ", " + str(av_g) + ", " + str(av_r) + ")")

        # average of HSV components around pointing area
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        av_h = np.mean(hsv[y - 2:y + 2, x - 2:x + 2, 0])
        av_s = np.mean(hsv[y - 2:y + 2, x - 2:x + 2, 1])
        av_v = np.mean(hsv[y - 2:y + 2, x - 2:x + 2, 2])

        print("(H,S,V) = (" + str(av_h) + ", " + str(av_s) + ", " + str(av_v) + ")")
        createGradationImage(av_h, av_s, av_v)

        # create mask image
        mask = createMaskImage(hsv, [int(av_h)-10, int(av_h)+10], [50, 255], [50, 255])
        cv2.imshow("masked image", mask)
        cv2.moveWindow("masked image", posX+270, posY+img.shape[0]+40)

        # track bar
        cv2.namedWindow("color range", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        cv2.moveWindow("color range", posX, posY+img.shape[0]+40)
        cv2.createTrackbar("Hue", "color range", int(av_h), 179, changeTrackbarRange) # Max value of hue in openCV is 179 (RED)

        # show clicked area
        cv2.rectangle(cache, (x - 2, y - 2), (x + 2, y + 2), (0, 255, 0), 1)
        cv2.imshow("target image", cache)


# main function-----------------------------------------------------------------------------
def main():
    global img, cache
    global posX, posY

    # read image
    img = cv2.imread("./pockys.jpg")
    imh, imw, channels = img.shape  # get image size and the number of channels

    # pre-processing area ---------------------
    #img = cv2.GaussianBlur(img, (5, 5), 5)


    #-------------------------------------------
    cache = img.copy()

    # display image
    cv2.imshow("target image", img)
    cv2.moveWindow("target image", posX, posY)
    cv2.setMouseCallback("target image", mouse_event)

    # calc window position --------
    X = posX + imw + 10
    Y = posY
    # ------------------------------

    # separate color channel
    cv2.imshow("red-channel", img[:, :, 2])
    cv2.moveWindow("red-channel", X, Y)

    cv2.imshow("green-channel", img[:, :, 1])
    cv2.moveWindow("green-channel", X + 30, Y + 30)

    cv2.imshow("blue-channel", img[:, :, 0])
    cv2.moveWindow("blue-channel", X + 60, Y + 60)

    # calc window position --------
    X = posX + imw + 120
    Y = posY + 120
    # ------------------------------

    # convert to HSV (Hue, Saturation, Value(Brightness))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Hue", hsv[:, :, 0])
    cv2.moveWindow("Hue", X, Y)

    cv2.imshow("Saturation", hsv[:, :, 1])
    cv2.moveWindow("Saturation", X + 30, Y + 30)

    cv2.imshow("Value", hsv[:, :, 2])
    cv2.moveWindow("Value", X + 60, Y + 60)

    # calc window position --------
    X = posX + imw + 240
    Y = posY + 240
    # ------------------------------

    # gaussian blur it
    gaussian = cv2.GaussianBlur(img, (5, 5), 5)  # GaussianBlur(target, ksize, sigmaX)
    cv2.imshow("blur image", gaussian)
    cv2.moveWindow("blur image", X, Y)

    # upper and lower threshold image using brightness (V value of HSV)
    ret, thimg = cv2.threshold(hsv[:, :, 2], 160, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold image", thimg)
    cv2.moveWindow("threshold image", X + 30, Y + 30)

    # detect edge from gaussian image
    # -- Canny edge detection
    #edge = cv2.Canny(gaussian, 10, 80)
    # -- Sobel edge detection
    dx = cv2.convertScaleAbs(cv2.Sobel(hsv[:,:,2], cv2.CV_64F, 1, 0, ksize=3))
    dy = cv2.convertScaleAbs(cv2.Sobel(hsv[:,:,2], cv2.CV_64F, 0, 1, ksize=3))
    edge = cv2.addWeighted(dx,0.5, dy, 0.5, 0)
    cv2.imshow("edge detection", edge)
    cv2.moveWindow("edge detection", X + 60, Y + 60)

    # keep all windows until "ESC" button is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
