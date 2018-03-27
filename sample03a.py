# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
print(str(100/3) )
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

def detectA(hsv):
    target_hue=[55, 70]
    target_sat=[50, 200]
    target_val=[50, 200]
    maskA = createMaskImage(hsv,target_hue,target_sat,target_val)
    cv2.imshow("mask", maskA)
    # METHOD1: fill hole in the object using Closing process (Dilation next to Erosion) of mathematical morphology
    maskA = cv2.morphologyEx(maskA, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))) # using 9x9 ellipse kernel
    cv2.imshow("adapt closing", maskA)

    # METHOD2: lt and Pepper Noise Reduction using Opening process (Erosion next to Dilation) of mathematical morphology
    maskA = cv2.morphologyEx(maskA, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))) # using 4x4 ellipse kernel
    cv2.imshow("adapt opening", maskA)
    return maskA

def draw_text_red(x,y,size,mesg_list,target_img):
    for mesg in mesg_list:
        strSize = cv2.getTextSize(str(mesg), cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0]
        cv2.rectangle(target_img, (x, y - strSize[1]), (x + strSize[0], y), (0, 0, 255), -1)
        cv2.putText(target_img, str(mesg), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255))
        y=y+int(strSize[1]*12/10)

def draw_text_green(x,y,size,mesg_list,target_img):
    for mesg in mesg_list:
        strSize = cv2.getTextSize(str(mesg), cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0]
        cv2.rectangle(target_img, (x, y - strSize[1]), (x + strSize[0], y), (0, 255,0), -1)
        cv2.putText(target_img, str(mesg), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 255))
        y=y+int(strSize[1]*12/10)

# show label all object and grande objects show red
def draw_obj_label(contours,label_data, target_img):
    grande_aspect_ratio_min = 1/2
    grande_aspect_ratio_max = 1/1.5

    for i in range(len(contours)):
        # -- get information of bounding rect of each contours
        posx, posy, width, height = cv2.boundingRect(contours[i])
        # -- decide "Skal" object using aspect ratio of bounding area
        if grande_aspect_ratio_min < width/height and width / height < grande_aspect_ratio_maxv and width<30: # --
            cv2.rectangle(target_img, (posx, posy), (posx + width, posy + height), (0, 0, 255), 2)
            mesg_list=[]
            mesg_list.append(label_data+" grande")
            aspect_ratio = width / height
            mesg_list.append(int(100*aspect_ratio))  ### aspect ratio
            # 0.6 is text size
            draw_text_red(posx,posy,0.6,mesg_list,target_img)
        else:
            cv2.rectangle(target_img, (posx, posy), (posx + width, posy + height), (0, 255, 0), 2)
            mesg_list = []
            mesg_list.append(label_data )
            aspect_ratio = width / height
            mesg_list.append(int(100 * aspect_ratio))  ### aspect ratio
            # 0.6 is text size
            draw_text_green(posx, posy, 0.6, mesg_list, target_img)


# main function-----------------------------------------------------------------------------
def main():

    # read image
    img = cv2.imread("./shape.jpg")#objects.jpg")
    cv2.imshow("Image", img)

    # convert to HSV (Hue, Saturation, Value(Brightness))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Hue", hsv[:, :, 0])

    #detect green
    objA_mask = detectA(hsv)

    # METHOD3 : Object detecting
    # edge detection and extract contours
    im, objA_contours, hierarchy = cv2.findContours(objA_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # -- show extracted contours with gray line
#    cv2.drawContours(greens, contours, -1, (128, 128, 128), 2)
#    cv2.imshow("mask", greens)

    # show bounding box of extracted objects using contours
    draw_obj_label(objA_contours,"green", img)
    cv2.imshow("Image", img)

    # keep all windows until "ESC" button is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
