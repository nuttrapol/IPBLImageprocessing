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

def detectRed(hsv):
    target_hue=[-2, 7]
    target_sat=[200, 255]
    target_val=[50, 250]
    return detect(hsv, target_hue,target_sat,target_val)

def detectGreen(hsv):
    target_hue=[30, 46]
    target_sat=[20, 200]
    target_val=[15, 100]
    return detect(hsv,target_hue,target_sat,target_val)

def detectPink(hsv):
    target_hue1=[100, 180]
    target_hue2=[-10, 5]
    target_sat1=[50, 120]
    target_sat2 = [50, 120]
    target_val1=[50, 240]
    target_val2 = [50, 240]
    return detect(hsv,target_hue1,target_sat1,target_val1,target_hue2,target_sat2,target_val2)

def detectYellow(hsv):
    target_hue1=[22, 25]
    target_hue2=[0, 15]
    target_sat1=[50, 255]
    target_sat2=[100, 180]
    target_val1=[50, 240]
    target_val2=[100, 160]
    return detect(hsv,target_hue1,target_sat1,target_val1,target_hue2,target_sat2,target_val2)

def detect(hsv,target_hue,target_sat,target_val,target_hue2=[0,0],target_sat2=[0,0],target_val2=[0,0]):
    mask1 = createMaskImage(hsv,target_hue,target_sat,target_val)
    mask2 = createMaskImage(hsv,target_hue2,target_sat2,target_val2)
    maskA = mask2+mask1
    cv2.imshow("mask", maskA)
    if(target_hue == [30,46]):
    # METHOD1: fill hole in the object using Closing process (Dilation next to Erosion) of mathematical morphology
        maskA = cv2.morphologyEx(maskA, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,37))) # using 9x9 ellipse kernel
        maskA = cv2.morphologyEx(maskA, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))  # using 9x9 ellipse kernel
    else:
        maskA = cv2.morphologyEx(maskA, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))  # using 9x9 ellipse kernel
        cv2.imshow("adapt closing", maskA)

    # METHOD2: lt and Pepper Noise Reduction using Opening process (Erosion next to Dilation) of mathematical morphology
        maskA = cv2.morphologyEx(maskA, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))) # using 4x4 ellipse kernel
        cv2.imshow("adapt opening", maskA)
    return maskA

def draw_text_red(x,y,size,mesg_list,target_img,r,g,b):
    for mesg in mesg_list:
        strSize = cv2.getTextSize(str(mesg), cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0]
        cv2.rectangle(target_img, (x, y - strSize[1]), (x + strSize[0], y), (r, g, b), -1)
        cv2.putText(target_img, str(mesg), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255))
        y=y+int(strSize[1]*12/10)

def draw_text_green(x,y,size,mesg_list,target_img):
    for mesg in mesg_list:
        strSize = cv2.getTextSize(str(mesg), cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0]
        cv2.rectangle(target_img, (x, y - strSize[1]), (x + strSize[0], y), (0, 255,0), 2)
        cv2.putText(target_img, str(mesg), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 255))
        y=y+int(strSize[1]*12/10)

# show label all object
def draw_obj_label(contours,label_data, target_img,r,g,b):
    grande_aspect_ratio_min = 1/2
    grande_aspect_ratio_max = 1/1.5

    for i in range(len(contours)):
        # -- get information of bounding rect of each contours
        posx, posy, width, height = cv2.boundingRect(contours[i])
        # -- decide "Skal" object using aspect ratio of bounding area
        if grande_aspect_ratio_min < width/height and width / height < grande_aspect_ratio_max and width>30: # --
            cv2.rectangle(target_img, (posx, posy), (posx + width, posy + height), (r, g, b), 2)
            mesg_list=[]
            mesg_list.append(label_data)
            aspect_ratio = width / height
            mesg_list.append(int(100*aspect_ratio))  ### aspect ratio
            # 0.6 is text size
            draw_text_red(posx,posy,0.6,mesg_list,target_img,r,g,b)


# main function-----------------------------------------------------------------------------
def main():

    # read image
    img = cv2.imread("./pockys.jpg")#objects.jpg")
    cv2.imshow("Image", img)

    # convert to HSV (Hue, Saturation, Value(Brightness))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Hue", hsv[:, :, 0])

    # detect red
    objRed_mask = detectRed(hsv)
    objGreen_mask = detectGreen(hsv)
    objPink_mask = detectPink(hsv)
    objYellow_mask = detectYellow(hsv)

    # METHOD3 : Object detecting
    # edge detection and extract contours
    im, objRed_contours, hierarchy = cv2.findContours(objRed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im, objGreen_contours, hierarchy = cv2.findContours(objGreen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im, objPink_contours, hierarchy = cv2.findContours(objPink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im, objYellow_contours, hierarchy = cv2.findContours(objYellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # -- show extracted contours with gray line
    #    cv2.drawContours(greens, contours, -1, (128, 128, 128), 2)
    #    cv2.imshow("mask", greens)

    # show bounding box of extracted objects using contours
    draw_obj_label(objRed_contours, "red", img,0,0,255)
    draw_obj_label(objGreen_contours, "Green", img,0,255,0)
    draw_obj_label(objPink_contours, "Pink", img,204,153,255)
    draw_obj_label(objYellow_contours, "Yellow", img,140,255,255)

    cv2.imshow("Image", img)

    # keep all windows until "ESC" button is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
