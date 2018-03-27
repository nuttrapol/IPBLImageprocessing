import cv2
import numpy as np


def clipping(img, img_min, img_max):
    mask = img > img_max
    channelNum = img.shape[2]
    if channelNum > 1:
        mask = mask.sum(axis=2)
 #       mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
        mask = np.tile(mask[:, :, None], [1, 1, channelNum])
    res = 255 * (img - img_min)/(img_max - img_min) # normalization
    res[mask > 0] = 255  # paint over-exposure regions with white

    return res

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
    target_hue=[10, 46]
    target_sat=[10, 200]
    target_val=[15, 100]
    return detect(hsv,target_hue,target_sat,target_val)

def detectGreenLay(hsv):
    target_hue1 = [5, 46]
    target_hue2 = [106, 126]
    target_sat1 = [10, 200]
    target_sat2 = [50, 255]
    target_val1 = [15, 100]
    target_val2 = [50, 255]
    return detect(hsv,target_hue1,target_sat1,target_val1,target_hue2,target_sat2,target_val2)

def detectPink(hsv):
    target_hue1=[100, 180]
    target_hue2=[-10, 5]
    target_sat1=[50, 120]
    target_sat2=[50, 120]
    target_val1=[50, 240]
    target_val2=[50, 240]
    return detect(hsv,target_hue1,target_sat1,target_val1,target_hue2,target_sat2,target_val2)

def detectYellow(hsv):
    target_hue1=[10, 23]
    target_hue2=[0, 0]
    target_sat1=[50, 255]
    target_sat2=[0, 0]
    target_val1=[50, 200]
    target_val2=[0, 0]
    return detect(hsv,target_hue1,target_sat1,target_val1,target_hue2,target_sat2,target_val2)

def detectYellowLay(hsv):
    target_hue1=[10, 46]
    target_hue2=[106, 126]
    target_sat1=[10, 200]
    target_sat2=[50, 255]
    target_val1=[15, 100]
    target_val2=[50, 255]
    return detect(hsv,target_hue1,target_sat1,target_val1,target_hue2,target_sat2,target_val2)

def detect(hsv,target_hue,target_sat,target_val,target_hue2=[0,0],target_sat2=[0,0],target_val2=[0,0]):
    mask1 = createMaskImage(hsv,target_hue,target_sat,target_val)
    mask2 = createMaskImage(hsv,target_hue2,target_sat2,target_val2)
    maskA = mask2+mask1
    if(target_hue == [10,46]):
    # METHOD1: fill hole in the object using Closing process (Dilation next to Erosion) of mathematical morphology
        maskA = cv2.morphologyEx(maskA, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))) # using 9x9 ellipse kernel
        maskA = cv2.morphologyEx(maskA, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))  # using 9x9 ellipse kernel
    else:
        maskA = cv2.morphologyEx(maskA, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))  # using 9x9 ellipse kernel


    # METHOD2: lt and Pepper Noise Reduction using Opening process (Erosion next to Dilation) of mathematical morphology
        maskA = cv2.morphologyEx(maskA, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))) # using 4x4 ellipse kernel

    return maskA

def draw_text(x,y,size,mesg_list,target_img,r,g,b):
    for mesg in mesg_list:
        strSize = cv2.getTextSize(str(mesg), cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0]
        cv2.rectangle(target_img, (x, y - strSize[1]), (x + strSize[0], y), (r, g, b),-1)
        cv2.putText(target_img, str(mesg), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0))
        y=y+int(strSize[1]*12/10)

def draw_obj_label(contours,label_data, target_img,r,g,b):
    grande_aspect_ratio_min = 1/2
    grande_aspect_ratio_max = 1/1.5

    for i in range(len(contours)):
        # -- get information of bounding rect of each contours
        posx, posy, width, height = cv2.boundingRect(contours[i])
        # -- decide "Skal" object using aspect ratio of bounding area
        if grande_aspect_ratio_min < width/height and width / height < grande_aspect_ratio_max and width>57 : # --
            cv2.rectangle(target_img, (posx, posy), (posx + width, posy + height), (r, g, b), 2)
            mesg_list=[]
            mesg_list.append(label_data)
            aspect_ratio = width / height
            mesg_list.append(int(100*aspect_ratio))  ### aspect ratio
            # 0.6 is text size
            draw_text(posx,posy,0.6,mesg_list,target_img,r,g,b)
            print('the width: ', width)
            print('the height: ', height)

def draw_obj_labelLay(contours,label_data, target_img,r,g,b):
    grande_aspect_ratio_min = 1/2
    grande_aspect_ratio_max = 1/1.5

    for i in range(len(contours)):
        # -- get information of bounding rect of each contours
        posx, posy, width, height = cv2.boundingRect(contours[i])
        # -- decide "Skal" object using aspect ratio of bounding area
        if grande_aspect_ratio_min < width/height and width / height < grande_aspect_ratio_max and width>110 and width<130: # --
            cv2.rectangle(target_img, (posx, posy), (posx + width, posy + height), (r, g, b), 2)
            mesg_list=[]
            mesg_list.append(label_data)
            aspect_ratio = width / height
            mesg_list.append(int(100*aspect_ratio))  ### aspect ratio
            # 0.6 is text size
            draw_text(posx,posy,0.6,mesg_list,target_img,r,g,b)
            print('the width: ', width)


cap = cv2.VideoCapture('finaltest.avi')
while cap.isOpened():
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r, g, b = cv2.split(frame)
    ave = [r.mean(), g.mean(), b.mean()]
    res = frame.astype(float)

    for c in range(3):
        res[:,:,c] = res[:,:,c] / ave[c]

    tmp_r, tmp_g, tmp_b = cv2.split(res)
    tmp_ave_res = [tmp_r.mean(), tmp_g.mean(), tmp_b.mean()]

    th = np.nanpercentile(res, 99.5, interpolation='linear')
    res = clipping(res, res.min(), th)
    res = res.astype(np.uint8)

    res1, res2, res3 = cv2.split(res)
    ave_res = [res1.mean(), res2.mean(), res3.mean()]

    objRed_mask = detectRed(hsv)
    objGreen_mask = detectGreen(hsv)
    objPink_mask = detectPink(hsv)
    objYellow_mask = detectYellow(hsv)
    objGreenLay_mask = detectGreenLay(hsv)
    objYellowLay_mask = detectYellowLay(hsv)

    im, objRed_contours, hierarchy = cv2.findContours(objRed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im, objGreen_contours, hierarchy = cv2.findContours(objGreen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im, objPink_contours, hierarchy = cv2.findContours(objPink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im, objYellow_contours, hierarchy = cv2.findContours(objYellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im, objYellowLay_contours, hierarchy = cv2.findContours(objYellowLay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im, objGreenLay_contours, hierarchy = cv2.findContours(objGreenLay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_obj_label(objRed_contours, "Choco", res, 0, 0, 255)
    draw_obj_label(objGreen_contours, "Matcha", res, 0, 255, 0)
    draw_obj_labelLay(objGreenLay_contours, "Nori", res, 0, 255, 0)
    draw_obj_label(objPink_contours, "Straw", res, 204, 153, 255)
    draw_obj_label(objYellow_contours, "Almond", res, 140, 255, 255)
    draw_obj_labelLay(objYellowLay_contours, "OriginalLay", res, 140, 255, 255)

    cv2.imshow('video frame', frame)
    cv2.imshow('modified img', res)


    if cv2.waitKey(1)&0xFF == ord('q'):
        break
