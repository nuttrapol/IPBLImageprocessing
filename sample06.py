import cv2
import numpy as np

def main():
    # read two gray images
    img = cv2.imread('./bci.jpg')
    r,g,b=cv2.split(img)
    ave = [r.mean(), g.mean(), b.mean()]
    # create the image data set (channel '0' is grayImage01, channel '1' is grayImage02)

    # initialize the output image
    res = img.astype(float)


    print('average of cha[0]: ', ave[0])
    print('average of cha[1]: ', ave[1])
    print('average of cha[2]: ', ave[2])

    # make the average of two input images the same value
    ## align the all average of two input images to 1.0
    for c in range(3):
        res[:,:,c] = res[:,:,c] / ave[c]

    # display the tmp average of the two modified gray images
    ## all average of two input images is 1.0 (but image range is not [0,255] but [0,unknown])

    tmp_r, tmp_g,tmp_b = cv2.split(res)
    tmp_ave_res = [tmp_r.mean(), tmp_g.mean(),tmp_b.mean()]
    print('tmp average of grayImage01: ', tmp_ave_res[0])
    print('tmp average of grayImage02: ', tmp_ave_res[1])
    print('tmp average of grayImage03: ', tmp_ave_res[2])

    ## image normalization
    ### image range [0, unknown] -> [0,255]
    th = np.nanpercentile(res, 99.5, interpolation='linear')
    res = clipping(res, res.min(), th)
    res = res.astype(np.uint8)

    # display the average of the two modified gray images

    res1, res2,res3 = cv2.split(res)
    ave_res = [res1.mean(), res2.mean(),res3.mean()]
    print('the average of modified grayImage01: ', ave_res[0])
    print('the average of modified grayImage02: ', ave_res[1])
    print('the average of modified grayImage02: ', ave_res[2])

    # display the results on each window
    cv2.imshow('img', img)
    cv2.imshow('modified img', res)



    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

if __name__=='__main__':
    main()