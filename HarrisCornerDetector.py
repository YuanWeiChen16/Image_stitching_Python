import cv2 as cv
import numpy as np
import sys
import operator



def main(argc,argv):
    ##use like  python HarrisCornerDetector.py N1.png
    Mask_Size = 5
    Corner_Threshold = 100000

    #read image
    Img = cv.imread(str(argc[1]))
    if Img is not None:
        ##convert to gray scale
        if len(Img.shape) == 3:
            Img = cv.cvtColor(Img,cv.COLOR_RGB2GRAY)
        if len(Img.shape) == 4 :
            Img = cv.cvtColor(Img,cv.COLOR_RGBA2GRAY)
        ##get gradient of image
        Img_dy, Img_dx = np.gradient(Img)
        Img_dxx = Img_dx**2
        Img_dxy = Img_dy*Img_dx
        Img_dyy = Img_dy**2
        ##get height width
        Img_Height = Img.shape[0]
        Img_Width = Img_shape[1]

        CornerList = []
        color_img = Img
        Color_Img = cv.cvtColor(Color_Img,cv.COLOR_GRAY2RGB)
        offset = int(Mask_Size/2)

        for y in range(offset,Img_Height - offset):
            for x in range(offset, Img_Width - offset):
                Mask_xx = Img_dxx[(y - offset) : (y+offset + 1),(x - offset):(x + offset + 1)]
                Mask_xy = Img_dxy[(y - offset) : (y+offset + 1),(x - offset):(x + offset + 1)]
                Mask_yy = Img_dyy[(y - offset) : (y+offset + 1),(x - offset):(x + offset + 1)]
                Mask_xx = Mask_xx.sum()

