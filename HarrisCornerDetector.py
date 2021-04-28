import cv2 as cv
import numpy as np
import sys
import operator

def main():
    print("BBBBB")
    ##use like  python HarrisCornerDetector.py N1.png
    Mask_Size = 5
    Corner_Threshold = 100000
    if len(sys.argv) < 2:
        print("use like python HarrisCornerDetector.py N1.png")

    #read image
    Img = cv.imread(str(sys.argv[1]))
    print("Open Image "+ str(sys.argv[1]))
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
        Img_Width = Img.shape[1]

        #List of corner
        CornerList = []
        Color_Img = Img
        Color_Img = cv.cvtColor(Color_Img,cv.COLOR_GRAY2RGB)
        offset = int(Mask_Size/2)

        for y in range(offset,Img_Height - offset):
            for x in range(offset, Img_Width - offset):
                #Claculate sum of squares
                Mask_xx = Img_dxx[(y - offset) : (y+offset + 1),(x - offset):(x + offset + 1)]
                Mask_xy = Img_dxy[(y - offset) : (y+offset + 1),(x - offset):(x + offset + 1)]
                Mask_yy = Img_dyy[(y - offset) : (y+offset + 1),(x - offset):(x + offset + 1)]
                Mask_xx = Mask_xx.sum()
                Mask_xy = Mask_xy.sum()
                Mask_yy = Mask_yy.sum()

                #Find determinant
                det = (Mask_xx*Mask_yy) - (Mask_xy**2)
                Corner_trace = Mask_xx + Mask_yy
                Corner_r = det - 0.04*(Corner_trace**2)

                if Corner_r > Corner_Threshold:
                    #add corner to file
                    CornerList.append([x,y,Corner_r])
                    #add corner to output image 
                    Color_Img.itemset((y, x, 0), 0)
                    Color_Img.itemset((y, x, 1), 255)
                    Color_Img.itemset((y, x, 2), 0)

        if Color_Img is not None:
            cv.imwrite("CornerImage.png", Color_Img)
        ##sort corner list
        CornerList.sort(key=operator.itemgetter(2))
        ##openFile        
        CornerFile = open('corners.txt', 'w')
        
        for i in range(500):
            CornerFile.write(str(CornerList[i][0]) + ' ' + str(CornerList[i][1]) + ' ' + str(CornerList[i][2]) + '\n')
        CornerFile.close()

if __name__ == "__main__":
    print("AAAAA")
    main()
