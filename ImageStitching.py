from FeatureMatching import FeatureMatching as fts_match
import cv2, math
import numpy as np
import argparse as ap
from utils import read_image
from Stitch import warpTwoImages, stitching2, multiStitching

def read_imageList(path):
    fp = open(path, 'r')
    filenames = []
    for each in fp.readlines():
        filenames.append(each.rstrip('\r\n'))
    print(filenames)
    images = []
    for each in filenames:
        images.append(read_image(each))
    count = len(images)
    return filenames, images, count

def stitching(leftImage, warpImage):
    y1, x1 = leftImage.shape[:2]
    y2, x2 = warpImage.shape[:2]
    print(leftImage[-1, -1])

    t = time.time()
    black_l = np.where(leftImage == np.array([0, 0, 0]))
    black_w = np.where(warpImage == np.array([0, 0, 0]))
    
    for i in range(0, x1):
        for j in range(0, y1):
            try:
                if (np.array_equal(leftImage[j, i], np.array([0, 0, 0])) and np.array_equal(warpImage[j, i], np.array([0, 0, 0]))):
                    warpImage[j, i] = [0, 0, 0]
                else:
                    if (np.array_equal(warpImage[j, i], [0, 0, 0])):
                        warpImage[j. i] = leftImage[j, i]
                    else:
                        if not np.array_equal(leftImage[j, i], [0, 0, 0]):
                            bw, gw, rw = warpImage[j, i]
                            bl, gl, rl = leftImage[j, i]
                            warpImage[j, i] = [bl, gl, rl]
            except:
                pass
    return warpImage

def leftshift(leftlist, args):
    x = leftlist[0]
    for y in leftlist[1:]:
        H = fts_match(x, y, args.win_size, args.max_iters, args.epsilon)
        tmp = H[0][2]
        H[0][2] = H[1][2]
        H[1][2] = tmp
        print("Homography: ", H)
        hi = np.linalg.inv(H)
        print("Inverse Homography: ", hi)
        ds = np.dot(hi, np.array([x.shape[1], x.shape[0], 1]))
        ds = ds / ds[-1]
        print("final ds => ", ds)
        f1 = np.dot(hi, np.array([0, 0, 1]))
        f1 = f1 / f1[-1]
        hi[0][-1] += abs(f1[0])
        hi[1][-1] += abs(f1[1])
        ds = np.dot(hi, np.array([x.shape[1], x.shape[0], 1]))
        print(ds)
        offsety = abs(int(f1[1]))
        offsetx = abs(int(f1[0]))

        dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
        print("image dsize => ", dsize)
        tmp = cv2.warpPerspective(x, hi, dsize)
        tmp[offsety:y.shape[0] + offsety, offsetx:y.shape[1] + offsetx] = y
        x = tmp
    
    return tmp

def rightshift(rightlist, leftImage, args):
    for each in rightlist:
        H = fts_match(leftImage, each, args.win_size, args.max_iters, args.epsilon)
        tmp = H[0][2]
        H[0][2] = H[1][2]
        H[1][2] = tmp
        print("Homography: ", H)
        txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
        txyz = txyz / txyz[-1]
        dsize = (int(txyz[0]) + leftImage.shape[1], int(txyz[1] + leftImage.shape[0]))
        tmp = cv2.warpPerspective(each, H, dsize)
        tmp = stitching(leftImage, tmp)
        print("tmp shape: ", tmp.shape)
        print("leftImage shape: ", leftImage.shape)
        leftImage = tmp
    
    return leftImage

    
def main(args):
    filenames, images, count = read_imageList(args.path)
    print("Number of images : ", count)
    centerIdx = count / 2
    print("Center index image : ", centerIdx)
    center_im = images[int(centerIdx)]
    leftlist = []
    rightlist = []
    for i in range(count):
        if (i <= centerIdx):
            leftlist.append(images[i])
        else:
            rightlist.append(images[i])

    # test stitching 1 
    # leftImage = leftshift(leftlist, args)
    # warpImage = rightshift(rightlist, leftImage, args)

    # test stitching 2
    warpImage, _, _, _ = warpTwoImages(images[0], images[1], args)
    # warpImage = multiStitching(images, args)

    # test stitching 3
    # warpImage = stitching2(images, args)

    print("Finished")
    cv2.imwrite("pano.jpg", warpImage)
    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--path", 
                        default="./image/file.txt",
                        help="Path to file list") 
    parser.add_argument("-w", 
                        "--win_size", 
                        default=10,
                        type=int,
                        help="Window size for your feature detector algorithm") 
    
    parser.add_argument("-i", 
                        "--max_iters", 
                        default=1000,
                        type=int,
                        help="Maximum iterations to perform RANSAC") 
    
    parser.add_argument("-e", 
                        "--epsilon", 
                        default=100,
                        type=float,
                        help="SSD epsilon threshold for RANSAC") 

    args = parser.parse_args()
    main(args)