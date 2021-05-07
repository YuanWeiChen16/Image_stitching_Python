from FeatureMatching import FeatureMatching as fts_match
import cv2, math
import numpy as np
import argparse as ap
from utils import read_image

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

def warpTwoImages(src_img, dst_img, showstep=False):
    
    # generate Homography matrix
    H = fts_match(src_img, dst_img, args.win_size, args.max_iters, args.epsilon)

    # get height and width of two images
    height_src, width_src = src_img.shape[:2]
    height_dst, width_dst = dst_img.shape[:2]

    # extract conners of two images: top-left, bottom-left, bottom-right, top-right
    pts1 = np.float32(
        [[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]
    ).reshape(-1, 1, 2)
    pts2 = np.float32(
        [[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]
    ).reshape(-1, 1, 2)
    # aply homography to conners of src_img
    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts1_, pts2), axis=0)

    # find max min of x,y coordinate
    [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
    [_, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]

    # top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
    # otherwise side=right
    # source image is merged to the left side or right side of destination image
    if pts[0][0][0] < 0:
        side = "left"
        width_pano = width_dst + t[0]
    else:
        width_pano = int(pts1_[3][0][0])
        side = "right"
    height_pano = ymax - ymin

    # Translation
    # https://stackoverflow.com/a/20355545
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    src_img_warped = cv2.warpPerspective(
        src_img, Ht.dot(H), (width_pano, height_pano)
    )
    # generating size of dst_img_rz which has the same size as src_img_warped
    dst_img_rz = np.zeros((height_pano, width_pano, 3))
    if side == "left":
        dst_img_rz[t[1] : height_src + t[1], t[0] : width_dst + t[0]] = dst_img
    else:
        dst_img_rz[t[1] : height_src + t[1], :width_dst] = dst_img

    # blending panorama
    pano, nonblend, leftside, rightside = panoramaBlending(
        dst_img_rz, src_img_warped, width_dst, side, showstep=showstep
    )

    # croping black region
    pano = crop(pano, height_dst, pts)
    return pano, nonblend, leftside, rightside    
    # try:
       
    # except BaseException:
    #     raise Exception("Please try again with another image set!")

def panoramaBlending(dst_img_rz, src_img_warped, width_dst, side, showstep=False):
    """Given two aligned images @dst_img and @src_img_warped, and the @width_dst is width of dst_img
    before resize, that indicates where there is the discontinuity between the images,
    this function produce a smoothed transient in the overlapping.
    @smoothing_window is a parameter that determines the width of the transient
    left_biased is a flag that determines whether it is masked the left image,
    or the right one"""

    h, w, _ = dst_img_rz.shape
    smoothing_window = int(width_dst / 8)
    barrier = width_dst - int(smoothing_window / 2)
    mask1 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=True
    )
    mask2 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=False
    )

    if showstep:
        nonblend = src_img_warped + dst_img_rz
    else:
        nonblend = None
        leftside = None
        rightside = None

    if side == "left":
        dst_img_rz = cv2.flip(dst_img_rz, 1)
        src_img_warped = cv2.flip(src_img_warped, 1)
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        pano = cv2.flip(pano, 1)
        if showstep:
            leftside = cv2.flip(src_img_warped, 1)
            rightside = cv2.flip(dst_img_rz, 1)
    else:
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        if showstep:
            leftside = dst_img_rz
            rightside = src_img_warped

    return pano, nonblend, leftside, rightside
    
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

    #leftImage = leftshift(leftlist, args)
    #warpImage = rightshift(rightlist, leftImage, args)
    warpImage, _, _, _ = warpTwoImages(images[1], images[2])
    print("Finished")
    cv2.imwrite("pano.jpg", warpImage)
    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--path", 
                        default="./image/file.txt",
                        help="Path to file list") 
    parser.add_argument("-w", 
                        "--win_size", 
                        default=15,
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