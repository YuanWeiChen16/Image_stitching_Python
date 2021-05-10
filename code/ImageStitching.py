from FeatureMatching import FeatureMatching as fts_match
import cv2, math
import numpy as np
import argparse as ap
from utils import read_image

def changeXY(H):
    tmp = H[0][2]
    H[0][2] = H[1][2]
    H[1][2] = tmp

    tmp = H[0][0]
    H[0][0] = H[1][1]
    H[1][1] = tmp
    
    tmp = H[0][1]
    H[0][1] = H[1][0]
    H[1][0] = tmp

    tmp = H[2][0]
    H[2][0] = H[2][1]
    H[2][1] = tmp
    return H

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

def crop(panorama, h_dst, conners):
   
    # find max min of x,y coordinate
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    t = [-xmin, -ymin]
    conners = conners.astype(int)

    # conners[0][0][0] is the X coordinate of top-left point of warped image
    # If it has value < 0, warp image is merged to the left side of destination image
    # otherwise is merged to the right side of destination image
    if conners[0][0][0] < 0:
        n = abs(-conners[1][0][0] + conners[0][0][0])
        panorama = panorama[t[1] : h_dst + t[1], n:, :]
    else:
        if conners[2][0][0] < conners[3][0][0]:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[2][0][0], :]
        else:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[3][0][0], :]
    return panorama

def warpTwoImages(src_img, dst_img, args):
    
    # generate Homography matrix
    H = fts_match(src_img, dst_img, args)    
    H = changeXY(H)
    print("Homography: ", H)
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
    
    # apply homography to conners of src_img
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
    pano = panoramaBlending(dst_img_rz, src_img_warped, width_dst, side)

    # croping black region
    pano = crop(pano, height_dst, pts)
    return pano

def multiStitching(list_images, args):

    n = int(len(list_images) / 2 + 0.5)
    left = list_images[:n]
    right = list_images[n - 1 :]
    right.reverse()
    while len(left) > 1:
        dst_img = left.pop()
        src_img = left.pop()
        left_pano = warpTwoImages(src_img, dst_img, args)
        left_pano = left_pano.astype("uint8")
        cv2.imwrite("leftpano.jpg", left_pano)
        left.append(left_pano)

    while len(right) > 1:
        dst_img = right.pop()
        src_img = right.pop()
        right_pano = warpTwoImages(src_img, dst_img, args)
        right_pano = right_pano.astype("uint8")
        cv2.imwrite("rightpano.jpg", right_pano)
        right.append(right_pano)

    fullpano = warpTwoImages(left_pano, right_pano, args)

    return fullpano

def StitchingFromLeft(list_images, args):
    all_img = list_images
    all_img.reverse()
    while len(all_img) > 1:
        src_img = all_img.pop()
        dst_img = all_img.pop()
        tmp_pano = warpTwoImages(src_img, dst_img, args)
        tmp_pano = tmp_pano.astype("uint8")
        all_img.append(tmp_pano)
    
    return all_img[0]

def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothing_window / 2)
    try:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset + 1).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset + 1).T, (height, 1)
            )
            mask[:, barrier + offset :] = 1
    except BaseException:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height, 1)
            )
            mask[:, barrier + offset :] = 1

    return cv2.merge([mask, mask, mask])
def panoramaBlending(dst_img_rz, src_img_warped, width_dst, side):

    h, w, _ = dst_img_rz.shape
    smoothing_window = int(width_dst / 8)
    barrier = width_dst - int(smoothing_window / 2)
    mask1 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=True
    )
    mask2 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=False
    )

    if side == "left":
        dst_img_rz = cv2.flip(dst_img_rz, 1)
        src_img_warped = cv2.flip(src_img_warped, 1)
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        pano = cv2.flip(pano, 1)

    else:
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz

    return pano

    
def main(args):
    filenames, images, count = read_imageList(args.path)
    print("Number of images : ", count)
    centerIdx = count / 2
    print("Center index image : ", centerIdx)

    # test stitching
    # warpImage = warpTwoImages(images[0], images[1], args)
    if args.fromMid:
        warpImage = multiStitching(images, args)
    else:
        warpImage = StitchingFromLeft(images, args)
    print("Finished")
    cv2.imwrite("pano.jpg", warpImage)
    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--path", 
                        default="./image/file.txt",
                        help="Path to file list") 
    parser.add_argument("-w", 
                        "--win_size", 
                        default=50,
                        type=int,
                        help="Window size for your feature detector algorithm") 
    
    parser.add_argument("-i", 
                        "--max_iters", 
                        default=2000,
                        type=int,
                        help="Maximum iterations to perform RANSAC") 
    
    parser.add_argument("-e", 
                        "--epsilon", 
                        default=0.5,
                        type=float,
                        help="Epsilon threshold for RANSAC") 

    parser.add_argument("-d", 
                        "--draw", 
                        default=False,
                        type=bool,
                        help="Draw Corner") 
    parser.add_argument("-m", 
                        "--fromMid", 
                        default=False,
                        type=bool,
                        help="Stitching from Mid") 
    args = parser.parse_args()
    main(args)