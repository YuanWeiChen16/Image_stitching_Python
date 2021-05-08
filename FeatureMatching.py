import cv2, math, pywt
import numpy as np
import argparse as ap
from HarrisCornerDetector import HarrisCorner, draw_corners
import sys
from utils import read_image, draw_matches, Timer, dlt
from operator import itemgetter

def harris(img, sigma=1, threshold=0.01):
    height, width = img.shape
    shape = (height, width)
    # Calculate the dx,dy gradients of the image (np.gradient doesnt work)
    dx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=5)
    # Get angle for rotation
    _, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    # Square the derivatives (A,B,C of H) and apply apply gaussian filters to each
    sigma = (sigma, sigma)
    Ixx = cv2.GaussianBlur(dx * dx, sigma, 0)
    Ixy = cv2.GaussianBlur(dx * dy, sigma, 0)
    Iyy = cv2.GaussianBlur(dy * dy, sigma, 0)

    H = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    # Find the determinate
    num = (H[0, 0] * H[1, 1]) - (H[0, 1] * H[1, 0])
    # Find the trace
    denom = H[0,0] + H[1,1]
    # Find the response using harmonic mean of the eigenvalues (Brown et. al. variation) 
    R = np.nan_to_num(num / denom)
    
    # Adaptive non-maximum suppression, keep the top 1% of values and remove non-maximum points in a 9x9 neighbourhood
    R_flat = R[:].flatten()
    # Get number of values in top threshold %
    N = int(len(R_flat) * threshold)
    # Get values in top threshold %
    top_k_percentile = np.partition(R_flat, -N)[-N:]
    # Find lowest value in top threshold %
    minimum = np.min(top_k_percentile)
    # Set all values less than this to 0
    R[R < minimum] = 0
    # Set non-maximum points in an SxS neighbourhood to 0
    s = 9
    for h in range(R.shape[0] - s):
        for w in range(R.shape[1] - s):
            maximum = np.max(R[h:h+s+1, w:w+s+1])
            for i in range(h, h+s+1):
                for j in range(w, w+s+1):
                    if R[i, j] != maximum:
                        R[i, j] = 0
                        
    # Return harris corners in [H, W, R] format
    features = list(np.where(R > 0))
    features.append(ang[np.where(R > 0)])
    corners = zip(*features)
    return list(corners)

def ransac(pts1, pts2, max_iter=1000, epsilon=1):
    matches = []
    # Each num of samples
    N = 4
    bestH = np.zeros((3, 3))
    for i in range(max_iter):
        idx = np.random.randint(0, len(pts1) - 1, N)
        src = pts1[idx]
        dst = pts2[idx]

        # Calculate the homography matrix H
        H = cv2.getPerspectiveTransform(src, dst)
        Hp = cv2.perspectiveTransform(pts1[None], H)[0]

        # find the inliers
        inliers = []
        for i in range(len(pts1)):
            ssd = np.sum(np.square(pts2[i] - Hp[i]))
            if ssd < epsilon:
                inliers.append([pts1[i], pts2[i]])

        if len(inliers) > len(matches):
            matches = inliers
            bestH = H
    return matches, bestH

# def ransac(pts1, pts2, N = 10, t = 0.9, threshold = 4.0):
#     total = len(pts1)
#     bestH = np.zeros((3, 3))
#     maxInliers = 0
#     matches = []

#     for i in range(N):
#         randFour = [np.random.randint(0, total) for i in range(4)]
#         p1 = pts1[randFour]
#         p2 = pts2[randFour]
#         H = dlt(p1, p2)

#         inlier = 0
#         match = []
#         for index in range(total):
#             src = pts1[index]
#             dst = np.array([pts2[index][0], pts2[index][1]])

#             pred = np.dot(H, np.array([src[0], src[1], 1]))
#             predx = pred[0] / pred[2]
#             predy = pred[1] / pred[2]
#             pred = np.array([predx, predy])
#             pred = np.float32([point for point in pred])

#             if np.linalg.norm(dst - pred) < threshold:
#                 inlier += 1
#                 match.append([pts1[index], pts2[index]])

        
#         if maxInliers < inlier:
#             maxInliers = inlier
#             bestH = H
#             matches = match
#             if inlier > t * total:
#                 break
#     return match, bestH

def mops(img, truth, win_size, h, w, r):
    height, width = img.shape
    offset = win_size // 2

    # for debugging
    # length = 150
    # h2 = int(height - length * math.cos(math.radians(r)))
    # w2 = int(width - length * math.sin(math.radians(r)))
    # cv2.line(img, (w, h), (w2, h2), (0, 255, 0), 2)

    # gradient angle of feature
    M = cv2.getRotationMatrix2D((w, h), -1 * r, 1)
    img_rot = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_NEAREST)

    # get 40x40 window around feature
    win = img_rot[h - offset:h + offset, w - offset:w + offset]

    # (s, s) of each sample region
    s = win_size // 8
    i = 0
    rows = []
    while i < win_size:
        j = 0
        cols = []
        while j < win_size:
            sample = win[i:i + s, j:j + s]
            sample = np.sum(sample) / (s * s)
            cols.append(sample)
            j += s
        rows.append(cols)
        i += s
    
    feature = np.array(rows)

    # normalize
    feature = (feature - np.mean(feature)) / np.std(feature)

    coeffs = pywt.dwt2(feature, 'haar')
    feature = pywt.idwt2(coeffs, 'haar')
    return feature

def get_matches(ft1, ft2, left_match, right_match, img_left, img_right, max_matches, win_size):
    matches = []
    offset = win_size // 2

    # brute-force matching
    for h1, w1, r1 in ft1:
        img_left_tmp = np.copy(img_left)
        h1_tmp = h1
        w1_tmp = w1

        # get 40x40 window around feature
        win_left = img_left[int(h1 - offset):int(h1 + offset), int(w1 - offset):int(w1 + offset)]

        # border
        if win_left.shape != (win_size, win_size):
            
            diff_h, diff_w = np.subtract(win_left.shape, (win_size, win_size))
            p_h = abs(diff_h // 2)
            p_w = abs(diff_w // 2)
            img_left_tmp = cv2.copyMakeBorder(img_left_tmp, p_h, p_h, p_w, p_w, cv2.BORDER_REFLECT)
            h1 += p_h
            w1 += p_w
            win_left = img_left_tmp[int(h1 - offset):int(h1 + offset), int(w1 - offset):int(w1 + offset)]

        # multiscale oriented patches 
        feature_left = mops(img_left_tmp, win_left, win_size, h1, w1, r1)

        lowest_dist = math.inf
        potential_match = ()
        for h2, w2, r2 in ft2:
            img_right_tmp = np.copy(img_right)
            h2_tmp = h2
            w2_tmp = w2

            # get 40x40 window around feature
            win_right = img_right[h2 - offset:h2 + offset, w2 - offset:w2 + offset]

            # border
            if win_right.shape != (win_size, win_size):
                diff_h, diff_w = np.subtract(win_right.shape, (win_size, win_size))
                p_h = abs(diff_h // 2)
                p_w = abs(diff_w // 2)
                img_right_tmp = cv2.copyMakeBorder(img_right_tmp, p_h, p_h, p_w, p_w, cv2.BORDER_REFLECT)
                h2 += p_h
                w2 += p_w
                win_right = img_right_tmp[h2 - offset: h2 + offset, w2 - offset: w2 + offset]

            # multiscale oriented patches 
            feature_right = mops(img_right_tmp, win_right, win_size, h2, w2, r2)

            # Check distance between features
            curr_dist = np.linalg.norm(feature_left - feature_right)
            if curr_dist < lowest_dist:
                lowest_dist = curr_dist
                potential_match = ([h1_tmp, w1_tmp, r1], [h2_tmp, w2_tmp, r2], curr_dist)
        
        matches.append(potential_match)
    
    # sort by smallest distance up
    matches = sorted(matches, key=itemgetter(2))
    for match in matches:
        # Ensure no duplicates
        if match[0][0:2] not in left_match and match[1][0:2] not in right_match:
            # add to matches
            left_match.append(match[0][0:2])
            right_match.append(match[1][0:2])
            # remove from matches
            ft1.remove(tuple(match[0]))
            ft2.remove(tuple(match[1]))
    
    # Recursively until every point has a match
    while (len(left_match) < max_matches and len(right_match) < max_matches):
        print('get matches')
        get_matches(ft1, ft2, left_match, right_match, img_left, img_right, max_matches, win_size)
    
    return np.array(left_match, dtype=np.float32), np.array(right_match, dtype=np.float32)

def FeatureMatching(img_left_clr, img_right_clr, win_size, max_iters, epsilon):
    img_left = cv2.cvtColor(img_left_clr, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right_clr, cv2.COLOR_RGB2GRAY)

    print("Getting the features from the Harris Corner Detector")
    ft_left = harris(img_left, sigma=3, threshold=0.01)
    #ft_left, _, _ = HarrisCorner(img_left)
    draw_corners(ft_left, img_left_clr, 'corners_left')
    ft_right = harris(img_right, sigma=3, threshold=0.01)
    #ft_right, _, _ = HarrisCorner(img_right)
    draw_corners(ft_right, img_right_clr, 'corners_right')
    print("Number of features (left): ", len(ft_left))
    print("Number of features (right): ", len(ft_right))
    print("Finding the best matches")
    with Timer(verbose=True) as t:
        max_matches = min(len(ft_left), len(ft_right))
        pts_left, pts_right = get_matches(ft_left, ft_right, [], [], img_left, img_right, max_matches, win_size)
        print("Number of matches = ", len(pts_left))
        assert len(pts_left) == len(pts_right)
    
    print("RANSAC")
    # matches, H = ransac(pts_left, pts_right, img_left, img_right, max_iters, epsilon)
    matches, H = ransac(pts_left, pts_right)
    print("Number of pruned matches = ", len(matches))
    # print("Homography :", H)
    draw_matches(matches, img_left_clr, img_right_clr)
    return H

def main(args):

    img_left_clr = read_image(args.imgleft)
    img_right_clr = read_image(args.imgright)

    img_left = cv2.cvtColor(img_left_clr, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right_clr, cv2.COLOR_RGB2GRAY)

    print("Getting the features from the Harris Corner Detector")
    ft_left = harris(img_left, sigma=3, threshold=0.01)
    #ft_left, _, _ = HarrisCorner(img_left)
    draw_corners(ft_left, img_left_clr, 'corners_left')
    ft_right = harris(img_right, sigma=3, threshold=0.01)
    #ft_right, _, _ = HarrisCorner(img_right)
    draw_corners(ft_right, img_right_clr, 'corners_right')
    print("Number of features (left): ", len(ft_left))
    print("Number of features (right): ", len(ft_right))
    print("Finding the best matches")
    with Timer(verbose=True) as t:
        max_matches = min(len(ft_left), len(ft_right))
        pts_left, pts_right = get_matches(ft_left, ft_right, [], [], img_left, img_right, max_matches, win_size = args.win_size)
        print("Number of matches = ", len(pts_left))
        assert len(pts_left) == len(pts_right)
    
    print("RANSAC")
    # matches, H = ransac(pts_left, pts_right, img_left, img_right, args.max_iters, args.epsilon)
    matches, H = ransac(pts_left, pts_right)
    print("Number of pruned matches = ", len(matches))
    print("Homography :", H)
    draw_matches(matches, img_left_clr, img_right_clr)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--imgleft", 
                        default="imgs/left.jpg",
                        help="Path to left image") 
    parser.add_argument("--imgright", 
                        default="imgs/right.jpg",
                        help="Path to right image") 
    parser.add_argument("-w", 
                        "--win_size", 
                        default=50,
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
   
    parser.add_argument("-v", 
                        "--verbose", 
                        help="Turn on debugging statements",
                        action="store_true") 

    args = parser.parse_args()
    main(args)