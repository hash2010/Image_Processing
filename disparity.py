# -*- coding: utf-8 -*-

"""
Created on Wed Mar 18 17:40:06 2020

@author: chand
"""

import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

#image read
img_l=plt.imread("stereocorridor_left.gif")
img_r=plt.imread("stereocorridor_right.gif")
plt.imshow(img_l)
plt.imshow(img_r)


#converting to gray scale
Ilef_gray = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
Irit_gray=cv.cvtColor(img_r,cv.COLOR_BGR2GRAY)

#calculation of keypoints & descriptors
kaze = cv.KAZE_create()
(kps1, des1) = kaze.detectAndCompute(Ilef_gray , None)
(kps2, des2) = kaze.detectAndCompute(Irit_gray, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kps2[m.trainIdx].pt)
        pts1.append(kps1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
Fundamental, inliers = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(Ilef_gray,Irit_gray)
plt.imshow(disparity,'gray')
plt.show()


