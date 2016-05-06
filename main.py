__author__ = 'lukas.bitter'
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


numHarrisCorner = 1
numSIFT = 2
numORB = 3




# find Harris corners
def harrisCorner(img, gray):
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    #cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def findSift(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp)
    cv2.imwrite('sift_keypoints.jpg',img)

def nothing(x):
    pass

def loop(img):
    src = img.copy()

    # create switch for ON/OFF harris functionality
    selFeature = 'Feature selection' #'0 : OFF \n1 : HARRIS \n2 : SIFT'
    paramFeature = 'Feature parameter'
    cv2.createTrackbar(selFeature, 'Image',0,5,nothing)
    cv2.createTrackbar(paramFeature, 'Image',0,14,nothing)
    msg = 'Original Image'


    #grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    param = None

    while(1):
        cv2.imshow('Image',img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

        # get current positions of  trackbars
        feature = cv2.getTrackbarPos(selFeature,'Image')
        newParam = cv2.getTrackbarPos(paramFeature,'Image')

        if newParam != param:
            param = newParam
            current = None

        if feature == 0:
            current = None
            img = src.copy()

        elif (feature == numHarrisCorner and not current == numHarrisCorner):
            current = numHarrisCorner
            img = src.copy()

            dst = cv2.cornerHarris(gray,2,2*param-1,0.04)

            #result is dilated for marking the corners, not important
            dst = cv2.dilate(dst,None)

            # Threshold for an optimal value, it may vary depending on the image.
            img[dst>0.01*dst.max()]=[0,0,255]


        elif (feature == numSIFT and not current == numSIFT):
            current = numSIFT
            img = src.copy()

            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp = sift.detect(gray,None)

            img=cv2.drawKeypoints(gray,kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        elif (feature == numORB and not current == numORB):
            current = numORB
            img = src.copy()

            # Initiate STAR detector
            orb = cv2.ORB()

            # find the keypoints with ORB
            kp = orb.detect(img,None)

            # compute the descriptors with ORB
            kp, des = orb.compute(img, kp)

            # draw only keypoints location,not size and orientation
            img = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)


if __name__ == '__main__':
    img = cv2.imread("leafs/01/RGB/1. Quercus suber/iPAD2_C01_EX01.JPG")
    cv2.namedWindow('Image')

    loop(img)

    cv2.destroyAllWindows()
